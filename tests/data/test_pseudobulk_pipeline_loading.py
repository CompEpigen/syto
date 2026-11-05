import json
import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from syto.app.pseudobulk_pipeline import PseudoBulkPipeline


class _StubClassifier:
    # Stands in for a real classifier, which inherits both hooks from
    # AbstractReadClassifier. The pseudobulk generator only predicts, so it
    # asks for the predict-time columns.
    def required_fit_columns(self, config):
        return ["input_ids", "methylation_ids", "dmr_ctype_label"]

    def required_predict_columns(self, config):
        return self.required_fit_columns(config)


def _make_logger():
    import logging

    logger = logging.getLogger("test_pseudobulk_loading")
    logger.addHandler(logging.NullHandler())
    return logger


class TestRawSplitsLoading(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

        # Columnar dataset with two splits and no NCPGS column.
        ds = self.root / "dataset"
        part = ds / "region_bucket=0"
        part.mkdir(parents=True)
        pd.DataFrame(
            {
                "input_ids": ["seqT", "seqV"],
                "methylation_ids": ["0101", "1"],
                "dmr_ctype_label": [3, 3],
                "original_label": [3, 5],
                "chromosome": ["chr1", "chr1"],
                "read_start": [100, 200],
                "read_end": [150, 250],
                "soft_label_other": [[0.1, 0.9], [0.5, 0.5]],
                "split": ["train", "valid"],
                "region_bucket": [0, 0],
            }
        ).to_parquet(part / "part-0.parquet")

        labels_path = self.root / "labels.json"
        labels_path.write_text(json.dumps({"0": "a", "1": "b"}))

        self.config = {
            "input_type": "raw_splits",
            "data_path": str(ds),
            "data_format": "auto",
            "split_column": "split",
            "labels_dict_path": str(labels_path),
            "num_labels": 2,
            "output_dir": str(self.root / "out"),
            "batch_size": 10,
            "n_workers": 1,
            "n_reads_to_sample": 100,
            "class_label_column": "original_label",
            "grg_label_column": "dmr_ctype_label",
            "classifier_type": "dismir",
            "classifier_checkpoint": "unused",
            "classifier_config": {
                "grg_label_column": "dmr_ctype_label",
                "num_prediction_classes": 2,
            },
            "split_information": {
                "train": {"target_proportions_path": "unused_train.npz"},
                "valid": {"target_proportions_path": "unused_valid.npz"},
            },
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_raw_splits_projects_and_derives_ncpgs(self):
        pipeline = PseudoBulkPipeline(self.config, _make_logger())
        pipeline._classifier = _StubClassifier()  # avoid loading a real checkpoint

        splits = pipeline._load_raw_splits()

        self.assertEqual(set(splits), {"train", "valid"})
        train = splits["train"]
        self.assertEqual(len(train), 1)
        self.assertNotIn("soft_label_other", train.columns)
        self.assertEqual(train["NCPGS"].iloc[0], 4)


if __name__ == "__main__":
    unittest.main()


class TestPipelineGeneratedProportions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        labels = self.root / "labels.json"
        labels.write_text(json.dumps({"0": "a", "1": "b", "2": "c"}))
        self.base = {
            "labels_dict_path": str(labels),
            "batch_size": 10,
            "n_workers": 1,
            "n_reads_to_sample": 100,
            "class_label_column": "original_label",
            "grg_label_column": "dmr_ctype_label",
            "classifier_config": {"num_prediction_classes": 3},
        }
        self.logger = logging.getLogger("test_target_proportions")
        self.logger.addHandler(logging.NullHandler())

    def tearDown(self):
        self.tmp.cleanup()

    def _pipeline(self, out, split_information):
        cfg = {
            **self.base,
            "output_dir": str(self.root / out),
            "split_information": split_information,
        }
        return PseudoBulkPipeline(cfg, self.logger)

    def test_generated_proportions_are_saved_and_reusable(self):
        gen = {
            "train": {"generate_proportions": {"n_pseudobulks": 50, "seed": 1}},
            "valid": {"generate_proportions": {"n_pseudobulks": 20, "seed": 2}},
        }
        first = self._pipeline("run1", gen)._load_target_proportions()
        self.assertEqual(first["train"].shape, (50, 3))
        self.assertEqual(first["valid"].shape, (20, 3))
        self.assertFalse(np.array_equal(first["train"][:20], first["valid"]))

        saved = {s: self.root / "run1" / f"target_proportions_{s}.npz" for s in gen}
        reuse = {s: {"target_proportions_path": str(p)} for s, p in saved.items()}
        second = self._pipeline("run2", reuse)._load_target_proportions()
        for split in gen:
            np.testing.assert_array_equal(first[split], second[split])

    def test_existing_file_with_other_proportions_is_refused(self):
        gen = {"train": {"generate_proportions": {"n_pseudobulks": 50, "seed": 1}}}
        self._pipeline("run1", gen)._load_target_proportions()
        changed = {"train": {"generate_proportions": {"n_pseudobulks": 50, "seed": 3}}}
        with self.assertRaisesRegex(ValueError, "target_proportions_train.npz"):
            self._pipeline("run1", changed)._load_target_proportions()

    def test_regenerating_identical_proportions_resumes(self):
        gen = {"train": {"generate_proportions": {"n_pseudobulks": 50, "seed": 1}}}
        a = self._pipeline("run1", gen)._load_target_proportions()
        b = self._pipeline("run1", gen)._load_target_proportions()
        np.testing.assert_array_equal(a["train"], b["train"])


class TestPipelineCoverageSplits(unittest.TestCase):
    """Splits sampling one pre-predicted source at their own depths."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        labels = self.root / "labels.json"
        labels.write_text(json.dumps({"0": "a", "1": "b", "2": "c"}))

        rng = np.random.default_rng(0)
        n = 3 * 3 * 20
        reads = pd.DataFrame(
            {
                "original_label": np.repeat(np.arange(3), n // 3),
                "dmr_ctype_label": np.tile(np.repeat(np.arange(3), 20), 3),
                "NCPGS": rng.integers(1, 9, n),
            }
        )
        for i in range(3):
            reads[f"prediction_{i}"] = rng.random(n)
        self.data_path = self.root / "test_predicted.pkl"
        reads.to_pickle(self.data_path)
        proportions = self.root / "proportions.npz"
        np.savez(proportions, proportions=rng.dirichlet(np.ones(3), size=4))

        self.config = {
            "input_type": "pre_predicted",
            "output_format": "columnar",
            "labels_dict_path": str(labels),
            "batch_size": 2,
            "n_workers": 1,
            "n_reads_to_sample": 300,
            "class_label_column": "original_label",
            "grg_label_column": "dmr_ctype_label",
            "classifier_type": "dismir",
            "classifier_config": {"num_prediction_classes": 3},
            "substitution_method": "uniform_number",
            "grg_sampling_method": "uniform_multinomial",
            "split_information": {
                name: {
                    "source_split": "test",
                    "data_path": str(self.data_path),
                    "target_proportions_path": str(proportions),
                    **({"n_reads_to_sample": n_reads} if n_reads else {}),
                }
                for name, n_reads in (("test_5", 5), ("test_300", None))
            },
        }
        self.logger = logging.getLogger("test_coverage_splits")
        self.logger.addHandler(logging.NullHandler())

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, out, **overrides):
        cfg = {**self.config, "output_dir": str(self.root / out), **overrides}
        return PseudoBulkPipeline(cfg, self.logger).run()

    def test_sources_load_once_and_splits_get_their_depth(self):
        from syto.data.pseudobulk_store import open_pseudobulk_store

        pipeline = PseudoBulkPipeline(
            {**self.config, "output_dir": str(self.root / "run")}, self.logger
        )
        self.assertEqual(list(pipeline._load_splits()), ["test"])
        self.assertEqual(pipeline._n_reads_per_split(), {"test_5": 5, "test_300": 300})

        reader = open_pseudobulk_store(self._run("run"))
        self.assertEqual(sorted(reader.list_splits()), ["test_300", "test_5"])
        for split, n_reads in (("test_5", 5), ("test_300", 300)):
            for _, counts, _ in reader.iter_pseudobulk_params(split):
                # Each cell type's share is floored, so up to n_classes - 1
                # reads are lost.
                self.assertIn(int(counts.sum()), range(n_reads - 2, n_reads + 1))

    def test_splits_of_one_source_must_agree_on_data_path(self):
        self.config["split_information"]["test_5"]["data_path"] = "other.pkl"
        pipeline = PseudoBulkPipeline(
            {**self.config, "output_dir": str(self.root / "run")}, self.logger
        )
        with self.assertRaisesRegex(ValueError, "different data_path"):
            pipeline._load_splits()

    def test_prior_comes_from_the_configured_store(self):
        from syto.data.pseudobulk_store import open_pseudobulk_store

        reference = self._run("reference")
        blended = self._run(
            "blended",
            substitution_method="prior_blending",
            uniform_prior_source={"path": str(reference), "split": "test_300"},
        )

        # 5 reads over 3 cell types x 3 GR groups leave some groups empty;
        # those rows equal the reference prior exactly.
        prior = open_pseudobulk_store(reference).read_uniform_prior("test_300")
        expected = prior[[f"prediction_{i}_wavg" for i in range(3)]].to_numpy()
        features, _ = open_pseudobulk_store(blended).read_pseudobulk_matrices(
            "test_5", 3
        )
        reader = open_pseudobulk_store(blended)
        n_empty = 0
        for pb, (_, counts, _) in enumerate(reader.iter_pseudobulk_params("test_5")):
            for grg in np.flatnonzero(counts.sum(axis=0) == 0):
                np.testing.assert_allclose(features[pb, grg], expected[grg])
                n_empty += 1
        self.assertGreater(n_empty, 0)
