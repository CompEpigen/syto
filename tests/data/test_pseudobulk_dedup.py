"""De-duplicated pseudobulk generation: loading, count-weighted sampling, block prediction."""

import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from syto.app.pseudobulk_pipeline import PseudoBulkPipeline
from syto.data.pseudobulk_generator import (
    PseudobulkGenerator,
    _sample_read_ids_from_grouped_dataframe,
)
from syto.data.pseudobulk_input import load_raw_splits


def _reads(name, start, seq, pattern, label, grg, split, n, file="f0"):
    return pd.DataFrame(
        {
            "name": [name] * n,
            "read_start": [start] * n,
            "read_end": [start + len(seq)] * n,
            "chromosome": ["chr1"] * n,
            "input_ids": [seq] * n,
            "methylation_ids": [pattern] * n,
            "original_label": [label] * n,
            "dmr_ctype_label": [grg] * n,
            "file": [file] * n,
            "split": [split] * n,
        }
    )


class TestDeduplicatedLoading(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        part = self.dir / "region_bucket=0"
        part.mkdir()
        pd.concat(
            [
                _reads("r1", 10, "ACGTACGT", "0101", 0, 0, "train", 3, file="a"),
                _reads(
                    "r1", 10, "ACGTACGT", "0101", 0, 0, "train", 2, file="b"
                ),  # other sample
                _reads(
                    "r1", 10, "ACGTACGT", "0101", 5, 0, "train", 4
                ),  # other cell type
                _reads(
                    "r1", 10, "ACGTACGT", "0101", 0, 7, "train", 1
                ),  # shared region copy
                _reads("r1", 10, "ACGTACGT", "1", 0, 0, "train", 9),  # filtered out
                _reads("r1", 10, "ACGTACGT", "0101", 0, 0, "valid", 6),
            ]
        ).to_parquet(part / "part-0.parquet", index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def _load(self, **kwargs):
        return load_raw_splits(
            self.dir,
            ["train", "valid"],
            classifier_required_columns=[
                "input_ids",
                "methylation_ids",
                "dmr_ctype_label",
            ],
            min_pattern_length=2,
            **kwargs,
        )

    def test_identical_reads_collapse_across_samples(self):
        train = self._load(deduplicate=True)["train"]
        self.assertNotIn("file", train.columns)
        key = (train.original_label == 0) & (train.dmr_ctype_label == 0)
        self.assertEqual(train.loc[key, "read_count"].tolist(), [5])

    def test_class_and_grg_labels_keep_rows_apart(self):
        train = self._load(deduplicate=True)["train"]
        got = sorted(zip(train.original_label, train.dmr_ctype_label, train.read_count))
        self.assertEqual(got, [(0, 0, 5), (0, 7, 1), (5, 0, 4)])

    def test_counts_preserve_the_filtered_read_total(self):
        splits = self._load(deduplicate=True)
        plain = self._load()
        for split in ("train", "valid"):
            self.assertEqual(int(splits[split].read_count.sum()), len(plain[split]))
        self.assertIn("NCPGS", splits["train"].columns)

    def test_disabled_by_default(self):
        train = self._load()["train"]
        self.assertNotIn("read_count", train.columns)
        self.assertEqual(len(train), 10)


class TestCountWeightedSampling(unittest.TestCase):
    def test_rows_are_drawn_proportionally_to_counts(self):
        indices = {(0, 0): np.array([10, 11, 12])}
        counts = np.array([1.0, 3.0, 6.0])
        n = np.array([[200_000]])
        ids = _sample_read_ids_from_grouped_dataframe(
            n, indices, seed=0, count_cdf_per_class_and_grg={(0, 0): np.cumsum(counts)}
        )
        freq = np.array([(ids == i).mean() for i in (10, 11, 12)])
        np.testing.assert_allclose(freq, counts / counts.sum(), atol=0.005)

    def test_without_counts_sampling_is_unchanged(self):
        indices = {(0, 0): np.array([1, 2, 3]), (0, 1): np.array([4, 5])}
        n = np.array([[5, 3]])
        rng = np.random.Generator(np.random.PCG64(seed=7))
        expected = np.concatenate(
            [rng.choice(indices[(0, 0)], 5), rng.choice(indices[(0, 1)], 3)]
        )
        np.testing.assert_array_equal(
            _sample_read_ids_from_grouped_dataframe(n, indices, seed=7), expected
        )

    def test_deduplicated_pseudobulk_matches_expanded_one(self):
        """Aggregated features of de-duplicated vs expanded reads agree in expectation."""
        rng = np.random.default_rng(1)
        n_rows, n_classes, n_grg = 60, 3, 2
        dedup = pd.DataFrame(
            {
                "original_label": rng.integers(0, n_classes, n_rows),
                "dmr_ctype_label": rng.integers(0, n_grg, n_rows),
                "NCPGS": rng.integers(2, 9, n_rows),
                "read_count": rng.integers(1, 20, n_rows),
                **{f"prediction_{i}": rng.random(n_rows) for i in range(n_classes)},
            }
        )
        expanded = dedup.loc[dedup.index.repeat(dedup.read_count)].drop(
            columns="read_count"
        )
        expanded = expanded.reset_index(drop=True)

        def mean_features(df, n_pseudobulks=300):
            gen = PseudobulkGenerator.__new__(PseudobulkGenerator)
            gen.splits_df = {"s": df}
            gen.class_label_column, gen.grg_label_column = (
                "original_label",
                "dmr_ctype_label",
            )
            gen.num_prediction_classes = n_classes
            gen.parameters = MagicMock(gr_groups_mapping={})
            gen.logger = logging.getLogger("t")
            gen._precompute_group_indices()
            gen._preextract_numpy_arrays()
            np.random.seed(0)
            feats = [
                PseudobulkGenerator.generate_single_pseudobulk(
                    n_reads_to_sample=3_000,
                    n_gr_groups=n_grg,
                    n_classes=n_classes,
                    indices_per_class_and_grg=gen._indices_per_split["s"],
                    numpy_arrays=gen._numpy_arrays_per_split["s"],
                    target_proportions=np.full(n_classes, 1 / n_classes),
                    grg_label_column="dmr_ctype_label",
                )
                .aggregated_features[[f"prediction_{i}_wavg" for i in range(n_classes)]]
                .to_numpy()
                for _ in range(n_pseudobulks)
            ]
            return np.mean(feats, axis=0)

        np.testing.assert_allclose(
            mean_features(dedup), mean_features(expanded), atol=0.01
        )


class TestBlockPrediction(unittest.TestCase):
    def _pipeline(self, block):
        pipeline = PseudoBulkPipeline.__new__(PseudoBulkPipeline)
        pipeline.prediction_block_size = block
        pipeline.logger = logging.getLogger("t")
        return pipeline

    @staticmethod
    def _classifier():
        def predict_split(df, **_):
            assert "read_name" in df.columns
            return df.assign(prediction_0=df["x"] * 2.0)

        clf = MagicMock()
        clf.predict_split.side_effect = predict_split
        return clf

    def test_blocks_reproduce_a_single_call(self):
        df = pd.DataFrame({"x": np.arange(10), "read_name": np.arange(10)})
        whole = self._pipeline(None)._predict_in_blocks(self._classifier(), df, {})
        clf = self._classifier()
        blocks = self._pipeline(3)._predict_in_blocks(clf, df, {})
        self.assertEqual(clf.predict_split.call_count, 4)
        pd.testing.assert_frame_equal(blocks, whole)

    def test_reordered_block_output_is_placed_by_read_name(self):
        clf = MagicMock()
        clf.predict_split.side_effect = lambda df, **_: df.assign(
            prediction_0=df["x"] * 2.0
        ).iloc[::-1]
        df = pd.DataFrame({"x": np.arange(10.0)})
        out = self._pipeline(4)._predict_in_blocks(clf, df, {})
        np.testing.assert_array_equal(out["prediction_0"], np.arange(10.0) * 2)

    def test_reads_without_prediction_are_dropped(self):
        clf = MagicMock()
        clf.predict_split.side_effect = lambda df, **_: df.assign(
            prediction_0=df["x"] * 2.0
        )[df["x"] != 5]
        df = pd.DataFrame({"x": np.arange(10.0)})
        out = self._pipeline(4)._predict_in_blocks(clf, df, {})
        self.assertNotIn(5.0, out["x"].tolist())
        self.assertEqual(len(out), 9)
        np.testing.assert_array_equal(out["prediction_0"], out["x"] * 2)

    def test_read_names_are_unique_across_blocks(self):
        df = pd.DataFrame({"x": np.arange(7)})
        out = self._pipeline(2)._predict_in_blocks(self._classifier(), df, {})
        self.assertTrue(out["read_name"].is_unique)
        self.assertEqual(len(out), 7)


if __name__ == "__main__":
    unittest.main()
