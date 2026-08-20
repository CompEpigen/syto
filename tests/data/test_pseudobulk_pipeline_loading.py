import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from syto.app.pseudobulk_pipeline import PseudoBulkPipeline


class _StubClassifier:
    def required_fit_columns(self, config):
        return ["input_ids", "methylation_ids", "dmr_ctype_label"]


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
            "classifier_config": {"grg_label_column": "dmr_ctype_label", 
                                  "num_prediction_classes":2,},
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
