import unittest
import tempfile
from pathlib import Path

import pandas as pd

from syto.data.omics_signatures_handlers.binary_cpg_signature import (
    BinaryCpGSignatureHandler,
)
from syto.data.labelers.labeler_factory import LabelerContext
from syto.data.dataset_build.finalize import label_and_split, finalize_bucket

NUM_CLASSES = 3


def _context():
    h = BinaryCpGSignatureHandler(
        start_column="read_start", methylation_pattern_column="methylation_ids"
    )
    return LabelerContext(num_classes=NUM_CLASSES, signature_handler=h)


def _labelers_config():
    return {
        "soft_label": {
            "type": "data_driven_soft",
            "distance_name": "jaccard",
            "perform_pooling": True,
            "min_reads": 5,
            "max_distance": 0.7,
        },
        "label": {"type": "hard_with_background"},
    }


class TestLabelAndSplit(unittest.TestCase):
    def setUp(self):
        rows = []
        for _ in range(40):
            rows.append(
                {"name": "chr1:1-9", "read_start": 100, "methylation_ids": "01",
                 "original_label": 0, "dmr_ctype_label": 0, "file": "f0"}
            )
        for _ in range(40):
            rows.append(
                {"name": "chr1:1-9", "read_start": 100, "methylation_ids": "01",
                 "original_label": 1, "dmr_ctype_label": 0, "file": "f1"}
            )
        self.df = pd.DataFrame(rows)
        self.plan = {
            "file_level": {(0, "f0"): "train", (1, "f1"): "valid"},
            "read_level": {},
        }

    def test_adds_soft_label_hard_label_and_split(self):
        out = label_and_split(
            self.df,
            self.plan,
            labelers_config=_labelers_config(),
            context=_context(),
        )
        for col in ["soft_label", "label", "split"]:
            self.assertIn(col, out.columns)

    def test_hard_label_background_for_off_target(self):
        out = label_and_split(
            self.df,
            self.plan,
            labelers_config=_labelers_config(),
            context=_context(),
        )
        off = out[out["original_label"] == 1]
        self.assertTrue((off["label"] == NUM_CLASSES).all())


class TestFitSplits(unittest.TestCase):
    def test_valid_only_signature_labelled_from_train(self):
        # sigT appears in train (label 0); sigV appears only in valid (label 1).
        rows = []
        for _ in range(30):
            rows.append(
                {"name": "chr1:1-9", "read_start": 100, "methylation_ids": "00",
                 "original_label": 0, "dmr_ctype_label": 0, "file": "f0"}
            )
        for _ in range(30):
            rows.append(
                {"name": "chr1:1-9", "read_start": 100, "methylation_ids": "11",
                 "original_label": 1, "dmr_ctype_label": 0, "file": "f1"}
            )
        df = pd.DataFrame(rows)
        plan = {"file_level": {(0, "f0"): "train", (1, "f1"): "valid"}, "read_level": {}}
        labelers = {
            "soft_label": {
                "type": "data_driven_soft",
                "distance_name": "jaccard",
                "perform_pooling": False,
                "min_reads": 1,
                "max_distance": 0.0,
            }
        }
        out = label_and_split(
            df, plan, labelers_config=labelers, context=_context(),
            fit_splits=["train"], fallback="uniform",
        )
        valid_rows = out[out["split"] == "valid"]
        # sigV never appears in train -> uniform fallback over 3 classes
        for lab in valid_rows["soft_label"]:
            self.assertAlmostEqual(max(lab), 1.0 / NUM_CLASSES, places=6)


class TestFinalizeBucketFilter(unittest.TestCase):
    def test_filters_by_region_and_pattern_length(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            staged = d / "staged" / "region_bucket=0"
            staged.mkdir(parents=True)
            df = pd.DataFrame(
                {
                    "name": ["keep", "drop_region", "keep"],
                    "read_start": [100, 100, 100],
                    "methylation_ids": ["0101", "0101", "0"],
                    "original_label": [0, 0, 0],
                    "dmr_ctype_label": [0, 0, 0],
                    "file": ["f0", "f0", "f0"],
                }
            )
            df.to_parquet(staged / "s.parquet", index=False)
            plan = {"file_level": {(0, "f0"): "train"}, "read_level": {}}
            out_path = finalize_bucket(
                str(d / "staged"), 0, str(d / "final"), plan,
                num_classes=NUM_CLASSES,
                labelers_config={"label": {"type": "hard_with_background"}},
                signature_config={
                    "start_column": "read_start",
                    "methylation_pattern_column": "methylation_ids",
                },
                region_names={"keep"},
                min_pattern_length=2,
                pattern_column="methylation_ids",
            )
            res = pd.read_parquet(out_path)
            # only the first row survives (region "keep" AND length 4 >= 2)
            self.assertEqual(len(res), 1)
            self.assertEqual(res.iloc[0]["name"], "keep")

    def test_empty_bucket_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            staged = d / "staged" / "region_bucket=0"
            staged.mkdir(parents=True)
            pd.DataFrame(
                {
                    "name": ["drop"], "read_start": [100],
                    "methylation_ids": ["0101"], "original_label": [0],
                    "dmr_ctype_label": [0], "file": ["f0"],
                }
            ).to_parquet(staged / "s.parquet", index=False)
            plan = {"file_level": {(0, "f0"): "train"}, "read_level": {}}
            out = finalize_bucket(
                str(d / "staged"), 0, str(d / "final"), plan,
                num_classes=NUM_CLASSES,
                labelers_config={"label": {"type": "hard_with_background"}},
                signature_config={
                    "start_column": "read_start",
                    "methylation_pattern_column": "methylation_ids",
                },
                region_names={"keep"},
                pattern_column="methylation_ids",
            )
            self.assertIsNone(out)
