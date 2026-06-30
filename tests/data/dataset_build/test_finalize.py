import unittest
import pandas as pd

from syto.data.omics_signatures_handlers.binary_cpg_signature import (
    BinaryCpGSignatureHandler,
)
from syto.data.labelers.labeler_factory import LabelerContext
from syto.data.dataset_build.finalize import label_and_split

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
