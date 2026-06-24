import unittest
import pandas as pd
from syto.data.omics_signatures_handlers.binary_cpg_signature import BinaryCpGSignatureHandler
from syto.data.labelers.data_driven_soft_labeler import DataDrivenSoftLabeler
from syto.data.dataset_build.finalize import label_and_split

NUM_CLASSES = 3


def _soft_labeler():
    h = BinaryCpGSignatureHandler(start_column="read_start", methylation_pattern_column="methylation_ids")
    return DataDrivenSoftLabeler(distance_name="jaccard", signature_handler=h)


class TestLabelAndSplit(unittest.TestCase):
    def setUp(self):
        rows = []
        for i in range(40):
            rows.append({"name": "chr1:1-9", "read_start": 100, "methylation_ids": "01",
                         "original_label": 0, "dmr_ctype_label": 0, "file": "f0"})
        for i in range(40):
            rows.append({"name": "chr1:1-9", "read_start": 100, "methylation_ids": "01",
                         "original_label": 1, "dmr_ctype_label": 0, "file": "f1"})
        self.df = pd.DataFrame(rows)
        self.plan = {"file_level": {(0, "f0"): "train", (1, "f1"): "valid"}, "read_level": {}}

    def test_adds_soft_label_hard_label_and_split(self):
        out = label_and_split(self.df, self.plan, num_classes=NUM_CLASSES,
                              soft_labeler=_soft_labeler(), min_reads=5, max_distance=0.7)
        for col in ["soft_label", "label", "split"]:
            self.assertIn(col, out.columns)

    def test_hard_label_background_for_off_target(self):
        # label 1 reads sit in a region whose dmr_ctype_label is 0 -> background class (NUM_CLASSES).
        out = label_and_split(self.df, self.plan, num_classes=NUM_CLASSES,
                              soft_labeler=_soft_labeler(), min_reads=5, max_distance=0.7)
        off = out[out["original_label"] == 1]
        self.assertTrue((off["label"] == NUM_CLASSES).all())
