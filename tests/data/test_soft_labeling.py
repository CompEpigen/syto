import unittest
import numpy as np
import pandas as pd
from syto.data.soft_labeling import (
    extract_cpg_signature,
    signature_distance,
    apply_normalized_knn_smoothing,
)


class TestExtractCpgSignature(unittest.TestCase):
    def test_basic(self):
        row = {"trimmed_start": 100, "pattern": "01201"}
        # 2 is an unknown status, should be ignored
        sig = extract_cpg_signature(row)
        self.assertEqual(sig, ((100, 0), (101, 1), (103, 0), (104, 1)))


class TestSignatureDistance(unittest.TestCase):
    def test_one_mismatch(self):
        sig1 = ((100, 0), (101, 1))
        sig2 = ((100, 0), (101, 0))  # 1 mismatch, length 2, both share 2
        # Mismatch rate: 1/2 = 0.5. Length penalty: 1.0 - (2 / 2) = 0
        # Total distance: 0.5
        self.assertAlmostEqual(signature_distance(sig1, sig2), 2 / 3)

    def test_length_penalty(self):
        sig1 = ((100, 0), (101, 1))
        sig3 = (
            (100, 0),
        )  # Share 1 length 1, max length 2. Mismatch: 0/1. Penalty: 1 - 1/2 = 0.5
        self.assertEqual(signature_distance(sig1, sig3), 0.5)

    def test_no_shared_positions(self):
        sig1 = ((100, 0), (101, 1))
        sig4 = ((200, 0),)  # No shared. Distance 1.0
        self.assertEqual(signature_distance(sig1, sig4), 1.0)


class TestApplyNormalizedKnnSmoothing(unittest.TestCase):
    def setUp(self):
        sig1 = ((100, 0), (101, 1))
        sig2 = ((100, 0), (101, 0))
        sig3 = ((200, 0),)

        row1 = {"name": "RegionA", "cpg_sig": sig1, "total_reads": 20}
        for i in range(40):
            row1[i] = 0
        row1[0] = 20

        row2 = {"name": "RegionA", "cpg_sig": sig2, "total_reads": 15}
        for i in range(40):
            row2[i] = 0
        row2[1] = 15

        row3 = {"name": "RegionA", "cpg_sig": sig3, "total_reads": 50}
        for i in range(40):
            row3[i] = 0
        row3[2] = 50

        self.sig1 = sig1
        self.df = pd.DataFrame([row1, row2, row3])

    def test_result_length(self):
        res = apply_normalized_knn_smoothing(
            self.df, min_reads=30, max_distance=0.5, num_classes=40
        )
        self.assertEqual(len(res), 3)

    def test_pooled_reads_and_signatures(self):
        res = apply_normalized_knn_smoothing(
            self.df, min_reads=30, max_distance=0.7, num_classes=40
        )
        res1 = res[res["cpg_sig"] == self.sig1].iloc[0]
        self.assertEqual(res1["raw_reads_pooled"], 35)
        self.assertEqual(res1["num_signatures_pooled"], 2)

    def test_soft_label_normalization(self):
        # global counts: c0=20, c1=15, c2=50 -> median=20
        # weight: c0=1.0, c1=1.333, c2=0.4
        # accumulated for sig1: [20, 15, 0] -> normalized: ~[20, 20, 0] -> prob: ~0.5 for c0, 0.5 for c1
        res = apply_normalized_knn_smoothing(
            self.df, min_reads=30, max_distance=0.7, num_classes=40
        )
        res1 = res[res["cpg_sig"] == self.sig1].iloc[0]
        prob = res1["soft_label"]
        self.assertAlmostEqual(prob[0], 0.5, delta=0.05)
        self.assertAlmostEqual(prob[1], 0.5, delta=0.05)


if __name__ == "__main__":
    unittest.main()
