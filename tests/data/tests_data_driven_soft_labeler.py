import unittest

import numpy as np
import pandas as pd

from syto.data.labelers.data_driven_soft_labeler import DataDrivenSoftLabeler
from syto.data.omics_signatures_handlers.binary_cpg_signature import (
    BinaryCpGSignatureHandler,
)

NUM_CLASSES = 3


def _make_labeler():
    """Build a DataDrivenSoftLabeler backed by a binary CpG / jaccard handler."""
    handler = BinaryCpGSignatureHandler(
        start_column="trimmed_start", methylation_pattern_column="pattern"
    )
    return DataDrivenSoftLabeler(distance_name="jaccard", signature_handler=handler)


class TestComputeLabelsPrecomputedSignatures(unittest.TestCase):
    """Tests using a precomputed signature column.

    Layout (1 region "RegionA")::

        sig1 = ((100, 0), (101, 1)) -> 20 reads label 0
        sig2 = ((100, 0), (101, 0)) -> 15 reads label 1
        sig3 = ((200, 0),)          -> 50 reads label 2

    Global class balance: {0: 20, 1: 15, 2: 50} -> median = 20.
    Weights: c0=1.0, c1=20/15, c2=0.4.
    """

    def setUp(self):
        """Build a 3-signature, 3-class fixture with a precomputed 'sig' column."""
        self.sig1 = ((100, 0), (101, 1))
        self.sig2 = ((100, 0), (101, 0))
        self.sig3 = ((200, 0),)

        rows = []
        for _ in range(20):
            rows.append({"name": "RegionA", "sig": self.sig1, "original_label": 0})
        for _ in range(15):
            rows.append({"name": "RegionA", "sig": self.sig2, "original_label": 1})
        for _ in range(50):
            rows.append({"name": "RegionA", "sig": self.sig3, "original_label": 2})
        self.df = pd.DataFrame(rows)
        self.labeler = _make_labeler()

    def test_rows_preserved(self):
        """Every input read is kept and a soft_label column is produced."""
        res = self.labeler.compute_labels(
            self.df,
            min_reads=30,
            max_distance=0.7,
            num_classes=NUM_CLASSES,
            precomputed_signature_column="sig",
        )
        self.assertEqual(len(res), len(self.df))
        self.assertIn("soft_label", res.columns)

    def test_signature_column_matches_precomputed(self):
        """With a precomputed column, 'signature' equals that column verbatim."""
        res = self.labeler.compute_labels(
            self.df,
            min_reads=30,
            max_distance=0.7,
            num_classes=NUM_CLASSES,
            precomputed_signature_column="sig",
        )
        self.assertTrue((res["signature"] == res["sig"]).all())

    def test_pooled_reads_and_signatures(self):
        """sig1 pools its nearest neighbor sig2 to reach the min_reads threshold."""
        res = self.labeler.compute_labels(
            self.df,
            min_reads=30,
            max_distance=0.7,
            num_classes=NUM_CLASSES,
            precomputed_signature_column="sig",
        )
        res1 = res[res["signature"] == self.sig1].iloc[0]
        # sig1 (20 reads) pools its nearest neighbor sig2 (15 reads) -> 35 reads
        self.assertEqual(res1["raw_counts_pooled"], 35)
        self.assertEqual(res1["num_signatures_pooled"], 2)

    def test_soft_label_normalization_with_pooling(self):
        """Global class-frequency weighting balances the pooled c0/c1 counts."""
        res = self.labeler.compute_labels(
            self.df,
            min_reads=30,
            max_distance=0.7,
            num_classes=NUM_CLASSES,
            precomputed_signature_column="sig",
        )
        res1 = res[res["signature"] == self.sig1].iloc[0]
        prob = res1["soft_label"]
        # Global counts {0:20, 1:15, 2:50} -> median 20 -> weights c0=1.0, c1=20/15.
        # Pooled raw counts for sig1 are [20, 15, 0]; after weighting both classes
        # contribute ~20, so the normalized probabilities are ~[0.5, 0.5, 0].
        self.assertAlmostEqual(prob[0], 0.5, delta=0.05)
        self.assertAlmostEqual(prob[1], 0.5, delta=0.05)
        self.assertAlmostEqual(prob[2], 0.0, delta=0.05)

    def test_soft_labels_sum_to_one(self):
        """Each soft label is a valid probability distribution (sums to 1)."""
        res = self.labeler.compute_labels(
            self.df,
            min_reads=30,
            max_distance=0.7,
            num_classes=NUM_CLASSES,
            precomputed_signature_column="sig",
        )
        sums = res["soft_label"].apply(sum)
        np.testing.assert_allclose(sums.values, 1.0, atol=1e-6)

    def test_no_pooling_uses_raw_counts(self):
        """With pooling disabled, sig1 keeps only its own class-0 reads."""
        res = self.labeler.compute_labels(
            self.df,
            perform_pooling=False,
            num_classes=NUM_CLASSES,
            precomputed_signature_column="sig",
        )
        res1 = res[res["signature"] == self.sig1].iloc[0]
        prob = res1["soft_label"]
        # Without pooling sig1 only has class-0 reads -> one-hot on class 0
        self.assertAlmostEqual(prob[0], 1.0, delta=1e-6)
        self.assertAlmostEqual(prob[1], 0.0, delta=1e-6)
        self.assertAlmostEqual(prob[2], 0.0, delta=1e-6)

    def test_keep_intermediate_values_false(self):
        """keep_intermediate_values=False drops the raw/weighted count columns."""
        res = self.labeler.compute_labels(
            self.df,
            min_reads=30,
            max_distance=0.7,
            num_classes=NUM_CLASSES,
            keep_intermediate_values=False,
            precomputed_signature_column="sig",
        )
        self.assertNotIn("raw_counts_0", res.columns)
        self.assertNotIn("weighted_counts_0", res.columns)
        self.assertIn("soft_label", res.columns)

    def test_keep_intermediate_values_true_has_columns(self):
        """keep_intermediate_values=True exposes raw/weighted/pooled columns."""
        res = self.labeler.compute_labels(
            self.df,
            min_reads=30,
            max_distance=0.7,
            num_classes=NUM_CLASSES,
            precomputed_signature_column="sig",
        )
        for c in range(NUM_CLASSES):
            self.assertIn(f"raw_counts_{c}", res.columns)
            self.assertIn(f"weighted_counts_{c}", res.columns)
        self.assertIn("raw_counts_pooled", res.columns)
        self.assertIn("num_signatures_pooled", res.columns)


class TestComputeLabelsSignatureExtraction(unittest.TestCase):
    """Tests where signatures are computed from patterns by the handler."""

    def setUp(self):
        """Build a fixture of raw reads whose signatures the handler will extract."""
        rows = []
        # pattern "01" at start 100 -> ((100, 0), (101, 1))
        for _ in range(20):
            rows.append(
                {
                    "name": "RegionA",
                    "trimmed_start": 100,
                    "pattern": "01",
                    "original_label": 0,
                }
            )
        # pattern "00" at start 100 -> ((100, 0), (101, 0))
        for _ in range(15):
            rows.append(
                {
                    "name": "RegionA",
                    "trimmed_start": 100,
                    "pattern": "00",
                    "original_label": 1,
                }
            )
        # pattern "0" at start 200 -> ((200, 0),)
        for _ in range(50):
            rows.append(
                {
                    "name": "RegionA",
                    "trimmed_start": 200,
                    "pattern": "0",
                    "original_label": 2,
                }
            )
        self.df = pd.DataFrame(rows)
        self.labeler = _make_labeler()
        self.sig1 = ((100, 0), (101, 1))

    def test_extracted_signatures_match_handler(self):
        """Signatures extracted from patterns match the handler's output."""
        res = self.labeler.compute_labels(
            self.df,
            min_reads=30,
            max_distance=0.7,
            num_classes=NUM_CLASSES,
        )
        self.assertIn(self.sig1, set(res["signature"]))

    def test_extraction_matches_precomputed_path(self):
        """Extracting signatures gives the same labels as passing them precomputed."""
        # Compute via handler extraction
        res_extracted = self.labeler.compute_labels(
            self.df,
            min_reads=30,
            max_distance=0.7,
            num_classes=NUM_CLASSES,
        )
        # Compute via precomputed column built with the same handler
        df_pre = self.df.copy()
        df_pre["sig"] = df_pre.apply(
            self.labeler.signature_handler.extract_signature, axis=1
        )
        res_pre = self.labeler.compute_labels(
            df_pre,
            min_reads=30,
            max_distance=0.7,
            num_classes=NUM_CLASSES,
            precomputed_signature_column="sig",
        )
        # Both paths must produce identical soft labels for the same signature
        sl_extracted = res_extracted[res_extracted["signature"] == self.sig1].iloc[0][
            "soft_label"
        ]
        sl_pre = res_pre[res_pre["signature"] == self.sig1].iloc[0]["soft_label"]
        np.testing.assert_allclose(sl_extracted, sl_pre, atol=1e-9)


class TestComputeLabelsValidation(unittest.TestCase):
    """Tests for the input-validation assertions in compute_labels."""

    def setUp(self):
        """Create a labeler shared across the validation tests."""
        self.labeler = _make_labeler()

    def test_missing_required_columns_raises(self):
        """Missing the 'original_label' column triggers an assertion error."""
        df = pd.DataFrame({"name": ["R1"], "sig": [((100, 0),)]})
        with self.assertRaises(AssertionError):
            self.labeler.compute_labels(
                df, num_classes=NUM_CLASSES, precomputed_signature_column="sig"
            )

    def test_out_of_range_labels_raises(self):
        """A label outside [0, num_classes-1] triggers an assertion error."""
        df = pd.DataFrame({"name": ["R1"], "original_label": [5], "sig": [((100, 0),)]})
        with self.assertRaises(AssertionError):
            self.labeler.compute_labels(
                df, num_classes=NUM_CLASSES, precomputed_signature_column="sig"
            )

    def test_missing_precomputed_column_raises(self):
        """Referencing a non-existent precomputed column triggers an assertion error."""
        df = pd.DataFrame({"name": ["R1"], "original_label": [0]})
        with self.assertRaises(AssertionError):
            self.labeler.compute_labels(
                df, num_classes=NUM_CLASSES, precomputed_signature_column="sig"
            )


class TestComputeLabelsRegionScopedPooling(unittest.TestCase):
    """Tests verifying that pooling never crosses region boundaries.

    Two reads can share identical signature coordinates while belonging to
    different regions (e.g. same start position on different chromosomes).
    Such reads live in distinct regions and must never be pooled together.
    """

    def setUp(self):
        """Create a labeler shared across the region-scoping tests."""
        self.labeler = _make_labeler()
        # Same signature coordinates in two different regions
        self.shared_sig = ((100, 0), (101, 1))

    def test_identical_signature_different_regions_not_pooled(self):
        """A signature is never pooled with an identical one from another region."""
        rows = []
        # chr1 region: only class 0, below min_reads on its own
        for _ in range(5):
            rows.append(
                {"name": "chr1:100", "sig": self.shared_sig, "original_label": 0}
            )
        # chr2 region: same coordinates but only class 1
        for _ in range(5):
            rows.append(
                {"name": "chr2:100", "sig": self.shared_sig, "original_label": 1}
            )
        df = pd.DataFrame(rows)

        # min_reads=30 is never reachable within a single region here, so if
        # cross-region pooling happened the two would borrow from each other.
        res = self.labeler.compute_labels(
            df,
            min_reads=30,
            max_distance=1.0,
            num_classes=NUM_CLASSES,
            precomputed_signature_column="sig",
        )

        chr1_row = res[res["name"] == "chr1:100"].iloc[0]
        chr2_row = res[res["name"] == "chr2:100"].iloc[0]

        # Each region only ever sees its own single signature -> only its own reads
        self.assertEqual(chr1_row["raw_counts_pooled"], 5)
        self.assertEqual(chr1_row["num_signatures_pooled"], 1)
        self.assertEqual(chr2_row["raw_counts_pooled"], 5)
        self.assertEqual(chr2_row["num_signatures_pooled"], 1)

    def test_soft_labels_stay_separated_across_regions(self):
        """Soft labels remain class-0 / class-1 pure despite identical signatures."""
        rows = []
        for _ in range(5):
            rows.append(
                {"name": "chr1:100", "sig": self.shared_sig, "original_label": 0}
            )
        for _ in range(5):
            rows.append(
                {"name": "chr2:100", "sig": self.shared_sig, "original_label": 1}
            )
        df = pd.DataFrame(rows)

        res = self.labeler.compute_labels(
            df,
            min_reads=30,
            max_distance=1.0,
            num_classes=NUM_CLASSES,
            precomputed_signature_column="sig",
        )

        chr1_prob = res[res["name"] == "chr1:100"].iloc[0]["soft_label"]
        chr2_prob = res[res["name"] == "chr2:100"].iloc[0]["soft_label"]

        # chr1 only has class-0 reads, chr2 only class-1 reads: no mixing.
        self.assertAlmostEqual(chr1_prob[0], 1.0, delta=1e-6)
        self.assertAlmostEqual(chr1_prob[1], 0.0, delta=1e-6)
        self.assertAlmostEqual(chr2_prob[1], 1.0, delta=1e-6)
        self.assertAlmostEqual(chr2_prob[0], 0.0, delta=1e-6)

    def test_pooling_happens_within_region_but_not_across(self):
        """Neighbors pool inside a region while an identical sig elsewhere is excluded."""
        rows = []
        # chr1 has two distinct signatures that should pool together
        sig_a = ((100, 0), (101, 1))
        sig_b = ((100, 0), (101, 0))  # jaccard distance 2/3 from sig_a
        for _ in range(10):
            rows.append({"name": "chr1:100", "sig": sig_a, "original_label": 0})
        for _ in range(15):
            rows.append({"name": "chr1:100", "sig": sig_b, "original_label": 1})
        # chr2 holds a copy of sig_a that must stay isolated from chr1
        for _ in range(50):
            rows.append({"name": "chr2:100", "sig": sig_a, "original_label": 2})
        df = pd.DataFrame(rows)

        res = self.labeler.compute_labels(
            df,
            min_reads=20,
            max_distance=1.0,
            num_classes=NUM_CLASSES,
            precomputed_signature_column="sig",
        )

        chr1_sig_a = res[(res["name"] == "chr1:100") & (res["sig"] == sig_a)].iloc[0]
        # Within chr1, sig_a (10 reads) pools sig_b (15 reads) to reach min_reads=20.
        self.assertEqual(chr1_sig_a["num_signatures_pooled"], 2)
        self.assertEqual(chr1_sig_a["raw_counts_pooled"], 25)
        # The chr2 copy of sig_a (class 2) is never pooled in, so the weighted
        # (post-pooling) class-2 mass for chr1 sig_a stays exactly 0.
        self.assertEqual(chr1_sig_a["weighted_counts_2"], 0)


if __name__ == "__main__":
    unittest.main()
