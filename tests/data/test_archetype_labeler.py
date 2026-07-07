import unittest

import numpy as np
import pandas as pd

from syto.data.labelers.archetype_labeler import ArchetypeLabeler
from syto.data.omics_signatures_handlers.binary_cpg_signature import (
    BinaryCpGSignatureHandler,
)

NUM_CLASSES = 3


def _make_labeler(eps: float = 1e-10):
    """Build an ArchetypeLabeler backed by a binary CpG signature handler."""
    handler = BinaryCpGSignatureHandler(
        start_column="trimmed_start", methylation_pattern_column="pattern"
    )
    return ArchetypeLabeler(signature_handler=handler, eps=eps)


class TestComputeLabelsPrecomputedSignatures(unittest.TestCase):
    """Tests using a precomputed signature column.

    Layout (1 region "RegionA", 3 adjacent CpGs at 100/101/102)::

        sigA = ((100, 1), (101, 1), (102, 1)) -> 20 reads label 0
        sigB = ((100, 0), (101, 0), (102, 0)) -> 15 reads label 1
        sigC = ((100, 1), (101, 0), (102, 1)) -> 50 reads label 2

    Every ctype covers every CpG, so no mean filling is required and each
    archetype is a (near) deterministic methylation pattern:

        mu[0] ~ (1, 1, 1), mu[1] ~ (0, 0, 0), mu[2] ~ (1, 0, 1)

    so each signature is (near) one-hot under its own ctype.
    """

    def setUp(self):
        """Build a 3-signature, 3-class fixture with a precomputed 'sig' column."""
        self.sigA = ((100, 1), (101, 1), (102, 1))
        self.sigB = ((100, 0), (101, 0), (102, 0))
        self.sigC = ((100, 1), (101, 0), (102, 1))

        rows = []
        for _ in range(20):
            rows.append({"name": "RegionA", "sig": self.sigA, "original_label": 0})
        for _ in range(15):
            rows.append({"name": "RegionA", "sig": self.sigB, "original_label": 1})
        for _ in range(50):
            rows.append({"name": "RegionA", "sig": self.sigC, "original_label": 2})
        self.df = pd.DataFrame(rows)
        self.labeler = _make_labeler()

    def test_rows_preserved(self):
        """Every input read is kept and a soft_label column is produced."""
        res = self.labeler.compute_labels(
            self.df, num_classes=NUM_CLASSES, precomputed_signature_column="sig"
        )
        self.assertEqual(len(res), len(self.df))
        self.assertIn("soft_label", res.columns)

    def test_signature_column_matches_precomputed(self):
        """With a precomputed column, 'signature' equals that column verbatim."""
        res = self.labeler.compute_labels(
            self.df, num_classes=NUM_CLASSES, precomputed_signature_column="sig"
        )
        self.assertTrue((res["signature"] == res["sig"]).all())

    def test_soft_labels_sum_to_one(self):
        """Each soft label is a valid probability distribution (sums to 1)."""
        res = self.labeler.compute_labels(
            self.df, num_classes=NUM_CLASSES, precomputed_signature_column="sig"
        )
        sums = res["soft_label"].apply(sum)
        np.testing.assert_allclose(sums.values, 1.0, atol=1e-6)

    def test_soft_label_length_matches_num_classes(self):
        """Each soft label vector has exactly num_classes entries."""
        res = self.labeler.compute_labels(
            self.df, num_classes=NUM_CLASSES, precomputed_signature_column="sig"
        )
        lengths = res["soft_label"].apply(len)
        self.assertTrue((lengths == NUM_CLASSES).all())

    def test_each_signature_is_near_one_hot_on_its_ctype(self):
        """Under a uniform prior, each signature concentrates on its own ctype."""
        res = self.labeler.compute_labels(
            self.df,
            num_classes=NUM_CLASSES,
            precomputed_signature_column="sig",
            ctype_prior_type="uniform",
        )
        probA = res[res["signature"] == self.sigA].iloc[0]["soft_label"]
        probB = res[res["signature"] == self.sigB].iloc[0]["soft_label"]
        probC = res[res["signature"] == self.sigC].iloc[0]["soft_label"]
        np.testing.assert_allclose(probA, [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(probB, [0.0, 1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(probC, [0.0, 0.0, 1.0], atol=1e-6)

    def test_likelihoods_present_and_match_archetype(self):
        """The likelihoods column stores P(sig | ctype) = phi for each ctype."""
        res = self.labeler.compute_labels(
            self.df, num_classes=NUM_CLASSES, precomputed_signature_column="sig"
        )
        self.assertIn("likelihoods", res.columns)
        likA = res[res["signature"] == self.sigA].iloc[0]["likelihoods"]
        # phi[0] = (1 - eps)^3 ~ 1, phi[1] = eps^3 ~ 0, phi[2] = (1-eps)^2 * eps~ 0
        self.assertAlmostEqual(likA[0], 1.0, delta=1e-6)
        self.assertAlmostEqual(likA[1], 0.0, delta=1e-6)
        self.assertAlmostEqual(likA[2], 0.0, delta=1e-6)

    def test_keep_likelihoods_false_drops_column(self):
        """keep_likelihoods=False removes the likelihoods column."""
        res = self.labeler.compute_labels(
            self.df,
            num_classes=NUM_CLASSES,
            keep_likelihoods=False,
            precomputed_signature_column="sig",
        )
        self.assertNotIn("likelihoods", res.columns)
        self.assertIn("soft_label", res.columns)

    def test_no_intermediate_columns(self):
        """The labeler never exposes raw counts or the prior in its output."""
        res = self.labeler.compute_labels(
            self.df, num_classes=NUM_CLASSES, precomputed_signature_column="sig"
        )
        self.assertNotIn("raw_counts_0", res.columns)
        self.assertNotIn("ctype_prior", res.columns)


class TestComputeLabelsSignatureExtraction(unittest.TestCase):
    """Tests where signatures are computed from patterns by the handler."""

    def setUp(self):
        """Build a fixture of raw reads whose signatures the handler will extract."""
        rows = []
        # pattern "111" at start 100 -> ((100, 1), (101, 1), (102, 1))
        for _ in range(20):
            rows.append(
                {
                    "name": "RegionA",
                    "trimmed_start": 100,
                    "pattern": "111",
                    "original_label": 0,
                }
            )
        # pattern "000" at start 100 -> ((100, 0), (101, 0), (102, 0))
        for _ in range(15):
            rows.append(
                {
                    "name": "RegionA",
                    "trimmed_start": 100,
                    "pattern": "000",
                    "original_label": 1,
                }
            )
        # pattern "101" at start 100 -> ((100, 1), (101, 0), (102, 1))
        for _ in range(50):
            rows.append(
                {
                    "name": "RegionA",
                    "trimmed_start": 100,
                    "pattern": "101",
                    "original_label": 2,
                }
            )
        self.df = pd.DataFrame(rows)
        self.labeler = _make_labeler()
        self.sigA = ((100, 1), (101, 1), (102, 1))

    def test_extracted_signatures_match_handler(self):
        """Signatures extracted from patterns match the handler's output."""
        res = self.labeler.compute_labels(self.df, num_classes=NUM_CLASSES)
        self.assertIn(self.sigA, set(res["signature"]))

    def test_extraction_matches_precomputed_path(self):
        """Extracting signatures gives the same labels as passing them precomputed."""
        res_extracted = self.labeler.compute_labels(self.df, num_classes=NUM_CLASSES)

        df_pre = self.df.copy()
        df_pre["sig"] = df_pre.apply(
            self.labeler.signature_handler.extract_signature, axis=1
        )
        res_pre = self.labeler.compute_labels(
            df_pre, num_classes=NUM_CLASSES, precomputed_signature_column="sig"
        )

        sl_extracted = res_extracted[res_extracted["signature"] == self.sigA].iloc[0][
            "soft_label"
        ]
        sl_pre = res_pre[res_pre["signature"] == self.sigA].iloc[0]["soft_label"]
        np.testing.assert_allclose(sl_extracted, sl_pre, atol=1e-9)


class TestComputeLabelsPriorTypes(unittest.TestCase):
    """Tests for the four ``ctype_prior_type`` options.

    An empty signature has an identical likelihood under every ctype
    (phi[c] = 1), so its posterior equals the (normalized) prior exactly. This
    is exploited to read the prior back out of the soft labels.

    Two regions with different class balance are used to distinguish the global
    prior from the per-region prior::

        RegionA: 30 x label0, 10 x label1, 1 empty read (label0)
        RegionB:  5 x label0, 20 x label1, 1 empty read (label1)
    """

    def setUp(self):
        """Build a 2-region, 2-class fixture containing one empty read per region."""
        self.sig_meth = ((100, 1), (101, 1))
        self.sig_unmeth = ((100, 0), (101, 0))
        self.empty = ()

        rows = []
        for _ in range(30):
            rows.append({"name": "A", "sig": self.sig_meth, "original_label": 0})
        for _ in range(10):
            rows.append({"name": "A", "sig": self.sig_unmeth, "original_label": 1})
        rows.append({"name": "A", "sig": self.empty, "original_label": 0})

        for _ in range(5):
            rows.append({"name": "B", "sig": self.sig_meth, "original_label": 0})
        for _ in range(20):
            rows.append({"name": "B", "sig": self.sig_unmeth, "original_label": 1})
        rows.append({"name": "B", "sig": self.empty, "original_label": 1})

        self.df = pd.DataFrame(rows)
        self.labeler = _make_labeler()

    def _empty_soft_label(self, res, region):
        """Return the soft label of the empty-signature read in the given region."""
        mask = (res["name"] == region) & (res["signature"].apply(len) == 0)
        return np.asarray(res[mask].iloc[0]["soft_label"])

    def test_uniform_prior(self):
        """A uniform prior yields a uniform posterior on the empty signature."""
        res = self.labeler.compute_labels(
            self.df,
            num_classes=2,
            ctype_prior_type="uniform",
            precomputed_signature_column="sig",
        )
        np.testing.assert_allclose(
            self._empty_soft_label(res, "A"), [0.5, 0.5], atol=1e-6
        )

    def test_inv_global_freq_prior(self):
        """inv_global_freq uses the inverse of the pooled class frequencies."""
        res = self.labeler.compute_labels(
            self.df,
            num_classes=2,
            ctype_prior_type="inv_global_freq",
            precomputed_signature_column="sig",
        )
        # global counts: label0 = 30 + 5 + 1 = 36, label1 = 10 + 20 + 1 = 31
        inv = np.array([1 / 36, 1 / 31])
        expected = inv / inv.sum()
        np.testing.assert_allclose(
            self._empty_soft_label(res, "A"), expected, atol=1e-6
        )
        np.testing.assert_allclose(
            self._empty_soft_label(res, "B"), expected, atol=1e-6
        )

    def test_inv_region_freq_prior(self):
        """inv_region_freq uses the inverse of the per-region class frequencies."""
        res = self.labeler.compute_labels(
            self.df,
            num_classes=2,
            ctype_prior_type="inv_region_freq",
            precomputed_signature_column="sig",
        )
        # RegionA counts: label0 = 31, label1 = 10
        inv_a = np.array([1 / 31, 1 / 10])
        np.testing.assert_allclose(
            self._empty_soft_label(res, "A"), inv_a / inv_a.sum(), atol=1e-6
        )
        # RegionB counts: label0 = 5, label1 = 21
        inv_b = np.array([1 / 5, 1 / 21])
        np.testing.assert_allclose(
            self._empty_soft_label(res, "B"), inv_b / inv_b.sum(), atol=1e-6
        )

    def test_predefined_prior_is_normalized(self):
        """A predefined prior is normalized and used for every region."""
        res = self.labeler.compute_labels(
            self.df,
            num_classes=2,
            ctype_prior_type="predefined",
            predefined_prior=[3, 7],
            precomputed_signature_column="sig",
        )
        np.testing.assert_allclose(
            self._empty_soft_label(res, "A"), [0.3, 0.7], atol=1e-6
        )
        np.testing.assert_allclose(
            self._empty_soft_label(res, "B"), [0.3, 0.7], atol=1e-6
        )


class TestComputeLabelsRegionScoping(unittest.TestCase):
    """Archetypes (and therefore soft labels) are estimated per region.

    The same signature coordinates in two regions can map to opposite ctypes
    and must receive opposite soft labels.
    """

    def setUp(self):
        """Build two regions whose ctype archetypes are swapped."""
        self.sig_meth = ((100, 1), (101, 1))
        self.sig_unmeth = ((100, 0), (101, 0))
        self.labeler = _make_labeler()

        rows = []
        # RegionA: ctype0 is the methylated archetype, ctype1 the unmethylated one
        for _ in range(10):
            rows.append({"name": "A", "sig": self.sig_meth, "original_label": 0})
        for _ in range(10):
            rows.append({"name": "A", "sig": self.sig_unmeth, "original_label": 1})
        # RegionB: the archetypes are swapped
        for _ in range(10):
            rows.append({"name": "B", "sig": self.sig_unmeth, "original_label": 0})
        for _ in range(10):
            rows.append({"name": "B", "sig": self.sig_meth, "original_label": 1})
        self.df = pd.DataFrame(rows)

    def test_same_signature_opposite_labels_across_regions(self):
        """An identical signature gets opposite labels in the two regions."""
        res = self.labeler.compute_labels(
            self.df, num_classes=2, precomputed_signature_column="sig"
        )
        a_meth = res[(res["name"] == "A") & (res["signature"] == self.sig_meth)].iloc[0]
        b_meth = res[(res["name"] == "B") & (res["signature"] == self.sig_meth)].iloc[0]
        # In RegionA the methylated signature belongs to ctype0
        np.testing.assert_allclose(a_meth["soft_label"], [1.0, 0.0], atol=1e-6)
        # In RegionB the same signature belongs to ctype1
        np.testing.assert_allclose(b_meth["soft_label"], [0.0, 1.0], atol=1e-6)


class TestEstimateRegionArchetypes(unittest.TestCase):
    """Direct tests of the archetype-estimation helper."""

    def setUp(self):
        """Create a labeler with the default eps for the helper tests."""
        self.labeler = _make_labeler()

    def test_mle_estimate_on_covered_positions(self):
        """mu[c, k] is the maximum-likelihood o1 / (o1 + o0) where covered."""
        signatures = [((100, 1),), ((100, 0),)]
        count_matrix = np.array([[3, 1], [1, 1]], dtype=float)
        mu, pos_to_idx = self.labeler._estimate_region_archetypes(
            signatures, count_matrix, num_classes=2, region="R"
        )
        self.assertEqual(pos_to_idx, {100: 0})
        self.assertAlmostEqual(mu[0, 0], 0.75, places=9)  # ctype0: 3 / 4
        self.assertAlmostEqual(mu[1, 0], 0.5, places=9)  # ctype1: 1 / 2

    def test_uncovered_cpg_filled_with_ctype_mean(self):
        """An uncovered CpG is filled with the per-ctype mean of covered mu."""
        signatures = [((101, 1),), ((101, 0),), ((102, 1),), ((200, 1),)]
        # ctype1: pos101 -> 2 meth / 2 unmeth (mu 0.5), pos102 -> 4 meth (mu ~1),
        #         pos200 uncovered -> mean(0.5, 1 - eps) ~ 0.75
        count_matrix = np.array([[0, 2], [0, 2], [0, 4], [5, 0]], dtype=float)
        mu, pos_to_idx = self.labeler._estimate_region_archetypes(
            signatures, count_matrix, num_classes=2, region="R"
        )
        self.assertAlmostEqual(mu[1, pos_to_idx[101]], 0.5, places=9)
        self.assertAlmostEqual(mu[1, pos_to_idx[200]], 0.75, places=8)

    def test_values_clipped_to_eps_bounds(self):
        """Saturated archetypes are clipped to (eps, 1 - eps)."""
        eps = 1e-3
        labeler = _make_labeler(eps=eps)
        signatures = [((100, 1),), ((101, 0),)]
        count_matrix = np.array([[4, 4], [4, 4]], dtype=float)
        mu, _ = labeler._estimate_region_archetypes(
            signatures, count_matrix, num_classes=2, region="R"
        )
        self.assertAlmostEqual(mu.max(), 1 - eps, places=12)
        self.assertAlmostEqual(mu.min(), eps, places=12)

    def test_completely_uncovered_ctype_gets_pseudocoverage(self):
        """A ctype with no covered CpG gets pseudocoverage and warns instead of raising."""
        eps = 1e-3
        labeler = _make_labeler(eps=eps)
        signatures = [((100, 1),)]
        # ctype 0 has 3 reads at pos 100; ctype 1 has none -> completely uncovered
        count_matrix = np.array([[3, 0]], dtype=float)
        with self.assertWarns(UserWarning):
            mu, pos_to_idx = labeler._estimate_region_archetypes(
                signatures, count_matrix, num_classes=2, region="R"
            )
        # the uncovered ctype is assigned a fully methylated (~1) archetype
        self.assertAlmostEqual(mu[1, pos_to_idx[100]], 1 - eps, places=12)


class TestComputeLabelsValidation(unittest.TestCase):
    """Tests for the input validation in compute_labels."""

    def setUp(self):
        """Create a labeler shared across the validation tests."""
        self.labeler = _make_labeler()

    def test_missing_required_columns_raises(self):
        """Missing the 'original_label' column triggers an assertion error."""
        df = pd.DataFrame({"name": ["R1"], "sig": [((100, 1),)]})
        with self.assertRaises(AssertionError):
            self.labeler.compute_labels(
                df, num_classes=NUM_CLASSES, precomputed_signature_column="sig"
            )

    def test_out_of_range_labels_raises(self):
        """A label outside [0, num_classes-1] triggers an assertion error."""
        df = pd.DataFrame({"name": ["R1"], "original_label": [5], "sig": [((100, 1),)]})
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

    def test_invalid_prior_type_raises(self):
        """An unknown ctype_prior_type triggers an assertion error."""
        df = pd.DataFrame({"name": ["R1"], "original_label": [0], "sig": [((100, 1),)]})
        with self.assertRaises(AssertionError):
            self.labeler.compute_labels(
                df,
                num_classes=2,
                ctype_prior_type="not_a_prior",
                precomputed_signature_column="sig",
            )

    def test_predefined_without_prior_raises(self):
        """ctype_prior_type='predefined' without predefined_prior raises."""
        df = pd.DataFrame({"name": ["R1"], "original_label": [0], "sig": [((100, 1),)]})
        with self.assertRaises(AssertionError):
            self.labeler.compute_labels(
                df,
                num_classes=2,
                ctype_prior_type="predefined",
                precomputed_signature_column="sig",
            )

    def test_predefined_wrong_length_raises(self):
        """A predefined prior of the wrong length raises."""
        df = pd.DataFrame({"name": ["R1"], "original_label": [0], "sig": [((100, 1),)]})
        with self.assertRaises(AssertionError):
            self.labeler.compute_labels(
                df,
                num_classes=2,
                ctype_prior_type="predefined",
                predefined_prior=[1.0, 1.0, 1.0],
                precomputed_signature_column="sig",
            )

    def test_predefined_negative_raises(self):
        """A predefined prior with negative entries raises."""
        df = pd.DataFrame({"name": ["R1"], "original_label": [0], "sig": [((100, 1),)]})
        with self.assertRaises(AssertionError):
            self.labeler.compute_labels(
                df,
                num_classes=2,
                ctype_prior_type="predefined",
                predefined_prior=[-1.0, 2.0],
                precomputed_signature_column="sig",
            )

    def test_uncovered_ctype_in_region_warns(self):
        """A region missing a whole ctype warns and completes instead of raising."""
        df = pd.DataFrame(
            {
                "name": ["R1", "R1"],
                "original_label": [0, 0],
                "sig": [((100, 1),), ((100, 0),)],
            }
        )
        # num_classes=2 but only ctype 0 appears -> ctype 1 is completely uncovered
        with self.assertWarns(UserWarning):
            self.labeler.compute_labels(
                df, num_classes=2, precomputed_signature_column="sig"
            )


if __name__ == "__main__":
    unittest.main()
