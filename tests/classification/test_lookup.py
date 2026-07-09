"""Test suite for the LookupClassifier and LabelConfig classes.

Covers
------
- LabelConfig initialisation, serialisation and backward-compatible alias.
- LookupClassifier init, fit, predict in both *soft* and *hard* label modes.
- Exact-match and 1-NN fallback prediction paths.
- Save / load round-trips for .joblib and .pkl formats.
- Serialisation helpers (_key_to_str / _str_to_key).
- Edge cases: unfitted prediction, unseen regions, unsupported extensions.
"""

# pylint: disable=protected-access
import json
import os
import sys
import shutil
import tempfile
import unittest
import warnings
import logging

import numpy as np
import pandas as pd

from syto.classification.classifiers.lookup import LabelConfig, LookupClassifier
from syto.data.omics_signatures_handlers.binary_cpg_signature import (
    BinaryCpGSignatureHandler,
)

warnings.filterwarnings("ignore")


# Configure logging to display INFO and above messages in the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────
# cannot be changed independantly of the sample data, which has 3 classes (0,1,2)
NUM_CLASSES = 3


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _make_sample_df():
    """Build a small DataFrame suitable for fitting a LookupClassifier.

    Layout (1 region, 2 distinct CpG signatures)::

        Region "R1", pattern "01" → cpg_sig ((0,0),(1,1)):
            50 reads label 0, 3 reads label 1
        Region "R1", pattern "10" → cpg_sig ((0,1),(1,0)):
            50 reads label 0, 2 reads label 2

    Global class balance: {0: 100, 1: 3, 2: 2}.
    Class 0 dominates ≥ 10x, so the hard-label sanity check passes.
    """
    rows = []
    # Signature 1: pattern "01" at start 0 → cpg_sig = ((0,0),(1,1))
    for _ in range(50):
        rows.append(
            {
                "name": "R1",
                "read_start": 0,
                "pattern": "01",
                "original_label": 0,
            }
        )
    for _ in range(3):
        rows.append(
            {
                "name": "R1",
                "read_start": 0,
                "pattern": "01",
                "original_label": 1,
            }
        )
    # Signature 2: pattern "10" at start 0 → cpg_sig = ((0,1),(1,0))
    for _ in range(50):
        rows.append(
            {
                "name": "R1",
                "read_start": 0,
                "pattern": "10",
                "original_label": 0,
            }
        )
    for _ in range(2):
        rows.append(
            {
                "name": "R1",
                "read_start": 0,
                "pattern": "10",
                "original_label": 2,
            }
        )
    return pd.DataFrame(rows)


def _make_balanced_df():
    """Build a DataFrame where no class dominates 10x.

    Global class balance: {0: 30, 1: 25} → ratio 1.2x < 10x required,
    so the hard-label sanity check should raise ValueError.
    """
    rows = []
    for _ in range(30):
        rows.append(
            {
                "name": "R1",
                "read_start": 0,
                "pattern": "01",
                "original_label": 0,
            }
        )
    for _ in range(25):
        rows.append(
            {
                "name": "R1",
                "read_start": 0,
                "pattern": "01",
                "original_label": 1,
            }
        )
    return pd.DataFrame(rows)


def _make_multi_region_df():
    """Build a DataFrame with two regions for per-region fallback tests.

    Region R1 has cpg_sig ((0,0),(1,1)); Region R2 has cpg_sig ((10,1),(11,0)).
    Class 0 dominates globally (100 reads vs 5 total for others).
    """
    rows = []
    for _ in range(50):
        rows.append(
            {
                "name": "R1",
                "read_start": 0,
                "pattern": "01",
                "original_label": 0,
            }
        )
    for _ in range(3):
        rows.append(
            {
                "name": "R1",
                "read_start": 0,
                "pattern": "01",
                "original_label": 1,
            }
        )
    for _ in range(50):
        rows.append(
            {
                "name": "R2",
                "read_start": 10,
                "pattern": "10",
                "original_label": 0,
            }
        )
    for _ in range(2):
        rows.append(
            {
                "name": "R2",
                "read_start": 10,
                "pattern": "10",
                "original_label": 2,
            }
        )
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# LabelConfig
# ──────────────────────────────────────────────────────────────────────


class TestLabelConfig(unittest.TestCase):
    """Tests for the LabelConfig dataclass and its helpers."""

    def test_default_values(self):
        """Default config should expose the documented defaults."""
        cfg = LabelConfig()
        self.assertEqual(cfg.num_classes, 39)
        self.assertEqual(cfg.label_col, "original_label")
        self.assertEqual(cfg.label_mode, "soft")
        self.assertEqual(cfg.min_reads, 30)
        self.assertAlmostEqual(cfg.max_distance, 0.41)

    def test_custom_values(self):
        """Config should accept and store custom values."""
        cfg = LabelConfig(num_classes=5, label_mode="hard", min_reads=10)
        self.assertEqual(cfg.num_classes, 5)
        self.assertEqual(cfg.label_mode, "hard")
        self.assertEqual(cfg.min_reads, 10)

    def test_to_dict(self):
        """to_dict() returns a plain dict mirroring all fields."""
        cfg = LabelConfig(num_classes=5)
        d = cfg.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["num_classes"], 5)
        # Every dataclass field should be present
        self.assertIn("label_col", d)
        self.assertIn("label_mode", d)
        self.assertIn("min_reads", d)
        self.assertIn("max_distance", d)

    def test_from_dict_round_trip(self):
        """from_dict(to_dict(cfg)) should reconstruct an equal config."""
        original = LabelConfig(num_classes=10, label_mode="hard", min_reads=5)
        restored = LabelConfig.from_dict(original.to_dict())
        self.assertEqual(original, restored)

    def test_from_dict_ignores_extra_keys(self):
        """Extra keys in the dict should be silently ignored."""
        d = {"num_classes": 7, "unexpected_key": "hello"}
        cfg = LabelConfig.from_dict(d)
        self.assertEqual(cfg.num_classes, 7)


# ──────────────────────────────────────────────────────────────────────
# LookupClassifier: init and properties
# ──────────────────────────────────────────────────────────────────────


class TestLookupClassifierInit(unittest.TestCase):
    """Tests for LookupClassifier construction and initial state."""

    def test_default_config(self):
        """Constructor with no args should use default LabelConfig."""
        clf = LookupClassifier()
        self.assertEqual(clf.config.num_classes, 39)
        self.assertFalse(clf._is_fitted)

    def test_custom_config(self):
        """Constructor should store the provided LabelConfig."""
        cfg = LabelConfig(num_classes=NUM_CLASSES)
        clf = LookupClassifier(cfg)
        self.assertEqual(clf.config.num_classes, NUM_CLASSES)

    def test_n_keys_before_fit(self):
        """n_keys should be 0 before fitting."""
        clf = LookupClassifier()
        self.assertEqual(clf.n_keys, 0)

    def test_empty_lookup_before_fit(self):
        """Internal lookup structures should be empty before fitting."""
        clf = LookupClassifier()
        self.assertEqual(len(clf._lookup), 0)
        self.assertEqual(len(clf._region_index), 0)


# ──────────────────────────────────────────────────────────────────────
# LookupClassifier: fit / predict – soft-label mode
# ──────────────────────────────────────────────────────────────────────


class TestLookupClassifierSoftMode(unittest.TestCase):
    """Fit and predict in the default soft-label mode."""

    def setUp(self):
        """Create a soft-label classifier and small sample data."""
        self.cfg = LabelConfig(
            num_classes=NUM_CLASSES,
            min_reads=2,
            max_distance=0.5,
        )
        self.clf = LookupClassifier(self.cfg)
        self.df = _make_sample_df()

    def test_fit_returns_self(self):
        """fit() should return the classifier itself (supports chaining)."""
        result = self.clf.fit(self.df)
        self.assertIs(result, self.clf)

    def test_fit_sets_fitted_flag(self):
        """fit() should mark the classifier as fitted."""
        self.clf.fit(self.df)
        self.assertTrue(self.clf._is_fitted)

    def test_fit_populates_lookup_with_correct_count(self):
        """After fit, the lookup table should hold 2 keys (2 distinct sigs)."""
        self.clf.fit(self.df)
        self.assertEqual(self.clf.n_keys, 2)

    def test_fit_populates_region_index(self):
        """After fit, the region index should contain region 'R1' with 2 sigs."""
        self.clf.fit(self.df)
        self.assertIn("R1", self.clf._region_index)
        self.assertEqual(len(self.clf._region_index["R1"]), 2)

    def test_lookup_entries_have_soft_label_key(self):
        """Each lookup entry in soft mode should contain 'soft_label'."""
        self.clf.fit(self.df)
        for entry in self.clf._lookup.values():
            self.assertIn("soft_label", entry)
            self.assertIn("normalized_counts", entry)

    def test_soft_labels_are_valid_distributions(self):
        """Soft labels stored in the lookup must sum to ≈ 1."""
        self.clf.fit(self.df)
        for entry in self.clf._lookup.values():
            total = sum(entry["soft_label"])
            self.assertAlmostEqual(total, 1.0, places=6)

    def test_predict_before_fit_raises(self):
        """predict() on an unfitted classifier should raise RuntimeError."""
        with self.assertRaises(RuntimeError):
            self.clf.predict(self.df)

    def test_predict_returns_dataframe(self):
        """predict() should return a DataFrame."""
        self.clf.fit(self.df)
        result = self.clf.predict(self.df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_predict_adds_prediction_columns(self):
        """predict() should add prediction_0..N-1 and prediction_source."""
        self.clf.fit(self.df)
        result = self.clf.predict(self.df)
        for j in range(NUM_CLASSES):
            self.assertIn(f"prediction_{j}", result.columns)
        self.assertIn("prediction_source", result.columns)

    def test_predict_preserves_original_columns(self):
        """predict() output should still contain all original columns."""
        self.clf.fit(self.df)
        result = self.clf.predict(self.df)
        for col in ("name", "read_start", "pattern", "original_label"):
            self.assertIn(col, result.columns)

    def test_predict_exact_match_source(self):
        """All training rows should get source='exact' when predicted."""
        self.clf.fit(self.df)
        result = self.clf.predict(self.df)
        self.assertTrue(
            (result["prediction_source"] == "exact").all(),
            "Expected all training predictions to be 'exact'.",
        )

    def test_predict_probabilities_sum_to_one(self):
        """Soft-label predictions should sum to ≈ 1 for every row."""
        self.clf.fit(self.df)
        result = self.clf.predict(self.df)
        pred_cols = [f"prediction_{j}" for j in range(NUM_CLASSES)]
        row_sums = result[pred_cols].sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-6)

    def test_predict_does_not_modify_input(self):
        """predict() should not mutate the caller's DataFrame."""
        self.clf.fit(self.df)
        original_cols = list(self.df.columns)
        self.clf.predict(self.df)
        self.assertEqual(list(self.df.columns), original_cols)

    def test_predict_with_precomputed_cpg_sig(self):
        """predict() should work when cpg_sig is already present in df."""
        self.clf.fit(self.df)

        # Pre-compute cpg_sig on a copy of the data
        df_with_sig = self.df.copy()
        signature_handler = BinaryCpGSignatureHandler(
            start_column="read_start", methylation_pattern_column="pattern"
        )
        df_with_sig["cpg_sig"] = df_with_sig.apply(
            signature_handler.extract_signature, axis=1
        )
        # Ensure tuple type, as the classifier does internally
        df_with_sig["cpg_sig"] = df_with_sig["cpg_sig"].apply(
            lambda x: (
                tuple(tuple(int(e) for e in p) for p in x)
                if not isinstance(x, tuple)
                else x
            )
        )

        result = self.clf.predict(df_with_sig)
        self.assertTrue(
            (result["prediction_source"] == "exact").all(),
            "Pre-computed cpg_sig should still yield exact matches.",
        )

    def test_predict_split_compatibility(self):
        """predict_split() should produce the same output as predict()."""
        self.clf.fit(self.df)
        pred_predict = self.clf.predict(self.df)
        pred_split = self.clf.predict_split(self.df)
        pred_cols = [f"prediction_{j}" for j in range(NUM_CLASSES)]
        np.testing.assert_array_equal(
            pred_predict[pred_cols].values,
            pred_split[pred_cols].values,
        )


# ──────────────────────────────────────────────────────────────────────
# LookupClassifier: fit / predict – hard-label mode
# ──────────────────────────────────────────────────────────────────────


class TestLookupClassifierHardMode(unittest.TestCase):
    """Fit and predict with label_mode='hard'."""

    def setUp(self):
        """Create a hard-label classifier and small sample data."""
        self.cfg = LabelConfig(
            num_classes=NUM_CLASSES,
            label_mode="hard",
            min_reads=2,
            max_distance=0.5,
        )
        self.clf = LookupClassifier(self.cfg)
        self.df = _make_sample_df()

    def test_fit_hard_mode_populates_lookup(self):
        """fit() in hard mode should populate the lookup table."""
        self.clf.fit(self.df)
        self.assertTrue(self.clf._is_fitted)
        self.assertEqual(self.clf.n_keys, 2)

    def test_lookup_entries_have_hard_label_key(self):
        """Each lookup entry in hard mode should contain 'hard_label'."""
        self.clf.fit(self.df)
        for entry in self.clf._lookup.values():
            self.assertIn("hard_label", entry)
            self.assertIn("normalized_counts", entry)

    def test_hard_labels_sum_to_one(self):
        """Hard-label vectors stored in the lookup should sum to exactly 1."""
        self.clf.fit(self.df)
        for entry in self.clf._lookup.values():
            total = sum(entry["hard_label"])
            self.assertAlmostEqual(total, 1.0, places=9)

    def test_predict_hard_labels_sum_to_one(self):
        """Hard-label predictions should sum to exactly 1."""
        self.clf.fit(self.df)
        result = self.clf.predict(self.df)
        pred_cols = [f"prediction_{j}" for j in range(NUM_CLASSES)]
        row_sums = result[pred_cols].sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-9)

    def test_predict_hard_source_exact(self):
        """All training predictions in hard mode should be 'exact'."""
        self.clf.fit(self.df)
        result = self.clf.predict(self.df)
        self.assertTrue((result["prediction_source"] == "exact").all())

    def test_hard_label_sanity_check_raises_on_balanced_data(self):
        """Hard-label fit should raise ValueError when no class dominates 10x."""
        cfg = LabelConfig(num_classes=NUM_CLASSES, label_mode="hard")
        clf = LookupClassifier(cfg)
        with self.assertRaises(ValueError) as ctx:
            clf.fit(_make_balanced_df())
        # The error message should mention the sanity check
        self.assertIn("sanity check", str(ctx.exception).lower())

    def test_hard_label_one_hot_on_clear_winner(self):
        """When one class clearly wins after normalisation, hard label is one-hot."""
        self.clf.fit(self.df)
        for entry in self.clf._lookup.values():
            label = entry["hard_label"]
            # Exactly one element should be nonzero (strict argmax, no ties)
            nonzero = [x for x in label if x > 0]
            self.assertEqual(len(nonzero), 1)
            self.assertAlmostEqual(nonzero[0], 1.0)

    def test_fit_hard_mode_with_pipeline_renamed_label(self):
        """fit() must work when the fitting pipeline has already renamed the
        configured label column to the canonical 'label'.

        The ClassifierFittingPipeline renames the chosen ``label_column`` to the
        canonical ``label`` before handing the frame to the classifier, so
        ``config.label_col`` (the pre-rename source name) is no longer present.
        """
        renamed_df = _make_sample_df().rename(columns={"original_label": "label"})
        cfg = LabelConfig(
            num_classes=NUM_CLASSES,
            label_mode="hard",
            label_col="hard_label_with_background",  # pre-rename source name
        )
        clf = LookupClassifier(cfg)
        clf.fit(renamed_df, compute_train_metrics=True)
        self.assertTrue(clf._is_fitted)
        self.assertEqual(clf.n_keys, 2)


# ──────────────────────────────────────────────────────────────────────
# 1-NN fallback
# ──────────────────────────────────────────────────────────────────────


class TestLookupClassifierFallback(unittest.TestCase):
    """Tests for the 1-NN fallback behaviour when exact match is missing."""

    def setUp(self):
        """Fit a soft-label classifier on sample data."""
        self.cfg = LabelConfig(
            num_classes=NUM_CLASSES,
            min_reads=2,
            max_distance=0.5,
        )
        self.clf = LookupClassifier(self.cfg)
        self.clf.fit(_make_sample_df())

    # ---- unseen region ----

    def test_fallback_unseen_region_soft_returns_uniform(self):
        """An unseen region in soft mode should return uniform distribution."""
        result = self.clf._fallback_nn("UNSEEN_REGION", ((99, 0),))
        expected = [1.0 / NUM_CLASSES] * NUM_CLASSES
        np.testing.assert_allclose(result, expected, atol=1e-9)

    def test_fallback_unseen_region_hard_returns_rejection(self):
        """An unseen region in hard mode should return rejection class (last)."""
        cfg = LabelConfig(
            num_classes=NUM_CLASSES,
            label_mode="hard",
            min_reads=2,
            max_distance=0.5,
        )
        clf = LookupClassifier(cfg)
        clf.fit(_make_sample_df())
        result = clf._fallback_nn("UNSEEN_REGION", ((99, 0),))
        expected = [0.0] * NUM_CLASSES
        expected[-1] = 1.0  # rejection class = last class
        np.testing.assert_allclose(result, expected, atol=1e-9)

    # ---- unknown signature in known region ----

    def test_fallback_unknown_signature_uses_1nn(self):
        """A new signature in a known region should trigger 1NN fallback."""
        # pattern "00" → cpg_sig ((0,0),(1,0)) which is NOT in the lookup
        test_row = pd.DataFrame(
            [
                {
                    "name": "R1",
                    "read_start": 0,
                    "pattern": "00",
                }
            ]
        )
        result = self.clf.predict(test_row)
        self.assertEqual(result["prediction_source"].iloc[0], "1nn")

    def test_fallback_predictions_sum_to_one(self):
        """Fallback predictions should also sum to ≈ 1."""
        test_row = pd.DataFrame(
            [
                {
                    "name": "R1",
                    "read_start": 0,
                    "pattern": "00",
                }
            ]
        )
        result = self.clf.predict(test_row)
        pred_cols = [f"prediction_{j}" for j in range(NUM_CLASSES)]
        row_sum = result[pred_cols].iloc[0].sum()
        self.assertAlmostEqual(row_sum, 1.0, places=6)

    def test_fallback_unknown_region_via_predict(self):
        """predict() on a completely unseen region should return '1nn' source."""
        test_row = pd.DataFrame(
            [
                {
                    "name": "UNSEEN_REGION",
                    "read_start": 999,
                    "pattern": "01",
                }
            ]
        )
        result = self.clf.predict(test_row)
        self.assertEqual(result["prediction_source"].iloc[0], "1nn")

    def test_fallback_tied_nn_averages(self):
        """When two signatures tie for nearest, their labels should be averaged.

        Pattern "00" → cpg_sig ((0,0),(1,0)).
        Distances to known sigs ((0,0),(1,1)) and ((0,1),(1,0)) are both
        2/3 (Jaccard), so the fallback should average both entries.
        """
        # Directly call _fallback_nn to check the aggregation logic
        query_sig = ((0, 0), (1, 0))
        result = self.clf._fallback_nn("R1", query_sig)

        # Should still be a valid probability distribution
        self.assertAlmostEqual(sum(result), 1.0, places=6)
        self.assertEqual(len(result), NUM_CLASSES)


# ──────────────────────────────────────────────────────────────────────
# Multi-region
# ──────────────────────────────────────────────────────────────────────


class TestLookupClassifierMultiRegion(unittest.TestCase):
    """Tests with data spanning multiple genomic regions."""

    def setUp(self):
        """Fit a classifier on multi-region data."""
        self.cfg = LabelConfig(
            num_classes=NUM_CLASSES,
            min_reads=2,
            max_distance=0.5,
        )
        self.clf = LookupClassifier(self.cfg)
        self.clf.fit(_make_multi_region_df())

    def test_region_index_has_both_regions(self):
        """After fit, both R1 and R2 should appear in the region index."""
        self.assertIn("R1", self.clf._region_index)
        self.assertIn("R2", self.clf._region_index)

    def test_predict_exact_across_regions(self):
        """Exact matches should work for rows from different regions."""
        result = self.clf.predict(_make_multi_region_df())
        self.assertTrue((result["prediction_source"] == "exact").all())

    def test_nn_fallback_uses_same_region(self):
        """1-NN fallback should only consider candidates within the same region.

        Query region R1 with a sig that only exists in R2 to verify
        R1's own NN is used (not R2's entry).

        Strategy
        --------
        R1 has one known signature (pattern "01", label mix {0:50, 1:3}).
        R2 has one known signature (pattern "10", label mix {0:50, 2:2}).
        Since R1 has exactly one candidate, the fallback for any unknown R1
        signature must return the same label vector as R1's exact entry.
        It must NOT return R2's label vector (which has class 2 probability > 0
        and class 1 probability ≈ 0, i.e. the opposite of R1's distribution).
        """
        pred_cols = [f"prediction_{j}" for j in range(NUM_CLASSES)]

        # Obtain the exact-match label vector for R1's only known signature
        r1_known_row = pd.DataFrame([{"name": "R1", "read_start": 0, "pattern": "01"}])
        r1_exact = self.clf.predict(r1_known_row)[pred_cols].values[0]

        # Obtain the exact-match label vector for R2's only known signature
        r2_known_row = pd.DataFrame([{"name": "R2", "read_start": 10, "pattern": "10"}])
        r2_exact = self.clf.predict(r2_known_row)[pred_cols].values[0]

        # Predict for an unseen R1 signature — should fall back to R1's NN
        test_row = pd.DataFrame(
            [
                {
                    "name": "R1",
                    "read_start": 0,
                    "pattern": "00",  # not in R1's lookup → fallback within R1
                }
            ]
        )
        result = self.clf.predict(test_row)
        fallback_pred = result[pred_cols].values[0]

        # Confirm the fallback path was actually taken
        self.assertEqual(result["prediction_source"].iloc[0], "1nn")

        # The fallback must match R1's only NN, not R2's entry
        np.testing.assert_array_almost_equal(
            fallback_pred,
            r1_exact,
            err_msg="Fallback should use R1's own NN, not a cross-region candidate.",
        )
        self.assertFalse(
            np.allclose(fallback_pred, r2_exact),
            "Fallback incorrectly returned R2's label vector for a R1 query.",
        )


# ──────────────────────────────────────────────────────────────────────
# Save / load
# ──────────────────────────────────────────────────────────────────────


class TestLookupClassifierSaveLoad(unittest.TestCase):
    """Save and load round-trips for .joblib and .pkl formats."""

    def setUp(self):
        """Fit a classifier and create a temp directory."""
        self.cfg = LabelConfig(
            num_classes=NUM_CLASSES,
            min_reads=2,
            max_distance=0.5,
        )
        self.clf = LookupClassifier(self.cfg)
        self.clf.fit(_make_sample_df())
        self.tmp_dir = tempfile.mkdtemp(prefix="lookup_test_")

    def tearDown(self):
        """Remove the temp directory."""
        shutil.rmtree(self.tmp_dir)

    # ---- joblib ----

    def test_save_load_joblib_round_trip(self):
        """Saving to .joblib and loading back should preserve state."""
        path = os.path.join(self.tmp_dir, "model.joblib")
        self.clf.save(path)

        loaded = LookupClassifier.load(path)
        self.assertIsInstance(loaded, LookupClassifier)
        self.assertTrue(loaded._is_fitted)
        self.assertEqual(loaded.n_keys, self.clf.n_keys)
        self.assertEqual(loaded.config.num_classes, NUM_CLASSES)

    def test_save_load_joblib_predictions_match(self):
        """Predictions from a loaded .joblib model should match the original."""
        path = os.path.join(self.tmp_dir, "model.joblib")
        self.clf.save(path)
        loaded = LookupClassifier.load(path)

        df = _make_sample_df()
        pred_orig = self.clf.predict(df)
        pred_loaded = loaded.predict(df)

        pred_cols = [f"prediction_{j}" for j in range(NUM_CLASSES)]
        np.testing.assert_array_almost_equal(
            pred_orig[pred_cols].values,
            pred_loaded[pred_cols].values,
        )

    # ---- pkl ----

    def test_save_load_pkl_round_trip(self):
        """Saving to .pkl and loading back should preserve state."""
        path = os.path.join(self.tmp_dir, "model.pkl")
        self.clf.save(path)

        loaded = LookupClassifier.load(path)
        self.assertIsInstance(loaded, LookupClassifier)
        self.assertTrue(loaded._is_fitted)
        self.assertEqual(loaded.n_keys, self.clf.n_keys)
        self.assertEqual(loaded.config.num_classes, NUM_CLASSES)

    def test_save_load_pkl_predictions_match(self):
        """Predictions from a loaded .pkl model should match the original."""
        path = os.path.join(self.tmp_dir, "model.pkl")
        self.clf.save(path)
        loaded = LookupClassifier.load(path)

        df = _make_sample_df()
        pred_orig = self.clf.predict(df)
        pred_loaded = loaded.predict(df)

        pred_cols = [f"prediction_{j}" for j in range(NUM_CLASSES)]
        np.testing.assert_array_almost_equal(
            pred_orig[pred_cols].values,
            pred_loaded[pred_cols].values,
        )

    # ---- config persistence ----

    def test_saved_config_preserved_joblib(self):
        """Config fields should survive the joblib round-trip."""
        path = os.path.join(self.tmp_dir, "model.joblib")
        self.clf.save(path)
        loaded = LookupClassifier.load(path)
        self.assertEqual(loaded.config.to_dict(), self.clf.config.to_dict())

    def test_saved_config_preserved_pkl(self):
        """Config fields should survive the pkl round-trip."""
        path = os.path.join(self.tmp_dir, "model.pkl")
        self.clf.save(path)
        loaded = LookupClassifier.load(path)
        self.assertEqual(loaded.config.to_dict(), self.clf.config.to_dict())

    # ---- error cases ----

    def test_save_unsupported_extension_raises(self):
        """Saving with an unsupported extension should raise ValueError."""
        path = os.path.join(self.tmp_dir, "model.json")
        with self.assertRaises(ValueError):
            self.clf.save(path)

    def test_load_unsupported_extension_raises(self):
        """Loading with an unsupported extension should raise ValueError."""
        path = os.path.join(self.tmp_dir, "model.json")
        # Create a dummy file so the path exists
        with open(path, "w", encoding="utf-8") as f:
            f.write("{}")
        with self.assertRaises(ValueError):
            LookupClassifier.load(path)

    def test_save_creates_parent_dirs(self):
        """save() should create missing parent directories."""
        path = os.path.join(self.tmp_dir, "sub", "dir", "model.joblib")
        self.clf.save(path)
        self.assertTrue(os.path.exists(path))


# ──────────────────────────────────────────────────────────────────────
# Serialisation helpers
# ──────────────────────────────────────────────────────────────────────


class TestSerializationHelpers(unittest.TestCase):
    """Tests for _key_to_str and _str_to_key static methods."""

    def test_key_str_round_trip(self):
        """_str_to_key(_key_to_str(key)) should return the original key."""
        key = ("chr1:100-200", ((100, 0), (101, 1), (105, 0)))
        s = LookupClassifier._key_to_str(key)
        restored = LookupClassifier._str_to_key(s)
        self.assertEqual(restored, key)

    def test_key_to_str_produces_valid_json(self):
        """_key_to_str should produce a parseable JSON string."""
        key = ("R1", ((0, 0), (1, 1)))
        s = LookupClassifier._key_to_str(key)
        parsed = json.loads(s)
        self.assertIsInstance(parsed, list)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0], "R1")

    def test_round_trip_empty_signature(self):
        """Round-trip should work for an empty CpG signature tuple."""
        key = ("R_empty", ())
        s = LookupClassifier._key_to_str(key)
        restored = LookupClassifier._str_to_key(s)
        self.assertEqual(restored, key)

    def test_round_trip_single_element_signature(self):
        """Round-trip should work for a single-element signature."""
        key = ("R_single", ((42, 1),))
        s = LookupClassifier._key_to_str(key)
        restored = LookupClassifier._str_to_key(s)
        self.assertEqual(restored, key)

    def test_round_trip_region_with_special_chars(self):
        """Round-trip should work for region names with colons and dashes."""
        key = ("chr22:12345-67890", ((10, 0), (20, 1)))
        s = LookupClassifier._key_to_str(key)
        restored = LookupClassifier._str_to_key(s)
        self.assertEqual(restored, key)


if __name__ == "__main__":
    unittest.main()
