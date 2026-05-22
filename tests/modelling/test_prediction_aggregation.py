import unittest

import numpy as np
import pandas as pd

from methyldl.modelling.prediction_aggregation import (
    aggregate_chuncked_predictions_weighted,
    aggregate_predictions_by_dmr,
    aggregate_predictions_by_dmr_optimized,
    get_final_prediction,
    _fill_in_missing_labels,
    VALID_SUBSTITUTION_STRATEGIES,
)


def build_read_level_prediction_df() -> pd.DataFrame:
    """Create a compact read-level dataset used across aggregation tests."""
    return pd.DataFrame(
        [
            {
                "dmr_label": "dmr_a",
                "file": "sample_1",
                "original_label": "tumor",
                "dmr_ctype_label": 0,
                "dmr_ctype": "ctype_a",
                "prediction": 99,
                "prediction_0": 0.2,
                "prediction_1": 0.8,
                "methylation_level": 0.75,
                "methylated_CpGs": 1,
                "unmethylated_CpGs": 1,
                "ctype": "meta_a",
                "label": 7,
                "chromosome": "chr1",
            },
            {
                "dmr_label": "dmr_a",
                "file": "sample_1",
                "original_label": "tumor",
                "dmr_ctype_label": 0,
                "dmr_ctype": "ctype_a",
                "prediction": 42,
                "prediction_0": 0.6,
                "prediction_1": 0.4,
                "methylation_level": 0.25,
                "methylated_CpGs": 2,
                "unmethylated_CpGs": 4,
                "ctype": "meta_a",
                "label": 7,
                "chromosome": "chr1",
            },
            {
                "dmr_label": "dmr_b",
                "file": "sample_1",
                "original_label": "normal",
                "dmr_ctype_label": 1,
                "dmr_ctype": "ctype_b",
                "prediction": 11,
                "prediction_0": 0.9,
                "prediction_1": 0.1,
                "methylation_level": 0.9,
                "methylated_CpGs": 1,
                "unmethylated_CpGs": 0,
                "ctype": "meta_b",
                "label": 3,
                "chromosome": "chr2",
            },
            {
                "dmr_label": "dmr_b",
                "file": "sample_1",
                "original_label": "normal",
                "dmr_ctype_label": 1,
                "dmr_ctype": "ctype_b",
                "prediction": 15,
                "prediction_0": 0.3,
                "prediction_1": 0.7,
                "methylation_level": 0.1,
                "methylated_CpGs": 0,
                "unmethylated_CpGs": 2,
                "ctype": "meta_b",
                "label": 3,
                "chromosome": "chr2",
            },
        ]
    )


def build_zero_weight_prediction_df() -> pd.DataFrame:
    """Create a dataset whose computed weights are all zero for one group."""
    return pd.DataFrame(
        [
            {
                "dmr_label": "dmr_zero",
                "file": "sample_1",
                "original_label": "normal",
                "prediction_0": 0.9,
                "prediction_1": 0.1,
                "methylation_level": 0.9,
                "methylated_CpGs": 0,
                "unmethylated_CpGs": 0,
            },
            {
                "dmr_label": "dmr_zero",
                "file": "sample_1",
                "original_label": "normal",
                "prediction_0": 0.3,
                "prediction_1": 0.7,
                "methylation_level": 0.1,
                "methylated_CpGs": 0,
                "unmethylated_CpGs": 0,
            },
        ]
    )


def build_explicit_weight_df() -> pd.DataFrame:
    """Create a dataset that already contains the weight column."""
    return pd.DataFrame(
        [
            {
                "group": "g1",
                "prediction_0": 0.1,
                "prediction_1": 0.9,
                "methylation_level": 0.8,
                "custom_weight": 1.0,
            },
            {
                "group": "g1",
                "prediction_0": 0.5,
                "prediction_1": 0.5,
                "methylation_level": 0.2,
                "custom_weight": 3.0,
            },
        ]
    )


class PredictionAggregationDataFrameTestBase(unittest.TestCase):
    """Provide reusable dataframe fixtures for prediction aggregation tests."""

    def setUp(self):
        """Build fresh input dataframes for each test method."""
        self.read_level_prediction_df = build_read_level_prediction_df()
        self.zero_weight_prediction_df = build_zero_weight_prediction_df()
        self.explicit_weight_df = build_explicit_weight_df()

    def tearDown(self):
        """Clear dataframe fixtures after each test method."""
        self.read_level_prediction_df = None
        self.zero_weight_prediction_df = None
        self.explicit_weight_df = None


class TestAggregatePredictionsByDmr(PredictionAggregationDataFrameTestBase):
    """Exercise all branches of the DMR-level aggregation helper."""

    def test_aggregate_predictions_by_dmr_with_defaults(self):
        """Aggregate with inferred prediction columns and weights created from CpG counts."""
        result = aggregate_predictions_by_dmr(self.read_level_prediction_df)

        self.assertNotIn("index", result.columns)
        self.assertEqual(len(result), 2)

        dmr_a = result[result["dmr_label"] == "dmr_a"].iloc[0]
        self.assertAlmostEqual(dmr_a["prediction_0_avg"], 0.4)
        self.assertAlmostEqual(dmr_a["prediction_1_avg"], 0.6)
        self.assertAlmostEqual(dmr_a["methylation_level_avg"], 0.5)
        self.assertAlmostEqual(dmr_a["prediction_0_wavg"], 0.5)
        self.assertAlmostEqual(dmr_a["prediction_1_wavg"], 0.5)
        self.assertAlmostEqual(dmr_a["methylation_level_wavg"], 0.375)
        self.assertAlmostEqual(dmr_a["total_weight"], 8.0)
        self.assertEqual(dmr_a["n_reads"], 2)
        self.assertEqual(dmr_a["ctype"], "meta_a")
        self.assertEqual(dmr_a["dmr_ctype"], "ctype_a")
        self.assertEqual(dmr_a["label"], 7)
        self.assertEqual(dmr_a["chromosome"], "chr1")

        dmr_b = result[result["dmr_label"] == "dmr_b"].iloc[0]
        self.assertAlmostEqual(dmr_b["prediction_0_avg"], 0.6)
        self.assertAlmostEqual(dmr_b["prediction_1_avg"], 0.4)
        self.assertAlmostEqual(dmr_b["prediction_0_wavg"], 0.5)
        self.assertAlmostEqual(dmr_b["prediction_1_wavg"], 0.5)
        self.assertAlmostEqual(dmr_b["methylation_level_wavg"], 11 / 30)
        self.assertAlmostEqual(dmr_b["total_weight"], 3.0)
        self.assertEqual(dmr_b["n_reads"], 2)

    def test_aggregate_predictions_by_dmr_handles_zero_sum_weights_after_float_cast(
        self,
    ):
        """Use clipped float weights so all-zero CpG counts still produce valid averages."""
        result = aggregate_predictions_by_dmr(self.zero_weight_prediction_df)

        row = result.iloc[0]
        # After casting to float before clipping, both reads receive the same tiny
        # fallback weight, so the weighted averages match the unweighted means.
        self.assertAlmostEqual(row["prediction_0_avg"], 0.6)
        self.assertAlmostEqual(row["prediction_1_avg"], 0.4)
        self.assertAlmostEqual(row["methylation_level_avg"], 0.5)
        self.assertAlmostEqual(row["prediction_0_wavg"], 0.6)
        self.assertAlmostEqual(row["prediction_1_wavg"], 0.4)
        self.assertAlmostEqual(row["methylation_level_wavg"], 0.5)
        self.assertAlmostEqual(row["total_weight"], 2e-10)
        self.assertEqual(row["n_reads"], 2)

    def test_aggregate_predictions_by_dmr_uses_explicit_weights_and_prediction_columns(
        self,
    ):
        """Use caller-provided grouping, prediction columns, and weight column unchanged."""
        result = aggregate_predictions_by_dmr(
            self.explicit_weight_df,
            group_cols=["group"],
            prediction_cols=["prediction_0", "prediction_1"],
            weight_col="custom_weight",
            create_weight_from_cpgs=False,
        )

        row = result.iloc[0]
        self.assertEqual(row["group"], "g1")
        self.assertAlmostEqual(row["prediction_0_avg"], 0.3)
        self.assertAlmostEqual(row["prediction_1_avg"], 0.7)
        self.assertAlmostEqual(row["prediction_0_wavg"], 0.4)
        self.assertAlmostEqual(row["prediction_1_wavg"], 0.6)
        self.assertAlmostEqual(row["total_weight"], 4.0)
        self.assertEqual(row["n_reads"], 2)
        self.assertNotIn("methylation_level_avg", result.columns)

    def test_aggregate_predictions_by_dmr_requires_labels_dict_when_filling_missing_labels(
        self,
    ):
        """Reject missing label metadata when the caller requests synthetic rows."""
        with self.assertRaises(ValueError) as context:
            aggregate_predictions_by_dmr(
                self.read_level_prediction_df,
                fill_in_missing_labels=True,
            )

        self.assertIn("labels_dict must be provided", str(context.exception))

    def test_aggregate_predictions_by_dmr_raises_when_weight_cannot_be_created(self):
        """Fail when neither the requested weight column nor CpG count columns are available."""
        df = pd.DataFrame(
            [
                {
                    "group": "g1",
                    "prediction_0": 0.2,
                    "prediction_1": 0.8,
                    "methylation_level": 0.5,
                }
            ]
        )

        with self.assertRaises(ValueError) as context:
            aggregate_predictions_by_dmr(
                df,
                group_cols=["group"],
                weight_col="missing_weight",
                create_weight_from_cpgs=True,
            )

        self.assertIn("cannot create from CpG columns", str(context.exception))

    def test_aggregate_predictions_by_dmr_fills_missing_labels(self):
        """Add zeroed rows for label ids that are absent from the aggregated data."""
        df = self.read_level_prediction_df.loc[
            lambda frame: frame["dmr_ctype_label"] == 0,
            [
                "dmr_ctype_label",
                "dmr_ctype",
                "prediction_0",
                "prediction_1",
                "methylation_level",
                "methylated_CpGs",
                "unmethylated_CpGs",
                "label",
                "chromosome",
            ],
        ]

        result = aggregate_predictions_by_dmr(
            df,
            group_cols=["dmr_ctype_label", "dmr_ctype"],
            fill_in_missing_labels=True,
            labels_dict={0: "ctype_a", 1: "ctype_b"},
        )

        self.assertEqual(set(result["dmr_ctype_label"]), {0, 1})

        existing_row = result[result["dmr_ctype_label"] == 0].iloc[0]
        self.assertEqual(existing_row["dmr_ctype"], "ctype_a")
        self.assertAlmostEqual(existing_row["prediction_0_avg"], 0.4)
        self.assertAlmostEqual(existing_row["prediction_1_avg"], 0.6)
        self.assertAlmostEqual(existing_row["prediction_0_wavg"], 0.5)
        self.assertAlmostEqual(existing_row["prediction_1_wavg"], 0.5)
        self.assertAlmostEqual(existing_row["methylation_level_avg"], 0.5)
        self.assertAlmostEqual(existing_row["methylation_level_wavg"], 0.375)
        self.assertEqual(existing_row["n_reads"], 2)
        self.assertAlmostEqual(existing_row["total_weight"], 8.0)
        self.assertEqual(existing_row["label"], 7)
        self.assertEqual(existing_row["chromosome"], "chr1")

        missing_row = result[result["dmr_ctype_label"] == 1].iloc[0]
        self.assertEqual(missing_row["dmr_ctype"], "ctype_b")
        self.assertEqual(missing_row["prediction_0_avg"], 0)
        self.assertEqual(missing_row["prediction_1_avg"], 0)
        self.assertEqual(missing_row["prediction_0_wavg"], 0)
        self.assertEqual(missing_row["prediction_1_wavg"], 0)
        self.assertEqual(missing_row["n_reads"], 0)
        self.assertEqual(missing_row["label"], -1)
        self.assertEqual(missing_row["chromosome"], 0)
        # self.assertEqual(missing_row["total_weight"], ???)

    def test_aggregate_predictions_by_dmr_keeps_original_rows_when_no_labels_are_missing(
        self,
    ):
        """Leave the aggregated frame unchanged when every label in labels_dict is present."""
        df = self.read_level_prediction_df.loc[
            lambda frame: frame["dmr_ctype_label"] == 0,
            [
                "dmr_ctype_label",
                "dmr_ctype",
                "prediction_0",
                "prediction_1",
                "methylation_level",
                "methylated_CpGs",
                "unmethylated_CpGs",
            ],
        ]

        result = aggregate_predictions_by_dmr(
            df,
            group_cols=["dmr_ctype_label", "dmr_ctype"],
            fill_in_missing_labels=True,
            labels_dict={0: "ctype_a"},
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["dmr_ctype_label"], 0)
        self.assertEqual(result.iloc[0]["dmr_ctype"], "ctype_a")


class TestAggregatePredictionsByDmrOptimized(PredictionAggregationDataFrameTestBase):
    """Validate the vectorized aggregation implementation and its error path."""

    def test_aggregate_predictions_by_dmr_optimized_matches_expected_weighted_outputs(
        self,
    ):
        """Compute weighted averages, counts, and metadata with the optimized code path."""
        result = aggregate_predictions_by_dmr_optimized(
            self.read_level_prediction_df,
            group_cols=["dmr_label", "file", "original_label"],
        )

        dmr_a = result[result["dmr_label"] == "dmr_a"].iloc[0]
        self.assertAlmostEqual(dmr_a["prediction_0_wavg"], 0.5)
        self.assertAlmostEqual(dmr_a["prediction_1_wavg"], 0.5)
        self.assertAlmostEqual(dmr_a["methylation_level_wavg"], 0.375)
        self.assertEqual(dmr_a["n_reads"], 2)
        self.assertAlmostEqual(dmr_a["total_weight"], 8.0)
        self.assertEqual(dmr_a["label"], 7)
        self.assertEqual(dmr_a["chromosome"], "chr1")

        dmr_b = result[result["dmr_label"] == "dmr_b"].iloc[0]
        self.assertAlmostEqual(dmr_b["prediction_0_wavg"], 0.5)
        self.assertAlmostEqual(dmr_b["prediction_1_wavg"], 0.5)
        self.assertAlmostEqual(dmr_b["methylation_level_wavg"], 11 / 30)
        self.assertEqual(dmr_b["total_weight"], 3)
        self.assertEqual(dmr_b["n_reads"], 2)

    def test_aggregate_predictions_by_dmr_optimized_handles_zero_sum_weights_after_float_cast(
        self,
    ):
        """Use clipped float weights so the optimized path also returns valid averages."""
        result = aggregate_predictions_by_dmr_optimized(
            self.zero_weight_prediction_df,
            group_cols=["dmr_label", "file", "original_label"],
        )

        row = result.iloc[0]
        # As in the non-optimized path, equal fallback weights reduce the weighted
        # averages to the same values as the simple per-group means.
        self.assertAlmostEqual(row["prediction_0_wavg"], 0.6)
        self.assertAlmostEqual(row["prediction_1_wavg"], 0.4)
        self.assertAlmostEqual(row["methylation_level_wavg"], 0.5)
        self.assertEqual(row["total_weight"], 0)
        self.assertEqual(row["n_reads"], 2)

    def test_aggregate_predictions_by_dmr_optimized_raises_when_weight_is_missing(self):
        """Raise a clear error when no usable weight information exists."""
        df = pd.DataFrame(
            [
                {
                    "group": "g1",
                    "prediction_0": 0.2,
                    "prediction_1": 0.8,
                    "methylation_level": 0.5,
                }
            ]
        )

        with self.assertRaises(ValueError) as context:
            aggregate_predictions_by_dmr_optimized(df, group_cols=["group"])

        self.assertIn(
            "Weight column 'total_marked_cpgs' not found", str(context.exception)
        )


class TestAggregateChunkedPredictionsWeighted(unittest.TestCase):
    """Cover weighted aggregation for chunked per-read predictions."""

    def test_aggregate_chuncked_predictions_weighted_returns_weighted_read_averages(
        self,
    ):
        """Collapse chunk-level predictions into one weighted row per read."""
        pred_df = pd.DataFrame(
            [
                {
                    "read_name": "r1",
                    "ncpgs_marked": 2,
                    "prediction_0": 0.2,
                    "prediction_1": 0.8,
                },
                {
                    "read_name": "r1",
                    "ncpgs_marked": 4,
                    "prediction_0": 0.5,
                    "prediction_1": 0.5,
                },
                {
                    "read_name": "r2",
                    "ncpgs_marked": 0,
                    "prediction_0": 0.9,
                    "prediction_1": 0.1,
                },
                {
                    "read_name": "r2",
                    "ncpgs_marked": 0,
                    "prediction_0": 0.3,
                    "prediction_1": 0.7,
                },
            ]
        )

        result = aggregate_chuncked_predictions_weighted(pred_df)

        r1 = result[result["read_name"] == "r1"].iloc[0]
        self.assertAlmostEqual(r1["prediction_0"], 0.4)
        self.assertAlmostEqual(r1["prediction_1"], 0.6)

        r2 = result[result["read_name"] == "r2"].iloc[0]
        # The function fills NaN values after the division, which turns a 0/0 weighted
        # average into deterministic zeros instead of propagating NaNs.
        self.assertEqual(r2["prediction_0"], 0)
        self.assertEqual(r2["prediction_1"], 0)


class TestGetFinalPrediction(unittest.TestCase):
    """Verify final class extraction from aggregated probability columns."""

    def test_get_final_prediction_adds_prediction_and_confidence_for_each_method(self):
        """Compute argmax predictions and confidences for avg and wavg columns."""
        aggregated_df = pd.DataFrame(
            [
                {
                    "prediction_0_avg": 0.2,
                    "prediction_1_avg": 0.8,
                    "prediction_0_wavg": 0.7,
                    "prediction_1_wavg": 0.3,
                },
                {
                    "prediction_0_avg": 0.9,
                    "prediction_1_avg": 0.1,
                    "prediction_0_wavg": 0.4,
                    "prediction_1_wavg": 0.6,
                },
            ]
        )

        result = get_final_prediction(aggregated_df)

        np.testing.assert_array_equal(result["final_prediction_avg"].to_numpy(), [1, 0])
        np.testing.assert_allclose(
            result["final_confidence_avg"].to_numpy(), [0.8, 0.9]
        )
        np.testing.assert_array_equal(
            result["final_prediction_wavg"].to_numpy(), [0, 1]
        )
        np.testing.assert_allclose(
            result["final_confidence_wavg"].to_numpy(), [0.7, 0.6]
        )

    def test_get_final_prediction_respects_custom_method_list_and_prefix(self):
        """Restrict prediction extraction to the requested method and custom prefix."""
        aggregated_df = pd.DataFrame(
            [
                {
                    "score_0_avg": 0.45,
                    "score_1_avg": 0.55,
                    "score_0_raw": 0.1,
                    "score_1_raw": 0.9,
                }
            ]
        )

        # tests with avg method and "score_" prefix, so raw columns should be ignored
        result = get_final_prediction(
            aggregated_df,
            methods=["avg"],
            prediction_prefix="score_",
        )

        self.assertEqual(result.iloc[0]["final_prediction_avg"], 1)
        self.assertAlmostEqual(result.iloc[0]["final_confidence_avg"], 0.55)
        self.assertNotIn("final_prediction_raw", result.columns)

        # tests with "raw" method and "score_" prefix, so avg columns should be ignored
        result = get_final_prediction(
            aggregated_df,
            methods=["raw"],
            prediction_prefix="score_",
        )

        self.assertEqual(result.iloc[0]["final_prediction_raw"], 1)
        self.assertAlmostEqual(result.iloc[0]["final_confidence_raw"], 0.9)
        self.assertNotIn("final_prediction_avg", result.columns)

    def test_get_final_prediction_raises_when_no_matching_columns_exist(self):
        """Fail fast when the requested aggregation suffix does not exist."""
        aggregated_df = pd.DataFrame(
            [{"prediction_0_avg": 0.2, "prediction_1_avg": 0.8}]
        )

        with self.assertRaises(ValueError) as context:
            get_final_prediction(aggregated_df, methods=["wavg"])

        self.assertIn(
            "No prediction columns found with suffix '_wavg'", str(context.exception)
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Tests for missing-label substitution strategies
# ═══════════════════════════════════════════════════════════════════════════


def _build_aggregated_with_missing_label() -> tuple:
    """Build read-level data with only one DMR label, plus a matching prior.

    The data contains only dmr_ctype_label=0 reads.  When
    ``fill_in_missing_labels=True`` with ``labels_dict={0: 'ctype_a', 1: 'ctype_b'}``,
    a synthetic row for label 1 will be inserted.

    Returns (df, labels_dict, uniform_prior).
    """
    # Read-level data — only ctype_a present
    df = pd.DataFrame(
        [
            {
                "dmr_ctype_label": 0,
                "dmr_ctype": "ctype_a",
                "prediction_0": 0.8,
                "prediction_1": 0.2,
                "methylation_level": 0.6,
                "total_weight": 5.0,
                "label": 0,
                "chromosome": "chr1",
            },
            {
                "dmr_ctype_label": 0,
                "dmr_ctype": "ctype_a",
                "prediction_0": 0.7,
                "prediction_1": 0.3,
                "methylation_level": 0.6,
                "total_weight": 5.0,
                "label": 0,
                "chromosome": "chr1",
            },
        ]
    )
    labels_dict = {0: "ctype_a", 1: "ctype_b"}

    uniform_prior = pd.DataFrame(
        [
            {
                "dmr_ctype_label": 0,
                "dmr_ctype": "ctype_a",
                "prediction_0_wavg": 0.5,
                "prediction_1_wavg": 0.5,
                "methylation_level_wavg": 0.4,
            },
            {
                "dmr_ctype_label": 1,
                "dmr_ctype": "ctype_b",
                "prediction_0_wavg": 0.3,
                "prediction_1_wavg": 0.7,
                "methylation_level_wavg": 0.35,
            },
        ]
    )
    return df, labels_dict, uniform_prior


class TestSubstitutionStrategyZeroes(unittest.TestCase):
    """Confirm backward-compatible zeroes strategy."""

    def test_zeroes_strategy_fills_missing_with_zeros(self):
        df, labels_dict, _ = _build_aggregated_with_missing_label()
        result = aggregate_predictions_by_dmr(
            df,
            group_cols=["dmr_ctype_label", "dmr_ctype"],
            prediction_cols=["prediction_0", "prediction_1", "methylation_level"],
            weight_col="total_weight",
            create_weight_from_cpgs=False,
            fill_in_missing_labels=True,
            labels_dict=labels_dict,
            substitution_strategy="zeroes",
        )
        missing = result[result["dmr_ctype_label"] == 1].iloc[0]
        self.assertEqual(missing["prediction_0_wavg"], 0)
        self.assertEqual(missing["prediction_1_wavg"], 0)
        self.assertEqual(missing["n_reads"], 0)


class TestSubstitutionStrategyPriorBlending(unittest.TestCase):
    """Validate the prior_blending formula."""

    def test_blending_zero_reads_collapses_to_prior(self):
        """Rows with n_reads=0 should be entirely replaced by the prior."""
        df, labels_dict, prior = _build_aggregated_with_missing_label()
        result = aggregate_predictions_by_dmr(
            df,
            group_cols=["dmr_ctype_label", "dmr_ctype"],
            prediction_cols=["prediction_0", "prediction_1", "methylation_level"],
            weight_col="total_weight",
            create_weight_from_cpgs=False,
            fill_in_missing_labels=True,
            labels_dict=labels_dict,
            substitution_strategy="prior_blending",
            uniform_prior=prior,
            prior_weight=1.0,
        )
        missing = result[result["dmr_ctype_label"] == 1].iloc[0]
        # n_reads=0 → alpha=0 → fully prior
        self.assertAlmostEqual(missing["prediction_0_wavg"], 0.3)
        self.assertAlmostEqual(missing["prediction_1_wavg"], 0.7)
        self.assertAlmostEqual(missing["methylation_level_wavg"], 0.35)

    def test_blending_nonzero_reads_mixed(self):
        """Rows with n_reads>0 should be blended: alpha = n/(n+w)."""
        df, labels_dict, prior = _build_aggregated_with_missing_label()
        result = aggregate_predictions_by_dmr(
            df,
            group_cols=["dmr_ctype_label", "dmr_ctype"],
            prediction_cols=["prediction_0", "prediction_1", "methylation_level"],
            weight_col="total_weight",
            create_weight_from_cpgs=False,
            fill_in_missing_labels=True,
            labels_dict=labels_dict,
            substitution_strategy="prior_blending",
            uniform_prior=prior,
            prior_weight=1.0,
        )
        existing = result[result["dmr_ctype_label"] == 0].iloc[0]
        # n_reads=2 (two read-level rows in the fixture), prior_weight=1 → alpha = 2/3
        alpha = 2.0 / 3.0
        expected_p0 = alpha * 0.75 + (1 - alpha) * 0.5
        expected_p1 = alpha * 0.25 + (1 - alpha) * 0.5
        self.assertAlmostEqual(existing["prediction_0_wavg"], expected_p0, places=6)
        self.assertAlmostEqual(existing["prediction_1_wavg"], expected_p1, places=6)

    def test_blending_custom_prior_weight(self):
        """A higher prior_weight shifts blending toward the prior."""
        df, labels_dict, prior = _build_aggregated_with_missing_label()
        result = aggregate_predictions_by_dmr(
            df,
            group_cols=["dmr_ctype_label", "dmr_ctype"],
            prediction_cols=["prediction_0", "prediction_1", "methylation_level"],
            weight_col="total_weight",
            create_weight_from_cpgs=False,
            fill_in_missing_labels=True,
            labels_dict=labels_dict,
            substitution_strategy="prior_blending",
            uniform_prior=prior,
            prior_weight=5.0,
        )
        existing = result[result["dmr_ctype_label"] == 0].iloc[0]
        # n_reads=2, prior_weight=5 → alpha = 2/7
        alpha = 2.0 / 7.0
        expected_p0 = alpha * 0.75 + (1 - alpha) * 0.5
        self.assertAlmostEqual(existing["prediction_0_wavg"], expected_p0, places=6)


class TestSubstitutionStrategyPriorImputation(unittest.TestCase):
    """Validate prior_imputation: only zero-read rows replaced."""

    def test_imputation_replaces_zero_read_rows(self):
        df, labels_dict, prior = _build_aggregated_with_missing_label()
        result = aggregate_predictions_by_dmr(
            df,
            group_cols=["dmr_ctype_label", "dmr_ctype"],
            prediction_cols=["prediction_0", "prediction_1", "methylation_level"],
            weight_col="total_weight",
            create_weight_from_cpgs=False,
            fill_in_missing_labels=True,
            labels_dict=labels_dict,
            substitution_strategy="prior_imputation",
            uniform_prior=prior,
        )
        missing = result[result["dmr_ctype_label"] == 1].iloc[0]
        self.assertAlmostEqual(missing["prediction_0_wavg"], 0.3)
        self.assertAlmostEqual(missing["prediction_1_wavg"], 0.7)

    def test_imputation_leaves_nonzero_read_rows_intact(self):
        df, labels_dict, prior = _build_aggregated_with_missing_label()
        result = aggregate_predictions_by_dmr(
            df,
            group_cols=["dmr_ctype_label", "dmr_ctype"],
            prediction_cols=["prediction_0", "prediction_1", "methylation_level"],
            weight_col="total_weight",
            create_weight_from_cpgs=False,
            fill_in_missing_labels=True,
            labels_dict=labels_dict,
            substitution_strategy="prior_imputation",
            uniform_prior=prior,
        )
        existing = result[result["dmr_ctype_label"] == 0].iloc[0]
        # Original values should be untouched
        self.assertAlmostEqual(existing["prediction_0_wavg"], 0.75)
        self.assertAlmostEqual(existing["prediction_1_wavg"], 0.25)


class TestSubstitutionStrategyValidation(unittest.TestCase):
    """Verify error handling for invalid configurations."""

    def test_invalid_strategy_raises(self):
        df, labels_dict, _ = _build_aggregated_with_missing_label()
        with self.assertRaises(ValueError) as ctx:
            aggregate_predictions_by_dmr(
                df,
                group_cols=["dmr_ctype_label", "dmr_ctype"],
                prediction_cols=["prediction_0", "prediction_1", "methylation_level"],
                weight_col="total_weight",
                create_weight_from_cpgs=False,
                fill_in_missing_labels=True,
                labels_dict=labels_dict,
                substitution_strategy="unknown_strategy",
            )
        self.assertIn("Unknown substitution_strategy", str(ctx.exception))

    def test_prior_strategy_without_prior_raises(self):
        df, labels_dict, _ = _build_aggregated_with_missing_label()
        with self.assertRaises(ValueError) as ctx:
            aggregate_predictions_by_dmr(
                df,
                group_cols=["dmr_ctype_label", "dmr_ctype"],
                prediction_cols=["prediction_0", "prediction_1", "methylation_level"],
                weight_col="total_weight",
                create_weight_from_cpgs=False,
                fill_in_missing_labels=True,
                labels_dict=labels_dict,
                substitution_strategy="prior_blending",
                uniform_prior=None,
            )
        self.assertIn("uniform_prior must be provided", str(ctx.exception))


class TestComputeUniformPriorMatrix(unittest.TestCase):
    """Test the compute_uniform_prior_matrix utility from pure_profile_generation."""

    def _make_mock_pure_profiles(self):
        """Build minimal mock pure profiles: 2 cell types, 2 prediction columns."""
        from methyldl.data.pure_profile_generation import compute_uniform_prior_matrix

        # Cell type 0: predictions [0.8, 0.2] for both DMR rows
        subs_0 = {
            "train": pd.DataFrame(
                [
                    {
                        "dmr_ctype_label": 0,
                        "dmr_ctype": "ctype_a",
                        "prediction_0_wavg": 0.8,
                        "prediction_1_wavg": 0.2,
                        "methylation_level_wavg": 0.7,
                    },
                    {
                        "dmr_ctype_label": 1,
                        "dmr_ctype": "ctype_b",
                        "prediction_0_wavg": 0.6,
                        "prediction_1_wavg": 0.4,
                        "methylation_level_wavg": 0.5,
                    },
                ]
            )
        }
        profile_0 = (np.array([1.0, 0.0]), subs_0, None)

        # Cell type 1: predictions [0.2, 0.8] for both DMR rows
        subs_1 = {
            "train": pd.DataFrame(
                [
                    {
                        "dmr_ctype_label": 0,
                        "dmr_ctype": "ctype_a",
                        "prediction_0_wavg": 0.4,
                        "prediction_1_wavg": 0.6,
                        "methylation_level_wavg": 0.3,
                    },
                    {
                        "dmr_ctype_label": 1,
                        "dmr_ctype": "ctype_b",
                        "prediction_0_wavg": 0.2,
                        "prediction_1_wavg": 0.8,
                        "methylation_level_wavg": 0.1,
                    },
                ]
            )
        }
        profile_1 = (np.array([0.0, 1.0]), subs_1, None)

        return [profile_0, profile_1]

    def test_averaging_across_profiles(self):
        from methyldl.data.pure_profile_generation import compute_uniform_prior_matrix

        profiles = self._make_mock_pure_profiles()
        prior = compute_uniform_prior_matrix(
            profiles, split_key="train", num_input_labels=2
        )

        self.assertEqual(len(prior), 2)

        row_a = prior[prior["dmr_ctype_label"] == 0].iloc[0]
        # Average of 0.8 and 0.4
        self.assertAlmostEqual(row_a["prediction_0_wavg"], 0.6)
        # Average of 0.2 and 0.6
        self.assertAlmostEqual(row_a["prediction_1_wavg"], 0.4)
        # Average of 0.7 and 0.3
        self.assertAlmostEqual(row_a["methylation_level_wavg"], 0.5)

        row_b = prior[prior["dmr_ctype_label"] == 1].iloc[0]
        # Average of 0.6 and 0.2
        self.assertAlmostEqual(row_b["prediction_0_wavg"], 0.4)
        # Average of 0.4 and 0.8
        self.assertAlmostEqual(row_b["prediction_1_wavg"], 0.6)

    def test_save_load_roundtrip(self):
        import tempfile
        from methyldl.data.pure_profile_generation import (
            compute_uniform_prior_matrix,
            save_uniform_prior,
            load_uniform_prior,
        )

        profiles = self._make_mock_pure_profiles()
        prior = compute_uniform_prior_matrix(
            profiles, split_key="train", num_input_labels=2
        )

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name

        save_uniform_prior(prior, path)
        loaded = load_uniform_prior(path)

        pd.testing.assert_frame_equal(
            prior.reset_index(drop=True),
            loaded.reset_index(drop=True),
            check_dtype=False,
        )

    def test_all_none_profiles_raises(self):
        from methyldl.data.pure_profile_generation import compute_uniform_prior_matrix

        with self.assertRaises(ValueError):
            compute_uniform_prior_matrix(
                [None, None], split_key="train", num_input_labels=2
            )
