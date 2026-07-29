import unittest

import numpy as np
import pandas as pd

from syto.visualization.inference.deconvolution_data import build_report_model


def synthetic_df():
    """3 cell types (A>B>C by consensus), 3 methods: 1 baseline + 2 syto."""
    data = {
        ("methylbert", "uxm", "None"): {"A": 0.6, "B": 0.3, "C": 0.1},
        ("methylbert", "xgboost", "None"): {"A": 0.5, "B": 0.4, "C": 0.1},
        ("methylbert", "xgboost", "linear"): {"A": 0.4, "B": 0.5, "C": 0.1},
    }
    rows = []
    for (clf, dec, cal), props in data.items():
        for ct, p in props.items():
            rows.append(
                {
                    "FileName": "sample.bam",
                    "CellType": ct,
                    "Classifier": clf,
                    "Deconvolver": dec,
                    "Calibrator": cal,
                    "PredictedProportion": p,
                }
            )
    return pd.DataFrame(rows)


class TestBuildReportModel(unittest.TestCase):
    def setUp(self):
        self.model = build_report_model(synthetic_df(), top_n=10)

    def test_cell_types_ordered_by_consensus_desc(self):
        self.assertEqual(self.model.cell_types, ["A", "B", "C"])

    def test_consensus_is_median(self):
        np.testing.assert_allclose(self.model.consensus, [0.5, 0.4, 0.1])

    def test_mean(self):
        np.testing.assert_allclose(self.model.mean, [0.5, 0.4, 0.1])

    def test_spread(self):
        np.testing.assert_allclose(self.model.spread_min, [0.4, 0.3, 0.1])
        np.testing.assert_allclose(self.model.spread_max, [0.6, 0.5, 0.1])

    def test_methods_baseline_first(self):
        self.assertTrue(self.model.methods[0].is_baseline)
        self.assertEqual(self.model.methods[0].deconvolver, "uxm")
        self.assertFalse(self.model.methods[1].is_baseline)

    def test_method_order_follows_hierarchy(self):
        keys = [m.key for m in self.model.methods]
        self.assertEqual(
            keys,
            [
                "methylbert·uxm·None",
                "methylbert·xgboost·None",
                "methylbert·xgboost·linear",
            ],
        )

    def test_divergence_l1_from_consensus(self):
        div = {m.key: m.divergence for m in self.model.methods}
        self.assertAlmostEqual(div["methylbert·uxm·None"], 0.2)
        self.assertAlmostEqual(div["methylbert·xgboost·None"], 0.0)
        self.assertAlmostEqual(div["methylbert·xgboost·linear"], 0.2)

    def test_matrix_shape_and_alignment(self):
        self.assertEqual(self.model.matrix.shape, (3, 3))
        np.testing.assert_allclose(self.model.matrix[:, 0], [0.6, 0.3, 0.1])

    def test_top_n_and_other_bucket(self):
        m = build_report_model(synthetic_df(), top_n=1)
        self.assertEqual(m.top_cell_types, ["A"])
        self.assertAlmostEqual(m.other_consensus, 0.5)

    def test_file_name_captured(self):
        self.assertEqual(self.model.file_name, "sample.bam")

    def test_known_truth_carried(self):
        m = build_report_model(synthetic_df(), known_truth="B")
        self.assertEqual(m.known_truth, "B")

    def test_missing_column_raises(self):
        df = synthetic_df().drop(columns=["Calibrator"])
        with self.assertRaises(ValueError):
            build_report_model(df)


class TestEpidishAttributedToBaselines(unittest.TestCase):
    """EpiDISH results must land in the baseline group, not syto."""

    def _df(self, deconvolver):
        rows = [
            {
                "FileName": "s.bam",
                "CellType": ct,
                "Classifier": "None",
                "Deconvolver": deconvolver,
                "Calibrator": "None",
                "PredictedProportion": p,
            }
            for ct, p in {"A": 0.7, "B": 0.3}.items()
        ]
        return pd.DataFrame(rows)

    def test_default_epidish_labels_are_baseline(self):
        for label in ("epidish", "epidish_cbs", "epidish_cp"):
            model = build_report_model(self._df(label), top_n=10)
            info = next(m for m in model.methods if m.deconvolver == label)
            self.assertTrue(info.is_baseline, msg=label)

    def test_custom_name_baseline_via_config(self):
        # A custom EpiDISH label is baseline only when passed in the set.
        df = self._df("epidish_houseman")
        model = build_report_model(
            df, top_n=10, baseline_deconvolvers=["epidish_houseman"]
        )
        info = next(m for m in model.methods if m.deconvolver == "epidish_houseman")
        self.assertTrue(info.is_baseline)


class TestCollectBaselineLabels(unittest.TestCase):
    class _FakeDec:
        def __init__(self, name, em_checkpoints=None):
            self.name = name
            self.em_checkpoints = em_checkpoints

    def test_appends_custom_names_after_defaults(self):
        from syto.visualization.inference.deconvolution_data import (
            collect_baseline_labels,
        )

        labels = collect_baseline_labels(
            [self._FakeDec("epidish_houseman"), self._FakeDec("uxm")]
        )
        self.assertIn("epidish_houseman", labels)
        self.assertIn("uxm", labels)  # default, not duplicated
        self.assertEqual(labels[:3], ["uxm", "celfie", "celfieish"])
        self.assertEqual(labels.count("uxm"), 1)

    def test_expands_em_checkpoints(self):
        from syto.visualization.inference.deconvolution_data import (
            collect_baseline_labels,
        )

        labels = collect_baseline_labels(
            [self._FakeDec("celfieish", em_checkpoints=[10, 50])]
        )
        self.assertIn("celfieish_10_steps", labels)
        self.assertIn("celfieish_50_steps", labels)


if __name__ == "__main__":
    unittest.main()
