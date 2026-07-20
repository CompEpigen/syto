import os
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")

import pandas as pd

from syto.visualization.inference.deconvolution_data import build_report_model
from syto.visualization.inference.deconvolution_pdf import render_pdf
from tests.visualization.inference.test_deconvolution_data import synthetic_df


def multi_classifier_df():
    """Adds a second classifier so hierarchy/color branches are exercised."""
    base = synthetic_df()
    extra = base.copy()
    extra["Classifier"] = "dismir"
    return pd.concat([base, extra], ignore_index=True)


class TestRenderPdf(unittest.TestCase):
    def test_pdf_written_and_nonempty(self):
        model = build_report_model(synthetic_df())
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "report.pdf")
            render_pdf(model, path)
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)

    def test_pdf_with_known_truth_and_multi_classifier(self):
        model = build_report_model(multi_classifier_df(), known_truth="A")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "report.pdf")
            render_pdf(model, path)
            self.assertGreater(os.path.getsize(path), 0)


if __name__ == "__main__":
    unittest.main()
