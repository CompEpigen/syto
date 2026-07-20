import io
import os
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")

from rich.console import Console

from syto.visualization.inference import render_deconvolution_report
from tests.visualization.inference.test_deconvolution_data import synthetic_df


class TestOrchestrator(unittest.TestCase):
    def test_writes_pdf_and_prints_cli(self):
        buf = io.StringIO()
        console = Console(file=buf, width=250)
        with tempfile.TemporaryDirectory() as d:
            path = render_deconvolution_report(
                synthetic_df(),
                d,
                cfg={"top_n": 2, "known_truth": "A"},
                console=console,
            )
            self.assertTrue(os.path.exists(path))
            self.assertEqual(os.path.basename(str(path)), "deconvolution_report.pdf")
            self.assertGreater(os.path.getsize(path), 0)
        self.assertIn("Deconvolution", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
