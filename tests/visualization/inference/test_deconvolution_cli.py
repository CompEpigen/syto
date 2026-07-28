import io
import unittest

from rich.console import Console

from syto.visualization.inference.deconvolution_data import build_report_model
from syto.visualization.inference.deconvolution_cli import render_cli
from tests.visualization.inference.test_deconvolution_data import synthetic_df


class TestRenderCli(unittest.TestCase):
    def _render(self, **kwargs):
        buf = io.StringIO()
        console = Console(file=buf, width=250)
        model = build_report_model(synthetic_df(), **kwargs)
        render_cli(model, console=console)
        return buf.getvalue()

    def test_renders_without_error_and_includes_key_content(self):
        out = self._render()
        self.assertIn("Deconvolution", out)  # rule title
        self.assertIn("uxm", out)  # a method key fragment
        self.assertIn("baseline", out)  # group label
        self.assertIn("syto", out)  # group label

    def test_known_truth_line_present(self):
        out = self._render(known_truth="A")
        self.assertIn("rank", out.lower())

    def test_other_row_when_top_n_smaller(self):
        out = self._render(top_n=1)
        self.assertIn("other", out.lower())


if __name__ == "__main__":
    unittest.main()
