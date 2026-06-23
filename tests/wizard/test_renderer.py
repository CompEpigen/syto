import os
import tempfile
import unittest

import yaml

from App.wizard import renderer


class TestRenderer(unittest.TestCase):
    def test_dump_yaml_roundtrips_and_preserves_key_order(self):
        config = {"classifier": {"classifier_type": "dismir"}, "num_labels": 39}
        text = renderer.dump_yaml(config)
        self.assertEqual(yaml.safe_load(text), config)
        # block style, not flow style
        self.assertNotIn("{", text)
        # insertion order preserved (classifier before num_labels)
        self.assertLess(text.index("classifier"), text.index("num_labels"))

    def test_save_config_writes_file_and_creates_dirs(self):
        config = {"a": 1}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "nested", "out.yaml")
            renderer.save_config(config, path)
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as f:
                self.assertEqual(yaml.safe_load(f), config)

    def test_default_save_path_shape(self):
        path = renderer.default_save_path("inference")
        self.assertTrue(path.startswith("App/config/inference_"))
        self.assertTrue(path.endswith(".yaml"))

    def test_render_preview_and_run_command_do_not_raise(self):
        renderer.render_preview({"a": {"b": 1}})
        renderer.print_run_command("inference", "App/config/rrbs/x.yaml")


if __name__ == "__main__":
    unittest.main()
