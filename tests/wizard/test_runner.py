import os
import tempfile
import unittest

import yaml

from App.wizard import runner
from App.wizard.engine import WizardEngine


class ScriptedEngine(WizardEngine):
    def __init__(self, raw_by_key):
        self._raw_by_key = raw_by_key

    def _ask(self, spec):
        return self._raw_by_key.get(spec.key, spec.default)

    def _ask_expert_gate(self):
        return False

    def ask_checkbox(self, label, choices):
        return []


def _raw_with_real_paths(tmpdir):
    """Answer set whose path-validated fields point at real (empty) files."""
    labels = "App/labels_dict.json"
    ckpt = os.path.join(tmpdir, "weight.pt")
    open(ckpt, "w").close()
    mask = os.path.join(tmpdir, "mask.npz")
    open(mask, "w").close()
    atlas = os.path.join(tmpdir, "atlas.tsv")
    open(atlas, "w").close()
    bam = os.path.join(tmpdir, "x.bam")
    open(bam, "w").close()
    ref = os.path.join(tmpdir, "hg38.fa.gz")
    open(ref, "w").close()
    pb = os.path.join(tmpdir, "pb.h5")
    open(pb, "w").close()
    raw = {
        "run_syto": True,
        "classifier.classifier_type": "dismir",
        "classifier.dismir_flavor": "lstm",
        "classifier.classifier_head_implementation": "grg_attention_based",
        "classifier.labeling_scheme": "Soft Labels",
        "checkpoint_path": ckpt,
        "features_mask_path": mask,
        "labels_dict_path": labels,
        "num_labels": "39",
        "deconvolution.syto.atlas_path": atlas,
        "deconvolution.syto.atlas_name": "atlas",
        "input.type": "bam",
        "input.data_path": bam,
        "input.reference_path": ref,
        "input.data_type": "wgbs",
        "input.chromosomes": "all",
        "fill_in_missing_labels": True,
        "missing_label_strategy": "prior_blending",
        "pseudobulk_path": pb,
        "output_dir": "/tmp/out",
        "output.save_processed_reads": False,
        "output.save_predictions": True,
        "output.save_deconvolution": True,
    }
    return raw, bam


class TestRunner(unittest.TestCase):
    def test_run_wizard_saves_expected_config(self):
        self.assertTrue(os.path.exists("App/labels_dict.json"))
        with tempfile.TemporaryDirectory() as d:
            raw, bam = _raw_with_real_paths(d)
            out_path = os.path.join(d, "generated.yaml")

            saved = runner.run_wizard(
                select_mode=lambda: runner.MODE_FINE,
                engine=ScriptedEngine(raw),
                select_task=lambda: "inference",
                confirm_save=lambda: True,
                ask_path=lambda default: out_path,
            )

            self.assertEqual(saved, out_path)
            with open(out_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            self.assertEqual(cfg["classifier"]["classifier_type"], "dismir")
            self.assertIs(cfg["classifier"]["soft_labels"], True)
            self.assertEqual(cfg["num_labels"], 39)
            self.assertEqual(cfg["input"]["data_path"], bam)
            self.assertEqual(cfg["deconvolution"]["syto"]["methods"], [])

    def test_decline_save_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            raw, _ = _raw_with_real_paths(d)
            saved = runner.run_wizard(
                select_mode=lambda: runner.MODE_FINE,
                engine=ScriptedEngine(raw),
                select_task=lambda: "inference",
                confirm_save=lambda: False,
                ask_path=lambda default: os.path.join(d, "unused.yaml"),
            )
        self.assertIsNone(saved)


class TestTemplateMode(unittest.TestCase):
    def _run(self, tasks, d, classifier="dismir", confirm_overwrite=lambda p: False):
        self.classifier_asked = False

        def select_classifier(choices):
            self.classifier_asked = True
            return classifier

        return runner.run_wizard(
            select_mode=lambda: runner.MODE_TEMPLATE,
            select_tasks=lambda choices: tasks,
            select_classifier=select_classifier,
            ask_dir=lambda default: d,
            confirm_overwrite=confirm_overwrite,
        )

    def test_writes_one_file_per_selected_task(self):
        with tempfile.TemporaryDirectory() as d:
            paths = self._run(["fit_deconvolution", "inference"], d)
            self.assertEqual(
                [os.path.basename(p) for p in paths],
                ["fit_deconvolution.yaml", "inference.yaml"],
            )
            for path in paths:
                cfg = yaml.safe_load(open(path, encoding="utf-8"))
                self.assertIsInstance(cfg, dict)

    def test_file_carries_the_header_comment(self):
        with tempfile.TemporaryDirectory() as d:
            (path,) = self._run(["inference"], d)
            text = open(path, encoding="utf-8").read()
            self.assertTrue(text.startswith("# ---"))
            self.assertIn("Target classifier: dismir", text)
            self.assertIn("--task inference", text)

    def test_classifier_not_asked_when_no_task_needs_it(self):
        with tempfile.TemporaryDirectory() as d:
            self._run(["fit_calibration"], d)
            self.assertFalse(self.classifier_asked)

    def test_no_selection_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self._run([], d), [])
            self.assertEqual(os.listdir(d), [])

    def test_existing_file_is_kept_when_overwrite_declined(self):
        with tempfile.TemporaryDirectory() as d:
            existing = os.path.join(d, "fit_calibration.yaml")
            with open(existing, "w", encoding="utf-8") as f:
                f.write("hand written\n")

            paths = self._run(["fit_calibration", "fit_deconvolution"], d)

            self.assertEqual(
                [os.path.basename(p) for p in paths], ["fit_deconvolution.yaml"]
            )
            self.assertEqual(open(existing, encoding="utf-8").read(), "hand written\n")

    def test_existing_file_is_replaced_when_overwrite_confirmed(self):
        with tempfile.TemporaryDirectory() as d:
            existing = os.path.join(d, "fit_calibration.yaml")
            with open(existing, "w", encoding="utf-8") as f:
                f.write("hand written\n")

            paths = self._run(["fit_calibration"], d, confirm_overwrite=lambda p: True)

            self.assertEqual(paths, [existing])
            cfg = yaml.safe_load(open(existing, encoding="utf-8"))
            self.assertIn("deconvolvers", cfg)


if __name__ == "__main__":
    unittest.main()
