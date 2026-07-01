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
        "pseudobulk_h5_path": pb,
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
                engine=ScriptedEngine(raw),
                select_task=lambda: "inference",
                confirm_save=lambda: False,
                ask_path=lambda default: os.path.join(d, "unused.yaml"),
            )
        self.assertIsNone(saved)


if __name__ == "__main__":
    unittest.main()
