import unittest

from syto.classification.classifiers.lazy_classifier_factory import (
    _CLF_REGISTRY,
)


def _make(arch, **kwargs):
    import importlib

    module_path, cls_name = _CLF_REGISTRY[arch].split(":")
    cls = getattr(importlib.import_module(module_path), cls_name)
    return cls.load(None, **kwargs)


class TestRequiredFitColumns(unittest.TestCase):
    def test_dismir_grg_head(self):
        clf = _make(
            "dismir",
            num_labels=39,
            num_grg_labels=39,
            classifier_head_implementation="grg_attention_based",
            grg_label_column="dmr_ctype_label",
            soft_labels=True,
        )
        cols = clf.required_fit_columns(
            {
                "model": {
                    "classifier_head_implementation": "grg_attention_based",
                    "grg_label_column": "dmr_ctype_label",
                }
            }
        )
        self.assertEqual(cols, ["input_ids", "methylation_ids", "dmr_ctype_label"])

    def test_lookup(self):
        clf = _make(
            "lookup",
            lookup_config={"num_classes": 39, "label_col": "original_label"},
        )
        cols = clf.required_fit_columns({})
        self.assertEqual(
            cols, ["name", "trimmed_start", "methylation_ids", "original_label"]
        )

    def test_cancer_detector(self):
        clf = _make("cancer_detector")
        cols = clf.required_fit_columns(
            {
                "training": {
                    "col_label": "label",
                    "col_marker_label": "dmr_ctype_label",
                }
            }
        )
        self.assertEqual(cols, ["methylation_ids", "label", "dmr_ctype_label"])


class TestMlflowFitParams(unittest.TestCase):
    def test_dismir_surfaces_flavour_and_head_config(self):
        clf = _make(
            "dismir",
            num_labels=39,
            num_grg_labels=39,
            dismir_flavor="minigru",
            classifier_head_implementation="grg_attention_based",
            grg_label_column="dmr_ctype_label",
            soft_labels=True,
        )
        self.assertEqual(
            clf.mlflow_fit_params(),
            {
                "flavour": "minigru",
                "num_labels": 39,
                "classifier_type": "grg_attention_based",
                "num_grg_labels": 39,
                "grg_label_column": "dmr_ctype_label",
                "soft_labels": True,
                "max_sequence_length": 150,
            },
        )

    def test_lookup_surfaces_label_config(self):
        clf = _make(
            "lookup",
            lookup_config={"num_classes": 39, "label_col": "original_label"},
        )
        params = clf.mlflow_fit_params()
        self.assertEqual(params["num_classes"], 39)
        self.assertEqual(params["label_col"], "original_label")
        # No foreign architecture knobs leak in.
        self.assertNotIn("dismir_flavor", params)
        self.assertNotIn("flavour", params)

    def test_cancer_detector_defaults_to_empty(self):
        # Its fit hyperparameters flow through the training config, not init.
        self.assertEqual(_make("cancer_detector").mlflow_fit_params(), {})
