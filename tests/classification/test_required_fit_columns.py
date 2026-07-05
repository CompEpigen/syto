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
