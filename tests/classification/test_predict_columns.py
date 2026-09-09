"""Predict-time column requirements must not demand ground-truth labels.

A fitted classifier asked to label new reads needs features, not answers. The
pseudobulk generator only ever predicts, so if a classifier reports its
fit-time columns there, generation starts demanding label columns that the
data need not carry -- and fails outright when a checkpoint's stored label
column names something absent from the dataset.
"""

import unittest

from syto.classification.classifiers.abstract_read_classifier import (
    AbstractReadClassifier,
)
from syto.classification.classifiers.lookup import LabelConfig, LookupClassifier


class TestLookupPredictColumns(unittest.TestCase):
    def test_predict_columns_exclude_the_label(self):
        clf = LookupClassifier(LabelConfig(label_col="hard_label_with_background"))
        predict_cols = clf.required_predict_columns({})
        self.assertNotIn("hard_label_with_background", predict_cols)
        self.assertEqual(predict_cols, ["name", "read_start", "methylation_ids"])

    def test_fit_columns_still_carry_the_label(self):
        # fitting genuinely needs it: for soft-label runs label_col is
        # 'original_label' while the pipeline's own label_column is the soft
        # vector, so the classifier must ask for its own column.
        clf = LookupClassifier(LabelConfig(label_col="original_label"))
        self.assertIn("original_label", clf.required_fit_columns({}))

    def test_a_stale_stored_label_col_cannot_break_prediction(self):
        # checkpoints fitted before the config stopped being mutated in place
        # carry label_col='label', a column no published dataset has.
        clf = LookupClassifier(LabelConfig(label_col="label"))
        self.assertNotIn("label", clf.required_predict_columns({}))


class _LabelFreeClassifier(AbstractReadClassifier):
    """Minimal stand-in for a classifier whose fit columns carry no label."""

    def required_fit_columns(self, config):
        return ["input_ids", "methylation_ids"]

    # the rest of the interface is irrelevant here
    def fit_classificaton(self, *a, **k):  # pragma: no cover
        raise NotImplementedError

    def predict_split(self, *a, **k):  # pragma: no cover
        raise NotImplementedError

    def save(self, *a, **k):  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def load(cls, *a, **k):  # pragma: no cover
        raise NotImplementedError


class TestDefaultIsFitColumns(unittest.TestCase):
    def test_classifier_without_an_override_falls_back(self):
        clf = _LabelFreeClassifier()
        cfg = {"model": {}}
        self.assertEqual(
            clf.required_predict_columns(cfg), clf.required_fit_columns(cfg)
        )


if __name__ == "__main__":
    unittest.main()
