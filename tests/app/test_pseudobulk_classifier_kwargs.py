"""The pseudobulk pipeline must size the GRG head from its config."""

import logging
import unittest
from unittest.mock import patch

from syto.app.pseudobulk_pipeline import PseudoBulkPipeline


def _pipeline(classifier_config):
    pipeline = PseudoBulkPipeline.__new__(PseudoBulkPipeline)
    pipeline._classifier = None
    pipeline.labels_dict = {"0": "a"}
    pipeline.config = {
        "classifier_type": "dismir",
        "classifier_checkpoint": "weight.pt",
        "classifier_config": classifier_config,
    }
    pipeline.logger = logging.getLogger("t")
    return pipeline


class TestClassifierKwargs(unittest.TestCase):
    def test_num_grg_labels_is_forwarded_when_configured(self):
        """A checkpoint trained on 50 groups cannot load into a 39-group head."""
        pipeline = _pipeline(
            {"num_prediction_classes": 7, "num_grg_labels": 50, "seq_length": 256}
        )
        with patch("syto.app.pseudobulk_pipeline.read_classifier_factory") as factory:
            pipeline._get_classifier()
        self.assertEqual(factory.call_args.kwargs["num_grg_labels"], 50)

    def test_absent_num_grg_labels_is_not_passed_as_none(self):
        """Classifiers default it themselves; an explicit None would break that."""
        pipeline = _pipeline({"num_prediction_classes": 40})
        with patch("syto.app.pseudobulk_pipeline.read_classifier_factory") as factory:
            pipeline._get_classifier()
        self.assertNotIn("num_grg_labels", factory.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
