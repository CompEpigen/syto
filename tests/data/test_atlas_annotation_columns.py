"""Carrying extra atlas columns (e.g. GR-group labels) onto reads at inference."""

import logging
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from syto.app.inference import InferencePipeline
from syto.data.atlases.uxm_atlases import UXMMethylationAtlas

LABELS = {0: "GroupA", 1: "GroupB"}


def _atlas():
    df = pd.DataFrame(
        {
            "chr": ["chr1"] * 3,
            "start": [100, 200, 200],
            "end": [150, 250, 250],
            "startCpG": [10, 20, 20],
            "endCpG": [15, 25, 25],
            # r1 marks both groups: two atlas rows, as for a shared U250 region.
            "target": ["GroupA", "GroupA", "GroupB"],
            "name": ["r0", "r1", "r1"],
            "direction": ["U"] * 3,
            "GroupA": [0.9, 0.8, 0.1],
            "GroupB": [0.1, 0.2, 0.9],
            "grg_cluster25_label": [3, 4, 7],
        }
    )
    return UXMMethylationAtlas(
        "t", "hg19", atlas_df=df, cell_types=["GroupA", "GroupB"]
    )


def _reads():
    return pd.DataFrame({"name": ["r0", "r1", "r0"], "read_start": [101, 201, 120]})


class TestAnnotationColumns(unittest.TestCase):
    def test_columns_follow_each_reads_target(self):
        out = _atlas()._annotate_dmr_labels(
            _reads(), LABELS, {}, ["grg_cluster25_label"]
        )
        got = sorted(
            zip(out["name"], out["dmr_ctype_label"], out["grg_cluster25_label"])
        )
        # r1 is copied once per target, each copy with its own row's label.
        self.assertEqual(got, [("r0", 0, 3), ("r0", 0, 3), ("r1", 0, 4), ("r1", 1, 7)])

    def test_without_columns_behaviour_is_unchanged(self):
        out = _atlas()._annotate_dmr_labels(_reads(), LABELS, {})
        self.assertNotIn("grg_cluster25_label", out.columns)
        self.assertEqual(len(out), 4)

    def test_stale_annotation_columns_are_replaced(self):
        reads = _reads().assign(grg_cluster25_label=99)
        out = _atlas()._annotate_dmr_labels(reads, LABELS, {}, ["grg_cluster25_label"])
        self.assertNotIn(99, out["grg_cluster25_label"].tolist())

    def test_missing_atlas_column_is_reported(self):
        with self.assertRaisesRegex(ValueError, "absent from the atlas"):
            _atlas()._annotate_dmr_labels(_reads(), LABELS, {}, ["nope"])

    def test_annotation_columns_require_labels(self):
        with self.assertRaisesRegex(ValueError, "requires labels_dict"):
            _atlas().prepare_reads(_reads(), annotation_columns=["grg_cluster25_label"])


class TestInferenceWiring(unittest.TestCase):
    def _pipeline(self, config):
        pipeline = InferencePipeline.__new__(InferencePipeline)
        pipeline.config = config
        pipeline.logger = logging.getLogger("t")
        pipeline.labels_dict = LABELS
        pipeline.num_labels = 2
        pipeline.cell_type_match_dict = {}
        return pipeline

    def test_overlap_passes_annotation_columns_to_the_atlas(self):
        pipeline = self._pipeline({})
        pipeline.region_annotation_columns = ["grg_cluster25_label"]
        atlas = MagicMock()
        atlas.prepare_reads.return_value = pd.DataFrame()
        reads = pd.DataFrame({"chromosome": ["chr1"], "read_start": [1]})
        pipeline._overlap_reads_with_atlas(reads, atlas)
        self.assertEqual(
            atlas.prepare_reads.call_args.kwargs["annotation_columns"],
            ["grg_cluster25_label"],
        )

    def _built_kwargs(self, classifier_cfg):
        pipeline = self._pipeline(
            {"classifier": classifier_cfg, "checkpoint_path": "w.pt"}
        )
        with patch("syto.app.inference.read_classifier_factory") as factory:
            pipeline._build_classifier()
        return factory.call_args.kwargs

    def test_num_grg_labels_reaches_the_classifier(self):
        kwargs = self._built_kwargs(
            {
                "classifier_type": "dismir",
                "grg_label_column": "grg_cluster25_label",
                "num_grg_labels": 383,
            }
        )
        self.assertEqual(kwargs["num_grg_labels"], 383)
        self.assertEqual(kwargs["grg_label_column"], "grg_cluster25_label")

    def test_num_grg_labels_left_to_default_when_unset(self):
        self.assertNotIn(
            "num_grg_labels", self._built_kwargs({"classifier_type": "dismir"})
        )


if __name__ == "__main__":
    unittest.main()
