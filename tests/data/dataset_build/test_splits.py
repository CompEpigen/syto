import unittest
import numpy as np
import pandas as pd
from syto.data.dataset_build.splits import plan_splits, apply_splits


class TestPlanSplits(unittest.TestCase):
    def test_multi_file_class_every_split_covered(self):
        counts = pd.DataFrame(
            {
                "file": [f"f{i}" for i in range(5)],
                "original_label": [0] * 5,
                "n_reads": [100, 90, 80, 70, 60],
            }
        )
        plan = plan_splits(counts, seed=1)
        splits_used = set(plan["file_level"].values())
        self.assertEqual(splits_used, {"train", "valid", "test"})

    def test_single_file_class_is_read_level(self):
        counts = pd.DataFrame({"file": ["f0"], "original_label": [7], "n_reads": [100]})
        plan = plan_splits(counts)
        self.assertIn((7, "f0"), plan["read_level"])
        self.assertNotIn((7, "f0"), plan["file_level"])


class TestApplySplits(unittest.TestCase):
    def test_file_level_assignment(self):
        plan = {"file_level": {(0, "f0"): "train", (0, "f1"): "test"}, "read_level": {}}
        df = pd.DataFrame({"original_label": [0, 0], "file": ["f0", "f1"], "x": [1, 2]})
        out = apply_splits(df, plan)
        self.assertEqual(out.set_index("file").loc["f0", "split"], "train")
        self.assertEqual(out.set_index("file").loc["f1", "split"], "test")

    def test_read_level_is_deterministic_and_covers_three_splits(self):
        plan = {
            "file_level": {},
            "read_level": {
                (7, "f0"): {"seed": 42, "train_ratio": 0.7, "valid_ratio": 0.15}
            },
        }
        df = pd.DataFrame(
            {"original_label": [7] * 100, "file": ["f0"] * 100, "x": range(100)}
        )
        out1 = apply_splits(df, plan)
        out2 = apply_splits(df, plan)
        self.assertEqual(set(out1["split"]), {"train", "valid", "test"})
        pd.testing.assert_series_equal(out1["split"], out2["split"])
