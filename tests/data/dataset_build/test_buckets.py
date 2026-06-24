import unittest
import pandas as pd
from syto.data.dataset_build.buckets import build_region_index, assign_region_bucket


def _atlas(n):
    return pd.DataFrame(
        {
            "chr": ["chr1"] * n,
            "start": list(range(100, 100 + 10 * n, 10)),
            "end": list(range(105, 105 + 10 * n, 10)),
            "name": [f"chr1:{100 + 10 * i}-{105 + 10 * i}" for i in range(n)],
        }
    )


class TestBuildRegionIndex(unittest.TestCase):
    def test_one_row_per_region_and_bucket_range(self):
        idx = build_region_index(_atlas(10), n_buckets=4)
        self.assertEqual(len(idx), 10)
        self.assertListEqual(sorted(idx.columns), ["name", "region_bucket"])
        self.assertTrue(idx["region_bucket"].between(0, 3).all())

    def test_buckets_are_contiguous_balanced(self):
        idx = build_region_index(_atlas(10), n_buckets=5)
        counts = idx["region_bucket"].value_counts()
        self.assertEqual(counts.min(), 2)
        self.assertEqual(counts.max(), 2)

    def test_more_buckets_than_regions_no_empty_assignment_error(self):
        idx = build_region_index(_atlas(3), n_buckets=10)
        self.assertEqual(len(idx), 3)
        self.assertEqual(idx["region_bucket"].nunique(), 3)


class TestAssignRegionBucket(unittest.TestCase):
    def test_join_and_drop_unknown(self):
        idx = build_region_index(_atlas(4), n_buckets=2)
        reads = pd.DataFrame(
            {"name": ["chr1:100-105", "chr1:130-135", "UNKNOWN"], "x": [1, 2, 3]}
        )
        out = assign_region_bucket(reads, idx)
        self.assertEqual(len(out), 2)
        self.assertIn("region_bucket", out.columns)
