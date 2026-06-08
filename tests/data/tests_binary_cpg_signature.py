import unittest

import pandas as pd

from syto.data.omics_signatures_handlers.binary_cpg_signature import (
    BinaryCpGSignatureHandler,
)


class TestExtractSignature(unittest.TestCase):
    """Tests for BinaryCpGSignatureHandler.extract_signature."""

    def test_basic_default_columns(self):
        """A pattern is converted to (position, state) pairs with unknowns dropped."""
        handler = BinaryCpGSignatureHandler()
        row = pd.Series({"start": 100, "pattern": "01201"})
        # '2' is an unknown status and should be ignored
        sig = handler.extract_signature(row)
        self.assertEqual(sig, ((100, 0), (101, 1), (103, 0), (104, 1)))

    def test_custom_columns(self):
        """The handler reads from the configured start/pattern column names."""
        handler = BinaryCpGSignatureHandler(
            start_column="trimmed_start", methylation_pattern_column="meth"
        )
        row = pd.Series({"trimmed_start": 10, "meth": "10"})
        sig = handler.extract_signature(row)
        self.assertEqual(sig, ((10, 1), (11, 0)))

    def test_all_unknown_states_returns_empty(self):
        """A pattern with no 0/1 characters yields an empty signature."""
        handler = BinaryCpGSignatureHandler()
        row = pd.Series({"start": 5, "pattern": "...x"})
        self.assertEqual(handler.extract_signature(row), ())

    def test_empty_pattern_returns_empty(self):
        """An empty pattern string yields an empty signature."""
        handler = BinaryCpGSignatureHandler()
        row = pd.Series({"start": 0, "pattern": ""})
        self.assertEqual(handler.extract_signature(row), ())


class TestComputeDistanceDispatcher(unittest.TestCase):
    """Tests for BinaryCpGSignatureHandler.compute_distance_dispatcher."""

    def setUp(self):
        """Create a handler shared across the dispatcher tests."""
        self.handler = BinaryCpGSignatureHandler()

    def test_returns_jaccard_callable(self):
        """Requesting 'jaccard' returns the bound jaccard distance method."""
        fct = self.handler.compute_distance_dispatcher("jaccard")
        self.assertEqual(fct, self.handler.compute_jaccard_distance)

    def test_unknown_distance_raises(self):
        """An unsupported distance name raises ValueError."""
        with self.assertRaises(ValueError):
            self.handler.compute_distance_dispatcher("euclidean")

    def test_jaccard_in_valid_distances(self):
        """'jaccard' is advertised in the handler's VALID_DISTANCES set."""
        self.assertIn("jaccard", self.handler.VALID_DISTANCES)


class TestComputeJaccardDistance(unittest.TestCase):
    """Tests for BinaryCpGSignatureHandler.compute_jaccard_distance."""

    def setUp(self):
        """Create a handler shared across the jaccard distance tests."""
        self.handler = BinaryCpGSignatureHandler()

    def test_identical_signatures(self):
        """Two identical signatures have distance 0."""
        sig = ((100, 0), (101, 1))
        self.assertEqual(self.handler.compute_jaccard_distance(sig, sig), 0.0)

    def test_one_mismatch(self):
        """One differing (position, state) pair gives distance 1 - 1/3."""
        sig1 = ((100, 0), (101, 1))
        sig2 = ((100, 0), (101, 0))  # share (100,0); union has 3 elements
        self.assertAlmostEqual(
            self.handler.compute_jaccard_distance(sig1, sig2), 2 / 3
        )

    def test_subset_signature(self):
        """A signature that is a subset of another gives distance 1 - 1/2."""
        sig1 = ((100, 0), (101, 1))
        sig2 = ((100, 0),)  # intersection 1, union 2
        self.assertEqual(self.handler.compute_jaccard_distance(sig1, sig2), 0.5)

    def test_no_shared_positions(self):
        """Fully disjoint signatures have distance 1."""
        sig1 = ((100, 0), (101, 1))
        sig2 = ((200, 0),)  # disjoint
        self.assertEqual(self.handler.compute_jaccard_distance(sig1, sig2), 1.0)

    def test_both_empty(self):
        """Two empty signatures are treated as identical (distance 0)."""
        self.assertEqual(self.handler.compute_jaccard_distance((), ()), 0.0)

    def test_symmetry(self):
        """The jaccard distance is symmetric in its two arguments."""
        # pylint: disable=arguments-out-of-order
        sig1 = ((100, 0), (101, 1))
        sig2 = ((100, 0), (102, 0))
        self.assertEqual(
            self.handler.compute_jaccard_distance(sig1, sig2),
            self.handler.compute_jaccard_distance(sig2, sig1),
        )


if __name__ == "__main__":
    unittest.main()
