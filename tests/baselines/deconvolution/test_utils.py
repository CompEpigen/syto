import unittest

from baselines.deconvolution.utils import rearange_deconvolution_results


class TestRearangeDeconvolutionResults(unittest.TestCase):
    """Tests for the shared rearange_deconvolution_results utility."""

    def test_reorders_proportions_to_label_index_order(self):
        """Proportions should be reordered to match label indices 0…n-1."""
        labels_dict_reversed = {f"ct{i}": i for i in range(39)}
        perm = list(range(38, -1, -1))
        ref_cells = [f"ct{i}" for i in perm]
        proportions = [i / 100.0 for i in perm]

        out = rearange_deconvolution_results(
            labels_dict_reversed=labels_dict_reversed,
            proportions=proportions,
            ref_cells=ref_cells,
        )

        self.assertEqual(len(out), 39)
        self.assertEqual(out, [i / 100.0 for i in range(39)])

    def test_n_labels_truncates_output(self):
        """n_labels limits the number of returned proportions."""
        labels = {"ct0": 0, "ct1": 1, "ct2": 2}
        ref_cells = ["ct2", "ct1", "ct0"]
        proportions = [0.5, 0.3, 0.2]

        out = rearange_deconvolution_results(labels, proportions, ref_cells, n_labels=2)

        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out[0], 0.2)
        self.assertAlmostEqual(out[1], 0.3)

    def test_identity_when_ref_cells_already_sorted(self):
        """No reordering needed when ref_cells match label index order."""
        labels = {"ct0": 0, "ct1": 1, "ct2": 2}
        ref_cells = ["ct0", "ct1", "ct2"]
        proportions = [0.5, 0.3, 0.2]

        out = rearange_deconvolution_results(labels, proportions, ref_cells)

        self.assertEqual(out, proportions)
