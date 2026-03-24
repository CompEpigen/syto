import unittest

import torch

from methyldl.deconvolution.deep_deconvolvers.flatten_mlp_deconvolver import (
    FlattenMLPDeconvolver,
)


class TestFlattenMLPDeconvolver(unittest.TestCase):
    """Tests for the flatten-then-MLP deconvolver."""

    def setUp(self):
        """Create deterministic inputs used across constructor and forward tests."""
        torch.manual_seed(19)
        self.inputs = torch.rand(4, 3, 5, dtype=torch.float32)

    def test_init_builds_hidden_stack_for_each_requested_dimension(self):
        """Each hidden width should expand into linear, norm, activation, and dropout layers."""
        model = FlattenMLPDeconvolver(
            n_dmr_groups=3,
            n_classes=5,
            n_cell_types=2,
            hidden_dims=[7, 4],
        )

        self.assertEqual(len(model.network), 9)
        self.assertEqual(model.network[0].in_features, 15)
        self.assertEqual(model.network[0].out_features, 7)
        self.assertEqual(model.network[4].out_features, 4)
        self.assertEqual(model.network[-1].out_features, 2)

    def test_init_accepts_no_hidden_layers(self):
        """An empty hidden_dims list should fall back to a single linear output layer."""
        model = FlattenMLPDeconvolver(
            n_dmr_groups=3,
            n_classes=5,
            n_cell_types=2,
            hidden_dims=[],
        )

        self.assertEqual(len(model.network), 1)
        self.assertEqual(model.network[0].in_features, 15)
        self.assertEqual(model.network[0].out_features, 2)

    def test_forward_returns_probability_distribution(self):
        """The MLP forward pass should emit normalized cell-type proportions."""
        model = FlattenMLPDeconvolver(
            n_dmr_groups=3,
            n_classes=5,
            n_cell_types=2,
            hidden_dims=[6],
        )
        model.eval()

        outputs = model(self.inputs)

        self.assertEqual(outputs.shape, (4, 2))
        self.assertTrue(torch.all(outputs >= 0).item())
        torch.testing.assert_close(outputs.sum(dim=-1), torch.ones(4))
