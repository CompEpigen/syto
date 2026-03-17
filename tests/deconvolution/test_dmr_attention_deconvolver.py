import unittest

import torch

from methyldl.deconvolution.deep_deconvolvers.dmr_attention_deconvolver import (
    DMRAttentionDeconvolver,
)


class TestDMRAttentionDeconvolver(unittest.TestCase):
    """Tests for the attention-based deconvolver."""

    def setUp(self):
        """Create deterministic fixtures for attention-path tests."""
        torch.manual_seed(11)
        self.inputs = torch.rand(2, 4, 5, dtype=torch.float32)

    def test_init_creates_attention_components_with_expected_shapes(self):
        """The constructor should size embeddings and prediction head from the given arguments."""
        model = DMRAttentionDeconvolver(
            n_dmr_groups=4,
            n_pred_classes=5,
            n_cell_types=3,
            embed_dim=8,
            num_heads=2,
        )

        self.assertEqual(tuple(model.dmr_embeddings.shape), (4, 8))
        self.assertEqual(model.attention.embed_dim, 8)
        self.assertEqual(model.attention.num_heads, 2)
        self.assertEqual(model.proportion_head[-1].out_features, 3)

    def test_forward_returns_normalized_proportions(self):
        """The attention model should output one probability vector per batch element."""
        model = DMRAttentionDeconvolver(
            n_dmr_groups=4,
            n_pred_classes=5,
            n_cell_types=3,
            embed_dim=8,
            num_heads=2,
        )
        model.eval()

        outputs = model(self.inputs)

        self.assertEqual(outputs.shape, (2, 3))
        self.assertTrue(torch.all(outputs >= 0).item())
        torch.testing.assert_close(outputs.sum(dim=-1), torch.ones(2))
