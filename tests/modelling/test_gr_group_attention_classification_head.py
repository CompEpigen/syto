import unittest
from types import SimpleNamespace

import torch

from syto.modelling.gr_group_attention_classification_head import (
    GRGAttentionClassificationHead,
)


def build_test_config(
    hidden_size=8,
    num_labels=3,
    num_dmr_labels=4,
    attention_probs_dropout_prob=0.0,
    hidden_dropout_prob=0.0,
    layer_norm_eps=1e-12,
):
    """Create a minimal configuration object for GRGAttentionClassificationHead tests."""
    return SimpleNamespace(
        hidden_size=hidden_size,
        num_labels=num_labels,
        num_dmr_labels=num_dmr_labels,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        hidden_dropout_prob=hidden_dropout_prob,
        layer_norm_eps=layer_norm_eps,
    )


class TestGRGAttentionClassificationHead(unittest.TestCase):
    """Exercise the public behavior of the DMR attention classifier."""

    def setUp(self):
        """Create a deterministic classifier and reusable test inputs."""
        torch.manual_seed(7)
        self.config = build_test_config()
        self.classifier = GRGAttentionClassificationHead(self.config)
        self.classifier.eval()

    def test_initialization_builds_expected_layers(self):
        """Verify the classifier stores config values and wires submodules correctly."""
        self.assertEqual(self.classifier.hidden_size, self.config.hidden_size)
        self.assertEqual(self.classifier.num_labels, self.config.num_labels)
        self.assertEqual(self.classifier.num_dmr_labels, self.config.num_dmr_labels)

        self.assertEqual(
            self.classifier.dmr_embedding.num_embeddings, self.config.num_dmr_labels
        )
        self.assertEqual(
            self.classifier.dmr_embedding.embedding_dim, self.config.hidden_size
        )

        self.assertEqual(
            self.classifier.query_proj.in_features, self.config.hidden_size
        )
        self.assertEqual(
            self.classifier.query_proj.out_features, self.config.hidden_size
        )
        self.assertEqual(self.classifier.key_proj.in_features, self.config.hidden_size)
        self.assertEqual(
            self.classifier.value_proj.out_features, self.config.hidden_size
        )

        self.assertEqual(
            self.classifier.context_fusion[0].in_features, self.config.hidden_size * 2
        )
        self.assertEqual(
            self.classifier.context_fusion[0].out_features, self.config.hidden_size
        )
        self.assertIsInstance(self.classifier.context_fusion[1], torch.nn.LayerNorm)

        self.assertEqual(
            self.classifier.classifier[0].in_features, self.config.hidden_size
        )
        self.assertEqual(
            self.classifier.classifier[-1].out_features, self.config.num_labels
        )
        self.assertAlmostEqual(self.classifier.scale, self.config.hidden_size**-0.5)

    def test_forward_without_attention_mask_returns_logits_and_normalized_attention(
        self,
    ):
        """Ensure the unmasked forward path returns finite tensors with expected shapes."""
        sequence_output = torch.randn(2, 5, self.config.hidden_size)
        dmr_ids = torch.tensor([0, 3], dtype=torch.long)

        logits, attention_weights = self.classifier(sequence_output, dmr_ids)

        self.assertEqual(logits.shape, (2, self.config.num_labels))
        self.assertEqual(attention_weights.shape, (2, 5))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(attention_weights).all())
        self.assertTrue((attention_weights >= 0).all())

        # The module returns the mean of row-wise softmax outputs, so each sample
        # should still sum to 1 when dropout is disabled.
        self.assertTrue(
            torch.allclose(
                attention_weights.sum(dim=1),
                torch.ones(2),
                atol=1e-6,
            )
        )

    def test_forward_with_attention_mask_excludes_masked_positions(self):
        """Check that masked keys receive effectively zero attention in the masked path."""
        sequence_output = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.5, -0.5, 0.25, -0.25, 1.5, -1.5],
                    [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
                    [-7.0, -7.0, -7.0, -7.0, -7.0, -7.0, -7.0, -7.0],
                ]
            ]
        )
        dmr_ids = torch.tensor([1], dtype=torch.long)
        attention_mask = torch.tensor([[1, 0, 0]], dtype=torch.long)

        logits, attention_weights = self.classifier(
            sequence_output, dmr_ids, attention_mask=attention_mask
        )

        self.assertEqual(logits.shape, (1, self.config.num_labels))
        self.assertEqual(attention_weights.shape, (1, 3))

        # Using a single unmasked token makes the expected behavior unambiguous:
        # every query must attend entirely to position 0 after masking.
        expected_attention = torch.tensor([[1.0, 0.0, 0.0]])
        self.assertTrue(
            torch.allclose(attention_weights, expected_attention, atol=1e-6)
        )

    def test_forward_with_mask_changes_attention_distribution(self):
        """Confirm that providing a mask changes the returned attention profile."""
        sequence_output = torch.randn(1, 4, self.config.hidden_size)
        dmr_ids = torch.tensor([2], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)

        _, unmasked_attention = self.classifier(sequence_output, dmr_ids)
        _, masked_attention = self.classifier(
            sequence_output, dmr_ids, attention_mask=attention_mask
        )

        self.assertFalse(torch.allclose(unmasked_attention, masked_attention))
        self.assertTrue(
            torch.allclose(masked_attention[:, 2:], torch.zeros(1, 2), atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(masked_attention.sum(dim=1), torch.ones(1), atol=1e-6)
        )

    def test_forward_with_all_zero_mask_keeps_uniform_attention_and_finite_outputs(
        self,
    ):
        """Lock in the current all-zero-mask behavior as part of the public contract."""
        sequence_output = torch.randn(1, 4, self.config.hidden_size)
        dmr_ids = torch.tensor([1], dtype=torch.long)
        attention_mask = torch.zeros((1, 4), dtype=torch.long)

        logits, attention_weights = self.classifier(
            sequence_output, dmr_ids, attention_mask=attention_mask
        )

        self.assertEqual(logits.shape, (1, self.config.num_labels))
        self.assertEqual(attention_weights.shape, (1, 4))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.isfinite(attention_weights).all())

        # Because every score is replaced by the same large negative constant,
        # softmax yields a uniform distribution instead of zeros or NaNs.
        expected_attention = torch.full((1, 4), 0.25)
        self.assertTrue(
            torch.allclose(attention_weights, expected_attention, atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(attention_weights.sum(dim=1), torch.ones(1), atol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()
