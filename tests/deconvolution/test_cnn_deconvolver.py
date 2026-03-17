import unittest

import torch

from methyldl.deconvolution.deep_deconvolvers.cnn_deconvolver import CNNDeconvolver


class TestCNNDeconvolver(unittest.TestCase):
    """Tests for the convolutional deconvolver."""

    def setUp(self):
        """Create deterministic fixtures for forward-pass assertions."""
        torch.manual_seed(7)
        self.inputs = torch.rand(3, 39, 40, dtype=torch.float32)

    def test_init_uses_requested_output_dimensions(self):
        """The constructor should wire the feature extractor and head to the requested sizes."""
        model = CNNDeconvolver(n_cell_types=4)

        self.assertEqual(model.conv_layers[0].in_channels, 1)
        self.assertEqual(model.conv_layers[0].out_channels, 32)
        self.assertEqual(model.fc[-1].out_features, 4)

    def test_forward_returns_probabilities_for_each_sample(self):
        """The forward pass should emit one normalized proportion vector per batch item."""
        model = CNNDeconvolver(n_cell_types=4)
        model.eval()

        outputs = model(self.inputs)

        self.assertEqual(outputs.shape, (3, 4))
        self.assertTrue(torch.all(outputs >= 0).item())
        torch.testing.assert_close(outputs.sum(dim=-1), torch.ones(3))
