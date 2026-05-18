"""Tests for methyldl.deconvolution.deep_deconvolvers.swn.

Only covers behavior specific to SWNDeconvolver and _SWNDeconvolverModel.
Shared AbstractNNDeconvolver behavior (fit, predict, save/load round-trip,
CV interface, etc.) is already tested in tests/deconvolution/test_mlp.py.
"""

import unittest

import numpy as np
import torch
import torch.nn as nn

from methyldl.deconvolution.deep_deconvolvers.abstract_nn_deconvolver import (
    AbstractNNDeconvolver,
)
from methyldl.deconvolution.deep_deconvolvers.swn import (
    SWNDeconvolver,
    _SWNDeconvolverModel,
)

# ── Shared constants ──────────────────────────────────────────────────────────
N_FEATURES = 12
N_TARGETS = 4

SWN_HIDDEN_DIM = 1024
SWN_DROPOUT = 0.2


# ═══════════════════════════════════════════════════════════════════════════════
#  _SWNDeconvolverModel – architecture
# ═══════════════════════════════════════════════════════════════════════════════


class TestSWNDeconvolverModelArchitecture(unittest.TestCase):
    """Tests for the architecture of the internal _SWNDeconvolverModel."""

    def setUp(self):
        self.model = _SWNDeconvolverModel(N_FEATURES, N_TARGETS)
        self.linear_layers = [
            m for m in self.model.model.modules() if isinstance(m, nn.Linear)
        ]

    def test_has_exactly_two_linear_layers(self):
        """SWN is a single-hidden-layer network: exactly 2 Linear layers."""
        self.assertEqual(len(self.linear_layers), 2)

    def test_hidden_layer_width_is_1024(self):
        """The single hidden layer should have width 1024."""
        self.assertEqual(self.linear_layers[0].out_features, SWN_HIDDEN_DIM)

    def test_first_linear_accepts_input_features(self):
        """The first Linear layer should accept n_input_features as input."""
        self.assertEqual(self.linear_layers[0].in_features, N_FEATURES)

    def test_last_linear_emits_n_targets(self):
        """The output Linear layer should emit n_targets outputs."""
        self.assertEqual(self.linear_layers[-1].out_features, N_TARGETS)

    def test_ends_with_softmax(self):
        """The final module in the sequential chain should be Softmax."""
        last = list(self.model.model.children())[-1]
        self.assertIsInstance(last, nn.Softmax)

    def test_contains_gelu_activation(self):
        """The model should contain exactly one GELU activation."""
        gelu_layers = [m for m in self.model.model.modules() if isinstance(m, nn.GELU)]
        self.assertEqual(len(gelu_layers), 1)

    def test_contains_exactly_one_dropout_layer(self):
        """The model should contain exactly one Dropout layer."""
        dropout_layers = [
            m for m in self.model.model.modules() if isinstance(m, nn.Dropout)
        ]
        self.assertEqual(len(dropout_layers), 1)

    def test_dropout_rate_is_0_2(self):
        """The single Dropout layer should have p=0.2."""
        dropout_layers = [
            m for m in self.model.model.modules() if isinstance(m, nn.Dropout)
        ]
        self.assertAlmostEqual(dropout_layers[0].p, SWN_DROPOUT)

    def test_hidden_to_output_transition(self):
        """Second Linear layer should connect the hidden dim to n_targets."""
        self.assertEqual(self.linear_layers[1].in_features, SWN_HIDDEN_DIM)
        self.assertEqual(self.linear_layers[1].out_features, N_TARGETS)


class TestSWNDeconvolverModelForward(unittest.TestCase):
    """Forward-pass tests for _SWNDeconvolverModel."""

    def setUp(self):
        self.model = _SWNDeconvolverModel(N_FEATURES, N_TARGETS)
        self.model.eval()

    def test_output_shape(self):
        """Forward pass should return a tensor of shape (batch_size, n_targets)."""
        x = torch.rand(8, N_FEATURES)
        with torch.no_grad():
            out = self.model(x)
        self.assertEqual(tuple(out.shape), (8, N_TARGETS))

    def test_output_is_non_negative(self):
        """All output values should be non-negative."""
        x = torch.rand(16, N_FEATURES)
        with torch.no_grad():
            out = self.model(x)
        self.assertTrue(torch.all(out >= 0.0).item())

    def test_output_sums_to_one(self):
        """Outputs should sum to 1 across the target dimension (Softmax)."""
        x = torch.rand(16, N_FEATURES)
        with torch.no_grad():
            out = self.model(x)
        np.testing.assert_allclose(out.sum(dim=-1).numpy(), np.ones(16), atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
#  SWNDeconvolver – class identity and initialization
# ═══════════════════════════════════════════════════════════════════════════════


class TestSWNDeconvolverIdentity(unittest.TestCase):
    """Tests that SWNDeconvolver is a proper subclass with correct initialization."""

    def setUp(self):
        self.deconvolver = SWNDeconvolver(
            n_input_features=N_FEATURES, n_targets=N_TARGETS
        )

    def test_is_abstract_nn_deconvolver_subclass(self):
        """SWNDeconvolver must be a subclass of AbstractNNDeconvolver."""
        self.assertIsInstance(self.deconvolver, AbstractNNDeconvolver)

    def test_inner_model_is_swn_model(self):
        """The wrapped model should be an instance of _SWNDeconvolverModel."""
        self.assertIsInstance(self.deconvolver.model, _SWNDeconvolverModel)

    def test_model_property_returns_private_attribute(self):
        """The model property should expose the _model private attribute."""
        # pylint: disable-next=protected-access
        self.assertIs(self.deconvolver.model, self.deconvolver._model)

    def test_model_setter_updates_private_attribute(self):
        """Assigning to the model property should update _model."""
        new_model = _SWNDeconvolverModel(N_FEATURES, N_TARGETS)
        self.deconvolver.model = new_model
        # pylint: disable-next=protected-access
        self.assertIs(self.deconvolver._model, new_model)
        self.assertIs(self.deconvolver.model, new_model)

    def test_inner_model_dimensions_match_constructor_args(self):
        """The inner model must be built with the given feature and target dims."""
        linear_layers = [
            m
            for m in self.deconvolver.model.model.modules()
            if isinstance(m, nn.Linear)
        ]
        self.assertEqual(linear_layers[0].in_features, N_FEATURES)
        self.assertEqual(linear_layers[-1].out_features, N_TARGETS)


if __name__ == "__main__":
    unittest.main()
