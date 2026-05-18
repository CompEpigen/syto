"""
Implementation of the Multi-Layer Perceptron (MLP) deconvolver
used in the Syto paper.
"""

import logging

from torch import nn

from methyldl.deconvolution.deep_deconvolvers.abstract_nn_deconvolver import (
    AbstractNNDeconvolver,
)

_module_logger = logging.getLogger(__name__)


class _MLPDeconvolverModel(nn.Module):
    """The PyTorch MLP model used as deconvolver in the Syto paper.
    It is a simple feedforward neural network with the following architecture:
    - Input layer: size = n_input_features
    - Hidden layer 1: size = 512, activation = GELU, dropout = 0.2
    - Hidden layer 2: size = 256, activation = GELU, dropout = 0.2
    - Hidden layer 3: size = n_input_features, activation = GELU, dropout = 0.1
    - Output layer: size = n_targets, activation = Softmax
    """

    def __init__(
        self,
        n_input_features: int,
        n_targets: int,
    ):
        super().__init__()
        hidden_dims = [512, 256]
        layers = []
        current_input_dim = n_input_features
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_input_dim, h_dim),
                    nn.GELU(),
                    nn.Dropout(0.2),
                ]
            )
            current_input_dim = h_dim
        # Final extra hidden layer matching the input
        layers.extend(
            [
                nn.Linear(current_input_dim, n_input_features),
                nn.GELU(),
                nn.Dropout(0.1),
            ]
        )
        layers.append(nn.Linear(n_input_features, n_targets))
        layers.append(nn.Softmax(dim=-1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the MLP model."""
        return self.model(x)


class MLPDeconvolver(AbstractNNDeconvolver):
    """
    The MLP Deconvolver used in the Syto paper
    """

    def __init__(
        self,
        n_input_features: int,
        n_targets: int,
        logger: logging.Logger = _module_logger,
    ):
        super().__init__(n_input_features, n_targets, logger)
        self._model = _MLPDeconvolverModel(n_input_features, n_targets)

    @property
    def model(self):
        """Property to access the underlying PyTorch MLP model."""
        return self._model

    @model.setter
    def model(self, value):
        """Setter for the underlying PyTorch MLP model."""
        self._model = value
