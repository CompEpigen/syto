"""
Implementation of the Shallow Wide Network (SWN) deconvolver
used in the Syto paper.
"""

import logging

from torch import nn

from methyldl.deconvolution.deep_deconvolvers.abstract_nn_deconvolver import (
    AbstractNNDeconvolver,
)

_module_logger = logging.getLogger(__name__)


class _SWNDeconvolverModel(nn.Module):
    """The PyTorch Shallow Wide Network (SWN) model used as deconvolver.
    It is a single-hidden-layer feedforward neural network with the following architecture:
    - Input layer: size = n_input_features
    - Hidden layer: size = 1024, activation = GELU, dropout = 0.2
    - Output layer: size = n_targets, activation = Softmax
    """

    def __init__(
        self,
        n_input_features: int,
        n_targets: int,
    ):
        super().__init__()
        hidden_dim = 1024
        dropout = 0.2
        self.model = nn.Sequential(
            nn.Linear(n_input_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_targets),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        """Forward pass through the SWN model."""
        return self.model(x)


class SWNDeconvolver(AbstractNNDeconvolver):
    """
    The Shallow Wide Network (SWN) Deconvolver.
    """

    def __init__(
        self,
        n_input_features: int,
        n_targets: int,
        logger: logging.Logger = _module_logger,
    ):
        super().__init__(n_input_features, n_targets, logger)
        self._model = _SWNDeconvolverModel(n_input_features, n_targets)

    @property
    def model(self):
        """Property to access the underlying PyTorch SWN model."""
        return self._model

    @model.setter
    def model(self, value):
        """Setter for the underlying PyTorch SWN model."""
        self._model = value
