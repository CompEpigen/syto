""" """

from pathlib import Path
from typing import Union
import logging

import json
import numpy as np
import torch
from torch import nn

from methyldl.deconvolution.abstract_deconvolver import AbstractDeconvolver
from methyldl.deconvolution.deep_deconvolvers.training import train_matrix_deconvolver

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


class MLPDeconvolver(AbstractDeconvolver):
    """
    The MLP Deconvolver used in the Syto paper
    """

    def __init__(
        self,
        n_input_features: int,
        n_targets: int,
        logger: logging.Logger = _module_logger,
    ):
        super().__init__()
        self.n_input_features = n_input_features
        self.n_targets = n_targets
        self.logger = logger
        self.model = _MLPDeconvolverModel(n_input_features, n_targets)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "MLPDeconvolver":
        """Fit the MLP model to the data.

        Args:
            X: The input data (e.g., methylation profiles).
            y: The target cell type proportions.
            **kwargs: Additional keyword arguments for training, such as:
                - X_val: Validation input data (required).
                - y_val: Validation target data (required).
                - n_epochs: Number of training epochs.
                - batch_size: Training batch size.
                - lr: Learning rate for the optimizer.
                - weight_decay: Weight decay for regularization.
                - device: Device to train on ("cuda" or "cpu").
                - early_stopping_metric: Metric to monitor for early stopping (e.g., "val_mae").
                - early_stopping_patience: Number of epochs with no improvement before stopping.
                - scheduler_type: Type of learning rate scheduler to use (e.g., "plateau").
                - verbose: Verbosity level for training output.
        """
        # pylint: disable=attribute-defined-outside-init, invalid-name
        assert (
            "X_val" in kwargs and "y_val" in kwargs
        ), "Validation data (X_val, y_val) must be provided for training the MLP deconvolver."
        X_val = kwargs["X_val"]
        y_val = kwargs["y_val"]
        self.n_epochs_ = kwargs.get("n_epochs", 100)
        self.batch_size_ = kwargs.get("batch_size", 64)
        self.lr_ = kwargs.get("lr", 1e-3)
        self.weight_decay_ = kwargs.get("weight_decay", 1e-4)
        self.device_ = kwargs.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.early_stopping_metric_ = kwargs.get("early_stopping_metric", "val_mae")
        self.early_stopping_patience_ = kwargs.get("early_stopping_patience", 15)
        self.scheduler_type_ = kwargs.get("scheduler_type", "plateau")
        self.verbose_ = kwargs.get("verbose", 1)

        self.model, self.history_ = train_matrix_deconvolver(
            model=self.model,
            X_train=X,
            y_train=y,
            X_val=X_val,
            y_val=y_val,
            n_epochs=self.n_epochs_,
            batch_size=self.batch_size_,
            lr=self.lr_,
            weight_decay=self.weight_decay_,
            device=self.device_,
            early_stopping_metric=self.early_stopping_metric_,
            early_stopping_patience=self.early_stopping_patience_,
            scheduler_type=self.scheduler_type_,
            verbose=self.verbose_,
        )

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Predict the cell type proportions for the given input data."""
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()
        with torch.inference_mode():
            # pylint: disable=invalid-name
            X_tensor = torch.FloatTensor(X).to(device)
            predictions = self.model(X_tensor).cpu().numpy()

        return predictions

    def save(self, path: Union[str, Path], **kwargs) -> None:
        """Save the model to the specified path."""
        if not self.is_fitted:
            raise ValueError(
                "Cannot save an unfitted MLPDeconvolver. Call fit() first."
            )
        path = Path(path)
        file_extension = path.suffix
        assert file_extension == ".pt"

        metadata_path = kwargs.get("metadata_path", None)
        if metadata_path is None:
            metadata_path = path.parent / (path.stem + "_metadata.json")

        torch.save(self.model.state_dict(), path)
        self.logger.info("Model saved to %s", path)
        metadata = {
            "n_input_features": self.n_input_features,
            "n_targets": self.n_targets,
            "is_fitted": self.is_fitted,
            "params": {
                "n_epochs_": self.n_epochs_,
                "batch_size_": self.batch_size_,
                "lr_": self.lr_,
                "weight_decay_": self.weight_decay_,
                "device_": self.device_,
                "early_stopping_metric_": self.early_stopping_metric_,
                "early_stopping_patience_": self.early_stopping_patience_,
                "scheduler_type_": self.scheduler_type_,
            },
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "MLPDeconvolver":
        """Load the model from the specified path.

        Args:
            path: The path to the saved model file (should end with .pt).
            **kwargs: Additional keyword arguments, such as:
                - metadata_path: Optional path to the metadata JSON file. If not provided, it is assumed to be in the same directory as the model file with the same name but ending with "_metadata.json".
        """
        path = Path(path)
        file_extension = path.suffix
        assert file_extension == ".pt"

        metadata_path = kwargs.get("metadata_path", None)
        if metadata_path is None:
            metadata_path = path.parent / (path.stem + "_metadata.json")

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        n_input_features = metadata.get("n_input_features")
        n_targets = metadata["n_targets"] or metadata["n_cell_types"]
        mlp_deconv = cls(
            n_input_features=n_input_features,
            n_targets=n_targets,
        )
        mlp_deconv.model.load_state_dict(torch.load(path, weights_only=True))
        mlp_deconv.is_fitted = metadata["is_fitted"]
        for param, value in metadata.get("params", {}).items():
            setattr(mlp_deconv, param, value)

        return mlp_deconv

    def get_cv_metric(self, X, y, **kwargs) -> float:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before returning CV metric.")
        return self.history_.val_loss[self.history_.best_epoch]

    @property
    def cv_metric_name(self) -> str:
        return "Validation loss"
