"""
Common wrapper for neural network-based deconvolvers, used in the Syto paper
(namely MLPDeconvolver and SWNDeconvolver).
"""

from abc import abstractmethod
from pathlib import Path
from typing import Union
import logging

import json
import numpy as np
import torch

from syto.deconvolution.abstract_deconvolver import AbstractDeconvolver
from syto.deconvolution.deep_deconvolvers.training import train_matrix_deconvolver
from syto.torch_device import DEVICE

_module_logger = logging.getLogger(__name__)


class AbstractNNDeconvolver(AbstractDeconvolver):
    """
    Abstract base class for neural network-based deconvolvers.
    This class provides common functionality for training, saving, and loading
    neural network models used for deconvolution tasks.
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
        self.is_fitted = False

    @property
    @abstractmethod
    def model(self):
        """Property to access the underlying PyTorch model. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the model property.")

    @model.setter
    @abstractmethod
    def model(self, value):
        """Setter for the underlying PyTorch model. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the model setter.")

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> "AbstractNNDeconvolver":
        """Fit the NN model to the data.

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
        ), "Validation data (X_val, y_val) must be provided for training the NN deconvolver."
        X_val = kwargs["X_val"]
        y_val = kwargs["y_val"]
        self.n_epochs_ = kwargs.get("n_epochs", 100)
        self.batch_size_ = kwargs.get("batch_size", 64)
        self.lr_ = kwargs.get("lr", 1e-3)
        self.weight_decay_ = kwargs.get("weight_decay", 1e-4)
        self.device_ = kwargs.get("device", DEVICE)
        self.early_stopping_metric_ = kwargs.get("early_stopping_metric", "val_mae")
        self.early_stopping_patience_ = kwargs.get("early_stopping_patience", 15)
        self.scheduler_type_ = kwargs.get("scheduler_type", "plateau")
        self.verbose_ = kwargs.get("verbose", 1)

        self.model, self.history = train_matrix_deconvolver(
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
        device = kwargs.get("device", DEVICE)
        self.model.to(device)
        self.model.eval()
        with torch.inference_mode():
            X_tensor = torch.FloatTensor(X).to(device)  # pylint: disable=invalid-name
            predictions = self.model(X_tensor).cpu().numpy()

        return predictions

    def save(self, path: Union[str, Path], **kwargs) -> None:
        """Save the model to the specified path.

        Args:
            path: The path to save the model file (should end with .pt).
            **kwargs: Additional keyword arguments, such as:
                - metadata_path: Optional path to the metadata JSON file.
                If not provided, it is assumed to be in the same directory as the model
                file with the same name but ending with "_metadata.json".
        """
        if not self.is_fitted:
            raise ValueError("Cannot save an unfitted NNDeconvolver. Call fit() first.")
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
                "n_epochs": self.n_epochs_,
                "batch_size": self.batch_size_,
                "lr": self.lr_,
                "weight_decay": self.weight_decay_,
                "device": self.device_,
                "early_stopping_metric": self.early_stopping_metric_,
                "early_stopping_patience": self.early_stopping_patience_,
                "scheduler_type": self.scheduler_type_,
            },
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f)

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "AbstractNNDeconvolver":
        """Load the model from the specified path.

        Args:
            path: The path to the saved model file (should end with .pt).
            **kwargs: Additional keyword arguments, such as:
                - metadata_path: Optional path to the metadata JSON file.
                If not provided, it is assumed to be in the same directory as the model
                file with the same name but ending with "_metadata.json".
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
        nn_deconv = cls(
            n_input_features=n_input_features,
            n_targets=n_targets,
        )
        nn_deconv.model.load_state_dict(
            torch.load(path, map_location=DEVICE, weights_only=True)
        )
        nn_deconv.is_fitted = metadata["is_fitted"]
        for param, value in metadata.get("params", {}).items():
            if not param.endswith("_"):
                param += "_"
            setattr(nn_deconv, param, value)
        # The metadata records the device the model was *trained* on, which is
        # not necessarily one this machine has.
        nn_deconv.device_ = DEVICE
        nn_deconv.model.to(DEVICE)

        return nn_deconv

    def get_cv_metric(self, X, y, **kwargs) -> float:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before returning CV metric.")
        return self.history.val_loss[self.history.best_epoch]

    @property
    def cv_metric_name(self) -> str:
        return "Validation loss"
