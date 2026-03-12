from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from methyldl.inference.evaluation import compute_deconvolution_metrics


@dataclass
class DeconvolverOutput:
    """Structured output from the deconvolver containing predictions and intermediate features."""

    proportions: torch.Tensor  # (batch, n_cell_types) - final predicted proportions
    diag_features: torch.Tensor  # (batch, hidden_dim) - diagonal pathway features
    confusion_features: torch.Tensor  # (batch, hidden_dim) - confusion pathway features
    reject_features: torch.Tensor  # (batch, hidden_dim//2) - rejection pathway features
    combined_features: (
        torch.Tensor
    )  # (batch, combined_dim) - pre-head combined features
    logits: torch.Tensor  # (batch, n_cell_types) - pre-softmax logits


@dataclass
class TrainingHistory:
    """Stores training metrics over epochs."""

    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    val_mae: list = field(default_factory=list)
    val_mse: list = field(default_factory=list)
    val_kl: list = field(default_factory=list)
    val_max_error: list = field(default_factory=list)
    val_cosine_sim: list = field(default_factory=list)
    learning_rates: list = field(default_factory=list)
    best_epoch: int = 0
    stopped_early: bool = False

    def to_dict(self):
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_mae": self.val_mae,
            "val_mse": self.val_mse,
            "val_kl": self.val_kl,
            "val_max_error": self.val_max_error,
            "val_cosine_sim": self.val_cosine_sim,
            "learning_rates": self.learning_rates,
            "best_epoch": self.best_epoch,
            "stopped_early": self.stopped_early,
        }


class EarlyStopping:
    """
    Early stopping handler.

    Parameters
    ----------
    patience : int
        Number of epochs to wait for improvement before stopping.
    min_delta : float
        Minimum change to qualify as an improvement.
    mode : str
        'min' for metrics where lower is better (loss, MAE),
        'max' for metrics where higher is better (cosine_sim).
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-5,
        mode: Literal["min", "max"] = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if self.mode == "min":
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def train_matrix_deconvolver(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    early_stopping_metric: Literal[
        "val_loss", "val_mae", "val_mse", "val_kl", "val_max_error", "val_cosine_sim"
    ] = "val_mae",
    early_stopping_patience: int = 15,
    early_stopping_min_delta: float = 1e-5,
    scheduler_type: Literal["cosine", "plateau", "none"] = "plateau",
    loss_weights: dict = None,
    verbose: int = 1,
    save_path: Optional[str] = None,
) -> tuple[nn.Module, TrainingHistory]:
    """
    Train a deconvolution model with early stopping support.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    X_train, y_train : np.ndarray
        Training data and labels.
    X_val, y_val : np.ndarray
        Validation data and labels.
    n_epochs : int
        Maximum number of epochs.
    batch_size : int
        Batch size for training.
    lr : float
        Initial learning rate.
    weight_decay : float
        L2 regularization weight.
    device : str
        Device to train on ('cuda' or 'cpu').
    early_stopping_metric : str
        Metric to monitor for early stopping.
        Options: 'val_loss', 'val_mae', 'val_mse', 'val_kl', 'val_max_error', 'val_cosine_sim'
    early_stopping_patience : int
        Number of epochs to wait for improvement.
    early_stopping_min_delta : float
        Minimum change to qualify as improvement.
    scheduler_type : str
        Learning rate scheduler type: 'cosine', 'plateau', or 'none'.
    loss_weights : dict
        Weights for loss components: {'mse': float, 'kl': float}.
        Default: {'mse': 1.0, 'kl': 0.5}
    verbose : int
        Verbosity level (0=silent, 1=progress, 2=detailed).
    save_path : str, optional
        Path to save the best model.

    Returns
    -------
    model : nn.Module
        The trained model (loaded with best weights).
    history : TrainingHistory
        Training history with all metrics.
    """

    # Default loss weights
    if loss_weights is None:
        loss_weights = {"mse": 1.0, "kl": 0.5}

    # Determine early stopping mode
    maximize_metrics = {"val_cosine_sim"}
    es_mode = "max" if early_stopping_metric in maximize_metrics else "min"

    # Setup data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train), torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Setup model, optimizer, scheduler
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=es_mode, patience=5, factor=0.5, min_lr=1e-6
        )
    else:
        scheduler = None

    # Loss function
    def loss_fn(pred, target, eps=1e-8):
        mse = nn.functional.mse_loss(pred, target)
        kl = (
            (target * (target.clamp(min=eps).log() - pred.clamp(min=eps).log()))
            .sum(dim=-1)
            .mean()
        )
        return loss_weights["mse"] * mse + loss_weights["kl"] * kl

    # Initialize tracking
    history = TrainingHistory()
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        min_delta=early_stopping_min_delta,
        mode=es_mode,
    )
    best_model_state = None

    # Training loop
    for epoch in range(n_epochs):
        # === Training phase ===
        model.train()
        train_loss = 0
        n_batches = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= n_batches

        # === Validation phase ===
        model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)

                val_loss += loss_fn(pred, y).item()
                all_preds.append(pred.cpu())
                all_targets.append(y.cpu())

        val_loss /= len(val_loader)

        # Compute all metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = compute_deconvolution_metrics(all_preds, all_targets)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)
        history.val_mae.append(metrics["mae"])
        history.val_mse.append(metrics["mse"])
        history.val_kl.append(metrics["kl"])
        history.val_max_error.append(metrics["max_error"])
        history.val_cosine_sim.append(metrics["cosine_sim"])
        history.learning_rates.append(current_lr)

        # Get the metric value for early stopping
        metric_map = {
            "val_loss": val_loss,
            "val_mae": metrics["mae"],
            "val_mse": metrics["mse"],
            "val_kl": metrics["kl"],
            "val_max_error": metrics["max_error"],
            "val_cosine_sim": metrics["cosine_sim"],
        }
        current_metric = metric_map[early_stopping_metric]

        # Check if this is the best model
        is_best = False
        if early_stopping.best_score is None:
            is_best = True
        elif (
            es_mode == "min"
            and current_metric < early_stopping.best_score - early_stopping_min_delta
        ):
            is_best = True
        elif (
            es_mode == "max"
            and current_metric > early_stopping.best_score + early_stopping_min_delta
        ):
            is_best = True

        if is_best:
            best_model_state = deepcopy(model.state_dict())
            history.best_epoch = epoch
            if save_path:
                torch.save(model.state_dict(), save_path)

        # Update scheduler
        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(current_metric)
            else:
                scheduler.step()

        # Logging
        if verbose >= 1 and (epoch + 1) % max(1, n_epochs // 100) == 0:
            print(
                f"Epoch {epoch+1:3d}/{n_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val MAE: {metrics['mae']:.4f} | "
                f"Val Max Error: {metrics['max_error']:.4f} | "
                f"LR: {current_lr:.2e}" + (" *" if is_best else "")
            )

        if verbose >= 2:
            print(
                f"         Val MSE: {metrics['mse']:.4f} | "
                f"Val KL: {metrics['kl']:.4f} | "
                f"Val MaxErr: {metrics['max_error']:.4f}"
            )

        # Early stopping check
        if early_stopping(current_metric, epoch):
            history.stopped_early = True
            if verbose >= 1:
                print(
                    f"\nEarly stopping triggered at epoch {epoch+1}. "
                    f"Best epoch: {early_stopping.best_epoch+1} "
                    f"({early_stopping_metric}={early_stopping.best_score:.4f})"
                )
            break

    # Load best model weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    if verbose >= 1:
        print(f"\nTraining complete. Best epoch: {history.best_epoch+1}")
        print(f"Best {early_stopping_metric}: {early_stopping.best_score:.4f}")

    return model, history
