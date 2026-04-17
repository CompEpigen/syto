"""Vector scaling (diagonal) calibration for multiclass probability predictions.

Learns a calibration map of the form softmax(diag(w) @ ln(q) + b) that
transforms uncalibrated probability vectors q into calibrated ones.
This is the diagonal special case of Dirichlet calibration from
Kull et al. (NeurIPS 2019), with log-transformed inputs and softmax output.

Uses PyTorch for optimization with mini-batch SGD/Adam and early stopping.
"""

import json
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from sklearn.model_selection import KFold, ParameterGrid
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin


class CalibrationMethod(Enum):
    """Supported calibration map parametrisations.

    Each variant constrains the weight matrix W differently:

    - FULL: W is an unconstrained (k x k) matrix. There are k^2 + k parameters.
    - DIAGONAL: W is restricted to a diagonal matrix (k diagonal + k bias
      parameters). Equivalent to vector scaling.
    - TEMPERATURE: W = t * I with a single scalar t and no bias.
      Equivalent to temperature scaling.
    """

    FULL = "full"
    DIAGONAL = "diagonal"
    TEMPERATURE = "temperature"


class _TrainedLinearCalibrationModel(nn.Module):
    """Learnable calibration map with residual parametrisation.

    Uses a residual connection around the identity map so that the
    learnable parameters represent the *deviation* from the identity:

    - Full:        logits = (I + W_delta) @ x + b
    - Diagonal:    logits = (1 + diag_delta) * x + b
    - Temperature: logits = (1 + t_delta) * x

    All delta parameters are initialised to zero, so the starting point
    is the identity calibration map (no-op). This makes L2 regularisation
    directly penalise deviation from identity, without needing the ODIR
    workaround of excluding diagonal elements.

    Args:
        n_classes: Number of classes (k).
        method: Which parametrisation to use for the weight matrix.
    """

    def __init__(
        self,
        n_classes: int,
        method: CalibrationMethod,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.method = method

        # All deltas are zero-initialised => identity map at start:
        # Full:        (I + 0) @ x + 0 = x  =>  softmax(x) = q
        # Diagonal:    (1 + 0) * x + 0 = x  =>  softmax(x) = q
        # Temperature: (1 + 0) * x     = x  =>  softmax(x) = q
        if method == CalibrationMethod.FULL:
            self.W_delta = nn.Parameter(
                torch.zeros(n_classes, n_classes, dtype=torch.float64)
            )
            self.b = nn.Parameter(torch.zeros(n_classes, dtype=torch.float64))
        elif method == CalibrationMethod.DIAGONAL:
            self.diag_delta = nn.Parameter(torch.zeros(n_classes, dtype=torch.float64))
            self.b = nn.Parameter(torch.zeros(n_classes, dtype=torch.float64))
        elif method == CalibrationMethod.TEMPERATURE:
            self.t_delta = nn.Parameter(torch.zeros(1, dtype=torch.float64))
        else:
            raise ValueError(f"Unknown method: {method}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute calibration logits (before normalization).

        Args:
            x: (N, K) tensor of (log-)transformed input probabilities.

        Returns:
            (N, K) tensor of logits.
        """
        if self.method == CalibrationMethod.FULL:
            # Residual: logits = (I + W_delta) @ x + b = x + W_delta @ x + b
            return x + x @ self.W_delta.T + self.b
        elif self.method == CalibrationMethod.DIAGONAL:
            # Residual: logits = (1 + diag_delta) * x + b = x + diag_delta * x + b
            return x + x * self.diag_delta + self.b
        elif self.method == CalibrationMethod.TEMPERATURE:
            # Residual: logits = (1 + t_delta) * x
            return x + x * self.t_delta
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def activate(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the normalization function to logits.

        Args:
            logits: (N, K) tensor of logits from :meth:`forward`.

        Returns:
            (N, K) tensor of calibrated probabilities.
        """
        return torch.softmax(logits, dim=1)

    def get_W_matrix(self) -> torch.Tensor:
        """Return the full (k x k) effective weight matrix W = I + delta.

        For diagonal and temperature methods, this reconstructs the
        equivalent full matrix so that inspection code can treat all
        methods uniformly.

        Returns:
            (k, k) tensor representing the calibration weight matrix.
        """
        I = torch.eye(
            self.n_classes,
            dtype=torch.float64,
            device=next(self.parameters()).device,
        )
        if self.method == CalibrationMethod.FULL:
            return I + self.W_delta
        elif self.method == CalibrationMethod.DIAGONAL:
            return I + torch.diag(self.diag_delta)
        elif self.method == CalibrationMethod.TEMPERATURE:
            return (1.0 + self.t_delta) * I
        else:
            raise ValueError(f"Unknown method: {self.method}")


class VectorScalingCalibrator(BaseEstimator, RegressorMixin):
    """

    Args:
        reg_lambda: L2 regularisation strength
        optimizer: Optimizer to use - "adam" or "sgd".
        lr: Learning rate for the optimizer.
        scheduler: Learning rate schedule - "plateau" (default), "constant", "linear",
            or "cosine". Linear decays lr to 0 over max_iter epochs.
            Cosine uses cosine annealing to 0.
        max_iter: Maximum number of optimization epochs.
        batch_size: Mini-batch size. Use 0 or None for full-batch
            (entire dataset as one batch per epoch).
        patience: Early stopping patience (number of epochs without
            improvement before stopping).
        tol: Minimum improvement in loss to reset patience counter.
        device: Torch device string ("cpu", "cuda", etc.).
        verbose: If True (default), print training progress. Set to False
            to silence all output during fitting.
    """

    def __init__(
        self,
        reg_lambda: float = 0.0,
        lr: float = 0.01,
        max_iter: int = 500,
        optimizer: str = "adam",
        scheduler: str = "plateau",
        batch_size: int = 1000,
        patience: int = 50,
        tol: float = 1e-7,
        device: str = "cuda",
        verbose: bool = True,
        plateau_factor: float = 0.5,
        plateau_patience: int = 10,
    ):
        assert optimizer in {"adam", "sgd"}, "Unsupported optimizer"
        assert scheduler in {
            "constant",
            "linear",
            "cosine",
            "plateau",
        }, "Unsupported scheduler"
        self.reg_lambda = reg_lambda
        self.optimizer = optimizer
        self.lr = lr
        self.scheduler = scheduler
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.patience = patience
        self.tol = tol
        self.device = device
        self.verbose = verbose
        self.plateau_factor = plateau_factor
        self.plateau_patience = plateau_patience

        # Fitted attributes (set by fit())
        self.model_: Optional[_TrainedLinearCalibrationModel] = None
        self.n_classes_: int = 0
        self.final_loss_: float = float("inf")
        self.best_epoch_: int = -1
        self.best_metrics_: dict[str, float] = {}
        self.history_: dict[str, list] = {}

    @staticmethod
    def _clip_and_log(X: np.ndarray) -> np.ndarray:
        """Clip probabilities away from 0/1 and take the natural log.

        Clipping to [eps, 1 - eps] prevents -inf/+inf values in the
        log-transformed input, which would break optimisation.

        Args:
            X: Probability array of shape (n_samples, n_classes).

        Returns:
            Log-transformed array with the same shape.
        """
        eps = np.finfo(X.dtype).tiny
        return np.log(np.clip(X, eps, 1.0 - eps))

    def _prepare_input(self, X: np.ndarray) -> np.ndarray:
        """Prepare input probabilities: for now just log-transform.
        In the future we could add other transformations or normalizations here.

        Args:
            X: Probability array of shape (n_samples, n_classes).

        Returns:
            Transformed array with the same shape.
        """
        return self._clip_and_log(X)

    def _compute_loss(
        self,
        model: _TrainedLinearCalibrationModel,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the calibration loss + regularisation (L2 penalty on deviation from identity).

        Args:
            model: The calibration model.
            inputs: (N, K) transformed input probabilities.
            targets: (N, K) target distributions (hard or soft labels).

        Returns:
            Scalar loss tensor.
        """
        logits = model(inputs)
        probs = model.activate(logits)
        # Clamp to avoid log(0) which would give -inf and corrupt the loss
        probs = torch.clamp(probs, min=1e-20)
        # NLL: works with both hard (one-hot) and soft label targets
        loss = -torch.mean(torch.sum(targets * torch.log(probs), dim=1))

        # --- Regularisation ---
        reg = torch.tensor(0.0, dtype=torch.float64, device=inputs.device)
        for param in model.parameters():
            reg = reg + torch.sum(param**2)
        loss = loss + self.reg_lambda * reg

        return loss

    @staticmethod
    def _prepare_targets(y: np.ndarray, n_samples: int, n_classes: int) -> np.ndarray:
        """Convert labels to a (n_samples, n_classes) target matrix.

        Accepts integer class labels, one-hot vectors, or soft probability
        vectors.
        """
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            targets = np.zeros((n_samples, n_classes), dtype=np.float64)
            targets[np.arange(n_samples), y.astype(int)] = 1.0
        else:
            if y.shape != (n_samples, n_classes):
                raise ValueError(
                    f"Target shape {y.shape} does not match "
                    f"predictions shape ({n_samples}, {n_classes})."
                )
            targets = y
        return targets

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        report_every: int = 100,
    ) -> "VectorScalingCalibrator":
        """Fit the calibration map on a training set.

        When ``X_val`` and ``y_val`` are provided, early stopping and model
        selection are based on the validation loss (computed without
        regularisation). Otherwise they fall back to the training loss.

        Args:
            X: Uncalibrated probability predictions, shape (n_samples, n_classes).
            y: Ground-truth labels. Accepts:
                - Integer class labels of shape (n_samples,)
                - One-hot encoded vectors of shape (n_samples, n_classes)
                - Soft label probability vectors of shape (n_samples, n_classes)
                  where each row sums to 1 (e.g. mixture proportions).
            X_val: Optional validation predictions, same format as ``X``.
            y_val: Optional validation labels, same format as ``y``.
            report_every: Print metrics every this many epochs.

        Returns:
            self. The training history is stored in ``self.history_``.
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_classes = X.shape
        self.n_classes_ = n_classes

        targets = self._prepare_targets(y, n_samples, n_classes)

        input_np = self._prepare_input(X)

        dev = torch.device(self.device)
        input_all = torch.as_tensor(input_np, dtype=torch.float64, device=dev)
        targets_all = torch.as_tensor(targets, dtype=torch.float64, device=dev)

        # Prepare validation tensors if provided
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val = np.asarray(X_val, dtype=np.float64)
            n_val = X_val.shape[0]
            val_targets = self._prepare_targets(y_val, n_val, n_classes)
            input_val = torch.as_tensor(
                self._prepare_input(X_val), dtype=torch.float64, device=dev
            )
            targets_val = torch.as_tensor(val_targets, dtype=torch.float64, device=dev)

        # Build calibration model and optimiser
        model = _TrainedLinearCalibrationModel(
            n_classes, CalibrationMethod.DIAGONAL
        ).to(dev)

        if self.optimizer == "adam":
            optim = torch.optim.Adam(model.parameters(), lr=self.lr)
        elif self.optimizer == "sgd":
            optim = torch.optim.SGD(model.parameters(), lr=self.lr)
        else:
            raise ValueError(
                f"Unknown optimizer '{self.optimizer}'. Choose 'adam' or 'sgd'."
            )

        # Learning rate scheduler
        if self.scheduler == "constant":
            sched = None
        elif self.scheduler == "linear":
            # Linearly decay lr to 0 over max_iter epochs
            sched = torch.optim.lr_scheduler.LinearLR(
                optim, start_factor=1.0, end_factor=0.0, total_iters=self.max_iter
            )
        elif self.scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=self.max_iter, eta_min=0.0
            )
        elif self.scheduler == "plateau":
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim,
                mode="min",
                factor=self.plateau_factor,
                patience=self.plateau_patience,
            )
        else:
            raise ValueError(
                f"Unknown scheduler '{self.scheduler}'. "
                "Choose 'plateau', 'constant', 'linear', or 'cosine'."
            )

        # batch_size=0 means full-batch: use the entire dataset each epoch
        effective_bs = self.batch_size if self.batch_size else n_samples

        no_improve = 0

        # History: store metrics for every epoch
        history: dict[str, list] = {
            "epoch": [],
            "lr": [],
            "train_loss": [],
            "train_mse": [],
        }
        if has_val:
            history["val_loss"] = []
            history["val_mse"] = []

        # --- Epoch -1: metrics for the untrained (identity) model ---
        model.eval()
        with torch.no_grad():
            train_logits = model(input_all)
            train_calibrated = model.activate(train_logits)
            init_train_loss = self._compute_loss(model, input_all, targets_all).item()
            init_train_mse = torch.mean((train_calibrated - targets_all) ** 2).item()

            history["epoch"].append(-1)
            history["lr"].append(self.lr)
            history["train_loss"].append(init_train_loss)
            history["train_mse"].append(init_train_mse)

            if has_val:
                val_logits = model(input_val)
                val_calibrated = model.activate(val_logits)
                init_val_loss = self._compute_loss(model, input_val, targets_val).item()
                init_val_mse = torch.mean((val_calibrated - targets_val) ** 2).item()
                history["val_loss"].append(init_val_loss)
                history["val_mse"].append(init_val_mse)
                selection_loss_init = init_val_loss
            else:
                selection_loss_init = init_train_loss

            best_loss = selection_loss_init
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch_idx = 0  # index into history lists
        if self.verbose:
            if has_val:
                print(
                    f"Epoch -1: train_loss = {init_train_loss:.7e}, "
                    f"val_loss = {init_val_loss:.7e}, "
                    f"train_mse = {init_train_mse:.7e}, val_mse = {init_val_mse:.7e}"
                )
            else:
                print(
                    f"Epoch -1: train_loss = {init_train_loss:.7e}, "
                    f"train_mse = {init_train_mse:.7e}"
                )

        for epoch in range(self.max_iter):
            # --- Training step ---
            model.train()
            # Shuffle data each epoch for stochastic mini-batching
            perm = torch.randperm(n_samples, device=dev)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_samples, effective_bs):
                idx = perm[start : start + effective_bs]
                batch_input = input_all[idx]
                batch_tgt = targets_all[idx]

                optim.zero_grad()
                loss = self._compute_loss(model, batch_input, batch_tgt)
                loss.backward()
                optim.step()
                epoch_loss += loss.item()
                n_batches += 1

            # Current learning rate (first param group)
            current_lr = optim.param_groups[0]["lr"]

            # --- Selection loss: use val set if available, else train loss ---
            train_loss = epoch_loss / n_batches
            model.eval()
            with torch.no_grad():
                # Train MSE: mean squared error between calibrated probs and targets
                train_logits = model(input_all)
                train_calibrated = model.activate(train_logits)
                train_mse = torch.mean((train_calibrated - targets_all) ** 2).item()

                if has_val:
                    # Validation loss without regularisation for clean selection
                    val_logits = model(input_val)
                    val_calibrated = model.activate(val_logits)
                    selection_loss = self._compute_loss(
                        model, input_val, targets_val
                    ).item()
                    val_mse = torch.mean((val_calibrated - targets_val) ** 2).item()
                else:
                    selection_loss = train_loss

            # Step the learning rate scheduler after computing selection_loss
            if sched is not None:
                if self.scheduler == "plateau":
                    sched.step(selection_loss)
                else:
                    sched.step()

            # Record metrics for this epoch
            history["epoch"].append(epoch)
            history["lr"].append(current_lr)
            history["train_loss"].append(train_loss)
            history["train_mse"].append(train_mse)
            if has_val:
                history["val_loss"].append(selection_loss)
                history["val_mse"].append(val_mse)

            # Early stopping: restore the best model if loss stagnates
            if selection_loss < best_loss - self.tol:
                best_loss = selection_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch_idx = len(history["epoch"]) - 1
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.patience:
                if self.verbose:
                    print(
                        f"Early stopping at epoch {epoch} (best loss: {best_loss:.7e})"
                    )
                break

            if self.verbose and epoch % report_every == 0:
                if has_val:
                    print(
                        f"Epoch {epoch}: train_loss = {train_loss:.7e}, "
                        f"val_loss = {selection_loss:.7e}, "
                        f"train_mse = {train_mse:.7e}, val_mse = {val_mse:.7e}"
                    )
                else:
                    print(
                        f"Epoch {epoch}: train_loss = {train_loss:.7e}, "
                        f"train_mse = {train_mse:.7e}"
                    )

        # Restore the model parameters that achieved the lowest loss
        if best_state is not None:
            model.load_state_dict(best_state)

        self.model_ = model
        self.final_loss_ = best_loss
        self.history_ = history
        self.best_epoch_ = history["epoch"][best_epoch_idx]
        self.best_metrics_ = {
            key: values[best_epoch_idx] for key, values in history.items()
        }

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Apply the learned calibration map to new predictions.

        Args:
            X: Uncalibrated probability predictions, shape (n_samples, n_classes).

        Returns:
            Calibrated probabilities, shape (n_samples, n_classes).
        """
        X = np.asarray(X, dtype=np.float64)
        input_np = self._prepare_input(X)

        dev = next(self.model_.parameters()).device
        input_t = torch.as_tensor(input_np, dtype=torch.float64, device=dev)

        with torch.no_grad():
            logits = self.model_(input_t)
            calibrated = self.model_.activate(logits)

        return calibrated.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Apply the learned calibration map (alias for :meth:`predict_proba`).

        Args:
            X: Uncalibrated probability predictions, shape (n_samples, n_classes).

        Returns:
            Calibrated probabilities, shape (n_samples, n_classes).
        """
        return self.predict_proba(X)

    def save(self, path: Union[str, Path]) -> None:
        """Save the fitted calibrator to a .npz file.

        Args:
            path: File path to save to (typically .npz).

        Raises:
            RuntimeError: If the calibrator has not been fitted yet.
        """
        if self.model_ is None:
            raise RuntimeError("Cannot save an unfitted calibrator. Call fit() first.")

        # Separate JSON-serialisable metadata from numpy arrays
        metadata = {
            "params": self.get_params(),
            "n_classes_": self.n_classes_,
            "final_loss_": self.final_loss_,
            "best_epoch_": self.best_epoch_,
            "best_metrics_": self.best_metrics_,
            "history_": self.history_,
            "model_method": self.model_.method.value,
        }
        arrays = {
            name: param.detach().cpu().numpy()
            for name, param in self.model_.state_dict().items()
        }
        arrays["_metadata_json"] = np.array(json.dumps(metadata))
        np.savez(path, **arrays)

    def load(self, path: Union[str, Path]) -> "VectorScalingCalibrator":
        """Load a fitted calibrator from a .npz file into this instance.

        Args:
            path: File path to load from.

        Returns:
            self, with all fitted attributes restored.
        """
        data = np.load(path, allow_pickle=False)
        metadata = json.loads(str(data["_metadata_json"]))

        # Restore constructor params
        for key, value in metadata["params"].items():
            setattr(self, key, value)

        self.n_classes_ = metadata["n_classes_"]
        self.final_loss_ = metadata["final_loss_"]
        self.best_epoch_ = metadata["best_epoch_"]
        self.best_metrics_ = metadata["best_metrics_"]
        self.history_ = metadata["history_"]

        method_enum = CalibrationMethod(metadata["model_method"])
        model = _TrainedLinearCalibrationModel(self.n_classes_, method_enum)
        state_dict = {name: torch.as_tensor(data[name]) for name in model.state_dict()}
        model.load_state_dict(state_dict)
        model.to(torch.device(self.device))
        self.model_ = model
        return self


class VectorScalingCalibratorCV(BaseEstimator, RegressorMixin):
    """Grid-search with k-fold cross-validation for :class:`VectorScalingCalibrator`.

    Evaluates every combination of hyperparameters via k-fold CV, and
    retrains a final calibrator on the full dataset using the
    best-performing hyperparameters.

    Args:
        reg_lambda_list: List of L2 regularisation strengths to search over.
        lr_list: List of learning rates to search over.
        max_iter_list: List of maximum epoch counts to search over.
        optimizer: Optimizer passed to each inner calibrator ("adam" or "sgd").
        scheduler: LR scheduler passed to each inner calibrator
            ("plateau", "constant", "linear", or "cosine").
        patience: Early-stopping patience passed to each inner calibrator.
        tol: Minimum loss improvement for early stopping.
        n_folds: Number of cross-validation folds (default 5).

    Attributes:
        best_calibrator_: The :class:`VectorScalingCalibrator` retrained on the
            full dataset with the best hyperparameters. ``None`` before fitting.
        best_params_: Dictionary of the best hyperparameter combination found
            by CV. ``None`` before fitting.
        best_cv_val_loss_: Mean validation loss across folds for the best
            hyperparameter combination. ``None`` before fitting.
    """

    def __init__(
        self,
        reg_lambda_list: list[float] = None,
        lr_list: list[float] = None,
        max_iter_list: list[int] = None,
        optimizer: str = "adam",
        scheduler: str = "plateau",
        patience: int = 50,
        tol: float = 1e-7,
        n_folds: int = 5,
        batch_size: int = None,
        verbose: bool = False,
        plateau_factor: float = 0.5,
        plateau_patience: int = 10,
    ):
        if reg_lambda_list is None:
            reg_lambda_list = [0.0, 1e-4, 1e-3, 1e-2]
        if lr_list is None:
            lr_list = [1e-4, 1e-3, 1e-2]
        if max_iter_list is None:
            max_iter_list = [1000]
        self.reg_lambda_list = reg_lambda_list
        self.lr_list = lr_list
        self.max_iter_list = max_iter_list
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.patience = patience
        self.tol = tol
        self.n_folds = n_folds
        self.batch_size = batch_size
        self.verbose = verbose
        self.plateau_factor = plateau_factor
        self.plateau_patience = plateau_patience

        self.best_calibrator_: Optional[VectorScalingCalibrator] = None
        self.best_params_: Optional[dict] = None
        self.best_cv_val_loss_: Optional[float] = None
        self.best_metrics_per_param_: Optional[dict] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "VectorScalingCalibratorCV":
        """Run k-fold CV grid search, and retrain on all data.

        Evaluates every hyperparameter combination via k-fold CV,
        then retrains a final :class:`VectorScalingCalibrator` on the full
        dataset with the best hyperparameters.

        Args:
            X: Probability predictions, shape (n_samples, n_classes).
            y: Labels (integer, one-hot, or soft), shape (n_samples, n_classes).

        Returns:
            self, with ``best_calibrator_``, ``best_params_``, and
            ``best_cv_val_loss_`` set.
        """
        param_grid = {
            "reg_lambda": self.reg_lambda_list,
            "lr": self.lr_list,
            "max_iter": self.max_iter_list,
            "optimizer": [self.optimizer],
            "scheduler": [self.scheduler],
            "patience": [self.patience],
            "tol": [self.tol],
            "batch_size": [self.batch_size],
            "plateau_factor": [self.plateau_factor],
            "plateau_patience": [self.plateau_patience],
        }

        # n-fold CV to find best hyperparameters
        best_mean_val_loss = float("inf")
        best_params = None
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        # Store best metrics for each hyperparameter combination
        best_metrics_per_param = {}
        with tqdm(
            total=len(ParameterGrid(param_grid)) * self.n_folds,
            desc="CV grid search",
            disable=not self.verbose,
        ) as pbar:
            for params in ParameterGrid(param_grid):
                fold_val_losses = []
                for train_idx, val_idx in kf.split(X):
                    X_tr_fold, X_val_fold = X[train_idx], X[val_idx]
                    y_tr_fold, y_val_fold = y[train_idx], y[val_idx]

                    calibrator = VectorScalingCalibrator(**params, verbose=False)
                    calibrator.fit(X_tr_fold, y_tr_fold, X_val_fold, y_val_fold)
                    best_metrics_per_param[tuple(params.items())] = (
                        calibrator.best_metrics_
                    )
                    val_loss = calibrator.best_metrics_.get("val_loss", float("inf"))
                    fold_val_losses.append(val_loss)
                    pbar.update(1)

                mean_val_loss = np.mean(fold_val_losses)
                if mean_val_loss < best_mean_val_loss:
                    best_mean_val_loss = mean_val_loss
                    best_params = params
                    pbar.set_postfix(
                        {
                            "best_val_loss": f"{best_mean_val_loss:.7e}",
                            "best_params": best_params,
                        }
                    )

        self.best_metrics_per_param_ = best_metrics_per_param

        # Retrain on the full dataset with the best hyperparameters
        final_calibrator = VectorScalingCalibrator(**best_params, verbose=False)
        final_calibrator.fit(X, y)

        self.best_calibrator_ = final_calibrator
        self.best_params_ = best_params
        self.best_cv_val_loss_ = best_mean_val_loss
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Apply the best calibration map to new predictions.

        Args:
            X: Uncalibrated probability predictions, shape (n_samples, n_classes).

        Returns:
            Calibrated probabilities, shape (n_samples, n_classes).

        Raises:
            RuntimeError: If :meth:`fit` has not been called yet.
        """
        if self.best_calibrator_ is None:
            raise RuntimeError("Calibrator not fitted yet. Call fit() first.")
        return self.best_calibrator_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Apply the best calibration map (alias for :meth:`predict_proba`).

        Args:
            X: Uncalibrated probability predictions, shape (n_samples, n_classes).

        Returns:
            Calibrated probabilities, shape (n_samples, n_classes).

        Raises:
            RuntimeError: If :meth:`fit` has not been called yet.
        """
        if self.best_calibrator_ is None:
            raise RuntimeError("Calibrator not fitted yet. Call fit() first.")
        return self.best_calibrator_.predict(X)

    def save(self, path: Union[str, Path]) -> None:
        """Save the fitted CV calibrator to a .npz file.

        Persists all constructor parameters, CV results, and the best
        inner calibrator (including its model weights).

        Args:
            path: File path to save to (typically .npz).

        Raises:
            RuntimeError: If the calibrator has not been fitted yet.
        """
        if self.best_calibrator_ is None:
            raise RuntimeError("Cannot save an unfitted calibrator. Call fit() first.")

        # Serialise best_metrics_per_param_ with string keys
        serialisable_metrics = None
        if self.best_metrics_per_param_ is not None:
            serialisable_metrics = {
                json.dumps(list(k)): v for k, v in self.best_metrics_per_param_.items()
            }

        # Inner calibrator metadata (same structure as VectorScalingCalibrator.save)
        inner = self.best_calibrator_
        inner_metadata = {
            "params": inner.get_params(),
            "n_classes_": inner.n_classes_,
            "final_loss_": inner.final_loss_,
            "best_epoch_": inner.best_epoch_,
            "best_metrics_": inner.best_metrics_,
            "history_": inner.history_,
            "model_method": inner.model_.method.value,
        }

        metadata = {
            "cv_params": self.get_params(),
            "best_params_": self.best_params_,
            "best_cv_val_loss_": self.best_cv_val_loss_,
            "best_metrics_per_param_": serialisable_metrics,
            "inner_calibrator": inner_metadata,
        }

        arrays = {
            name: param.detach().cpu().numpy()
            for name, param in inner.model_.state_dict().items()
        }
        arrays["_metadata_json"] = np.array(json.dumps(metadata))
        np.savez(path, **arrays)

    def load(self, path: Union[str, Path]) -> "VectorScalingCalibratorCV":
        """Load a fitted CV calibrator from a .npz file.

        Args:
            path: File path to load from.

        Returns:
            self, with all fitted attributes restored.
        """
        data = np.load(path, allow_pickle=False)
        metadata = json.loads(str(data["_metadata_json"]))

        # Restore constructor params
        cv_params = metadata["cv_params"]
        for key, value in cv_params.items():
            setattr(self, key, value)

        self.best_params_ = metadata["best_params_"]
        self.best_cv_val_loss_ = metadata["best_cv_val_loss_"]

        # Restore best_metrics_per_param_ with tuple keys
        raw = metadata.get("best_metrics_per_param_")
        if raw is not None:
            self.best_metrics_per_param_ = {
                tuple(tuple(pair) for pair in json.loads(k)): v for k, v in raw.items()
            }

        # Reconstruct inner calibrator
        inner_meta = metadata["inner_calibrator"]
        inner = VectorScalingCalibrator(**inner_meta["params"])
        inner.n_classes_ = inner_meta["n_classes_"]
        inner.final_loss_ = inner_meta["final_loss_"]
        inner.best_epoch_ = inner_meta["best_epoch_"]
        inner.best_metrics_ = inner_meta["best_metrics_"]
        inner.history_ = inner_meta["history_"]

        method_enum = CalibrationMethod(inner_meta["model_method"])
        model = _TrainedLinearCalibrationModel(inner.n_classes_, method_enum)
        state_dict = {name: torch.as_tensor(data[name]) for name in model.state_dict()}
        model.load_state_dict(state_dict)
        model.to(torch.device(inner.device))
        inner.model_ = model

        self.best_calibrator_ = inner
        return self
