"""Dirichlet calibration for multiclass probability predictions.

Implements the method from Kull et al. (NeurIPS 2019):
"Beyond temperature scaling: Obtaining well-calibrated multiclass
probabilities with Dirichlet calibration."

The linear parametrisation is: mu(q; W, b) = softmax(W @ ln(q) + b)
which is equivalent to log-transforming the uncalibrated probabilities,
followed by one linear layer and softmax.

This implementation uses PyTorch for optimization and supports:
- Full Dirichlet calibration (full W matrix)
- Vector scaling (diagonal W)
- Temperature scaling (scalar * I diagonal)
- L2 and ODIR (Off-Diagonal and Intercept Regularisation)
- Mini-batch training for large datasets
"""

import logging
from enum import Enum
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin

logger = logging.getLogger(__name__)


class CalibrationMethod(Enum):
    """Supported calibration map parametrisations.

    Each variant constrains the weight matrix W differently:

    - FULL: W is an unconstrained (k x k) matrix. This is the full
      Dirichlet calibration with k^2 + k parameters.
    - DIAGONAL: W is restricted to a diagonal matrix (k diagonal + k bias
      parameters). Equivalent to vector scaling.
    - TEMPERATURE: W = t * I with a single scalar t and no bias.
      Equivalent to temperature scaling.
    """

    FULL = "full"
    DIAGONAL = "diagonal"
    TEMPERATURE = "temperature"


class _DirichletCalibrationModel(nn.Module):
    """Learnable calibration map: softmax(W @ ln(q) + b).

    This is the "linear parametrisation" from Kull et al. (2019), Eq. 7.
    The model takes log-transformed probability vectors as input and
    produces calibrated probability vectors via a linear layer + softmax.

    All parameters are initialised to the identity map (W=I, b=0) so that
    the uncalibrated model is the starting point of optimisation.

    Args:
        n_classes: Number of classes (k).
        method: Which parametrisation to use for the weight matrix.
    """

    def __init__(self, n_classes: int, method: CalibrationMethod):
        super().__init__()
        self.n_classes = n_classes
        self.method = method

        # Initialise to the identity calibration map (no-op):
        # Full: W = I, b = 0  =>  softmax(I @ ln(q) + 0) = softmax(ln(q)) = q
        # Diagonal: diag = 1, b = 0  =>  same as above
        # Temperature: t = 1  =>  softmax(1 * ln(q)) = q
        if method == CalibrationMethod.FULL:
            self.W = nn.Parameter(torch.eye(n_classes, dtype=torch.float64))
            self.b = nn.Parameter(torch.zeros(n_classes, dtype=torch.float64))
        elif method == CalibrationMethod.DIAGONAL:
            self.diag = nn.Parameter(torch.ones(n_classes, dtype=torch.float64))
            self.b = nn.Parameter(torch.zeros(n_classes, dtype=torch.float64))
        elif method == CalibrationMethod.TEMPERATURE:
            self.temperature = nn.Parameter(torch.ones(1, dtype=torch.float64))
        else:
            raise ValueError(f"Unknown method: {method}")

    def forward(self, log_probs: torch.Tensor) -> torch.Tensor:
        """Apply calibration map to log-transformed probabilities.

        Args:
            log_probs: (N, K) tensor of log(clipped probabilities).

        Returns:
            (N, K) tensor of calibrated probabilities.
        """
        if self.method == CalibrationMethod.FULL:
            # Full matrix multiply: logits_i = sum_j W_ij * ln(q_j) + b_i
            logits = log_probs @ self.W.T + self.b
        elif self.method == CalibrationMethod.DIAGONAL:
            # Element-wise scaling: logits_i = diag_i * ln(q_i) + b_i
            logits = log_probs * self.diag + self.b
        elif self.method == CalibrationMethod.TEMPERATURE:
            # Single scalar scaling with no bias: logits = t * ln(q)
            logits = log_probs * self.temperature
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return torch.softmax(logits, dim=1)

    def get_W_matrix(self) -> torch.Tensor:
        """Return the full (k x k) weight matrix W.

        For diagonal and temperature methods, this reconstructs the
        equivalent full matrix (diagonal or scalar * identity) so that
        regularisation and inspection code can treat all methods uniformly.

        Returns:
            (k, k) tensor representing the calibration weight matrix.
        """
        if self.method == CalibrationMethod.FULL:
            return self.W
        elif self.method == CalibrationMethod.DIAGONAL:
            return torch.diag(self.diag)
        elif self.method == CalibrationMethod.TEMPERATURE:
            return self.temperature * torch.eye(
                self.n_classes,
                dtype=self.temperature.dtype,
                device=self.temperature.device,
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")


class DirichletCalibrator(BaseEstimator, RegressorMixin):
    """Post-hoc Dirichlet calibration for multiclass classifiers.

    Learns a calibration map of the form softmax(W @ ln(q) + b) that
    transforms uncalibrated probability vectors q into calibrated ones.
    Uses PyTorch with mini-batch SGD for scalability to large datasets.

    Args:
        method: Calibration method - "full", "diagonal", or "temperature".
        reg_lambda: L2 regularisation strength for off-diagonal elements
            (ODIR) or all elements (when reg_mu is None).
        reg_mu: Separate regularisation strength for the intercept (bias)
            vector. When set, ODIR regularisation is used: reg_lambda
            controls off-diagonal W elements, reg_mu controls bias.
            When None, reg_lambda is applied as standard L2 to all params.
        optimizer: Optimizer to use - "adam" or "sgd".
        lr: Learning rate for the optimizer.
        scheduler: Learning rate schedule - "constant" (default), "linear",
            or "cosine". Linear decays lr to 0 over max_iter epochs.
            Cosine uses cosine annealing to 0.
        max_iter: Maximum number of optimization epochs.
        batch_size: Mini-batch size. Use 0 or None for full-batch
            (entire dataset as one batch per epoch).
        patience: Early stopping patience (number of epochs without
            improvement before stopping).
        tol: Minimum improvement in loss to reset patience counter.
        device: Torch device string ("cpu", "cuda", etc.).
    """

    def __init__(
        self,
        method: str = "full",
        reg_lambda: float = 0.0,
        reg_mu: Optional[float] = None,
        optimizer: str = "adam",
        lr: float = 1e-2,
        scheduler: str = "constant",
        max_iter: int = 2048,
        batch_size: int = 1000,
        patience: int = 50,
        tol: float = 1e-7,
        device: str = "cpu",
    ):
        self.method = method
        self.reg_lambda = reg_lambda
        self.reg_mu = reg_mu
        self.optimizer = optimizer
        self.lr = lr
        self.scheduler = scheduler
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.patience = patience
        self.tol = tol
        self.device = device

        # Fitted attributes (set by fit())
        self.model_: Optional[_DirichletCalibrationModel] = None
        self.n_classes_: int = 0
        self.final_loss_: float = float("inf")
        self.history_: dict[str, list] = {}

    def _to_method_enum(self) -> CalibrationMethod:
        """Convert the string ``self.method`` to a :class:`CalibrationMethod` enum.

        Raises:
            ValueError: If ``self.method`` is not one of the supported strings.
        """
        mapping = {
            "full": CalibrationMethod.FULL,
            "diagonal": CalibrationMethod.DIAGONAL,
            "temperature": CalibrationMethod.TEMPERATURE,
        }
        if self.method not in mapping:
            raise ValueError(
                f"Unknown method '{self.method}'. "
                f"Choose from {list(mapping.keys())}."
            )
        return mapping[self.method]

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

    def _compute_loss(
        self,
        model: _DirichletCalibrationModel,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the calibration loss: cross-entropy + regularisation.

        The cross-entropy is computed as -mean(sum(y * log(p_cal))) which
        generalises to soft targets (y can be any probability vector).

        Two regularisation modes are supported:

        - **Standard L2** (when ``reg_mu`` is None): ``reg_lambda * sum(param^2)``
          over all parameters.
        - **ODIR** (when ``reg_mu`` is set): ``reg_lambda * mean(W_offdiag^2)``
          for off-diagonal W elements + ``reg_mu * mean(b^2)`` for the bias.
          The diagonal of W is left unregularised so each class can freely
          adjust its own scaling.

        Args:
            model: The calibration model.
            log_probs: (N, K) log-transformed input probabilities.
            targets: (N, K) target distributions (hard or soft labels).

        Returns:
            Scalar loss tensor.
        """
        calibrated = model(log_probs)
        # Clamp to avoid log(0) which would give -inf and corrupt the loss
        calibrated = torch.clamp(calibrated, min=1e-20)
        # Cross-entropy: works with both hard (one-hot) and soft label targets
        nll = -torch.mean(torch.sum(targets * torch.log(calibrated), dim=1))

        # --- Regularisation ---
        if self.reg_mu is not None:
            # ODIR (Off-Diagonal and Intercept Regularisation):
            # Penalise off-diagonal W entries (class interactions) and bias
            # separately, leaving diagonal entries (per-class scaling) free.
            W = model.get_W_matrix()
            k = W.shape[0]
            off_diag_mask = ~torch.eye(k, dtype=torch.bool, device=W.device)
            reg_off_diag = self.reg_lambda * torch.mean(W[off_diag_mask] ** 2)
            if hasattr(model, "b"):
                reg_intercept = self.reg_mu * torch.mean(model.b**2)
            else:
                reg_intercept = 0.0
            nll = nll + reg_off_diag + reg_intercept
        elif self.reg_lambda > 0:
            # Standard L2 penalty on all learnable parameters
            reg = torch.tensor(0.0, dtype=torch.float64, device=log_probs.device)
            for param in model.parameters():
                reg = reg + torch.sum(param**2)
            nll = nll + self.reg_lambda * reg

        return nll

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
    ) -> "DirichletCalibrator":
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

        log_probs_np = self._clip_and_log(X)

        dev = torch.device(self.device)
        log_probs_all = torch.as_tensor(log_probs_np, dtype=torch.float64, device=dev)
        targets_all = torch.as_tensor(targets, dtype=torch.float64, device=dev)

        # Prepare validation tensors if provided
        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val = np.asarray(X_val, dtype=np.float64)
            n_val = X_val.shape[0]
            val_targets = self._prepare_targets(y_val, n_val, n_classes)
            log_probs_val = torch.as_tensor(
                self._clip_and_log(X_val), dtype=torch.float64, device=dev
            )
            targets_val = torch.as_tensor(val_targets, dtype=torch.float64, device=dev)

        # Build calibration model and optimiser
        method_enum = self._to_method_enum()
        model = _DirichletCalibrationModel(n_classes, method_enum).to(dev)

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
        else:
            raise ValueError(
                f"Unknown scheduler '{self.scheduler}'. "
                "Choose 'constant', 'linear', or 'cosine'."
            )

        # batch_size=0 means full-batch: use the entire dataset each epoch
        effective_bs = self.batch_size if self.batch_size else n_samples

        best_loss = float("inf")
        best_state = None
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

        for epoch in range(self.max_iter):
            # --- Training step ---
            model.train()
            # Shuffle data each epoch for stochastic mini-batching
            perm = torch.randperm(n_samples, device=dev)
            epoch_loss = 0.0
            n_batches = 0
            for start in range(0, n_samples, effective_bs):
                idx = perm[start : start + effective_bs]
                batch_log = log_probs_all[idx]
                batch_tgt = targets_all[idx]

                optim.zero_grad()
                loss = self._compute_loss(model, batch_log, batch_tgt)
                loss.backward()
                optim.step()
                epoch_loss += loss.item()
                n_batches += 1

            # Step the learning rate scheduler after each epoch
            if sched is not None:
                sched.step()

            # Current learning rate (first param group)
            current_lr = optim.param_groups[0]["lr"]

            # --- Selection loss: use val set if available, else train loss ---
            train_loss = epoch_loss / n_batches
            model.eval()
            with torch.no_grad():
                # Train MSE: mean squared error between calibrated probs and targets
                train_calibrated = model(log_probs_all)
                train_mse = torch.mean((train_calibrated - targets_all) ** 2).item()

                if has_val:
                    # Validation loss without regularisation for clean selection
                    val_calibrated = model(log_probs_val)
                    val_calibrated_clamped = torch.clamp(val_calibrated, min=1e-20)
                    selection_loss = -torch.mean(
                        torch.sum(
                            targets_val * torch.log(val_calibrated_clamped), dim=1
                        )
                    ).item()
                    val_mse = torch.mean((val_calibrated - targets_val) ** 2).item()
                else:
                    selection_loss = train_loss

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
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.patience:
                print(f"Early stopping at epoch {epoch} (best loss: {best_loss:.7e})")
                break

            if epoch % report_every == 0:
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

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Apply the learned calibration map to new predictions.

        Args:
            X: Uncalibrated probability predictions, shape (n_samples, n_classes).

        Returns:
            Calibrated probabilities, shape (n_samples, n_classes).
        """
        X = np.asarray(X, dtype=np.float64)
        log_probs_np = self._clip_and_log(X)

        dev = next(self.model_.parameters()).device
        log_probs_t = torch.as_tensor(log_probs_np, dtype=torch.float64, device=dev)

        with torch.no_grad():
            calibrated = self.model_(log_probs_t)

        return calibrated.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Apply the learned calibration map (alias for :meth:`predict_proba`).

        Args:
            X: Uncalibrated probability predictions, shape (n_samples, n_classes).

        Returns:
            Calibrated probabilities, shape (n_samples, n_classes).
        """
        return self.predict_proba(X)

    @property
    def coef_(self) -> np.ndarray:
        """The (k x k) weight matrix W of the learned calibration map.

        For diagonal / temperature methods this is the equivalent full matrix.
        """
        return self.model_.get_W_matrix().detach().cpu().numpy()

    @property
    def intercept_(self) -> np.ndarray:
        """The bias vector b of the learned calibration map.

        Returns a zero vector for temperature scaling (which has no bias).
        """
        if hasattr(self.model_, "b"):
            return self.model_.b.detach().cpu().numpy()
        return np.zeros(self.n_classes_)

    def canonical_params(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the canonical parametrisation (A, c) from Kull et al. Eq. 8.

        The canonical form is unique and interpretable:

        - **A**: Non-negative (k x k) matrix where each column has at least
          one zero entry. ``A_ij = W_ij - min_i(W_ij)``.
          Increasing ``A_ij`` boosts the calibrated probability of class *i*
          when the uncalibrated probability of class *j* is high.
        - **c**: Probability vector of length k. It is the calibrated output
          when the input is the uniform distribution ``(1/k, ..., 1/k)``.
          ``c = softmax(W @ ln(u) + b)``.

        Returns:
            Tuple ``(A, c)`` where A has shape (k, k) and c has shape (k,).
        """
        W = self.coef_
        b = self.intercept_
        k = W.shape[0]

        # Shift each column so its minimum is 0 => non-negative entries
        col_min = W.min(axis=0)
        A = W - col_min

        # Evaluate the calibration map at the simplex centre u = (1/k,...,1/k)
        u = np.full(k, 1.0 / k)
        logits = W @ np.log(u) + b
        # Numerically stable softmax
        c = np.exp(logits - logits.max())
        c = c / c.sum()

        return A, c
