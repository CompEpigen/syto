"""
THIS MODULE IS EXPERIMENTAL, IT IS ONLY USED FOR PROOF OF CONCEPTS IN THE NOTEBOOKS.
IT SHOULD **NOT** BE CONSIDERED PRODUCTION-READY CODE, ESPECIALLY IN TERMS OF API STABILITY
AND TESTING (THERE ARE NO TESTS FOR THIS MODULE).



Dirichlet calibration for multiclass probability predictions.

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

from enum import Enum
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, RegressorMixin
from entmax import sparsemax, entmax15


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
        init_noise_std: Standard deviation of Gaussian noise added to the
            zero-initialised delta parameters. Default 0.0 (no noise).
    """

    def __init__(
        self,
        n_classes: int,
        method: CalibrationMethod,
        init_noise_std: float = 0.0,
        normalization: str = "softmax",
    ):
        super().__init__()
        self.n_classes = n_classes
        self.method = method
        self.normalization = normalization
        assert normalization in {
            "softmax",
            "sparsemax",
            "entmax15",
        }, "Unsupported normalization"

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

        # Add Gaussian noise to break symmetry around zero init
        if init_noise_std > 0.0:
            with torch.no_grad():
                for param in self.parameters():
                    param.add_(torch.randn_like(param) * init_noise_std)

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
        if self.normalization == "softmax":
            return torch.softmax(logits, dim=1)
        elif self.normalization == "sparsemax":
            return sparsemax(logits, dim=1)
        elif self.normalization == "entmax15":
            return entmax15(logits, dim=1)
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")

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


class DirichletCalibrator(BaseEstimator, RegressorMixin):
    """Post-hoc Dirichlet calibration for multiclass classifiers.

    Learns a calibration map of the form softmax(W @ ln(q) + b) that
    transforms uncalibrated probability vectors q into calibrated ones.
    Uses PyTorch with mini-batch SGD for scalability to large datasets.

    Args:
        method: Calibration method - "full", "diagonal", or "temperature".
        log_transform: If True (default), fit on log-transformed probabilities
            (Dirichlet calibration: softmax(W @ ln(q) + b)). If False, fit
            on raw probabilities (softmax(W @ q + b)).
        reg_lambda: L2 regularisation strength for off-diagonal elements
            (ODIR) or all elements (when reg_mu is None).
        reg_mu: Separate regularisation strength for the intercept (bias)
            vector. When set, ODIR regularisation is used: reg_lambda
            controls off-diagonal W elements, reg_mu controls bias.
            When None, reg_lambda is applied as standard L2 to all params.
        optimizer: Optimizer to use - "adam" or "sgd".
        lr: Learning rate for the optimizer.
        scheduler: Learning rate schedule - "constant" (default), "linear",
            "cosine", or "plateau". Linear decays lr to 0 over max_iter
            epochs. Cosine uses cosine annealing to 0. Plateau reduces lr
            when the selection loss stops improving.
        max_iter: Maximum number of optimization epochs.
        batch_size: Mini-batch size. Use 0 or None for full-batch
            (entire dataset as one batch per epoch).
        patience: Early stopping patience (number of epochs without
            improvement before stopping).
        tol: Minimum improvement in loss to reset patience counter.
        device: Torch device string ("cpu", "cuda", etc.).
        verbose: If True (default), print training progress. Set to False
            to silence all output during fitting.
        init_noise_std: Standard deviation of Gaussian noise added to the
            zero-initialised delta parameters. Helps break symmetry.
            Default 0.0 (no noise).
        plateau_factor: Factor by which the learning rate is reduced when
            using the "plateau" scheduler. New lr = old lr * factor.
            Default 0.5.
        plateau_patience: Number of epochs with no improvement after which
            the "plateau" scheduler reduces the learning rate. Default 10.
    """

    def __init__(
        self,
        method: str = "full",
        log_transform: bool = True,
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
        verbose: bool = True,
        init_noise_std: float = 0.0,
        normalization: str = "softmax",
        plateau_factor: float = 0.5,
        plateau_patience: int = 10,
    ):
        assert method in {"full", "diagonal", "temperature"}, "Unsupported method"
        assert optimizer in {"adam", "sgd"}, "Unsupported optimizer"
        assert scheduler in {"constant", "linear", "cosine", "plateau"}, "Unsupported scheduler"
        assert normalization in {
            "softmax",
            "sparsemax",
            "entmax15",
        }, "Unsupported normalization"
        self.method = method
        self.log_transform = log_transform
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
        self.verbose = verbose
        self.init_noise_std = init_noise_std
        self.normalization = normalization
        self.plateau_factor = plateau_factor
        self.plateau_patience = plateau_patience

        # Fitted attributes (set by fit())
        self.model_: Optional[_DirichletCalibrationModel] = None
        self.n_classes_: int = 0
        self.final_loss_: float = float("inf")
        self.best_epoch_: int = -1
        self.best_metrics_: dict[str, float] = {}
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

    def _prepare_input(self, X: np.ndarray) -> np.ndarray:
        """Prepare input probabilities: log-transform or pass through raw.

        Args:
            X: Probability array of shape (n_samples, n_classes).

        Returns:
            Transformed array with the same shape.
        """
        if self.log_transform:
            return self._clip_and_log(X)
        return X.copy()

    @staticmethod
    def _fenchel_young_loss(
        logits: torch.Tensor,
        p_star: torch.Tensor,
        targets: torch.Tensor,
        normalization: str,
    ) -> torch.Tensor:
        """Fenchel-Young loss for all normalizations, supporting soft targets.

        Computes the proper Fenchel-Young loss (Blondel et al. 2020):
            L_FY(z, y) = -Omega(p*) + <p* - y, z> + Omega(y)  >= 0

        where Omega is the negative Tsallis entropy and p* = mapping(z).
        The loss is always non-negative and equals 0 iff p* = y.

        - softmax (alpha -> 1): reduces to KL divergence D_KL(y || p*).
        - sparsemax (alpha=2): Omega(p) = (||p||^2 - 1) / 2
        - entmax-1.5 (alpha=1.5): Omega(p) = (sum(p^1.5) - 1) / (1.5 * 0.5)

        Args:
            logits: (N, K) pre-activation logits.
            p_star: (N, K) activated probabilities (softmax/sparsemax/entmax output).
            targets: (N, K) target distributions (soft labels).
            normalization: One of "softmax", "sparsemax", "entmax15".

        Returns:
            Scalar mean loss (non-negative).
        """
        if normalization == "softmax":
            # KL divergence: sum y_i * log(y_i / p_i)
            # Using xlogy to handle y_i = 0 correctly (0 * log(0) = 0)
            p_clamped = torch.clamp(p_star, min=1e-20)
            return torch.mean(
                torch.sum(
                    torch.xlogy(targets, targets) - targets * torch.log(p_clamped),
                    dim=1,
                )
            )
        elif normalization == "sparsemax":
            # -Omega(p*) = (1 - ||p*||^2) / 2;  Omega(y) = (||y||^2 - 1) / 2
            neg_omega_p = (1.0 - torch.sum(p_star**2, dim=1)) / 2.0
            omega_y = (torch.sum(targets**2, dim=1) - 1.0) / 2.0
            inner = torch.sum((p_star - targets) * logits, dim=1)
            return torch.mean(neg_omega_p + inner + omega_y)
        elif normalization == "entmax15":
            # -Omega(p*) = (1 - sum p^1.5) * 4/3;  Omega(y) = (sum y^1.5 - 1) * 4/3
            neg_omega_p = (1.0 - torch.sum(p_star**1.5, dim=1)) * (4.0 / 3.0)
            omega_y = (torch.sum(targets**1.5, dim=1) - 1.0) * (4.0 / 3.0)
            inner = torch.sum((p_star - targets) * logits, dim=1)
            return torch.mean(neg_omega_p + inner + omega_y)
        else:
            raise ValueError(f"Unknown normalization: {normalization}")

    def _compute_loss(
        self,
        model: _DirichletCalibrationModel,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        with_regularization: bool = True,
    ) -> torch.Tensor:
        """Compute the calibration loss + regularisation.

        Uses cross-entropy for softmax and Fenchel-Young loss for
        sparsemax/entmax, all supporting soft targets.

        Two regularisation modes are supported:

        - **Standard L2** (when ``reg_mu`` is None): ``reg_lambda * sum(param^2)``
          over all parameters.
        - **ODIR** (when ``reg_mu`` is set): ``reg_lambda * mean(W_offdiag^2)``
          for off-diagonal W elements + ``reg_mu * mean(b^2)`` for the bias.
          The diagonal of W is left unregularised so each class can freely
          adjust its own scaling.

        Args:
            model: The calibration model.
            inputs: (N, K) transformed input probabilities.
            targets: (N, K) target distributions (hard or soft labels).
            with_regularization: If False, skip adding the regularisation term

        Returns:
            Scalar loss tensor.
        """
        logits = model(inputs)
        if self.normalization == "softmax":
            probs = model.activate(logits)
            # Clamp to avoid log(0) which would give -inf and corrupt the loss
            probs = torch.clamp(probs, min=1e-20)
            # NLL: works with both hard (one-hot) and soft label targets
            loss = -torch.mean(torch.sum(targets * torch.log(probs), dim=1))
        else:
            p_star = model.activate(logits)
            loss = self._fenchel_young_loss(logits, p_star, targets, model.normalization)

        # --- Regularisation ---
        if with_regularization:
            if self.reg_mu is not None:
                W = model.get_W_matrix()
                k = W.shape[0]
                off_diag_mask = ~torch.eye(k, dtype=torch.bool, device=W.device)
                reg_off_diag = self.reg_lambda * torch.mean(W[off_diag_mask] ** 2)
                if hasattr(model, "b"):
                    reg_intercept = self.reg_mu * torch.mean(model.b**2)
                else:
                    reg_intercept = 0.0
                loss = loss + reg_off_diag + reg_intercept
            elif self.reg_lambda > 0:
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
        method_enum = self._to_method_enum()
        model = _DirichletCalibrationModel(
            n_classes,
            method_enum,
            init_noise_std=self.init_noise_std,
            normalization=self.normalization,
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
                "Choose 'constant', 'linear', 'cosine', or 'plateau'."
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
            init_train_loss = self._compute_loss(
                model, train_logits, targets_all, with_regularization=False
            ).item()
            init_train_mse = torch.mean((train_calibrated - targets_all) ** 2).item()

            history["epoch"].append(-1)
            history["lr"].append(self.lr)
            history["train_loss"].append(init_train_loss)
            history["train_mse"].append(init_train_mse)

            if has_val:
                val_logits = model(input_val)
                val_calibrated = model.activate(val_logits)
                init_val_loss = self._compute_loss(
                    model, val_logits, targets_val, with_regularization=False
                ).item()
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
                loss = self._compute_loss(model, batch_input, batch_tgt, with_regularization=True)
                loss.backward()
                optim.step()
                epoch_loss += loss.item()
                n_batches += 1

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
                        model, val_logits, targets_val, with_regularization=False
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

            # Current learning rate (first param group)
            current_lr = optim.param_groups[0]["lr"]

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
            # pylint: disable-next=not-callable
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
