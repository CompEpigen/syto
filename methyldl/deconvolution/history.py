"""Unified, flexible training history for deconvolution models.

This module provides a single ``DeconvolutionHistory`` dataclass shared by
all deconvolver implementations (deep-learning, XGBoost, etc.).  Metric
series are stored in plain dictionaries so the history automatically
adapts to whatever ``compute_deconvolution_metrics`` returns — no
hard-coded field per metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DeconvolutionHistory:
    """Flexible training history for any deconvolution model.

    Metric series are stored in dictionaries keyed by metric name.
    Whatever ``compute_deconvolution_metrics`` returns is recorded
    automatically — no hard-coded field per metric.

    Attributes
    ----------
    train_metrics : dict[str, list]
        Per-epoch training metric series (key → list of scalar values).
    val_metrics : dict[str, list]
        Per-epoch validation metric series.
    train_loss : list[float]
        Per-epoch training loss (model-specific, always recorded).
    val_loss : list[float]
        Per-epoch validation loss.
    learning_rates : list[float]
        Per-epoch learning-rate snapshots.
    best_epoch : int
        Index of the best epoch (0-based).
    stopped_early : bool
        Whether training was terminated by early stopping.
    """

    # Per-epoch metric series (key → list[float])
    train_metrics: dict[str, list] = field(default_factory=dict)
    val_metrics: dict[str, list] = field(default_factory=dict)

    # Model-specific losses (always recorded)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)

    # Extras
    learning_rates: list[float] = field(default_factory=list)
    best_epoch: int = 0
    stopped_early: bool = False

    # ── helpers ──────────────────────────────────────────────

    def record_metrics(
        self,
        metrics: dict[str, object],
        split: Literal["train", "val"],
    ) -> None:
        """Append one epoch's worth of metrics.

        Non-scalar entries (dicts, arrays, etc.) returned by
        ``compute_deconvolution_metrics`` are silently skipped so that
        per-epoch series remain clean lists of numbers.

        Parameters
        ----------
        metrics : dict[str, object]
            Dictionary returned by ``compute_deconvolution_metrics``.
        split : ``"train"`` or ``"val"``
            Which metric store to append to.
        """
        target = self.train_metrics if split == "train" else self.val_metrics
        for key, value in metrics.items():
            # Keep only scalar entries
            if not isinstance(value, (int, float)):
                continue
            target.setdefault(key, []).append(value)

    def get(self, name: str, split: Literal["train", "val"] = "val") -> list:
        """Retrieve a metric series by name and split.

        The special name ``"loss"`` is forwarded to the dedicated
        ``train_loss`` / ``val_loss`` lists.

        Parameters
        ----------
        name : str
            Metric name (e.g. ``"mae"``, ``"loss"``).
        split : ``"train"`` or ``"val"``
            Which split to query.

        Returns
        -------
        list
            The requested metric series, or an empty list if not found.
        """
        if name == "loss":
            return self.val_loss if split == "val" else self.train_loss
        store = self.val_metrics if split == "val" else self.train_metrics
        return store.get(name, [])

    def to_dict(self) -> dict:
        """Flat dictionary suitable for serialisation or logging.

        Keys follow the ``{split}_{metric}`` convention (e.g.
        ``"val_mae"``, ``"train_mse"``) to stay compatible with
        downstream consumers.
        """
        out: dict = {}
        out["train_loss"] = self.train_loss
        out["val_loss"] = self.val_loss
        for key, vals in self.train_metrics.items():
            out[f"train_{key}"] = vals
        for key, vals in self.val_metrics.items():
            out[f"val_{key}"] = vals
        out["learning_rates"] = self.learning_rates
        out["best_epoch"] = self.best_epoch
        out["stopped_early"] = self.stopped_early
        return out
