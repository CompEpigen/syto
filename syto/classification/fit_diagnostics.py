"""Diagnostic artifacts produced during read-classifier fitting.

Architecture-neutral, like ``hf_training``: nothing here is MethylBERT
specific, so any classifier training through ``transformers.Trainer`` can opt
in. Currently provides per-evaluation KDE plots of the *on-target* score
distribution — the probability the model assigns to a chunk's own
``dmr_ctype_label``, restricted to chunks where that label equals the read's
``original_label``.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # fits run headless; must precede the pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

_module_logger = logging.getLogger(__name__)


def data_list_column(data_list, name: str) -> np.ndarray:
    """Extract one column from a ``[header_row, *data_rows]`` table.

    ``prepare_methylbert_list`` returns exactly that shape, and
    ``MethylBertFinetuneDataset`` is 1:1 and order-preserving with
    ``data_list[1:]``, so the returned array is aligned with dataset (and
    therefore evaluation-output) row order.
    """
    header = list(data_list[0])
    if name not in header:
        raise ValueError(f"Column {name!r} is not in the data-list header: {header}")
    idx = header.index(name)
    return np.array([row[idx] for row in data_list[1:]])


def on_target_scores(
    predictions: np.ndarray,
    dmr_labels: np.ndarray,
    on_target_mask: np.ndarray,
) -> np.ndarray:
    """Probability assigned to each on-target chunk's own ``dmr_ctype_label``.

    The vectorised form of selecting ``prediction_{dmr_ctype_label}`` on the
    rows where ``dmr_ctype_label == original_label``.
    """
    rows = np.flatnonzero(np.asarray(on_target_mask).astype(bool))
    if rows.size == 0:
        return np.empty(0, dtype=float)
    labels = np.asarray(dmr_labels)[rows].astype(int)
    return np.asarray(predictions)[rows, labels].astype(float)


class OnTargetScoreRecorder:
    """Writes one on-target score KDE plot per completed evaluation pass.

    Per-chunk metadata cannot come from the evaluation output itself:
    ``eval_pred.label_ids`` carries the training ``ctype``, not
    ``dmr_ctype_label`` / ``original_label``. So callers ``register`` the two
    arrays alongside the dataset object, and this class looks them up by
    dataset identity when the evaluation output arrives.

    Keying on identity rather than on the metric prefix is what lets one
    recorder serve both the periodic validation evaluations and the single
    final train-set pass (``evaluate_train_metrics``), and it degrades
    gracefully when both roles are the same object.
    """

    def __init__(self, plot_dir, *, bw_adjust: float = 0.01, logger=None):
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.bw_adjust = bw_adjust
        self._logger = logger or _module_logger
        self._metadata = {}

    def register(self, dataset, *, dmr_labels, on_target_mask) -> None:
        """Associate per-chunk metadata with a dataset, by object identity."""
        self._metadata[id(dataset)] = (
            np.asarray(dmr_labels),
            np.asarray(on_target_mask).astype(bool),
        )

    def on_eval(
        self, dataset, predictions, *, step: int, prefix: str = "eval"
    ) -> Optional[Path]:
        """Plot the on-target score distribution for one evaluation pass.

        Returns the written path, or ``None`` (with a warning) when the inputs
        cannot produce a meaningful plot.
        """
        meta = self._metadata.get(id(dataset))
        if meta is None:
            self._logger.warning(
                "No on-target metadata registered for the evaluated dataset; "
                "skipping the %s on-target score plot at step %s.",
                prefix,
                step,
            )
            return None

        if isinstance(predictions, (tuple, list)):
            predictions = predictions[0] if predictions else None
        if predictions is None:
            self._logger.warning(
                "Evaluation produced no predictions (prediction_loss_only?); "
                "skipping the %s on-target score plot at step %s.",
                prefix,
                step,
            )
            return None

        dmr_labels, on_target_mask = meta
        predictions = np.asarray(predictions)
        if len(predictions) != len(on_target_mask):
            self._logger.warning(
                "Prediction/metadata length mismatch (%d vs %d); skipping the "
                "%s on-target score plot at step %s.",
                len(predictions),
                len(on_target_mask),
                prefix,
                step,
            )
            return None

        scores = on_target_scores(predictions, dmr_labels, on_target_mask)
        if scores.size == 0:
            self._logger.warning(
                "No on-target chunks in the evaluated dataset; skipping the "
                "%s on-target score plot at step %s.",
                prefix,
                step,
            )
            return None

        path = self.plot_dir / f"{prefix}_step_{step:07d}.png"
        self._plot(scores, path, prefix=prefix, step=step)
        self._logger.info("Wrote %s on-target score plot: %s", prefix, path)
        return path

    def _plot(self, scores: np.ndarray, path: Path, *, prefix: str, step: int):
        fig, ax = plt.subplots(figsize=(7, 4))
        try:
            sns.kdeplot(x=scores, bw_adjust=self.bw_adjust, ax=ax)
            ax.set_xlabel("P(own dmr_ctype_label)")
            ax.set_ylabel("density")
            ax.set_title(
                f"{prefix} on-target scores — step {step}\n"
                f"n={scores.size}, mean={scores.mean():.3f}, "
                f"median={np.median(scores):.3f}"
            )
            fig.tight_layout()
            fig.savefig(path, dpi=120)
        finally:
            plt.close(fig)
