"""Diagnostic artifacts produced during read-classifier fitting.

Architecture-neutral, like ``hf_training``: nothing here is MethylBERT
specific, so any classifier training through ``transformers.Trainer`` can opt
in. Currently provides per-evaluation KDE plots of the *on-target* score
distribution — the probability the model assigns to a chunk's own
``dmr_ctype_label``, restricted to chunks where that label equals the read's
``original_label``.
"""

import logging

import numpy as np

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
