"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""

import numpy as np
import sklearn
import torch
from typing import Union, Tuple, Any, Callable, Optional
from scipy.special import softmax


def _safe_average_precision(
    labels: np.ndarray,
    predictions_proba: np.ndarray,
    append_background_probability: bool = False,
) -> float:
    """Compute average_precision_score, handling multi-class edge cases.

    When ``append_background_probability`` is ``True``, a pseudo-probability
    column for the background class is appended as
    ``1 - max(predictions_proba, axis=1)``.  This lets AP measure how well the
    model separates confident (real-class) predictions from uncertain
    (background) ones when background class is not explicitely represented by
    the original probability vector

    Parameters
    ----------
    labels : np.ndarray
        1-D array of integer class labels (may include a background class ID
        that exceeds the number of probability columns).
    predictions_proba : np.ndarray
        Either a 1-D array (binary) or a 2-D array of shape
        ``(n_samples, n_classes)`` with predicted probabilities.
    append_background_probability : bool, default False
        If ``True``, synthesize and append a background probability column
        (``1 - max(proba)``) before computing per-class AP.
    """
    if predictions_proba.ndim >= 2:
        if append_background_probability:
            # Append a pseudo-probability column for the background class:
            # high value when the model is unconfident (low max probability)
            background_proba = 1.0 - np.max(predictions_proba, axis=1, keepdims=True)
            predictions_proba = np.concatenate(
                [predictions_proba, background_proba], axis=1
            )

        # Compute macro AP by taking the mean of class-wise AP scores.
        # This prevents shape mismatches/IndexError in sklearn if some classes are missing in labels.
        class_aps = []
        for c in range(predictions_proba.shape[1]):
            y_true_c = (labels == c).astype(int)
            if len(np.unique(y_true_c)) == 2:
                ap_c = sklearn.metrics.average_precision_score(
                    y_true_c, predictions_proba[:, c]
                )
                class_aps.append(ap_c)

        if len(class_aps) == 0:
            return float("nan")

        return float(np.mean(class_aps))
    else:
        # Binary: predictions_proba is 1-D
        if len(labels) == 0 or len(np.unique(labels)) < 2:
            return float("nan")

        return sklearn.metrics.average_precision_score(labels, predictions_proba)


def calculate_metric_with_sklearn(
    predictions_proba: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    append_background_probability: bool = False,
):
    valid_mask = (
        labels != -100
    )  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    valid_predictions_proba = predictions_proba[valid_mask]

    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "average_precision": _safe_average_precision(
            valid_labels,
            valid_predictions_proba,
            append_background_probability=append_background_probability,
        ),
    }


def preprocess_logits_for_prediction(
    logits: Union[torch.Tensor, Tuple[torch.Tensor]], _
):
    """
    Preprocess logits for predictions.
    Returns softmax probabilities across all classes (sums to 1).
    Applies uniformly for both binary (num_labels=2) and multi-class settings.
    """
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]

    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])

    return torch.softmax(logits, dim=-1)


def make_compute_metrics(
    background_threshold_func: Optional[Callable[[int], float]] = None,
):
    """Factory returning a HuggingFace Trainer-compatible compute_metrics callable.

    Parameters
    ----------
    background_threshold_func : callable(int) -> float, optional
        When provided, predictions and labels whose maximum probability falls
        below this threshold are assigned to a synthetic background class
        (index = ``num_classes``).  The callable receives the number of model
        output classes and returns the decision threshold.
        When ``None``, no background class is synthesized (standard hard-label
        evaluation).

    Returns
    -------
    callable
        ``compute_metrics(eval_pred) -> dict``
    """
    append_background_probability = background_threshold_func is not None

    def compute_metrics(eval_pred):
        predictions_proba, labels = eval_pred
        num_classes = predictions_proba.shape[-1]

        if append_background_probability:
            # --- Background thresholding path ---
            threshold = background_threshold_func(num_classes)
            background_class_id = num_classes

            # Predictions: assign background when max probability is below threshold
            max_probs = np.max(predictions_proba, axis=-1)
            pred_indices = np.argmax(predictions_proba, axis=-1)
            predictions = np.where(
                max_probs < threshold, background_class_id, pred_indices
            )

            # Labels: convert soft labels to hard, with background thresholding
            if labels.ndim >= 2:
                max_labels = np.max(labels, axis=-1)
                label_indices = np.argmax(labels, axis=-1)
                hard_labels = np.where(
                    max_labels < threshold, background_class_id, label_indices
                )
                # Handle HuggingFace Padding (-100)
                # If the soft labels were padded, the row sum will be negative instead of 1.0
                row_sums = np.sum(labels, axis=-1)
                hard_labels = np.where(row_sums < 0, -100, hard_labels)
            else:
                # Fallback: hard labels passed directly
                hard_labels = labels

            labels = hard_labels

        else:
            # --- Standard path (no background class) ---
            # Handle soft labels: convert [B, C] probabilities to [B] hard indices
            if labels.ndim >= 2:
                labels = np.argmax(labels, axis=-1)

            if num_classes == 1:
                # Binary with single output
                predictions_proba = predictions_proba.squeeze(-1)
                predictions = (predictions_proba > 0.5).astype(int)
            elif num_classes == 2:
                # Binary with 2 outputs (use argmax or softmax)
                predictions = np.argmax(predictions_proba, axis=-1)
                predictions_proba = predictions_proba[:, 1]
            else:
                # Multi-class
                predictions = np.argmax(predictions_proba, axis=-1)

        return calculate_metric_with_sklearn(
            predictions_proba,
            predictions,
            labels,
            append_background_probability=append_background_probability,
        )

    return compute_metrics


# Backward-compatible module-level callables
compute_metrics = make_compute_metrics(background_threshold_func=None)
compute_metrics_soft_labels = make_compute_metrics(
    background_threshold_func=lambda n: 0.5
)


def extract_trainer_metrics(trainer) -> dict:
    """Extract the train/validation metrics a HuggingFace ``Trainer`` already computed.

    Scans ``trainer.state.log_history`` for training-summary entries
    (``train_loss``, ``train_runtime``, ... — including any explicit
    ``evaluate(..., metric_key_prefix="train")`` pass), and the evaluation entry
    corresponding to the **best** checkpoint. With early stopping the model
    trains several evaluations past the best step and ``load_best_model_at_end``
    restores the best checkpoint, so reporting the *last* eval would disagree
    with the saved artifact; ``trainer.state.best_global_step`` identifies the
    right entry. When it is unset (no ``metric_for_best_model`` /
    ``load_best_model_at_end``), the last eval entry is used as a fallback.
    ``eval_*`` keys are renamed to ``val_*`` for consistency with the
    ``train_*``/``val_*`` convention used by other classifiers' ``history``
    records.
    """
    log_history = trainer.state.log_history
    best_step = getattr(trainer.state, "best_global_step", None)

    eval_entries = [e for e in log_history if any(k.startswith("eval_") for k in e)]
    chosen_eval = None
    if best_step is not None:
        chosen_eval = next(
            (e for e in eval_entries if e.get("step") == best_step), None
        )
    if chosen_eval is None and eval_entries:
        chosen_eval = eval_entries[-1]

    metrics: dict = {}
    # Training-summary entries (end-of-train summary and/or an explicit
    # train-set evaluation) carry ``train_*`` keys directly.
    for entry in log_history:
        if "train_loss" in entry:
            metrics.update(
                {k: v for k, v in entry.items() if isinstance(v, (int, float))}
            )
    if chosen_eval is not None:
        metrics.update(
            {
                (f"val_{k[len('eval_'):]}" if k.startswith("eval_") else k): v
                for k, v in chosen_eval.items()
                if isinstance(v, (int, float))
            }
        )
    return metrics
