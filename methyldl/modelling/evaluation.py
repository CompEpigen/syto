import numpy as np
import sklearn
import torch
from typing import Union, Tuple, Any
from scipy.special import softmax

"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""


def calculate_metric_with_sklearn(predictions: np.ndarray, labels: np.ndarray):
    valid_mask = (
        labels != -100
    )  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    # print(valid_labels, valid_predictions)
    # print(sklearn.metrics.f1_score(
    #         valid_labels, valid_predictions, average="macro", zero_division=0
    #     ))
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
    }


def preprocess_logits_for_prediction(
    logits: Union[torch.Tensor, Tuple[torch.Tensor]], _
):
    """
    Preprocess logits for predictions.
    Returns probabilities:
    - For binary (2 classes): probability of positive class
    - For multi-class (>2 classes): probability distribution across all classes
    """
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]

    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])

    num_classes = logits.shape[-1]

    if num_classes == 2:
        # Binary classification: return probability of positive class (class 1)
        # Using sigmoid for compatibility with BCE loss, or softmax for CE loss
        # Sigmoid approach (works with both):
        return torch.sigmoid(logits)

        # Alternative softmax approach (more consistent with CE loss):
        # return torch.softmax(logits, dim=-1)[:, 1]
    else:
        # Multi-class classification: return full probability distribution
        return torch.softmax(logits, dim=-1)


def keep_logits_only(raw_model_output, labels):
    """
    We want to keep only the first item so that the Trainer
    concatenates an (N, num_labels) tensor nothing else.
    """
    if isinstance(raw_model_output, tuple):
        raw_model_output = raw_model_output[0]  # grab logits
    # (if it is already a Tensor, we just fall through)
    return raw_model_output


"""
Compute metrics used for huggingface trainer.
"""


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Handle soft labels: convert [B, C] probabilities to [B] hard indices
    if labels.ndim >= 2:
        labels = np.argmax(labels, axis=-1)
    num_classes = logits.shape[-1]
    if num_classes == 1:
        # Binary with single output
        predictions = predictions > 0.5
    elif num_classes == 2:
        # Binary with 2 outputs (use argmax or softmax)
        predictions = np.argmax(logits, axis=-1)
    else:
        # Multi-class
        predictions = np.argmax(logits, axis=-1)

    return calculate_metric_with_sklearn(predictions, labels)


"""
Compute metrics soft labels.
"""


def compute_metrics_soft_labels(eval_pred, threshold_func=lambda n: 0.5):
    """
    Computes metrics for soft labels by dynamically assigning a "Rejection" class
    if the maximum probability of a read falls below a certain threshold.
    """
    logits, soft_labels = eval_pred
    num_classes = logits.shape[-1]

    # 1. Calculate the threshold (e.g., 1/sqrt(39) ≈ 0.16)
    threshold = threshold_func(num_classes)

    # The rejection class will be assigned the index N (e.g., 39, if classes are 0-38)
    rejection_class_id = num_classes

    # ==========================================
    # PROCESS PREDICTIONS
    # ==========================================
    # Convert logits to probabilities
    probs = softmax(logits, axis=-1)
    max_probs = np.max(probs, axis=-1)
    pred_indices = np.argmax(probs, axis=-1)

    # If the highest probability is below the threshold, assign it to the Rejection class
    predictions = np.where(max_probs < threshold, rejection_class_id, pred_indices)

    # ==========================================
    # PROCESS GROUND TRUTH
    # ==========================================
    if soft_labels.ndim >= 2:
        max_labels = np.max(soft_labels, axis=-1)
        label_indices = np.argmax(soft_labels, axis=-1)

        # Apply the same threshold logic to the ground truth
        hard_labels = np.where(
            max_labels < threshold, rejection_class_id, label_indices
        )

        # Handle HuggingFace Padding (-100)
        # If the soft labels were padded, the row sum will be negative instead of 1.0
        row_sums = np.sum(soft_labels, axis=-1)
        hard_labels = np.where(row_sums < 0, -100, hard_labels)
    else:
        # Fallback just in case hard labels were passed somehow
        hard_labels = soft_labels

    return calculate_metric_with_sklearn(predictions, hard_labels)
