import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class ConfidenceWeightedCrossEntropy(_Loss):
    """
    Custom loss function that applies a dynamic penalty to reads
    based on their confidence (sharpness) in the target distribution.
    """

    def __init__(
        self,
        num_classes=39,
        penalty_scale=1.0,
        on_target_weight=None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
    ) -> None:
        """
        Args:
            num_classes: Number of distinct cell types.
            penalty_scale: Scaling factor for the penalty applied to high entropy reads.
            on_target_weight: Optional fixed weight for on-target reads.
                None means on-target reads are not treated differently from off-target reads.
                If set to a value, on-target reads will be weighted by this value.
        """
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)
        assert 0 <= penalty_scale and penalty_scale <= 1.0
        self.num_classes = num_classes
        self.min_prob = 1.0 / num_classes  # Baseline for a perfectly flat distribution
        self.penalty_scale = penalty_scale
        self.on_target_weight = on_target_weight
        self.use_specific_on_target_weight = on_target_weight is not None
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        soft_targets: torch.Tensor,
        is_on_target: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: Unnormalized predictions from the model, shape (batch_size, num_classes)
            soft_targets: Ground truth probabilities, shape (batch_size, num_classes)
        """
        assert (
            is_on_target is None or self.use_specific_on_target_weight
        ), "is_on_target should be provided iff on_target_weight is set."
        if is_on_target is not None:
            assert (
                is_on_target.shape[0] == logits.shape[0]
            ), "is_on_target must have the same batch size as logits."

        # Calculate standard Cross Entropy loss per item in the batch
        ce_loss = F.cross_entropy(logits, soft_targets, reduction="none")

        # Determine the "Sharpness" (confidence) of each ground truth target
        # by extracting the highest probability in each target vector
        target_max_probs = soft_targets.max(dim=1)[0]

        # Normalize the weights
        # A perfectly flat target (e.g., 1/39) gets a weight of ~0.0
        # A perfectly sharp target (e.g., 1.0) gets a weight of 1.0
        weights = (target_max_probs - self.min_prob) / (1.0 - self.min_prob)

        # scale the weights by the penalty factor
        # If penalty factor
        weights = (1 - self.penalty_scale) * torch.ones_like(
            weights
        ) + self.penalty_scale * weights

        #
        if self.use_specific_on_target_weight:
            weights = torch.where(
                is_on_target,
                torch.full_like(weights, self.on_target_weight),
                weights,
            )

        # Ensure weights don't drop exactly to zero to maintain slight background learning,
        # but keep them very small
        weights = torch.clamp(weights, min=0.01)

        # Apply weights to the loss
        weighted_loss = ce_loss * weights

        # Apply reduction
        if self.reduction == "mean":
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        else:
            return weighted_loss


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.1,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    This code is from : https://pytorch.org/vision/main/_modules/torchvision/ops/focal_loss.html

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (float): Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default: ``0.25``.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples. Default: ``2``.
        reduction (string): ``'none'`` | ``'mean'`` | ``'sum'``
                ``'none'``: No reduction will be applied to the output.
                ``'mean'``: The output will be averaged.
                ``'sum'``: The output will be summed. Default: ``'none'``.
    Returns:
        Loss tensor with the reduction option applied.
    """
    # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py

    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss


class FocalLoss(_Loss):

    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return sigmoid_focal_loss(input, target, reduction=self.reduction)
