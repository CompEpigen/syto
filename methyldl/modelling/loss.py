import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceWeightedCrossEntropy(nn.Module):
    def __init__(self, num_classes=39, penalty_scale=1.0, on_target_weight=None):
        """
        Args:
            num_classes: Number of distinct cell types.
            penalty_scale: Scaling factor for the penalty applied to background reads.
            on_target_weight: Optional fixed weight for on-target reads (default is 1.0).
        """
        super().__init__()
        self.num_classes = num_classes
        self.min_prob = 1.0 / num_classes # Baseline for a perfectly flat distribution
        self.penalty_scale = penalty_scale
        self.on_target_weight = on_target_weight
        self.use_specific_on_target_weight = on_target_weight is not None

    def forward(self, logits: torch.Tensor, soft_targets: torch.Tensor, ) -> torch.Tensor:
        """
        Args:
            logits: Unnormalized predictions from the model, shape (batch_size, num_classes)
            soft_targets: Ground truth probabilities, shape (batch_size, num_classes)
        """
        # 1. Calculate standard Cross Entropy loss per item in the batch
        # Note: PyTorch's cross_entropy supports soft targets natively in recent versions
        ce_loss = F.cross_entropy(logits, soft_targets, reduction="none")

        # 2. Determine the "Sharpness" (confidence) of each ground truth target
        # Extract the highest probability in each target vector
        target_max_probs = soft_targets.max(dim=1)[0]

        # 3. Normalize the weights
        # A perfectly flat target (e.g., 1/39) gets a weight of ~0.0
        # A perfectly sharp target (e.g., 1.0) gets a weight of 1.0
        weights = (target_max_probs - self.min_prob) / (1.0 - self.min_prob)

        # Ensure weights don't drop exactly to zero to maintain slight background learning,
        # but keep them very small (e.g., clamp at 0.01)
        weights = torch.clamp(weights, min=0.01)

        # 4. Apply weights to the loss
        weighted_loss = ce_loss * weights

        # Return the mean loss across the batch
        return weighted_loss.mean()


class OnTargetSoftLoss(nn.Module):
    def __init__(self, num_classes=39, min_bg_weight=0.01):
        """
        Args:
            num_classes: Number of distinct cell types.
            min_bg_weight: The floor multiplier for pure background/flat reads.
        """
        super().__init__()
        self.num_classes = num_classes
        self.min_prob = 1.0 / num_classes  # Baseline for a perfectly flat distribution
        self.min_bg_weight = min_bg_weight

    def forward(
        self,
        logits: torch.Tensor,
        soft_targets: torch.Tensor,
        is_on_target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, num_classes) Unnormalized predictions
            soft_targets: (batch_size, num_classes) Ground truth soft labels
            is_on_target: (batch_size,) Boolean tensor where True means
                          original_label == dmr_ctype_label
        """
        # 1. Calculate standard Cross Entropy loss per item in the batch
        ce_loss = F.cross_entropy(logits, soft_targets, reduction="none")

        # 2. Calculate the penalty for OFF-TARGET (background) reads based on their sharpness
        target_max_probs = soft_targets.max(dim=1)[0]

        # Scale background weights: flat target approaches 0.0, sharp target approaches 1.0
        bg_weights = (target_max_probs - self.min_prob) / (1.0 - self.min_prob)
        bg_weights = torch.clamp(bg_weights, min=self.min_bg_weight, max=1.0)

        # 3. The Core Logic:
        # If on-target -> Weight is exactly 1.0 (No penalty)
        # If off-target -> Weight is the penalized bg_weights
        final_weights = torch.where(
            is_on_target, torch.ones_like(bg_weights), bg_weights
        )

        # 4. Apply weights to the loss and return the mean
        weighted_loss = ce_loss * final_weights
        return weighted_loss.mean()
