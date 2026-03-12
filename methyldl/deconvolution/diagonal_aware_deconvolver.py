import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .training import DeconvolverOutput


class DiagonalAwareDeconvolver(nn.Module):
    """
    Explicitly model the diagonal (DMR_i → CellType_i) relationship
    plus off-diagonal confusion patterns.

    Enhanced with:
    - Configurable encoder pathways (diagonal, confusion, reject)
    - Feature extraction for all intermediate representations
    - Ablation controls to zero-out feature pathways at inference
    - Interpretability-focused forward passes

    Args:
        n_dmr_groups: Number of DMR groups (default: 39)
        n_pred_classes: Number of prediction classes including rejection (default: 40)
        n_cell_types: Number of output cell types (default: 39)
        hidden_dim: Hidden dimension for encoders (default: 128)
        use_diagonal: Whether to include diagonal encoder pathway (default: True)
        use_confusion: Whether to include confusion/off-diagonal encoder pathway (default: True)
        use_reject: Whether to include rejection column encoder pathway (default: True)

    Examples:
        # Full model (all pathways)
        model = DiagonalAwareDeconvolver()

        # Diagonal only (like XGBoost baseline but neural)
        model = DiagonalAwareDeconvolver(use_confusion=False, use_reject=False)

        # Diagonal + rejection (matches XGBoost feature set)
        model = DiagonalAwareDeconvolver(use_confusion=False)

        # Confusion patterns only
        model = DiagonalAwareDeconvolver(use_diagonal=False, use_reject=False)
    """

    def __init__(
        self,
        n_dmr_groups: int = 39,
        n_pred_classes: int = 40,
        n_cell_types: int = 39,
        hidden_dim: int = 128,
        use_diagonal: bool = True,
        use_confusion: bool = True,
        use_reject: bool = True,
    ):
        super().__init__()

        # Validate at least one pathway is enabled
        if not any([use_diagonal, use_confusion, use_reject]):
            raise ValueError("At least one encoder pathway must be enabled")

        # Store configuration
        self.n_dmr = n_dmr_groups
        self.n_pred_classes = n_pred_classes
        self.n_cell_types = n_cell_types
        self.hidden_dim = hidden_dim

        # Pathway flags
        self.use_diagonal = use_diagonal
        self.use_confusion = use_confusion
        self.use_reject = use_reject

        # Feature dimensions (0 if pathway disabled)
        self.diag_feature_dim = hidden_dim if use_diagonal else 0
        self.confusion_feature_dim = hidden_dim if use_confusion else 0
        self.reject_feature_dim = (hidden_dim // 2) if use_reject else 0
        self.combined_dim = (
            self.diag_feature_dim + self.confusion_feature_dim + self.reject_feature_dim
        )

        # Build only the enabled encoders
        if use_diagonal:
            self.diagonal_encoder = nn.Sequential(
                nn.Linear(n_dmr_groups, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.diagonal_encoder = None

        if use_confusion:
            self.confusion_encoder = nn.Sequential(
                nn.Linear(n_dmr_groups * n_pred_classes, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
        else:
            self.confusion_encoder = None

        if use_reject:
            self.reject_encoder = nn.Sequential(
                nn.Linear(n_dmr_groups, hidden_dim // 2), nn.GELU()
            )
        else:
            self.reject_encoder = None

        # Combine all features and predict proportions
        self.proportion_head = nn.Sequential(
            nn.Linear(self.combined_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_cell_types),
        )

    @property
    def config(self) -> Dict:
        """Return model configuration."""
        return {
            "n_dmr_groups": self.n_dmr,
            "n_pred_classes": self.n_pred_classes,
            "n_cell_types": self.n_cell_types,
            "hidden_dim": self.hidden_dim,
            "use_diagonal": self.use_diagonal,
            "use_confusion": self.use_confusion,
            "use_reject": self.use_reject,
            "combined_dim": self.combined_dim,
            "n_parameters": sum(p.numel() for p in self.parameters()),
        }

    def __repr__(self) -> str:
        pathways = []
        if self.use_diagonal:
            pathways.append("diagonal")
        if self.use_confusion:
            pathways.append("confusion")
        if self.use_reject:
            pathways.append("reject")

        return (
            f"{self.__class__.__name__}(\n"
            f"  pathways={pathways},\n"
            f"  combined_dim={self.combined_dim},\n"
            f"  n_parameters={sum(p.numel() for p in self.parameters()):,}\n"
            f")"
        )

    def extract_input_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract raw input features before encoding.

        Args:
            x: Input tensor of shape (batch, n_dmr_groups, n_pred_classes)

        Returns:
            Dictionary containing available features based on enabled pathways
        """
        features = {"full_matrix": x}

        if self.use_diagonal:
            features["diagonal"] = torch.diagonal(x[:, :, : self.n_dmr], dim1=1, dim2=2)

        if self.use_reject:
            features["reject_col"] = x[:, :, -1]

        if self.use_confusion:
            cell_type_matrix = x[:, :, : self.n_dmr].clone()
            mask = torch.eye(self.n_dmr, device=x.device, dtype=torch.bool)
            cell_type_matrix[:, mask] = 0
            features["off_diagonal"] = cell_type_matrix

        return features

    def encode_features(
        self,
        x: torch.Tensor,
        zero_diagonal: bool = False,
        zero_confusion: bool = False,
        zero_reject: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Encode input through enabled pathways with optional ablation.

        Args:
            x: Input tensor of shape (batch, n_dmr_groups, n_pred_classes)
            zero_diagonal: If True, zero out diagonal features (only if pathway enabled)
            zero_confusion: If True, zero out confusion features (only if pathway enabled)
            zero_reject: If True, zero out rejection features (only if pathway enabled)

        Returns:
            Tuple of (diag_features, confusion_features, reject_features)
            None for disabled pathways
        """
        diag_features = None
        confusion_features = None
        reject_features = None

        # Diagonal pathway
        if self.use_diagonal:
            diagonal = torch.diagonal(x[:, :, : self.n_dmr], dim1=1, dim2=2)
            diag_features = self.diagonal_encoder(diagonal)
            if zero_diagonal:
                diag_features = torch.zeros_like(diag_features)

        # Confusion pathway
        if self.use_confusion:
            full_flat = x.flatten(start_dim=1)
            confusion_features = self.confusion_encoder(full_flat)
            if zero_confusion:
                confusion_features = torch.zeros_like(confusion_features)

        # Rejection pathway
        if self.use_reject:
            reject_col = x[:, :, -1]
            reject_features = self.reject_encoder(reject_col)
            if zero_reject:
                reject_features = torch.zeros_like(reject_features)

        return diag_features, confusion_features, reject_features

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        zero_diagonal: bool = False,
        zero_confusion: bool = False,
        zero_reject: bool = False,
    ) -> torch.Tensor | DeconvolverOutput:
        """
        Forward pass with optional feature extraction and ablation.

        Args:
            x: Input tensor of shape (batch, n_dmr_groups, n_pred_classes)
            return_features: If True, return DeconvolverOutput with all features
            zero_diagonal: If True, zero out diagonal pathway (if enabled)
            zero_confusion: If True, zero out confusion pathway (if enabled)
            zero_reject: If True, zero out rejection pathway (if enabled)

        Returns:
            If return_features=False: Tensor of proportions (batch, n_cell_types)
            If return_features=True: DeconvolverOutput dataclass
        """
        # Encode with optional ablation
        diag_features, confusion_features, reject_features = self.encode_features(
            x, zero_diagonal, zero_confusion, zero_reject
        )

        # Combine only enabled features
        features_to_cat = []
        if diag_features is not None:
            features_to_cat.append(diag_features)
        if confusion_features is not None:
            features_to_cat.append(confusion_features)
        if reject_features is not None:
            features_to_cat.append(reject_features)

        combined = torch.cat(features_to_cat, dim=-1)

        # Get logits and proportions
        logits = self.proportion_head(combined)
        proportions = torch.softmax(logits, dim=-1)

        if return_features:
            return DeconvolverOutput(
                proportions=proportions,
                diag_features=diag_features,
                confusion_features=confusion_features,
                reject_features=reject_features,
                combined_features=combined,
                logits=logits,
            )
        return proportions

    def ablation_study(
        self, x: torch.Tensor, y_true: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Run systematic ablation study on enabled feature pathways.

        Only tests ablations that are valid for the current configuration.

        Args:
            x: Input tensor of shape (batch, n_dmr_groups, n_pred_classes)
            y_true: Optional ground truth for computing metrics

        Returns:
            Dictionary with predictions under each valid ablation condition
        """
        results = {}

        with torch.no_grad():
            # Always include full model (no ablation)
            results["full_model"] = self.forward(x)

            # Single pathway ablations (only for enabled pathways)
            if self.use_diagonal:
                results["no_diagonal"] = self.forward(x, zero_diagonal=True)
            if self.use_confusion:
                results["no_confusion"] = self.forward(x, zero_confusion=True)
            if self.use_reject:
                results["no_reject"] = self.forward(x, zero_reject=True)

            # "Only X" conditions (requires at least 2 pathways enabled)
            n_enabled = sum([self.use_diagonal, self.use_confusion, self.use_reject])

            if n_enabled >= 2:
                if self.use_diagonal:
                    results["only_diagonal"] = self.forward(
                        x,
                        zero_confusion=self.use_confusion,
                        zero_reject=self.use_reject,
                    )
                if self.use_confusion:
                    results["only_confusion"] = self.forward(
                        x, zero_diagonal=self.use_diagonal, zero_reject=self.use_reject
                    )
                if self.use_reject:
                    results["only_reject"] = self.forward(
                        x,
                        zero_diagonal=self.use_diagonal,
                        zero_confusion=self.use_confusion,
                    )

        return results

    def get_feature_importance(
        self, x: torch.Tensor, target_cell_type: int, method: str = "gradient"
    ) -> Dict[str, torch.Tensor]:
        """
        Compute feature importance for a target cell type.

        Args:
            x: Input tensor (batch, n_dmr_groups, n_pred_classes)
            target_cell_type: Index of cell type to analyze
            method: 'gradient' or 'integrated_gradient'

        Returns:
            Dictionary with importance scores for enabled input features
        """
        x = x.clone().requires_grad_(True)

        output = self.forward(x, return_features=True)
        target_prob = output.proportions[:, target_cell_type].sum()
        target_prob.backward()

        importance = {"input_gradient": x.grad.detach()}

        if self.use_diagonal:
            importance["diagonal_importance"] = (
                x.grad[:, :, : self.n_dmr].diagonal(dim1=1, dim2=2).detach()
            )

        if self.use_reject:
            importance["reject_importance"] = x.grad[:, :, -1].detach()

        return importance

