
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.gridspec import GridSpec

from methyldl.deconvolution.diagonal_aware_deconvolver import DiagonalAwareDeconvolver
from methyldl.deconvolution.training import DeconvolverOutput




class DeconvolverVisualizer:
    """
    Comprehensive visualization tools for interpretability analysis.
    """

    def __init__(
        self,
        model: DiagonalAwareDeconvolver,
        cell_type_names: Optional[List[str]] = None,
        dmr_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.cell_type_names = cell_type_names or [
            f"CT_{i}" for i in range(model.n_cell_types)
        ]
        self.dmr_names = dmr_names or [f"DMR_{i}" for i in range(model.n_dmr)]

    def plot_ablation_comparison(
        self,
        ablation_results: Dict[str, torch.Tensor],
        y_true: Optional[torch.Tensor] = None,
        sample_idx: int = 0,
        figsize: Tuple[int, int] = (16, 10),
    ) -> plt.Figure:
        """
        Visualize predictions under different ablation conditions.

        Shows how zeroing out different feature pathways affects predictions,
        revealing the contribution of each pathway.
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

        conditions = list(ablation_results.keys())

        for idx, condition in enumerate(conditions):
            ax = fig.add_subplot(gs[idx // 4, idx % 4])

            pred = ablation_results[condition][sample_idx].detach().cpu().numpy()

            colors = plt.cm.viridis(pred / pred.max() if pred.max() > 0 else pred)
            bars = ax.bar(range(len(pred)), pred, color=colors)

            if y_true is not None:
                true = y_true[sample_idx].detach().cpu().numpy()
                ax.scatter(
                    range(len(true)),
                    true,
                    color="red",
                    marker="x",
                    s=50,
                    zorder=5,
                    label="True",
                )
                ax.legend(loc="upper right", fontsize=8)

            ax.set_title(condition.replace("_", " ").title(), fontsize=10)
            ax.set_xlabel("Cell Type")
            ax.set_ylabel("Proportion")
            ax.set_ylim(0, 1)

            # Show top 3 predicted cell types
            top3 = np.argsort(pred)[-3:][::-1]
            ax.set_xticks(top3)
            ax.set_xticklabels(
                [self.cell_type_names[i] for i in top3], rotation=45, fontsize=8
            )

        fig.suptitle("Ablation Study: Feature Pathway Contributions", fontsize=14)
        return fig

    def plot_feature_space(
        self,
        outputs: List[DeconvolverOutput],
        labels: Optional[torch.Tensor] = None,
        feature_type: str = "combined",
        method: str = "pca",
        figsize: Tuple[int, int] = (12, 5),
    ) -> plt.Figure:
        """
        Visualize learned feature representations using dimensionality reduction.

        Useful for understanding:
        - How well features separate different cell type compositions
        - Clustering structure in the learned representations

        Args:
            outputs: List of DeconvolverOutput from multiple samples
            labels: Optional labels for coloring (e.g., dominant cell type)
            feature_type: 'diag', 'confusion', 'reject', or 'combined'
            method: 'pca', 'tsne', or 'umap'
        """
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE

        # Stack features
        feature_map = {
            "diag": "diag_features",
            "confusion": "confusion_features",
            "reject": "reject_features",
            "combined": "combined_features",
        }

        features = (
            torch.cat([getattr(o, feature_map[feature_type]) for o in outputs], dim=0)
            .detach()
            .cpu()
            .numpy()
        )

        # Dimensionality reduction
        if method == "pca":
            reducer = PCA(n_components=2)
            embedded = reducer.fit_transform(features)
            var_explained = reducer.explained_variance_ratio_
        elif method == "tsne":
            reducer = TSNE(n_components=2, perplexity=min(30, len(features) - 1))
            embedded = reducer.fit_transform(features)
            var_explained = None
        else:
            raise ValueError(f"Unknown method: {method}")

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Colored by labels if provided
        ax1 = axes[0]
        if labels is not None:
            labels_np = (
                labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels
            )
            scatter = ax1.scatter(
                embedded[:, 0],
                embedded[:, 1],
                c=labels_np,
                cmap="tab20",
                alpha=0.7,
                s=50,
            )
            plt.colorbar(scatter, ax=ax1, label="Label")
        else:
            ax1.scatter(embedded[:, 0], embedded[:, 1], alpha=0.7, s=50)

        ax1.set_xlabel(f"{method.upper()} 1")
        ax1.set_ylabel(f"{method.upper()} 2")
        ax1.set_title(f"{feature_type.title()} Features - {method.upper()}")

        if var_explained is not None:
            ax1.set_xlabel(f"{method.upper()} 1 ({var_explained[0]:.1%} var)")
            ax1.set_ylabel(f"{method.upper()} 2 ({var_explained[1]:.1%} var)")

        # Plot 2: Feature statistics
        ax2 = axes[1]
        feature_norms = np.linalg.norm(features, axis=1)
        ax2.hist(feature_norms, bins=30, edgecolor="black", alpha=0.7)
        ax2.axvline(
            feature_norms.mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {feature_norms.mean():.2f}",
        )
        ax2.set_xlabel("Feature Norm")
        ax2.set_ylabel("Count")
        ax2.set_title(f"{feature_type.title()} Feature Magnitude Distribution")
        ax2.legend()

        plt.tight_layout()
        return fig

    def plot_diagonal_analysis(
        self,
        x: torch.Tensor,
        output: DeconvolverOutput,
        sample_idx: int = 0,
        figsize: Tuple[int, int] = (14, 10),
    ) -> plt.Figure:
        """
        Detailed analysis of diagonal pathway contribution.

        Visualizes:
        - Raw diagonal values from input
        - Correlation between diagonal and predictions
        - Per-cell-type diagonal importance
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # Extract data
        diag_input = (
            torch.diagonal(x[sample_idx, :, : self.model.n_dmr], dim1=0, dim2=1)
            .detach()
            .cpu()
            .numpy()
        )
        pred = output.proportions[sample_idx].detach().cpu().numpy()
        diag_feat = output.diag_features[sample_idx].detach().cpu().numpy()

        # Plot 1: Diagonal input values
        ax1 = fig.add_subplot(gs[0, 0])
        bars = ax1.bar(
            range(len(diag_input)),
            diag_input,
            color=plt.cm.Blues(diag_input / diag_input.max()),
        )
        ax1.set_xlabel("DMR Group / Cell Type Index")
        ax1.set_ylabel("Diagonal Value (P(CT_i | DMR_i))")
        ax1.set_title("Raw Diagonal Input Values")

        # Highlight top values
        top5 = np.argsort(diag_input)[-5:]
        for i in top5:
            ax1.annotate(
                self.cell_type_names[i], (i, diag_input[i]), fontsize=8, rotation=45
            )

        # Plot 2: Diagonal vs Prediction scatter
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(diag_input, pred, alpha=0.7, s=60)

        # Add correlation line
        z = np.polyfit(diag_input, pred, 1)
        p = np.poly1d(z)
        x_line = np.linspace(diag_input.min(), diag_input.max(), 100)
        ax2.plot(
            x_line,
            p(x_line),
            "r--",
            alpha=0.8,
            label=f"r={np.corrcoef(diag_input, pred)[0,1]:.3f}",
        )

        ax2.set_xlabel("Diagonal Input Value")
        ax2.set_ylabel("Predicted Proportion")
        ax2.set_title("Diagonal-Prediction Correlation")
        ax2.legend()

        # Plot 3: Encoded diagonal features heatmap
        ax3 = fig.add_subplot(gs[1, 0])
        diag_feat_2d = (
            diag_feat.reshape(-1, 16)
            if len(diag_feat) >= 16
            else diag_feat.reshape(1, -1)
        )
        sns.heatmap(
            diag_feat_2d,
            cmap="RdBu_r",
            center=0,
            ax=ax3,
            cbar_kws={"label": "Activation"},
        )
        ax3.set_title("Encoded Diagonal Features")
        ax3.set_xlabel("Feature Dimension")
        ax3.set_ylabel("Feature Block")

        # Plot 4: Per-cell-type contribution analysis
        ax4 = fig.add_subplot(gs[1, 1])
        contribution = diag_input * pred  # Simple interaction
        colors = ["green" if c > 0.01 else "gray" for c in contribution]
        ax4.barh(range(len(contribution)), contribution, color=colors)
        ax4.set_xlabel("Diagonal × Prediction")
        ax4.set_ylabel("Cell Type Index")
        ax4.set_title("Diagonal Contribution Score")
        ax4.set_yticks(range(0, len(contribution), 5))

        fig.suptitle(f"Diagonal Pathway Analysis (Sample {sample_idx})", fontsize=14)
        return fig

    def plot_confusion_patterns(
        self,
        x: torch.Tensor,
        output: DeconvolverOutput,
        sample_idx: int = 0,
        figsize: Tuple[int, int] = (16, 6),
    ) -> plt.Figure:
        """
        Visualize off-diagonal confusion patterns.

        Shows:
        - Full prediction matrix heatmap
        - Off-diagonal structure that might indicate cell type similarities
        - Confusion feature activation patterns
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # Get the cell type portion of input matrix
        matrix = x[sample_idx, :, : self.model.n_dmr].detach().cpu().numpy()

        # Plot 1: Full matrix heatmap
        ax1 = axes[0]
        sns.heatmap(
            matrix,
            cmap="YlOrRd",
            ax=ax1,
            xticklabels=5,
            yticklabels=5,
            cbar_kws={"label": "Prediction Probability"},
        )
        ax1.set_xlabel("Predicted Cell Type")
        ax1.set_ylabel("DMR Group")
        ax1.set_title("Full Prediction Matrix")

        # Highlight diagonal
        for i in range(min(matrix.shape)):
            ax1.add_patch(
                plt.Rectangle((i, i), 1, 1, fill=False, edgecolor="blue", linewidth=2)
            )

        # Plot 2: Off-diagonal only (zeroed diagonal)
        ax2 = axes[1]
        off_diag = matrix.copy()
        np.fill_diagonal(off_diag, 0)
        sns.heatmap(
            off_diag,
            cmap="YlOrRd",
            ax=ax2,
            xticklabels=5,
            yticklabels=5,
            cbar_kws={"label": "Off-Diagonal Value"},
        )
        ax2.set_xlabel("Predicted Cell Type")
        ax2.set_ylabel("DMR Group")
        ax2.set_title("Off-Diagonal Confusion Patterns")

        # Plot 3: Confusion features
        ax3 = axes[2]
        conf_feat = output.confusion_features[sample_idx].detach().cpu().numpy()
        conf_feat_2d = (
            conf_feat.reshape(-1, 16)
            if len(conf_feat) >= 16
            else conf_feat.reshape(1, -1)
        )
        sns.heatmap(
            conf_feat_2d,
            cmap="RdBu_r",
            center=0,
            ax=ax3,
            cbar_kws={"label": "Activation"},
        )
        ax3.set_title("Encoded Confusion Features")
        ax3.set_xlabel("Feature Dimension")
        ax3.set_ylabel("Feature Block")

        plt.suptitle(f"Confusion Pattern Analysis (Sample {sample_idx})", fontsize=14)
        plt.tight_layout()
        return fig

    def plot_rejection_analysis(
        self,
        x: torch.Tensor,
        output: DeconvolverOutput,
        sample_idx: int = 0,
        figsize: Tuple[int, int] = (14, 5),
    ) -> plt.Figure:
        """
        Analyze rejection column patterns.

        Shows:
        - Rejection probability per DMR group
        - Relationship between rejection and prediction confidence
        - Rejection feature activations
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        reject_col = x[sample_idx, :, -1].detach().cpu().numpy()
        pred = output.proportions[sample_idx].detach().cpu().numpy()
        reject_feat = output.reject_features[sample_idx].detach().cpu().numpy()

        # Plot 1: Rejection values per DMR
        ax1 = axes[0]
        colors = plt.cm.Reds(reject_col / max(reject_col.max(), 0.01))
        ax1.bar(range(len(reject_col)), reject_col, color=colors)
        ax1.set_xlabel("DMR Group")
        ax1.set_ylabel("Rejection Probability")
        ax1.set_title("Per-DMR Rejection Values")
        ax1.axhline(
            reject_col.mean(),
            color="blue",
            linestyle="--",
            label=f"Mean: {reject_col.mean():.3f}",
        )
        ax1.legend()

        # Plot 2: Rejection vs Prediction Entropy
        ax2 = axes[1]
        pred_entropy = -np.sum(pred * np.log(pred + 1e-10))
        total_rejection = reject_col.sum()

        # Create a summary visualization
        metrics = {
            "Total Rejection": total_rejection,
            "Mean Rejection": reject_col.mean(),
            "Max Rejection": reject_col.max(),
            "Pred Entropy": pred_entropy,
            "Max Pred": pred.max(),
        }

        bars = ax2.bar(metrics.keys(), metrics.values(), color="steelblue")
        ax2.set_ylabel("Value")
        ax2.set_title("Rejection & Confidence Metrics")
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Annotate bars
        for bar, val in zip(bars, metrics.values()):
            ax2.annotate(
                f"{val:.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Plot 3: Rejection features
        ax3 = axes[2]
        ax3.bar(
            range(len(reject_feat)),
            reject_feat,
            color=plt.cm.RdBu_r(
                (reject_feat - reject_feat.min())
                / (reject_feat.max() - reject_feat.min() + 1e-10)
            ),
        )
        ax3.set_xlabel("Feature Dimension")
        ax3.set_ylabel("Activation")
        ax3.set_title("Encoded Rejection Features")
        ax3.axhline(0, color="black", linewidth=0.5)

        plt.suptitle(f"Rejection Pattern Analysis (Sample {sample_idx})", fontsize=14)
        plt.tight_layout()
        return fig

    def plot_pathway_contributions(
        self, x: torch.Tensor, sample_idx: int = 0, figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Compare predictions with each pathway in isolation and combined.

        This is a key interpretability visualization showing how each
        pathway independently predicts and how they combine.
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        with torch.no_grad():
            # Get predictions with different ablations
            full_pred = (
                self.model(x[sample_idx : sample_idx + 1]).squeeze().cpu().numpy()
            )

            # Only diagonal
            diag_only = (
                self.model(
                    x[sample_idx : sample_idx + 1],
                    zero_confusion=True,
                    zero_reject=True,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

            # Only confusion
            conf_only = (
                self.model(
                    x[sample_idx : sample_idx + 1], zero_diagonal=True, zero_reject=True
                )
                .squeeze()
                .cpu()
                .numpy()
            )

            # Only rejection
            rej_only = (
                self.model(
                    x[sample_idx : sample_idx + 1],
                    zero_diagonal=True,
                    zero_confusion=True,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        predictions = {
            "Full Model": full_pred,
            "Diagonal Only": diag_only,
            "Confusion Only": conf_only,
            "Rejection Only": rej_only,
        }

        colors = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]

        for ax, (name, pred), color in zip(axes.flat, predictions.items(), colors):
            bars = ax.bar(range(len(pred)), pred, color=color, alpha=0.7)
            ax.set_title(name, fontsize=12)
            ax.set_xlabel("Cell Type")
            ax.set_ylabel("Proportion")
            ax.set_ylim(0, 1)

            # Annotate top 3
            top3 = np.argsort(pred)[-3:][::-1]
            for i in top3:
                if pred[i] > 0.01:
                    ax.annotate(
                        f"{self.cell_type_names[i]}\n{pred[i]:.2f}",
                        (i, pred[i]),
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        fig.suptitle(
            f"Pathway Contribution Analysis (Sample {sample_idx})", fontsize=14
        )
        plt.tight_layout()
        return fig

    def plot_feature_correlation_matrix(
        self, outputs: List[DeconvolverOutput], figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Visualize correlations between different feature types.

        Useful for understanding redundancy and independence of pathways.
        """
        # Stack all features
        diag = (
            torch.cat([o.diag_features for o in outputs], dim=0).detach().cpu().numpy()
        )
        conf = (
            torch.cat([o.confusion_features for o in outputs], dim=0)
            .detach()
            .cpu()
            .numpy()
        )
        rej = (
            torch.cat([o.reject_features for o in outputs], dim=0)
            .detach()
            .cpu()
            .numpy()
        )

        # Compute mean activations per sample
        diag_mean = diag.mean(axis=1)
        conf_mean = conf.mean(axis=1)
        rej_mean = rej.mean(axis=1)

        # Stack and compute correlation
        all_means = np.stack([diag_mean, conf_mean, rej_mean], axis=1)

        # Also add feature norms
        diag_norm = np.linalg.norm(diag, axis=1)
        conf_norm = np.linalg.norm(conf, axis=1)
        rej_norm = np.linalg.norm(rej, axis=1)

        all_features = np.stack(
            [diag_mean, conf_mean, rej_mean, diag_norm, conf_norm, rej_norm], axis=1
        )

        feature_names = [
            "Diag Mean",
            "Conf Mean",
            "Rej Mean",
            "Diag Norm",
            "Conf Norm",
            "Rej Norm",
        ]

        fig, ax = plt.subplots(figsize=figsize)
        corr_matrix = np.corrcoef(all_features.T)

        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            ax=ax,
            xticklabels=feature_names,
            yticklabels=feature_names,
            vmin=-1,
            vmax=1,
        )

        ax.set_title("Feature Pathway Correlation Matrix")
        plt.tight_layout()
        return fig

    def create_interpretability_report(
        self,
        x: torch.Tensor,
        y_true: Optional[torch.Tensor] = None,
        sample_indices: List[int] = None,
        save_dir: str = None,
    ) -> Dict[str, plt.Figure]:
        """
        Generate comprehensive interpretability report.

        Creates all visualizations for selected samples.

        Args:
            x: Input batch
            y_true: Optional ground truth
            sample_indices: Which samples to analyze (default: first 3)
            save_dir: Optional directory to save figures

        Returns:
            Dictionary of figure names to figures
        """
        if sample_indices is None:
            sample_indices = list(range(min(3, len(x))))

        figures = {}

        # Get outputs for all samples
        with torch.no_grad():
            outputs = [
                self.model(x[i : i + 1], return_features=True) for i in sample_indices
            ]

        # Ablation study
        ablation_results = self.model.ablation_study(x)
        figures["ablation"] = self.plot_ablation_comparison(
            ablation_results, y_true, sample_indices[0]
        )

        # Per-sample analyses
        for idx, sample_idx in enumerate(sample_indices):
            output = outputs[idx]

            figures[f"diagonal_s{sample_idx}"] = self.plot_diagonal_analysis(
                x, output, sample_idx
            )
            figures[f"confusion_s{sample_idx}"] = self.plot_confusion_patterns(
                x, output, sample_idx
            )
            figures[f"rejection_s{sample_idx}"] = self.plot_rejection_analysis(
                x, output, sample_idx
            )
            figures[f"pathways_s{sample_idx}"] = self.plot_pathway_contributions(
                x, sample_idx
            )

        # Cross-sample analyses
        all_outputs = [
            self.model(x[i : i + 1], return_features=True) for i in range(len(x))
        ]
        figures["feature_correlation"] = self.plot_feature_correlation_matrix(
            all_outputs
        )

        # Feature space visualization
        if len(x) > 10:
            labels = x[:, :, : self.model.n_dmr].diagonal(dim1=1, dim2=2).argmax(dim=1)
            figures["feature_space"] = self.plot_feature_space(
                all_outputs, labels, feature_type="combined", method="pca"
            )

        # Save if directory provided
        if save_dir:
            import os

            os.makedirs(save_dir, exist_ok=True)
            for name, fig in figures.items():
                fig.savefig(
                    os.path.join(save_dir, f"{name}.png"), dpi=150, bbox_inches="tight"
                )

        return figures

