import torch
import torch.nn as nn


class DMRAttentionDeconvolver(nn.Module):
    """
    Process each DMR row independently, then use attention
    to weight their contributions to final prediction.
    """

    def __init__(
        self,
        n_dmr_groups=39,
        n_pred_classes=40,
        n_cell_types=39,
        embed_dim=64,
        num_heads=4,
    ):
        super().__init__()

        self.n_dmr_groups = n_dmr_groups
        self.n_cell_types = n_cell_types

        # Encode each DMR's prediction distribution
        self.row_encoder = nn.Sequential(
            nn.Linear(n_pred_classes, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Learnable DMR embeddings (captures which DMR we're looking at)
        self.dmr_embeddings = nn.Parameter(torch.randn(n_dmr_groups, embed_dim) * 0.02)

        # Self-attention across DMRs
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=0.1, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(embed_dim)

        # Final prediction head
        self.proportion_head = nn.Sequential(
            nn.Linear(embed_dim * n_dmr_groups, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_cell_types),
        )

    def forward(self, x):
        # x: (batch, 39, 40) — each row is a distribution over predictions
        batch_size = x.shape[0]

        # Encode each DMR row
        row_features = self.row_encoder(x)  # (batch, 39, embed_dim)

        # Add DMR positional embeddings
        row_features = row_features + self.dmr_embeddings.unsqueeze(0)

        # Self-attention: DMRs attend to each other
        attn_out, _ = self.attention(row_features, row_features, row_features)
        row_features = self.attn_norm(row_features + attn_out)

        # Flatten and predict
        pooled = row_features.flatten(start_dim=1)  # (batch, 39 * embed_dim)
        logits = self.proportion_head(pooled)

        return torch.softmax(logits, dim=-1)
