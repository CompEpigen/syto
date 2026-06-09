import torch
import torch.nn as nn
from typing import Dict
import torch.nn.functional as F
import math


class GRGAttentionClassificationHead(nn.Module):
    """
    Attention-based classifier that uses GRG information as contextual labels
    for sequence-level classification.
    """

    def __init__(self, config):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        self.num_grg_labels = config.num_grg_labels

        # GRG embedding layer
        self.grg_embedding = nn.Embedding(
            num_embeddings=self.num_grg_labels, embedding_dim=self.hidden_size
        )

        # Single-head attention components
        self.query_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.key_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.value_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # Attention dropout
        self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # Context fusion layer - combines attended features with GRG context
        self.context_fusion = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
        )

        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size // 2, self.num_labels),
        )
        # Focal loss bias initialization
        if getattr(config, "focal_init", False):
            print(
                "focal_init has been set to True. The biases for the final classification layer will be set such that p for target classes is focal_prior_prob/n_classes"
            )
            bg_index = getattr(config, "bg_class_index", 0)
            prior_prob = getattr(config, "focal_prior_prob", 0.01)
            num_fg = self.num_labels - 1

            # bias so that softmax gives ~(1-prior_prob) to bg,
            # ~prior_prob spread across foreground classes
            bg_bias = math.log((1.0 - prior_prob) / prior_prob * num_fg)

            final_layer = self.classifier[-1]  # last nn.Linear
            nn.init.normal_(final_layer.weight, std=0.01)
            nn.init.constant_(final_layer.bias, 0.0)
            final_layer.bias.data[bg_index] = bg_bias

        # Scale factor for attention scores
        self.scale = config.hidden_size**-0.5

    def forward(self, sequence_output, grg_ids, attention_mask=None):
        """
        Args:
            sequence_output: [batch_size, seq_len, hidden_size] - BERT output
            grg_ids: [batch_size] - GRG labels for each sequence
            attention_mask: [batch_size, seq_len] - attention mask for padding

        Returns:
            logits: [batch_size, num_labels] - classification logits
            attention_weights: [batch_size, seq_len] - attention weights for interpretability
        """
        batch_size, seq_len, hidden_size = sequence_output.shape

        # Get GRG embeddings and expand to sequence length
        grg_embeds = self.grg_embedding(grg_ids)  # [batch_size, hidden_size]
        grg_context = grg_embeds.unsqueeze(1).expand(
            -1, seq_len, -1
        )  # [batch_size, seq_len, hidden_size]

        # Compute attention using GRG context as query
        # This allows the model to attend to sequence positions relevant to the GR
        queries = self.query_proj(grg_context)  # [batch_size, seq_len, hidden_size]
        keys = self.key_proj(sequence_output)  # [batch_size, seq_len, hidden_size]
        values = self.value_proj(sequence_output)  # [batch_size, seq_len, hidden_size]

        # Compute attention scores
        attention_scores = (
            torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        )  # [batch_size, seq_len, seq_len]

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for attention computation
            extended_mask = attention_mask.unsqueeze(1).expand(
                -1, seq_len, -1
            )  # [batch_size, seq_len, seq_len]
            attention_scores = attention_scores.masked_fill(extended_mask == 0, -1e4)

        # Compute attention weights
        attention_weights = F.softmax(
            attention_scores, dim=-1
        )  # [batch_size, seq_len, seq_len]
        attention_weights = self.attn_dropout(attention_weights)

        # Apply attention to values
        attended_features = torch.matmul(
            attention_weights, values
        )  # [batch_size, seq_len, hidden_size]

        # Aggregate attended features (mean pooling over sequence length)
        # We use mean of the attended features as the sequence representation
        aggregated_features = attended_features.mean(dim=1)  # [batch_size, hidden_size]

        # Combine with GRG embedding for final context
        combined_features = torch.cat(
            [aggregated_features, grg_embeds], dim=-1
        )  # [batch_size, hidden_size * 2]

        # Apply context fusion
        fused_features = self.context_fusion(
            combined_features
        )  # [batch_size, hidden_size]

        # Final classification
        logits = self.classifier(fused_features)  # [batch_size, num_labels]

        # Return mean attention weights for interpretability
        mean_attention_weights = attention_weights.mean(dim=1)  # [batch_size, seq_len]

        return logits, mean_attention_weights
