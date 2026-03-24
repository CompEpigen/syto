import torch
import torch.nn as nn


class FlattenMLPDeconvolver(nn.Module):
    """
    Flatten 39x40 → 1560 features, then MLP.
    Simple but ignores spatial structure.
    """

    def __init__(
        self,
        n_dmr_groups=39,
        n_classes=40,
        n_cell_types=39,
        hidden_dims=[512, 256, 128],
    ):
        super().__init__()

        input_dim = n_dmr_groups * n_classes

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.GELU(),
                    nn.Dropout(0.2),
                ]
            )
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, n_cell_types))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, 39, 40)
        x = x.flatten(start_dim=1)  # (batch, 1560)
        logits = self.network(x)
        return torch.softmax(logits, dim=-1)
