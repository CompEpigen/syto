import torch
import torch.nn as nn

class CNNDeconvolver(nn.Module):
    """
    Treat 39×40 matrix as a single-channel image.
    """

    def __init__(self, n_dmr_groups=39, n_pred_classes=40, n_cell_types=39):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # (1, 39, 40) → (32, 39, 40)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            # (32, 39, 40) → (64, 19, 20)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # (64, 19, 20) → (128, 9, 10)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # (128, 4, 4)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_cell_types),
        )

    def forward(self, x):
        # x: (batch, 39, 40)
        x = x.unsqueeze(1)  # (batch, 1, 39, 40)
        x = self.conv_layers(x)
        logits = self.fc(x)
        return torch.softmax(logits, dim=-1)
