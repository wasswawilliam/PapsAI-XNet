# src/models/attention.py

import torch
import torch.nn as nn


class SEChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style Channel Attention Module.

    This module learns channel-wise importance weights from CNN feature maps
    and recalibrates feature responses before projection to the ANFIS module.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        if channels < reduction:
            reduction = 1

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.gating = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.size()

        descriptor = self.global_avg_pool(x).view(batch_size, channels)
        attention_weights = self.gating(descriptor).view(batch_size, channels, 1, 1)

        return x * attention_weights
