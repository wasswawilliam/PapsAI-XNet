# src/models/feature_norm.py

import torch
import torch.nn as nn


class L2FeatureNormalization(nn.Module):
    """
    L2 normalization layer for stabilizing CNN embeddings before ANFIS reasoning.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(h, p=2, dim=1, keepdim=True)
        return h / (norm + self.eps)
