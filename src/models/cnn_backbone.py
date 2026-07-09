"""
===========================================================
PapsAI XNet CNN Backbone
===========================================================

Implements the attention-enhanced CNN feature extractor described
in Section 2.3 of the manuscript.

Architecture

Input
↓

Conv Block 1 (32)

↓

Conv Block 2 (64)

↓

Conv Block 3 (128)

↓

Conv Block 4 (256)

↓

Global Average Pooling

↓

SE Channel Attention

↓

Fully Connected (32)

↓

L2 Feature Normalization

↓

32-dimensional embedding to ANFIS
===========================================================
"""

import torch
import torch.nn as nn

from .attention import SEChannelAttention
from .feature_norm import L2FeatureNormalization


class ConvBlock(nn.Module):
    """
    Standard CNN block

    Conv
    ↓
    BatchNorm
    ↓
    ReLU
    ↓
    MaxPool
    """

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )

        )

    def forward(self, x):

        return self.block(x)


class PapsAICNN(nn.Module):
    """
    Attention-enhanced CNN backbone.

    Output:
        32-dimensional normalized feature embedding.
    """

    def __init__(self):

        super().__init__()

        ####################################################
        # Feature extractor
        ####################################################

        self.layer1 = ConvBlock(3, 32)

        self.layer2 = ConvBlock(32, 64)

        self.layer3 = ConvBlock(64, 128)

        self.layer4 = ConvBlock(128, 256)

        ####################################################
        # Global Average Pooling
        ####################################################

        self.gap = nn.AdaptiveAvgPool2d(1)

        ####################################################
        # Channel Attention Module
        ####################################################

        self.attention = SEChannelAttention(
            channels=256,
            reduction=16
        )

        ####################################################
        # Bottleneck
        ####################################################

        self.fc = nn.Sequential(

            nn.Linear(
                256,
                32
            ),

            nn.ReLU(inplace=True)

        )

        ####################################################
        # Feature normalization
        ####################################################

        self.normalize = L2FeatureNormalization()

    def forward(self, x):

        ##############################################
        # CNN feature extraction
        ##############################################

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        ##############################################
        # Attention
        ##############################################

        x = self.attention(x)

        ##############################################
        # Global Average Pooling
        ##############################################

        x = self.gap(x)

        x = torch.flatten(x, 1)

        ##############################################
        # Bottleneck
        ##############################################

        embedding = self.fc(x)

        ##############################################
        # L2 normalization
        ##############################################

        embedding = self.normalize(embedding)

        return embedding
