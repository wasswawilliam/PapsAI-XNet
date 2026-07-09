"""
===========================================================
PapsAI XNet ANFIS Module
===========================================================

Implements a first-order Sugeno Adaptive Neuro-Fuzzy Inference
System for seven-class cervical cytology classification.

This module receives the 32-dimensional normalized CNN feature
embedding and performs interpretable fuzzy reasoning using:

- Gaussian membership functions
- Compact 16-rule fuzzy rule base
- Product T-norm rule firing
- Normalized firing strengths
- First-order Sugeno consequents
- Weighted rule aggregation
===========================================================
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianMembership(nn.Module):
    """
    Gaussian membership function.

    mu(z) = exp(-((z - c)^2) / (2 * sigma^2))

    centers and sigmas are learnable.
    """

    def __init__(
        self,
        num_rules: int,
        input_dim: int,
        init_centers: Optional[torch.Tensor] = None,
        init_sigmas: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.num_rules = num_rules
        self.input_dim = input_dim
        self.eps = eps

        if init_centers is None:
            init_centers = torch.randn(num_rules, input_dim) * 0.1

        if init_sigmas is None:
            init_sigmas = torch.ones(num_rules, input_dim)

        if init_centers.shape != (num_rules, input_dim):
            raise ValueError(
                f"init_centers must have shape {(num_rules, input_dim)}, "
                f"got {tuple(init_centers.shape)}"
            )

        if init_sigmas.shape != (num_rules, input_dim):
            raise ValueError(
                f"init_sigmas must have shape {(num_rules, input_dim)}, "
                f"got {tuple(init_sigmas.shape)}"
            )

        self.centers = nn.Parameter(init_centers.float())
        self.log_sigmas = nn.Parameter(torch.log(init_sigmas.float().clamp_min(eps)))

    @property
    def sigmas(self) -> torch.Tensor:
        return torch.exp(self.log_sigmas).clamp_min(self.eps)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Tensor of shape [batch_size, input_dim]

        Returns:
            membership degrees of shape [batch_size, num_rules, input_dim]
        """

        if z.dim() != 2:
            raise ValueError("Input z must have shape [batch_size, input_dim].")

        if z.size(1) != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, got {z.size(1)}."
            )

        z_expanded = z.unsqueeze(1)
        centers = self.centers.unsqueeze(0)
        sigmas = self.sigmas.unsqueeze(0)

        memberships = torch.exp(
            -((z_expanded - centers) ** 2) / (2.0 * sigmas**2)
        )

        return memberships


class SugenoANFIS(nn.Module):
    """
    First-order Sugeno ANFIS classifier.

    For rule r:

        f_r(z) = a_r0 + a_r1*z1 + ... + a_rD*zD

    Final class score:

        y = sum_r normalized_w_r * f_r(z)
    """

    def __init__(
        self,
        input_dim: int = 32,
        num_rules: int = 16,
        num_classes: int = 7,
        init_centers: Optional[torch.Tensor] = None,
        init_sigmas: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_rules = num_rules
        self.num_classes = num_classes
        self.eps = eps

        self.membership = GaussianMembership(
            num_rules=num_rules,
            input_dim=input_dim,
            init_centers=init_centers,
            init_sigmas=init_sigmas,
            eps=eps,
        )

        # Consequent parameters:
        # one linear Sugeno function per rule and per class.
        # Shape: [num_rules, num_classes, input_dim + 1]
        self.consequents = nn.Parameter(
            torch.randn(num_rules, num_classes, input_dim + 1) * 0.01
        )

    def compute_rule_strengths(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute raw and normalized rule firing strengths.

        Args:
            z: Tensor of shape [batch_size, input_dim]

        Returns:
            raw_strengths: [batch_size, num_rules]
            normalized_strengths: [batch_size, num_rules]
        """

        memberships = self.membership(z)

        raw_strengths = torch.prod(memberships, dim=2)

        normalized_strengths = raw_strengths / (
            torch.sum(raw_strengths, dim=1, keepdim=True) + self.eps
        )

        return raw_strengths, normalized_strengths

    def compute_rule_outputs(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute first-order Sugeno consequent outputs.

        Args:
            z: Tensor of shape [batch_size, input_dim]

        Returns:
            rule_outputs: Tensor of shape [batch_size, num_rules, num_classes]
        """

        batch_size = z.size(0)

        ones = torch.ones(batch_size, 1, device=z.device, dtype=z.dtype)
        z_augmented = torch.cat([ones, z], dim=1)

        rule_outputs = torch.einsum(
            "bd,rcd->brc",
            z_augmented,
            self.consequents,
        )

        return rule_outputs

    def forward(
        self,
        z: torch.Tensor,
        return_interpretability: bool = False,
    ):
        """
        Args:
            z: 32-dimensional normalized CNN embedding.
            return_interpretability: if True, returns fuzzy reasoning details.

        Returns:
            logits or dictionary containing logits and interpretability outputs.
        """

        raw_strengths, normalized_strengths = self.compute_rule_strengths(z)
        rule_outputs = self.compute_rule_outputs(z)

        class_scores = torch.sum(
            normalized_strengths.unsqueeze(-1) * rule_outputs,
            dim=1,
        )

        if return_interpretability:
            return {
                "logits": class_scores,
                "raw_rule_strengths": raw_strengths,
                "normalized_rule_strengths": normalized_strengths,
                "rule_outputs": rule_outputs,
                "membership_centers": self.membership.centers,
                "membership_sigmas": self.membership.sigmas,
            }

        return class_scores

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.forward(z)
        return torch.argmax(logits, dim=1)

    def predict_proba(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.forward(z)
        return F.softmax(logits, dim=1)
