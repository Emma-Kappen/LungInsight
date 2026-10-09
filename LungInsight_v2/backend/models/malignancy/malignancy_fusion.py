"""Ordinal fusion head for radiologist malignancy suspicion ratings."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalHead(nn.Module):
    """Cumulative ordinal head with ordered thresholds for ratings 1 through 5."""

    def __init__(self, in_features: int):
        super().__init__()
        self.score = nn.Linear(in_features, 1)
        self.first_threshold = nn.Parameter(torch.tensor(0.0))
        self.threshold_gaps = nn.Parameter(torch.full((3,), -1.0))

    def thresholds(self) -> torch.Tensor:
        gaps = F.softplus(self.threshold_gaps) + 1e-4
        return torch.cat(
            (self.first_threshold.view(1), self.first_threshold + torch.cumsum(gaps, dim=0))
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.score(features) - self.thresholds().unsqueeze(0)

    @staticmethod
    def probabilities(logits: torch.Tensor) -> torch.Tensor:
        """Return [P(1), ..., P(5)] from cumulative logits [P(>=2), ..., P(>=5)]."""
        cumulative = torch.sigmoid(logits.float())
        probabilities = torch.cat(
            (
                1.0 - cumulative[:, :1],
                cumulative[:, :-1] - cumulative[:, 1:],
                cumulative[:, -1:],
            ),
            dim=1,
        )
        return probabilities.clamp_min(0.0)


class MalignancyFusion(nn.Module):
    """Map attribute scores and optional uncertainties to an ordinal rating."""

    def __init__(self, input_dim: int = 12, dropout: float = 0.2):
        super().__init__()
        if input_dim not in (6, 12):
            raise ValueError("input_dim must be 6 (scores) or 12 (scores and uncertainties)")
        self.input_dim = int(input_dim)
        self.dropout = float(dropout)
        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(32, 16),
            nn.GELU(),
        )
        self.ordinal = OrdinalHead(16)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(f"Expected [batch,{self.input_dim}] features, got {tuple(features.shape)}")
        return self.ordinal(self.backbone(features))

    def predict(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        cumulative_logits = self(features)
        distribution = self.ordinal.probabilities(cumulative_logits)
        ratings = torch.arange(1, 6, device=distribution.device, dtype=distribution.dtype)
        return {
            "class_probabilities": distribution,
            "malignancy_expected": (distribution * ratings).sum(dim=-1),
            "probability_malignant_ge4": torch.sigmoid(cumulative_logits[:, 2]),
            "probability_malignant_ge5": torch.sigmoid(cumulative_logits[:, 3]),
        }


def cumulative_targets(ratings: torch.Tensor) -> torch.Tensor:
    """Convert rating values in [1,5] to four binary P(M >= threshold) targets."""
    thresholds = torch.arange(2, 6, device=ratings.device, dtype=ratings.dtype)
    return (ratings.unsqueeze(-1) >= thresholds).to(torch.float32)


def ordinal_reader_loss(logits: torch.Tensor, ratings: torch.Tensor) -> torch.Tensor:
    targets = cumulative_targets(ratings.float())
    return F.binary_cross_entropy_with_logits(logits.float(), targets)