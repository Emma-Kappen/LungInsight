#!/usr/bin/env python3
"""LungInsight v2 -- matched stride-8 multi-objective 3D nodule model.

The shared image architecture is identical between objective modes. Only the
six ordered-attribute output/loss heads differ:
  * ordinal_reader: reader-level cumulative-link ordinal supervision
  * mean_regression: direct regression to the nodule reader mean

The two nominal attributes always use empirical reader distributions with
soft categorical cross-entropy.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAME = "LungInsightSpatialObjectiveV1"
MODEL_VERSION = "1.0"

MODEL_TARGETS = [
    "subtlety",
    "internal_structure",
    "calcification",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
]

ORDERED_TARGETS = [
    "subtlety",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
]
NOMINAL_TARGETS = ["internal_structure", "calcification"]

RATING_MAX = {
    "subtlety": 5,
    "internal_structure": 4,
    "calcification": 6,
    "sphericity": 5,
    "margin": 5,
    "lobulation": 5,
    "spiculation": 5,
    "texture": 5,
}

READER_NAMES = [
    "calcification",
    "internalStructure",
    "lobulation",
    "malignancy",
    "margin",
    "sphericity",
    "spiculation",
    "subtlety",
    "texture",
]
READER_TO_MODEL = [7, 1, 0, 5, 4, 2, 6, 8]

INPUT_SHAPE = (64, 64, 64)
HU_LOW, HU_HIGH, HU_SHIFT = -1000.0, 400.0, 1000.0
FOLD_MEAN, FOLD_STD = 357.87565, 288.47657

ENCODER_CFG = {
    "stem_channels": 16,
    "channels": [32, 64, 128],
    "blocks": [1, 2, 2],
}
SPATIAL_SHAPE = (8, 8, 8)
SPATIAL_CHANNELS = 128
NUM_TASKS = len(MODEL_TARGETS)
ATTN_DIM_DEFAULT = 96
HEAD_HIDDEN_DEFAULT = 64
DROPOUT_DEFAULT = 0.2


def preprocess_hu(x_hu: torch.Tensor) -> torch.Tensor:
    """Raw HU tensor -> exact LungInsight normalized input."""
    x = x_hu.float().clamp(HU_LOW, HU_HIGH) + HU_SHIFT
    return (x - FOLD_MEAN) / FOLD_STD


def _gn(channels: int, groups: int = 8) -> nn.GroupNorm:
    return nn.GroupNorm(min(groups, channels), channels)


class ResBlock3D(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.norm1 = _gn(cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.norm2 = _gn(cout)
        self.act = nn.GELU()
        self.shortcut = (
            None
            if cin == cout and stride == 1
            else nn.Sequential(
                nn.Conv3d(cin, cout, 1, stride, bias=False),
                _gn(cout),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        residual = x if self.shortcut is None else self.shortcut(x)
        return self.act(y + residual)


class Encoder3DStride8(nn.Module):
    """Existing 3D residual CNN through the 128x8x8x8 feature tap.

    The historical encoder has a later 192-channel 4^3 stage. It is not
    evaluated here because the experiment is explicitly a stride-8 tap.
    """

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        cfg = cfg or ENCODER_CFG
        stem_channels = int(cfg["stem_channels"])
        channels = [int(v) for v in cfg["channels"]]
        blocks = [int(v) for v in cfg["blocks"]]
        if channels != [32, 64, 128] or blocks != [1, 2, 2]:
            raise ValueError("This experiment requires encoder channels [32,64,128] and blocks [1,2,2].")

        self.stem = nn.Sequential(
            nn.Conv3d(1, stem_channels, 3, 1, 1, bias=False),
            _gn(stem_channels),
            nn.GELU(),
        )
        stages: list[nn.Module] = []
        cin = stem_channels
        for cout, nblocks in zip(channels, blocks):
            block_list: list[nn.Module] = []
            for i in range(nblocks):
                block_list.append(ResBlock3D(cin, cout, stride=2 if i == 0 else 1))
                cin = cout
            stages.append(nn.Sequential(*block_list))
        self.stage1, self.stage2, self.stage3 = stages
        self.out_channels = 128

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(self.stem(x))
        x = self.stage2(x)
        return self.stage3(x)


class SpatialTaskAttention(nn.Module):
    """Eight task-specific attention readouts over 8x8x8 spatial tokens."""

    def __init__(self, in_channels: int = SPATIAL_CHANNELS, attn_dim: int = ATTN_DIM_DEFAULT, num_tasks: int = NUM_TASKS):
        super().__init__()
        self.in_channels = int(in_channels)
        self.attn_dim = int(attn_dim)
        self.num_tasks = int(num_tasks)
        self.norm = nn.LayerNorm(in_channels)
        self.key = nn.Linear(in_channels, attn_dim, bias=False)
        self.value = nn.Linear(in_channels, attn_dim, bias=False)
        self.coord = nn.Linear(3, attn_dim, bias=False)
        self.query = nn.Parameter(torch.empty(num_tasks, attn_dim))
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        self.scale = attn_dim ** -0.5

    @staticmethod
    def coordinates(z: int, y: int, x: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        zz = torch.linspace(-1.0, 1.0, z, device=device, dtype=dtype)
        yy = torch.linspace(-1.0, 1.0, y, device=device, dtype=dtype)
        xx = torch.linspace(-1.0, 1.0, x, device=device, dtype=dtype)
        grid = torch.stack(torch.meshgrid(zz, yy, xx, indexing="ij"), dim=-1)
        return grid.reshape(-1, 3)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim != 5:
            raise ValueError(f"Expected [B,C,Z,Y,X], got {tuple(feat.shape)}")
        b, c, z, y, x = feat.shape
        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} feature channels, got {c}")

        tokens = self.norm(feat.flatten(2).transpose(1, 2))  # [B,T,C]
        coords = self.coordinates(z, y, x, feat.device, feat.dtype)
        keys = self.key(tokens) + self.coord(coords).unsqueeze(0)
        values = self.value(tokens)
        queries = self.query.unsqueeze(0).expand(b, -1, -1)
        scores = torch.einsum("bad,btd->bat", queries, keys) * self.scale
        weights = torch.softmax(scores.float(), dim=-1).to(values.dtype)
        return torch.einsum("bat,btd->bad", weights, values)


class OrdinalHead(nn.Module):
    """Cumulative-link ordinal head: latent score + strictly increasing thresholds."""

    def __init__(self, in_dim: int, k: int, hidden: int = HEAD_HIDDEN_DEFAULT, dropout: float = DROPOUT_DEFAULT):
        super().__init__()
        self.k = int(k)
        if self.k < 3:
            raise ValueError("This head expects K>=3 for the configured threshold parameterization.")
        self.score = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.theta0 = nn.Parameter(torch.tensor(0.0))
        self.gap_raw = nn.Parameter(torch.full((self.k - 2,), -1.0))

    def thresholds(self) -> torch.Tensor:
        gaps = F.softplus(self.gap_raw) + 0.05
        return torch.cat([self.theta0.view(1), self.theta0 + torch.cumsum(gaps, dim=0)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self.score(x)
        theta = self.thresholds().view(1, -1)
        return score - theta

    @staticmethod
    def probabilities(logits: torch.Tensor) -> torch.Tensor:
        q = torch.sigmoid(logits.float())  # q_k = P(Y > k)
        b, km1 = q.shape
        p = torch.empty(b, km1 + 1, device=q.device, dtype=q.dtype)
        p[:, 0] = 1.0 - q[:, 0]
        if km1 > 1:
            p[:, 1:-1] = q[:, :-1] - q[:, 1:]
        p[:, -1] = q[:, -1]
        return p.clamp_min(1e-7)

    def expected_rating(self, logits: torch.Tensor) -> torch.Tensor:
        return 1.0 + torch.sigmoid(logits.float()).sum(dim=-1)


class MeanRegressionHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = HEAD_HIDDEN_DEFAULT, dropout: float = DROPOUT_DEFAULT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class NominalHead(nn.Module):
    def __init__(self, in_dim: int, k: int, hidden: int = HEAD_HIDDEN_DEFAULT, dropout: float = DROPOUT_DEFAULT):
        super().__init__()
        self.k = int(k)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.k),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialObjectiveModel(nn.Module):
    """Shared stride-8 architecture with interchangeable ordered objectives."""

    def __init__(
        self,
        objective: str,
        attn_dim: int = ATTN_DIM_DEFAULT,
        head_hidden: int = HEAD_HIDDEN_DEFAULT,
        dropout: float = DROPOUT_DEFAULT,
    ):
        super().__init__()
        if objective not in {"ordinal_reader", "mean_regression"}:
            raise ValueError("objective must be 'ordinal_reader' or 'mean_regression'")
        self.objective = objective
        self.attn_dim = int(attn_dim)
        self.head_hidden = int(head_hidden)
        self.dropout = float(dropout)
        self.encoder = Encoder3DStride8()
        self.attention = SpatialTaskAttention(SPATIAL_CHANNELS, self.attn_dim, NUM_TASKS)

        self.heads = nn.ModuleList()
        for name in MODEL_TARGETS:
            k = RATING_MAX[name]
            if name in ORDERED_TARGETS:
                if objective == "ordinal_reader":
                    self.heads.append(OrdinalHead(self.attn_dim, k, head_hidden, dropout))
                else:
                    self.heads.append(MeanRegressionHead(self.attn_dim, head_hidden, dropout))
            else:
                self.heads.append(NominalHead(self.attn_dim, k, head_hidden, dropout))

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            x = x.unsqueeze(1)
        if x.ndim != 5 or x.shape[1] != 1 or tuple(x.shape[2:]) != INPUT_SHAPE:
            raise ValueError(
                f"Expected [B,64,64,64] or [B,1,64,64,64], got {tuple(x.shape)}"
            )
        return self.encoder(x)

    def task_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(self.feature_map(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        z = self.task_features(x)
        return tuple(head(z[:, i]) for i, head in enumerate(self.heads))

    def ordered_probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return OrdinalHead.probabilities(logits)

    def expected_ratings(self, outputs: Sequence[torch.Tensor]) -> torch.Tensor:
        values: list[torch.Tensor] = []
        for name, out in zip(MODEL_TARGETS, outputs):
            if name in ORDERED_TARGETS:
                if self.objective == "ordinal_reader":
                    values.append(1.0 + torch.sigmoid(out.float()).sum(dim=-1))
                else:
                    values.append(1.0 + (RATING_MAX[name] - 1) * torch.sigmoid(out.float()))
            else:
                p = torch.softmax(out.float(), dim=-1)
                rating_values = torch.arange(1, RATING_MAX[name] + 1, device=p.device, dtype=p.dtype)
                values.append((p * rating_values).sum(dim=-1))
        return torch.stack(values, dim=1)

    def nominal_probabilities(self, output: torch.Tensor) -> torch.Tensor:
        return torch.softmax(output.float(), dim=-1)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def architecture_config(
    objective: str,
    attn_dim: int = ATTN_DIM_DEFAULT,
    head_hidden: int = HEAD_HIDDEN_DEFAULT,
    dropout: float = DROPOUT_DEFAULT,
) -> dict:
    return {
        "name": MODEL_NAME,
        "version": MODEL_VERSION,
        "objective": objective,
        "spatial_variant": "stride8",
        "encoder": dict(ENCODER_CFG),
        "spatial_output": [SPATIAL_CHANNELS, *SPATIAL_SHAPE],
        "spatial_tokens": int(torch.tensor(SPATIAL_SHAPE).prod().item()),
        "attention_dim": int(attn_dim),
        "head_hidden": int(head_hidden),
        "dropout": float(dropout),
        "attention": "task-query spatial attention + normalized 3D coordinate keys",
        "targets": list(MODEL_TARGETS),
        "ordered_targets": list(ORDERED_TARGETS),
        "nominal_targets": list(NOMINAL_TARGETS),
        "rating_max": dict(RATING_MAX),
        "input_shape": list(INPUT_SHAPE),
        "preprocessing": {
            "clip_low_hu": HU_LOW,
            "clip_high_hu": HU_HIGH,
            "shift": HU_SHIFT,
            "mean": FOLD_MEAN,
            "std": FOLD_STD,
            "formula": "(clip(HU,-1000,400)+1000-357.87565)/288.47657",
        },
    }
