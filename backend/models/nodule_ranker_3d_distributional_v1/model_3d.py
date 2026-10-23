"""
LungInsight v2 -- distributional_v1 3D CNN for reader-aware LIDC attribute prediction.

This module intentionally preserves the existing Encoder3D architecture and replaces
the proportional-odds cumulative ordinal heads with eight independent categorical
distribution heads.  Each head predicts the empirical distribution of reader ratings.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


MODEL_IDENTITY = "distributional_v1"

MODEL_TARGETS = [
    "subtlety", "internal_structure", "calcification", "sphericity",
    "margin", "lobulation", "spiculation", "texture",
]

RATING_MAX = {
    "subtlety": 5,
    "internal_structure": 5,
    "calcification": 6,
    "sphericity": 5,
    "margin": 5,
    "lobulation": 5,
    "spiculation": 5,
    "texture": 5,
}

READER_NAMES = [
    "calcification", "internalStructure", "lobulation", "malignancy",
    "margin", "sphericity", "spiculation", "subtlety", "texture",
]
READER_TO_MODEL = [7, 1, 0, 5, 4, 2, 6, 8]

INPUT_SHAPE = (64, 64, 64)
HU_LOW, HU_HIGH, HU_SHIFT = -1000.0, 400.0, 1000.0
FOLD_MEAN, FOLD_STD = 357.87565, 288.47657
PREPROCESSING = {
    "source_units": "HU",
    "clip_low_hu": HU_LOW,
    "clip_high_hu": HU_HIGH,
    "shift_after_clip": HU_SHIFT,
    "mean": FOLD_MEAN,
    "std": FOLD_STD,
    "formula": "(clip(HU,-1000,400) + 1000 - 357.87565) / 288.47657",
}

assert RATING_MAX["calcification"] == 6
assert all(RATING_MAX[t] == 5 for t in MODEL_TARGETS if t != "calcification")
assert [READER_NAMES[i] if i != 1 else "internalStructure" for i in READER_TO_MODEL] == [
    "subtlety", "internalStructure", "calcification", "sphericity",
    "margin", "lobulation", "spiculation", "texture",
]


def preprocess_hu(x_hu: torch.Tensor) -> torch.Tensor:
    """Raw HU -> normalised network input, preserving tensor shape."""
    x = x_hu.float().clamp(HU_LOW, HU_HIGH) + HU_SHIFT
    return (x - FOLD_MEAN) / FOLD_STD


def _gn(c: int, groups: int = 8) -> nn.GroupNorm:
    return nn.GroupNorm(min(groups, c), c)


class ResBlock3D(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = _gn(cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = _gn(cout)
        self.act = nn.GELU()
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv3d(cin, cout, 1, stride, bias=False),
                _gn(cout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.n1(self.conv1(x)))
        out = self.n2(self.conv2(out))
        return self.act(out + (x if self.short is None else self.short(x)))


class Encoder3D(nn.Module):
    """The existing proven encoder; no architectural search is introduced."""

    def __init__(
        self,
        stem_channels: int = 16,
        channels: Sequence[int] = (32, 64, 128, 192),
        blocks: Sequence[int] = (1, 2, 2, 1),
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, stem_channels, 3, 1, 1, bias=False),
            _gn(stem_channels),
            nn.GELU(),
        )
        layers, cin = [], stem_channels
        for c, nb in zip(channels, blocks):
            for i in range(nb):
                layers.append(ResBlock3D(cin, c, 2 if i == 0 else 1))
                cin = c
        self.stages = nn.Sequential(*layers)
        self.out_channels = cin
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stages(self.stem(x))


class ReconstructionDecoder(nn.Module):
    """Legacy pre-training head retained so old encoder checkpoints remain loadable."""

    def __init__(self, cin: int, chans: Sequence[int] = (128, 64, 32, 16)):
        super().__init__()
        layers, c = [], cin
        for co in chans:
            layers += [
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(c, co, 3, 1, 1, bias=False),
                _gn(co),
                nn.GELU(),
            ]
            c = co
        layers.append(nn.Conv3d(c, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        return self.net(f)


class MaskedReconstructionModel(nn.Module):
    """Legacy wrapper used only when loading/continuing the existing encoder pretraining."""

    def __init__(self, encoder_cfg: dict):
        super().__init__()
        self.encoder = Encoder3D(**encoder_cfg)
        self.decoder = ReconstructionDecoder(self.encoder.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class DistributionHead(nn.Module):
    """Independent categorical rating-distribution head."""

    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_classes = int(n_classes)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class NoduleRanker3D(nn.Module):
    """
    Shared 3D encoder + shared embedding + eight independent distribution heads.

    forward() returns a tuple of eight logits tensors, each with its own K.
    expected_rating() derives E[Y] from the categorical probabilities.
    probabilities() returns a tuple of normalized rating distributions.
    """

    def __init__(
        self,
        targets: Sequence[str] = MODEL_TARGETS,
        rating_max: dict | None = None,
        stem_channels: int = 16,
        channels: Sequence[int] = (32, 64, 128, 192),
        blocks: Sequence[int] = (1, 2, 2, 1),
        embed_dim: int = 256,
        head_hidden: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        rating_max = rating_max or RATING_MAX
        self.targets = list(targets)
        if self.targets != MODEL_TARGETS:
            raise ValueError(f"targets must be exactly {MODEL_TARGETS}, got {self.targets}")
        if "malignancy" in self.targets:
            raise ValueError("malignancy is not a neural-network target")

        self.ks = [int(rating_max[t]) for t in self.targets]
        self.encoder = Encoder3D(stem_channels, channels, blocks)
        c = self.encoder.out_channels
        self.embed = nn.Sequential(
            nn.Linear(2 * c, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList(
            [
                DistributionHead(embed_dim, k, head_hidden, dropout)
                for k in self.ks
            ]
        )

    def embed_features(self, x: torch.Tensor) -> torch.Tensor:
        f = self.encoder(x)
        pooled = torch.cat(
            [f.mean(dim=(2, 3, 4)), f.amax(dim=(2, 3, 4))],
            dim=1,
        )
        return self.embed(pooled)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        z = self.embed_features(x)
        return tuple(head(z) for head in self.heads)

    def probabilities(
        self, logits: tuple[torch.Tensor, ...] | list[torch.Tensor]
    ) -> tuple[torch.Tensor, ...]:
        if len(logits) != len(self.ks):
            raise ValueError(f"expected {len(self.ks)} head outputs, got {len(logits)}")
        return tuple(torch.softmax(lg.float(), dim=-1) for lg in logits)

    def expected_rating(
        self, logits: tuple[torch.Tensor, ...] | list[torch.Tensor]
    ) -> torch.Tensor:
        probs = self.probabilities(logits)
        expected = []
        for p, k in zip(probs, self.ks):
            values = torch.arange(1, k + 1, device=p.device, dtype=p.dtype)
            expected.append((p * values).sum(dim=-1))
        return torch.stack(expected, dim=1)

    def expected_rating_from_probabilities(
        self, probs: tuple[torch.Tensor, ...] | list[torch.Tensor]
    ) -> torch.Tensor:
        expected = []
        for p, k in zip(probs, self.ks):
            if p.shape[-1] != k:
                raise ValueError(f"probability head has K={p.shape[-1]}, expected K={k}")
            values = torch.arange(1, k + 1, device=p.device, dtype=p.dtype)
            expected.append((p * values).sum(dim=-1))
        return torch.stack(expected, dim=1)


DEFAULT_ENCODER_CFG = {
    "stem_channels": 16,
    "channels": [32, 64, 128, 192],
    "blocks": [1, 2, 2, 1],
}


def architecture_config(
    dropout: float = 0.1,
    embed_dim: int = 256,
    head_hidden: int = 64,
) -> dict:
    return {
        "name": "NoduleRanker3D",
        "identity": MODEL_IDENTITY,
        "targets": list(MODEL_TARGETS),
        "rating_max": dict(RATING_MAX),
        "encoder": dict(DEFAULT_ENCODER_CFG),
        "embed_dim": int(embed_dim),
        "head_hidden": int(head_hidden),
        "dropout": float(dropout),
        "head_type": "independent_categorical_distribution",
        "loss": "mean_abs_cdf_emd_plus_small_mean_huber",
        "pooling": "global_avg+global_max",
        "norm": "GroupNorm(8)",
        "input_shape": list(INPUT_SHAPE),
    }


def build_model(cfg: dict) -> NoduleRanker3D:
    if cfg.get("identity") != MODEL_IDENTITY:
        raise ValueError(
            f"checkpoint architecture identity {cfg.get('identity')!r} is not {MODEL_IDENTITY!r}"
        )
    e = cfg["encoder"]
    return NoduleRanker3D(
        cfg["targets"],
        cfg["rating_max"],
        e["stem_channels"],
        e["channels"],
        e["blocks"],
        cfg["embed_dim"],
        cfg["head_hidden"],
        cfg["dropout"],
    )


def load_trained_model(path, device="cpu"):
    ck = torch.load(str(path), map_location=device, weights_only=False)
    if ck.get("model_identity") != MODEL_IDENTITY:
        raise ValueError(
            f"Not a {MODEL_IDENTITY} checkpoint: model_identity={ck.get('model_identity')!r}"
        )
    model = build_model(ck["architecture"])
    model.load_state_dict(ck["model_state_dict"], strict=True)
    return model.to(device).eval(), ck


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
