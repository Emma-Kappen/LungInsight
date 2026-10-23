#!/usr/bin/env python3
"""LungInsight v2 spatial_v1: matched 3D spatial-attention models.

Variants:
  spatial16: tap 64 x 16 x 16 x 16 after encoder stage 2 (4096 tokens)
  stride8:   tap 128 x 8 x 8 x 8 after encoder stage 3 (512 tokens)
  stride4:   tap 192 x 4 x 4 x 4 after encoder stage 4 (64 tokens)

The downstream attention/readout and prediction heads are identical.
Input contract: raw HU patches [B,64,64,64] or [B,1,64,64,64].
"""
from __future__ import annotations
from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_TARGETS = ["subtlety","internal_structure","calcification","sphericity","margin","lobulation","spiculation","texture"]
RATING_MAX = {"subtlety":5,"internal_structure":4,"calcification":6,"sphericity":5,"margin":5,"lobulation":5,"spiculation":5,"texture":5}
ORDERED_TARGETS = {"subtlety","sphericity","margin","lobulation","spiculation","texture"}
READER_TO_MODEL = [7,1,0,5,4,2,6,8]
INPUT_SHAPE = (64,64,64)
HU_LOW, HU_HIGH, HU_SHIFT = -1000.0, 400.0, 1000.0
FOLD_MEAN, FOLD_STD = 357.87565, 288.47657

ENCODER_CFG = {"stem_channels":16,"channels":[32,64,128,192],"blocks":[1,2,2,1]}


def preprocess_hu(x_hu: torch.Tensor) -> torch.Tensor:
    x = x_hu.float().clamp(HU_LOW, HU_HIGH) + HU_SHIFT
    return (x - FOLD_MEAN) / FOLD_STD


def _gn(c: int, groups: int = 8) -> nn.GroupNorm:
    return nn.GroupNorm(min(groups, c), c)


class ResBlock3D(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.norm1 = _gn(cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.norm2 = _gn(cout)
        self.act = nn.GELU()
        self.short = None if (cin == cout and stride == 1) else nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False), _gn(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return self.act(y + (x if self.short is None else self.short(x)))


class Encoder3DStages(nn.Module):
    def __init__(self):
        super().__init__()
        stem = ENCODER_CFG["stem_channels"]
        chans, blocks = ENCODER_CFG["channels"], ENCODER_CFG["blocks"]
        self.stem = nn.Sequential(nn.Conv3d(1, stem, 3, 1, 1, bias=False), _gn(stem), nn.GELU())
        stages = []
        cin = stem
        for c, nb in zip(chans, blocks):
            layers = []
            for i in range(nb):
                layers.append(ResBlock3D(cin, c, 2 if i == 0 else 1))
                cin = c
            stages.append(nn.Sequential(*layers))
        self.stage1, self.stage2, self.stage3, self.stage4 = stages

    def forward_to(self, x: torch.Tensor, variant: str) -> torch.Tensor:
        y = self.stage1(self.stem(x))
        y = self.stage2(y)
        if variant == "spatial16":
            return y
        y = self.stage3(y)
        if variant == "stride8":
            return y
        if variant == "stride4":
            return self.stage4(y)
        raise ValueError("variant must be spatial16, stride8 or stride4")


class SpatialTaskAttention(nn.Module):
    def __init__(self, in_channels: int, attn_dim: int = 96, num_tasks: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.key = nn.Linear(in_channels, attn_dim, bias=False)
        self.value = nn.Linear(in_channels, attn_dim, bias=False)
        self.coord = nn.Linear(3, attn_dim, bias=False)
        self.query = nn.Parameter(torch.empty(num_tasks, attn_dim))
        nn.init.normal_(self.query, 0.0, 0.02)
        self.scale = attn_dim ** -0.5

    @staticmethod
    def _coords(shape, device, dtype):
        z,y,x = shape
        zz = torch.linspace(-1,1,z,device=device,dtype=dtype)
        yy = torch.linspace(-1,1,y,device=device,dtype=dtype)
        xx = torch.linspace(-1,1,x,device=device,dtype=dtype)
        return torch.stack(torch.meshgrid(zz,yy,xx,indexing="ij"),dim=-1).reshape(-1,3)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        b,c,z,y,x = feat.shape
        tokens = self.norm(feat.flatten(2).transpose(1,2))
        coords = self._coords((z,y,x), feat.device, feat.dtype)
        keys = self.key(tokens) + self.coord(coords)[None]
        values = self.value(tokens)
        q = self.query[None].expand(b,-1,-1)
        scores = torch.einsum("bad,btd->bat", q, keys) * self.scale
        weights = scores.softmax(-1)
        return torch.einsum("bat,btd->bad", weights, values)


class OrdinalHead(nn.Module):
    """Cumulative-link ordinal head: one latent score + ordered thresholds."""
    def __init__(self, in_dim: int, k: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.k = int(k)
        self.score = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        # Positive threshold gaps guarantee theta_1 < theta_2 < ... .
        self.theta0 = nn.Parameter(torch.tensor(0.0))
        self.gap_raw = nn.Parameter(torch.full((self.k - 2,), -1.0)) if self.k > 2 else None

    def thresholds(self):
        if self.k == 2:
            return self.theta0.view(1)
        gaps = F.softplus(self.gap_raw) + 0.05
        return torch.cat([self.theta0.view(1), self.theta0 + torch.cumsum(gaps, 0)])

    def forward(self, x):
        score = self.score(x)
        theta = self.thresholds().view(1, -1)
        # P(Y > k) = sigmoid(score - theta_k)
        return score - theta


class NominalHead(nn.Module):
    def __init__(self, in_dim: int, k: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, k))
    def forward(self, x): return self.net(x)


class SpatialNoduleRanker3D(nn.Module):
    def __init__(self, spatial_variant="stride8", attn_dim=96, head_hidden=64, dropout=0.2):
        super().__init__()
        if spatial_variant not in {"spatial16","stride8","stride4"}: raise ValueError("spatial_variant must be spatial16, stride8 or stride4")
        self.spatial_variant = spatial_variant
        self.attn_dim = attn_dim
        self.encoder = Encoder3DStages()
        channels = 64 if spatial_variant == "spatial16" else 128 if spatial_variant == "stride8" else 192
        self.attention = SpatialTaskAttention(channels, attn_dim, len(MODEL_TARGETS))
        self.heads = nn.ModuleList()
        for name in MODEL_TARGETS:
            if name in ORDERED_TARGETS:
                self.heads.append(OrdinalHead(attn_dim, RATING_MAX[name], head_hidden, dropout))
            else:
                self.heads.append(NominalHead(attn_dim, RATING_MAX[name], head_hidden, dropout))

    def feature_map(self,x):
        if x.ndim == 4: x = x.unsqueeze(1)
        if x.ndim != 5 or x.shape[1] != 1: raise ValueError(f"Expected [B,64,64,64] or [B,1,64,64,64], got {tuple(x.shape)}")
        return self.encoder.forward_to(x,self.spatial_variant)
    def forward_features(self,x): return self.attention(self.feature_map(x))
    def forward(self,x):
        z = self.forward_features(x)
        return tuple(h(z[:,i]) for i,h in enumerate(self.heads))

    @staticmethod
    def ordinal_probabilities(logits):
        q = torch.sigmoid(logits.float())
        b = q.shape[0]; k = q.shape[1]+1
        p = torch.empty((b,k),device=q.device,dtype=q.dtype)
        p[:,0] = 1-q[:,0]
        if k > 2: p[:,1:-1] = q[:,:-1]-q[:,1:]
        p[:,-1] = q[:,-1]
        return p.clamp_min(1e-7)

    def expected_rating(self, outputs):
        preds=[]
        for name, logits in zip(MODEL_TARGETS,outputs):
            if name in ORDERED_TARGETS:
                preds.append(1.0 + torch.sigmoid(logits.float()).sum(-1))
            else:
                p = torch.softmax(logits.float(),-1)
                values = torch.arange(1,RATING_MAX[name]+1,device=p.device,dtype=p.dtype)
                preds.append((p*values).sum(-1))
        return torch.stack(preds,1)

    def parameter_count(self): return sum(p.numel() for p in self.parameters())


def architecture_config(spatial_variant, attn_dim=96, head_hidden=64, dropout=0.2):
    return {"name":"SpatialNoduleRanker3D","version":"spatial_v1","spatial_variant":spatial_variant,"attention_dim":attn_dim,"head_hidden":head_hidden,"dropout":dropout,"encoder":ENCODER_CFG,"targets":MODEL_TARGETS,"rating_max":RATING_MAX,"input_shape":list(INPUT_SHAPE),"attention":"task-query spatial attention + 3D coordinate keys","preprocessing":{"clip_low_hu":HU_LOW,"clip_high_hu":HU_HIGH,"shift":HU_SHIFT,"mean":FOLD_MEAN,"std":FOLD_STD}}
