"""
vitdet3d.py

3D CNN + Vision-Transformer nodule DETECTOR, ported from
rlsn/LungNoduleDetection (https://github.com/rlsn/LungNoduleDetection,
MIT licensed; pretrained checkpoint at
https://huggingface.co/rlsn/DeTr4LungNodule).

This is an independent, dependency-light re-implementation of that
repo's model.py (class VitDet3D). The upstream model.py builds its
transformer encoder from `transformers.models.vit.modeling_vit`
(ViTEncoder / ViTPooler), but those are treated as PRIVATE internals by
HuggingFace and have already been renamed/removed between transformers
versions (there is no more standalone `ViTEncoder` export in the
transformers version available in this environment). To load the
upstream checkpoint reliably regardless of transformers version, this
file reimplements the standard pre-LayerNorm ViT encoder block directly
in plain PyTorch -- functionally identical to HF's ViTLayer/ViTEncoder/
ViTPooler (same pre-norm attention -> pre-norm MLP -> residual
structure, same GELU, same CLS-token pooler), and it produces the exact
same set of named parameters (`encoder.layer.{i}.attention...`,
`pooler.dense...`, etc.) so the original state_dict still loads with
`load_state_dict(...)`.

Architecture (matches rlsn's model_config.json exactly):
    - CNNFeatureExtractor: a small 3D ResNet stem+3 stages (stride 2,
      2, 2) that turns a [1, 40, 128, 128] patch into a
      [256, 3, 8, 8] feature volume.
    - PosEmbedding: flattens the feature volume into a sequence of
      3*8*8 = 192 tokens, linearly projects each to hidden_size,
      prepends a learned [CLS] token, and adds learned absolute
      position embeddings.
    - A 6-layer, 8-head, hidden_size-256 ViT encoder processes the
      token sequence.
    - Two small 3-layer MLP heads read off the pooled [CLS] token:
        classification_head -> 1 logit (nodule vs. not, in THIS patch)
        bbox_head           -> 6 values (z_low, y_low, x_low,
                                z_high, y_high, x_high), each
                                normalized to [0, 1] as a FRACTION of
                                the patch size (crop_size), not
                                absolute voxels.

This is a whole-patch, single-object detector (does the patch contain
a nodule, and if so roughly where), evaluated with a sliding window
over the full volume in 04_detect_candidates.py -- it is NOT a
region-proposal / per-anchor detector like Faster-RCNN or YOLO.

IMPORTANT -- normalization statistics: the upstream checkpoint was
trained on HU volumes normalized with LUNA16_Dataset.mean/std from
rlsn's dataset.py (computed over the LUNA16 training set):
    mean = -775.657161489884
    std  = 962.3208802005623
04_detect_candidates.py applies these exact constants before running
the detector. Do NOT substitute your own dataset statistics here --
doing so would silently shift every input away from what the
pretrained weights expect.
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# LUNA16 training-set normalization stats the upstream checkpoint expects.
# Source: rlsn/LungNoduleDetection/dataset.py, LUNA16_Dataset.mean/std.
LUNA16_MEAN = -775.657161489884
LUNA16_STD = 962.3208802005623

# Default model_config.json values from rlsn/LungNoduleDetection.
DEFAULT_CROP_SIZE = (40, 128, 128)   # (Z, Y, X) voxels per detector patch


@dataclass
class VitDet3DConfig:
    """Plain-dataclass stand-in for transformers.ViTConfig -- holds the
    exact fields rlsn/LungNoduleDetection's model_config.json sets, no
    transformers dependency required.
    """
    hidden_size: int = 256
    image_size: List[int] = field(default_factory=lambda: [40, 128, 128])
    patch_size: List[int] = field(default_factory=lambda: [4, 16, 16])
    num_labels: int = 1
    num_channels: int = 1
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 1024
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    qkv_bias: bool = True


# ----------------------------------------------------------------------
# CNN stem (identical to model.py's ResBlock / CNNFeatureExtractor)
# ----------------------------------------------------------------------

class ResBlock3D(nn.Module):
    """Basic (non-bottleneck) 3D residual block: two 3x3x3 convs, stride
    applied on the first conv only. Matches model.py's ResBlock.
    """

    def __init__(self, in_channels: int, out_channels: int, stride,
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=[3, 3, 3],
                                stride=stride, padding=1, bias=False)
        self.downsample = downsample
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=[3, 3, 3],
                                padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class CNNFeatureExtractor(nn.Module):
    """3D ResNet stem + 3 downsampling stages: [1,40,128,128] -> [256,3,8,8].

    Matches model.py's CNNFeatureExtractor exactly: stem stride-2 conv +
    stride-2 maxpool, then 3 stages of 2 ResBlocks each at strides
    1, 2, 2.
    """

    def __init__(self, config: VitDet3DConfig):
        super().__init__()
        self.in_channels = 64
        self.out_size = [3, 8, 8]  # CNN output [D,H,W] for the default 40x128x128 input

        self.conv1 = nn.Conv3d(config.num_channels, self.in_channels,
                                kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)

    def _make_layer(self, num_channels: int, num_layers: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, num_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(num_channels),
            )
        layers = [ResBlock3D(self.in_channels, num_channels, stride, downsample)]
        self.in_channels = num_channels
        for _ in range(1, num_layers):
            layers.append(ResBlock3D(self.in_channels, num_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class PosEmbedding(nn.Module):
    """Flatten CNN feature volume -> token sequence, prepend [CLS],
    add learned absolute position embeddings. Matches model.py exactly.
    """

    def __init__(self, config: VitDet3DConfig, in_channels: int, in_size):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.seq_len = int(np.prod(in_size))
        self.projection = nn.Linear(in_channels, config.hidden_size)
        self.position_embeddings = nn.Parameter(torch.randn(1, self.seq_len + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, c, d, h, w = x.shape
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = x.flatten(2).transpose(1, 2)  # (B, D*H*W, C)
        x = self.projection(x)
        embeddings = torch.cat((cls_tokens, x), dim=1)
        embeddings = embeddings + self.position_embeddings
        return self.dropout(embeddings)


# ----------------------------------------------------------------------
# Self-contained ViT encoder (functionally == HF ViTEncoder/ViTLayer,
# reimplemented so this file has no dependency on transformers-internal
# classes that get renamed/removed across versions). Parameter names
# below (attention.attention.{query,key,value}, attention.output.dense,
# intermediate.dense, output.dense, layernorm_before/after) intentionally
# mirror HF's ViTLayer naming so an upstream `encoder.layer.*` state_dict
# loads into this module without any key remapping.
# ----------------------------------------------------------------------

class ViTSelfAttention(nn.Module):
    def __init__(self, config: VitDet3DConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        x = x.view(b, n, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        q = self._split_heads(self.query(hidden_states))
        k = self._split_heads(self.key(hidden_states))
        v = self._split_heads(self.value(hidden_states))

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn_probs = self.dropout(F.softmax(attn_scores, dim=-1))
        context = torch.matmul(attn_probs, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        b, n = context.shape[0], context.shape[1]
        return context.view(b, n, self.all_head_size)


class ViTSelfOutput(nn.Module):
    def __init__(self, config: VitDet3DConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.dense(hidden_states))


class ViTAttention(nn.Module):
    def __init__(self, config: VitDet3DConfig):
        super().__init__()
        self.attention = ViTSelfAttention(config)
        self.output = ViTSelfOutput(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.output(self.attention(hidden_states))


class ViTIntermediate(nn.Module):
    def __init__(self, config: VitDet3DConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.act(self.dense(hidden_states))


class ViTOutput(nn.Module):
    def __init__(self, config: VitDet3DConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.dense(hidden_states)) + input_tensor


class ViTLayer(nn.Module):
    """Pre-LayerNorm transformer block: LN -> MHSA -> residual,
    LN -> MLP(GELU) -> residual. Matches HF's ViTLayer structure.
    """

    def __init__(self, config: VitDet3DConfig):
        super().__init__()
        self.attention = ViTAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention(self.layernorm_before(hidden_states))
        hidden_states = attn_out + hidden_states
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        return self.output(layer_output, hidden_states)


class ViTEncoder(nn.Module):
    def __init__(self, config: VitDet3DConfig):
        super().__init__()
        self.layer = nn.ModuleList([ViTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)
        return hidden_states


class ViTPooler(nn.Module):
    """Linear + Tanh on the [CLS] token (index 0). Matches HF ViTPooler."""

    def __init__(self, config: VitDet3DConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.activation(self.dense(hidden_states[:, 0]))


class MLP(nn.Module):
    """(num_layers-1) hidden Linear+ReLU blocks at `in_dim`, then a
    final Linear to `out_dim`. Matches model.py's MLP exactly.
    """

    def __init__(self, in_dim: int, out_dim: int, num_layers: int):
        super().__init__()
        layers: List[nn.Module] = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, in_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class VitDet3DOutput:
    """Lightweight attribute-access output container (no transformers
    ModelOutput dependency)."""

    def __init__(self, logits, bbox, last_hidden_state, pooler_output, loss=None):
        self.logits = logits
        self.bbox = bbox
        self.last_hidden_state = last_hidden_state
        self.pooler_output = pooler_output
        self.loss = loss


class VitDet3D(nn.Module):
    """3D CNN+ViT single-object detector: per-patch nodule confidence
    logit + normalized 6-value bounding box.

    forward(pixel_values) returns a VitDet3DOutput with:
        logits: (N, 1) raw nodule-presence logit (apply sigmoid once
                 for probability, matching BCEWithLogitsLoss training)
        bbox:   (N, 6) predicted box as [z_lo, y_lo, x_lo, z_hi, y_hi,
                 x_hi], each a FRACTION of crop_size in [0, 1]
                 (multiply by crop_size, elementwise, to get voxel
                 offsets within the input patch).
    """

    def __init__(self, config: VitDet3DConfig, add_pooling_layer: bool = True):
        super().__init__()
        self.config = config
        self.cnn = CNNFeatureExtractor(config)
        self.embeddings = PosEmbedding(config, self.cnn.in_channels, self.cnn.out_size)
        self.encoder = ViTEncoder(config)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None
        self.classification_head = MLP(config.hidden_size, config.num_labels, 3)
        self.bbox_head = MLP(config.hidden_size, 6, 3)

    def forward(self, pixel_values: torch.Tensor, labels=None, bbox=None) -> VitDet3DOutput:
        feature_maps = self.cnn(pixel_values)
        embeddings = self.embeddings(feature_maps)
        sequence_output = self.layernorm(self.encoder(embeddings))
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        logits = self.classification_head(pooled_output)
        bbox_pred = self.bbox_head(pooled_output)

        loss = None
        if labels is not None and bbox is not None:
            from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
            loss_bbox_fn = MSELoss(reduction='none')
            if self.config.num_labels == 1:
                loss = BCEWithLogitsLoss()(logits.view(-1), labels.float())
            else:
                loss = CrossEntropyLoss()(logits, labels)
            mask = labels.unsqueeze(-1).bool()
            loss = loss + (loss_bbox_fn(bbox_pred, bbox) * mask).mean()

        return VitDet3DOutput(logits, bbox_pred, sequence_output, pooled_output, loss)


def build_vitdet3d(config_overrides: Optional[dict] = None) -> VitDet3D:
    """Construct a VitDet3D with rlsn/LungNoduleDetection's default
    model_config.json hyperparameters (optionally overridden).
    """
    config = VitDet3DConfig()
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(config, k, v)
    return VitDet3D(config)


def load_vitdet3d_checkpoint(checkpoint_path: str, device: torch.device,
                              config_overrides: Optional[dict] = None) -> VitDet3D:
    """Build the detector and load a checkpoint's state_dict (a raw
    .pt/.pth/.bin file, or a HuggingFace safetensors file).

    If the checkpoint was saved from the ORIGINAL rlsn/LungNoduleDetection
    (which builds encoder/pooler from transformers' ViTEncoder/ViTPooler),
    the parameter names line up 1:1 with this reimplementation, so
    load_state_dict should succeed without remapping.
    """
    model = build_vitdet3d(config_overrides)
    if checkpoint_path.endswith('.safetensors'):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path, device=str(device))
    else:
        state_dict = torch.load(checkpoint_path, map_location=device)
        if isinstance(state_dict, dict):
            for key in ('model_state_dict', 'state_dict', 'model'):
                if isinstance(state_dict.get(key), dict):
                    state_dict = state_dict[key]
                    break
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[warn] detector checkpoint load: {len(missing)} missing, "
              f"{len(unexpected)} unexpected keys "
              f"(missing e.g. {missing[:3]}, unexpected e.g. {unexpected[:3]})")
    model.to(device)
    return model


if __name__ == '__main__':
    m = build_vitdet3d()
    x = torch.randn(2, 1, *DEFAULT_CROP_SIZE)
    out = m(x)
    print('logits', out.logits.shape)
    print('bbox', out.bbox.shape)
    n_params = sum(p.numel() for p in m.parameters())
    print(f'params: {n_params / 1e6:.2f}M')
