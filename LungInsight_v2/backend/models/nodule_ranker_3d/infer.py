#!/usr/bin/env python3
"""
Inference for the LungInsight 8-attribute LIDC nodule model.

Input:
    A single .npy nodule patch, normally shaped [64, 64, 64],
    or a directory containing .npy patches.

Output:
    Eight LIDC-IDRI radiological characteristics:
      subtlety
      internalStructure
      calcification
      sphericity
      margin
      lobulation
      spiculation
      texture

Malignancy is intentionally NOT predicted by this neural network.

The preprocessing is matched to train_h5.py:
  - 64^3 input
  - 16 evenly spaced axial slices
  - if values look like HU: clip [-1000, 400] and map to [0, 1]
  - otherwise leave already-normalized values unchanged
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


ATTRIBUTE_NAMES = [
    "subtlety",
    "internalStructure",
    "calcification",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
]

ATTRIBUTE_RANGES = {
    "subtlety": (1.0, 5.0),
    "internalStructure": (1.0, 4.0),
    "calcification": (1.0, 6.0),
    "sphericity": (1.0, 5.0),
    "margin": (1.0, 5.0),
    "lobulation": (1.0, 5.0),
    "spiculation": (1.0, 5.0),
    "texture": (1.0, 5.0),
}


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class OrdinalNetwork(nn.Module):
    """
    Architecture corresponding to train_h5.py.

    Sixteen axial slices are encoded by a shared 2-D CNN.
    Slice features are attention-pooled and combined with one
    learned embedding per attribute. Each attribute has its own
    cumulative ordinal head.
    """

    def __init__(self, base=24):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock(1, base),
            ConvBlock(base, base * 2),
            ConvBlock(base * 2, base * 4),
            ConvBlock(base * 4, base * 4),
        )

        feature_dim = base * 4

        self.slice_attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1),
        )

        self.attribute_embedding = nn.Parameter(
            torch.randn(len(ATTRIBUTE_NAMES), feature_dim) * 0.02
        )

        self.heads = nn.ModuleList()
        for name in ATTRIBUTE_NAMES:
            lo, hi = ATTRIBUTE_RANGES[name]
            n_thresholds = int(hi - lo)
            self.heads.append(
                nn.Sequential(
                    nn.Linear(feature_dim, feature_dim // 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.10),
                    nn.Linear(feature_dim // 2, n_thresholds),
                )
            )

    def forward(self, x):
        # x: [B, 16, H, W]
        b, s, h, w = x.shape

        z = x.reshape(b * s, 1, h, w)
        z = self.encoder(z)
        z = torch.nn.functional.adaptive_avg_pool2d(z, 1)
        z = z.flatten(1)
        z = z.reshape(b, s, -1)

        attention_logits = self.slice_attention(z).squeeze(-1)
        attention = torch.softmax(attention_logits, dim=1)
        pooled = (z * attention.unsqueeze(-1)).sum(dim=1)

        outputs = []
        for i, head in enumerate(self.heads):
            attr_feature = pooled + self.attribute_embedding[i]
            outputs.append(head(attr_feature))

        return outputs


def clean_state_dict(state):
    """Handle common checkpoint wrappers/prefixes."""
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]

    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value
    return cleaned


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    config = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}
    base = int(config.get("base", 24))

    model = OrdinalNetwork(base=base)

    state = clean_state_dict(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint does not match the expected 8-attribute architecture.\n"
            f"Missing keys: {missing}\n"
            f"Unexpected keys: {unexpected}"
        )

    model.to(device)
    model.eval()
    return model, checkpoint


def normalize_patch(patch):
    patch = np.asarray(patch, dtype=np.float32)

    finite = np.isfinite(patch)
    if not finite.all():
        if not finite.any():
            raise ValueError("The .npy patch contains no finite values.")
        replacement = float(np.nanmedian(patch))
        patch = np.where(finite, patch, replacement).astype(np.float32)

    # Match train_h5.py automatic preprocessing.
    pmin = float(patch.min())
    pmax = float(patch.max())

    if pmin < -2.0 or pmax > 2.0:
        patch = np.clip(patch, -1000.0, 400.0)
        patch = (patch + 1000.0) / 1400.0

    return patch.astype(np.float32)


def prepare_patch(path):
    patch = np.load(path)

    # Remove harmless singleton dimensions, e.g. [1,64,64,64].
    patch = np.squeeze(patch)

    if patch.ndim != 3:
        raise ValueError(
            f"{path}: expected a 3-D nodule patch, got shape {patch.shape}"
        )

    if patch.shape != (64, 64, 64):
        raise ValueError(
            f"{path}: expected shape (64, 64, 64), got {patch.shape}. "
            "Resize/re-extract the patch to 64^3 before inference."
        )

    patch = normalize_patch(patch)

    # ZYX convention. Use the same 16 evenly spaced axial slices
    # as the HDF5 training pipeline.
    indices = np.linspace(0, 63, 16).round().astype(np.int64)
    slices = patch[indices]

    # [16,H,W] -> [1,16,H,W]
    tensor = torch.from_numpy(slices).unsqueeze(0)
    return tensor


@torch.no_grad()
def predict(model, tensor, device):
    tensor = tensor.to(device, non_blocking=True)

    logits = model(tensor)

    result = {}

    for name, logit in zip(ATTRIBUTE_NAMES, logits):
        lo, hi = ATTRIBUTE_RANGES[name]

        # Cumulative ordinal decoding.
        probability = torch.sigmoid(logit)
        score = lo + probability.sum(dim=1)

        # Keep the continuous score and provide a conventional
        # nearest-integer LIDC-style presentation.
        score_value = float(score.item())
        rounded = int(np.clip(np.rint(score_value), lo, hi))

        result[name] = {
            "score": score_value,
            "rounded": rounded,
            "range": [int(lo), int(hi)],
        }

    return result


def infer_one(model, path, device):
    tensor = prepare_patch(path)
    prediction = predict(model, tensor, device)

    return {
        "file": str(path),
        "shape": [64, 64, 64],
        "attributes": prediction,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run the trained LungInsight 8-attribute model on .npy nodule patches."
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to best.pth produced by train_h5.py",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--npy",
        help="Single .npy nodule patch",
    )
    group.add_argument(
        "--npy-dir",
        help="Directory containing .npy nodule patches",
    )
    parser.add_argument(
        "--output",
        help="Output JSON/CSV file for batch inference",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cuda, cuda:0, or cpu. Defaults to CUDA when available.",
    )
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    checkpoint_path = Path(args.checkpoint)

    if args.npy:
        paths = [Path(args.npy)]
    else:
        root = Path(args.npy_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"Not a directory: {root}")
        paths = sorted(root.glob("*.npy"))

        if not paths:
            raise FileNotFoundError(f"No .npy files found in {root}")

    model, checkpoint = load_checkpoint(checkpoint_path, device)

    results = []
    for path in paths:
        results.append(infer_one(model, path, device))

    # Single-patch: print clean JSON directly.
    if args.npy:
        payload = results[0]
        print(json.dumps(payload, indent=2))
    else:
        payload = {"count": len(results), "results": results}
        print(json.dumps(payload, indent=2))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)

        if output.suffix.lower() == ".csv":
            with output.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                header = ["file"]
                for name in ATTRIBUTE_NAMES:
                    header += [f"{name}_score", f"{name}_rounded"]
                writer.writerow(header)

                for item in results:
                    row = [item["file"]]
                    for name in ATTRIBUTE_NAMES:
                        row += [
                            item["attributes"][name]["score"],
                            item["attributes"][name]["rounded"],
                        ]
                    writer.writerow(row)
        else:
            with output.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
