#!/usr/bin/env python3
"""Run the five frozen hybrid checkpoints on one raw-HU NPY patch."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model_objective import MODEL_TARGETS, SpatialObjectiveModel, preprocess_hu


CHECKPOINT_GROUPS = {
    "lobulation_spiculation": ("lobulation", "spiculation"),
    "subtlety": ("subtlety",),
    "sphericity": ("sphericity",),
    "margin": ("margin",),
    "texture": ("texture",),
}
OUTPUT_ATTRIBUTES = ("subtlety", "sphericity", "margin", "lobulation", "spiculation", "texture")


def choose_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def load_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    model = SpatialObjectiveModel(
        architecture["objective"],
        architecture["attention_dim"],
        architecture["head_hidden"],
        architecture["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval(), architecture


def main(args: argparse.Namespace) -> int:
    device = choose_device(args.device)
    patch = np.load(Path(args.input_npy).resolve(), allow_pickle=False)
    if patch.ndim == 5 and patch.shape[:2] == (1, 1):
        patch = patch[0, 0]
    elif patch.ndim == 4 and patch.shape[0] == 1 and patch.shape[1:] == (64, 64, 64):
        patch = patch[0]
    if patch.shape != (64, 64, 64):
        raise ValueError(f"Expected one [64,64,64] patch, got {patch.shape}")
    x = preprocess_hu(torch.from_numpy(np.asarray(patch, dtype=np.float32))).unsqueeze(0).unsqueeze(0).to(device)

    checkpoints = {
        "lobulation_spiculation": Path(args.lobulation_spiculation_checkpoint).resolve(),
        "subtlety": Path(args.subtlety_checkpoint).resolve(),
        "sphericity": Path(args.sphericity_checkpoint).resolve(),
        "margin": Path(args.margin_checkpoint).resolve(),
        "texture": Path(args.texture_checkpoint).resolve(),
    }
    predictions: dict[str, float] = {}
    common_architecture = None
    with torch.inference_mode():
        for group, checkpoint_path in checkpoints.items():
            model, architecture = load_model(checkpoint_path, device)
            if architecture["objective"] != "ordinal_reader":
                raise ValueError(f"{group} checkpoint must use ordinal_reader")
            if common_architecture is None:
                common_architecture = architecture
            elif architecture != common_architecture:
                raise ValueError("Hybrid checkpoints must share an identical architecture configuration")
            expected = model.expected_ratings(model(x))[0].cpu().numpy()
            for name in CHECKPOINT_GROUPS[group]:
                predictions[name] = float(expected[MODEL_TARGETS.index(name)])

    output = {name: predictions[name] for name in OUTPUT_ATTRIBUTES}
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"Saved {output_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-npy", required=True)
    parser.add_argument("--lobulation-spiculation-checkpoint", required=True)
    parser.add_argument("--subtlety-checkpoint", required=True)
    parser.add_argument("--sphericity-checkpoint", required=True)
    parser.add_argument("--margin-checkpoint", required=True)
    parser.add_argument("--texture-checkpoint", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="auto")
    raise SystemExit(main(parser.parse_args()))
