#!/usr/bin/env python3
"""Predict radiologist malignancy suspicion from one raw-HU nodule patch."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SPATIAL_DIR = Path(__file__).resolve().parent.parent / "nodule_ranker_3d_distributional_v1" / "spatial_v1"
sys.path.insert(0, str(SPATIAL_DIR))

from malignancy_fusion import MalignancyFusion
from model_objective import MODEL_TARGETS, SpatialObjectiveModel, preprocess_hu
from train_malignancy import choose_device


ATTRIBUTE_NAMES = ("subtlety", "sphericity", "margin", "lobulation", "spiculation", "texture")
GROUPS = {
    "lobulation_spiculation": ("lobulation", "spiculation"),
    "subtlety": ("subtlety",),
    "sphericity": ("sphericity",),
    "margin": ("margin",),
    "texture": ("texture",),
}


def load_attribute_model(path: str, device: torch.device) -> SpatialObjectiveModel:
    checkpoint = torch.load(Path(path).resolve(), map_location="cpu", weights_only=False)
    architecture = checkpoint["architecture"]
    if architecture.get("objective") != "ordinal_reader":
        raise ValueError(f"{path} must be an ordinal_reader hybrid checkpoint")
    model = SpatialObjectiveModel(
        architecture["objective"], architecture["attention_dim"],
        architecture["head_hidden"], architecture["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()
    return model.to(device)


def predict_attribute(model: SpatialObjectiveModel, x: torch.Tensor, passes: int) -> np.ndarray:
    draws = []
    with torch.inference_mode():
        for _ in range(passes):
            expected = model.expected_ratings(model(x))[0]
            selected = [MODEL_TARGETS.index(name) for name in ATTRIBUTE_NAMES]
            draws.append(expected[selected])
    values = torch.stack(draws)
    return torch.stack((values.mean(dim=0), values.std(dim=0, unbiased=False)), dim=1).cpu().numpy()


def main(args: argparse.Namespace) -> int:
    if args.mc_passes < 2:
        raise ValueError("--mc-passes must be at least 2")
    device = choose_device(args.device)
    patch = np.load(Path(args.input_npy).resolve(), allow_pickle=False)
    if patch.ndim == 5 and patch.shape[:2] == (1, 1):
        patch = patch[0, 0]
    elif patch.ndim == 4 and patch.shape[0] == 1:
        patch = patch[0]
    if patch.shape != (64, 64, 64):
        raise ValueError(f"Expected one raw-HU [64,64,64] patch, got {patch.shape}")
    x = preprocess_hu(torch.from_numpy(np.asarray(patch, dtype=np.float32))).unsqueeze(0).unsqueeze(0).to(device)

    checkpoint_paths = {
        "lobulation_spiculation": args.lobulation_spiculation,
        "subtlety": args.subtlety,
        "sphericity": args.sphericity,
        "margin": args.margin,
        "texture": args.texture,
    }
    outputs: dict[str, np.ndarray] = {}
    for group, names in GROUPS.items():
        model = load_attribute_model(checkpoint_paths[group], device)
        predictions = predict_attribute(model, x, args.mc_passes)
        for name in names:
            outputs[name] = predictions[ATTRIBUTE_NAMES.index(name)]
        del model
    scores = [float(outputs[name][0]) for name in ATTRIBUTE_NAMES]
    uncertainties = [float(outputs[name][1]) for name in ATTRIBUTE_NAMES]
    feature = torch.tensor([scores + uncertainties], dtype=torch.float32, device=device)

    malignancy_checkpoint = torch.load(
        Path(args.malignancy_checkpoint).resolve(), map_location="cpu", weights_only=False
    )
    input_dim = int(malignancy_checkpoint["architecture"]["input_dim"])
    model = MalignancyFusion(input_dim=input_dim).to(device)
    model.load_state_dict(malignancy_checkpoint["model_state_dict"], strict=True)
    model.eval()
    mean = torch.as_tensor(malignancy_checkpoint["feature_mean"], dtype=feature.dtype, device=device)
    std = torch.as_tensor(malignancy_checkpoint["feature_std"], dtype=feature.dtype, device=device)
    with torch.inference_mode():
        prediction = model.predict((feature[:, :input_dim] - mean) / std)
    result = {
        "malignancy_expected": float(prediction["malignancy_expected"][0].cpu()),
        "malignancy_distribution": prediction["class_probabilities"][0].cpu().tolist(),
        "probability_malignant_ge4": float(prediction["probability_malignant_ge4"][0].cpu()),
        "probability_malignant_ge5": float(prediction["probability_malignant_ge5"][0].cpu()),
        "attributes": {
            name: {"score": float(outputs[name][0]), "uncertainty": float(outputs[name][1])}
            for name in ATTRIBUTE_NAMES
        },
        "interpretation": "Radiologist malignancy suspicion score, not a cancer diagnosis.",
        "rating_interpretation": ["very unlikely", "unlikely", "indeterminate", "suspicious", "highly suspicious"],
    }
    serialized = json.dumps(result, indent=2)
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized, encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-npy", required=True, help="One raw-HU 64x64x64 CT patch")
    parser.add_argument("--lobulation-spiculation", "--lobulation-spiculation-checkpoint", dest="lobulation_spiculation", required=True)
    parser.add_argument("--subtlety", "--subtlety-checkpoint", required=True)
    parser.add_argument("--sphericity", "--sphericity-checkpoint", required=True)
    parser.add_argument("--margin", "--margin-checkpoint", required=True)
    parser.add_argument("--texture", "--texture-checkpoint", required=True)
    parser.add_argument("--malignancy-checkpoint", required=True)
    parser.add_argument("--mc-passes", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None)
    raise SystemExit(main(parser.parse_args()))