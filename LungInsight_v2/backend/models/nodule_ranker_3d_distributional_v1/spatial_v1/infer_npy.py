#!/usr/bin/env python3
"""Run a spatial_objective_v1 checkpoint on detected-nodule NPY patches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model_objective import MODEL_TARGETS, NOMINAL_TARGETS, ORDERED_TARGETS, RATING_MAX, SpatialObjectiveModel, preprocess_hu


def choose_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def load_checkpoint(path: Path, device: torch.device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    arch = ck["architecture"]
    model = SpatialObjectiveModel(
        arch["objective"],
        arch["attention_dim"],
        arch["head_hidden"],
        arch["dropout"],
    )
    model.load_state_dict(ck["model_state_dict"], strict=True)
    return model.to(device).eval(), ck


def main(a):
    device = choose_device(a.device)
    checkpoint = Path(a.checkpoint).resolve()
    input_path = Path(a.input_npy).resolve()
    output_path = Path(a.output_json).resolve()
    x_np = np.load(input_path, mmap_mode="r")
    if x_np.ndim == 3:
        x_np = x_np[None, ...]
    elif x_np.ndim == 5 and x_np.shape[1] == 1:
        x_np = x_np[:, 0]
    if x_np.ndim != 4 or tuple(x_np.shape[1:]) != (64, 64, 64):
        raise ValueError(f"Expected [64,64,64], [N,64,64,64], or [N,1,64,64,64], got {x_np.shape}")

    model, ck = load_checkpoint(checkpoint, device)
    objective = ck["architecture"]["objective"]
    expected_batches: list[np.ndarray] = []
    nominal_batches: dict[str, list[np.ndarray]] = {n: [] for n in NOMINAL_TARGETS}
    ordinal_prob_batches: dict[str, list[np.ndarray]] = {n: [] for n in ORDERED_TARGETS} if objective == "ordinal_reader" else {}

    with torch.inference_mode():
        for start in range(0, len(x_np), a.batch_size):
            arr = np.array(x_np[start:start + a.batch_size], dtype=np.float32, copy=True)
            x = preprocess_hu(torch.from_numpy(arr)).to(device).unsqueeze(1)
            outputs = model(x)
            expected_batches.append(model.expected_ratings(outputs).cpu().numpy().astype(np.float32))
            for h, name in enumerate(MODEL_TARGETS):
                if name in NOMINAL_TARGETS:
                    nominal_batches[name].append(torch.softmax(outputs[h].float(), dim=-1).cpu().numpy().astype(np.float32))
                elif objective == "ordinal_reader":
                    from model_objective import OrdinalHead
                    ordinal_prob_batches[name].append(OrdinalHead.probabilities(outputs[h]).cpu().numpy().astype(np.float32))

    expected = np.concatenate(expected_batches, axis=0)
    nominal_np = {k: np.concatenate(v, axis=0) for k, v in nominal_batches.items()}
    ordinal_np = {k: np.concatenate(v, axis=0) for k, v in ordinal_prob_batches.items()} if objective == "ordinal_reader" else {}
    output = []
    for i in range(len(expected)):
        record = {"patch_index": i, "attributes": {}}
        for h, name in enumerate(MODEL_TARGETS):
            attr = {"expected_rating": float(expected[i, h])}
            if name in NOMINAL_TARGETS:
                probs = nominal_np[name][i]
                attr["probabilities"] = {str(r): float(probs[r - 1]) for r in range(1, RATING_MAX[name] + 1)}
                attr["predicted_class"] = int(np.argmax(probs) + 1)
            elif objective == "ordinal_reader":
                probs = ordinal_np[name][i]
                attr["probabilities"] = {str(r): float(probs[r - 1]) for r in range(1, RATING_MAX[name] + 1)}
            else:
                attr["normalized_regression_output"] = float((expected[i, h] - 1.0) / (RATING_MAX[name] - 1))
            record["attributes"][name] = attr
        output.append(record)

    report = {
        "checkpoint": str(checkpoint),
        "objective": objective,
        "input_npy": str(input_path),
        "n_patches": int(len(output)),
        "input_shape": list(x_np.shape),
        "attributes": output,
        "test_or_h5_dependency": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {output_path} | patches={len(output)} | objective={objective}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input-npy", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=16)
    raise SystemExit(main(p.parse_args()))
