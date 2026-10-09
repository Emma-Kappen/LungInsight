"""
LungInsight v2 -- Stage 03: Nodule Attribute Characterization (3D CNN, v3.0)
============================================================================

Runs the genuine 3D multi-headed nodule-attribute network
(backend/models/nodule_ranker_3d/model_3d.py, trained by train_nodule_ranker_3d.py)
on the Stage 02 nodule patches.

    python backend/Imaging/03_nodule_classification.py LIDC-IDRI-0001 \
        --checkpoint backend/models/nodule_ranker_3d/checkpoints/best.pth

Input  : backend/output/<patient_id>/02_detect_and_patch/patches/{arrays/patch_XXX.npy, patches_manifest.json}
         float32 HU, ZYX, exactly 64x64x64 (the shape the 3D model was trained on).
Output : backend/output/<patient_id>/03_nodule_classification/{predictions.json, predictions.csv,
         classification_metadata.json}

Differences from the retired 2D slice-wise Stage 03 (contract changes, all deliberate):
  * 8 attributes only. `malignancy` is NOT predicted here (reserved for a downstream stage).
  * internal_structure uses the 1..5 scale (not 1..4); calcification 1..6; others 1..5.
  * No slice/attribute attention is produced (the 3D CNN has none) -> no attention/ directory;
    `auxiliary_score_0_1` is kept for CSV/JSON compatibility but is always null.
  * Preprocessing constants are read from the checkpoint:
        normalized = (clip(HU,-1000,400) + 1000 - 357.87565) / 288.47657
  * score_0_1 = (rating - 1) / (K - 1) is kept as a normalised view of the rating.
Stage 02 geometry / provenance fields are copied through unchanged.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
V2_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT_ROOT = V2_ROOT / "backend" / "output"
DEFAULT_MODEL_DIR = V2_ROOT / "backend" / "models" / "nodule_ranker_3d"
DEFAULT_CHECKPOINT = DEFAULT_MODEL_DIR / "checkpoints" / "best.pth"


def json_default(o: Any):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=json_default, allow_nan=False)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def import_model_module(model_dir: Path):
    path = model_dir / "model_3d.py"
    if not path.is_file():
        raise FileNotFoundError(f"Missing 3D model definition: {path}")
    spec = importlib.util.spec_from_file_location("lunginsight_model_3d", str(path))
    mod = importlib.util.module_from_spec(spec) # type: ignore
    sys.modules["lunginsight_model_3d"] = mod
    spec.loader.exec_module(mod) # type: ignore
    return mod


def load_patch_manifest(stage02_dir: Path) -> dict[str, Any]:
    p = stage02_dir / "patches" / "patches_manifest.json"
    if not p.is_file():
        raise FileNotFoundError(f"Missing Stage 02 patch manifest: {p}")
    m = load_json(p)
    if not isinstance(m, dict):
        raise ValueError("patches_manifest.json must contain an object")
    return m


def load_patch_array(patch_root: Path, record: dict[str, Any]) -> tuple[Path, np.ndarray]:
    options = []
    if record.get("array_path"):
        c = Path(str(record["array_path"]))
        options = [c] if c.is_absolute() else [c, patch_root / c]
    options.append(patch_root / "arrays" / f"patch_{int(record['patch_id']):03d}.npy")
    for p in options:
        if p.is_file():
            return p, np.asarray(np.load(p, allow_pickle=False), dtype=np.float32)
    raise FileNotFoundError(f"Patch array not found. Tried: {options}")


def validate_patch(record: dict[str, Any], patch: np.ndarray, expected: tuple[int, int, int]) -> None:
    if str(record.get("coordinate_order", "ZYX")).upper() != "ZYX":
        raise ValueError(f"Patch {record.get('patch_id')} is not ZYX")
    if tuple(patch.shape) != expected:
        raise ValueError(f"Patch {record.get('patch_id')} has shape {patch.shape}; the 3D model requires {expected}.")
    if not np.all(np.isfinite(patch)):
        raise ValueError(f"Patch {record.get('patch_id')} contains non-finite values")


@torch.inference_mode()
def infer_batch(model, mm, patches_hu: list[np.ndarray], device: torch.device) -> np.ndarray:
    x = torch.from_numpy(np.stack(patches_hu)).to(device)
    x = mm.preprocess_hu(x).unsqueeze(1)  # (B,1,64,64,64), ZYX
    return model.expected_rating(model(x)).cpu().numpy().astype(np.float64)  # (B,8)


def build_attributes(ratings: np.ndarray, targets: list[str], scales: dict) -> dict[str, Any]:
    out = {}
    for i, name in enumerate(targets):
        lo, hi = float(scales[name]["min"]), float(scales[name]["max"])
        r = float(np.clip(ratings[i], lo, hi))
        out[name] = {
            "feature_id": i,
            "score_0_1": (r - lo) / (hi - lo),
            "rating_original_scale": r,
            "rating_rounded": int(np.clip(np.floor(r + 0.5), lo, hi)),
            "rating_scale": [lo, hi],
            "auxiliary_score_0_1": None,
        }
    return out


def build_record(pr: dict[str, Any], patch_path: Path, attrs: dict[str, Any], ck_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "patch_id": int(pr["patch_id"]),
        "candidate_id": pr["candidate_id"],
        "prediction_space": pr.get("space", "patch_local_ct"),
        "source_ct_space": pr.get("source_space", "stage01_final_cropped_ct"),
        "coordinate_order": "ZYX",
        "patch_size_vox_zyx": pr.get("patch_size_vox_zyx"),
        "spacing_zyx_mm": pr.get("spacing_zyx_mm"),
        "origin_lps_mm": pr.get("origin_lps_mm"),
        "patch_origin_lps_mm": pr.get("patch_origin_lps_mm"),
        "direction_lps": pr.get("direction_lps"),
        "center_zyx": pr.get("center_zyx"),
        "center_lps_mm": pr.get("center_lps_mm", pr.get("center_world_mm")),
        "patch_center_local_zyx": pr.get("patch_center_local_zyx"),
        "detector_score": pr.get("detector_score"),
        "raw_logit": pr.get("raw_logit"),
        "diameter_mm": pr.get("diameter_mm"),
        "anatomical_location": pr.get("anatomical_location"),
        "attributes": attrs,
        "patch_array_path": str(patch_path),
        "reference_model": ck_meta,
    }


def write_csv(path: Path, records: list[dict[str, Any]], targets: list[str]) -> None:
    fields = ["patch_id", "candidate_id", "anatomical_primary", "anatomical_region", "laterality",
              "compartment", "detector_score"]
    for n in targets:
        fields += [f"{n}_score_0_1", f"{n}_rating_original_scale", f"{n}_auxiliary_score_0_1"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            loc = r.get("anatomical_location") or {}
            row = {"patch_id": r["patch_id"], "candidate_id": r["candidate_id"],
                   "anatomical_primary": loc.get("primary"), "anatomical_region": loc.get("region"),
                   "laterality": loc.get("laterality"), "compartment": loc.get("compartment"),
                   "detector_score": r.get("detector_score")}
            for n in targets:
                a = r["attributes"][n]
                row[f"{n}_score_0_1"] = a["score_0_1"]
                row[f"{n}_rating_original_scale"] = a["rating_original_scale"]
                row[f"{n}_auxiliary_score_0_1"] = a["auxiliary_score_0_1"]
            w.writerow(row)


def run_pipeline(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root).resolve()
    patient_root = output_root / args.patient_id
    stage02_dir = patient_root / "02_detect_and_patch"
    stage03_dir = patient_root / "03_nodule_classification"
    patch_root = stage02_dir / "patches"
    if not stage02_dir.is_dir():
        raise FileNotFoundError(f"Stage 02 output not found: {stage02_dir}")

    ck_path = Path(args.checkpoint).resolve()
    if not ck_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ck_path}")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device in (None, "auto") else args.device)

    mm = import_model_module(Path(args.model_dir).resolve())
    model, ck = mm.load_trained_model(ck_path, device)
    targets = list(ck["targets"])
    if targets != list(mm.MODEL_TARGETS) or "malignancy" in targets:
        raise RuntimeError(f"Checkpoint targets {targets} do not match the 8 required model targets")
    scales = ck["rating_scales"]
    expected_shape = tuple(ck["input_shape"][1:])
    pre = ck["preprocessing"]

    print("=" * 78)
    print("LUNGINSIGHT STAGE 03 -- 3D NODULE ATTRIBUTE CHARACTERIZATION")
    print("=" * 78)
    print(f"Patient     : {args.patient_id}\nDevice      : {device}\nCheckpoint  : {ck_path} (epoch {ck.get('epoch')})")

    manifest = load_patch_manifest(stage02_dir)
    records_in = manifest.get("patches", [])
    if not isinstance(records_in, list):
        raise ValueError("patches_manifest.json field 'patches' must be a list")

    ck_meta = {
        "architecture": ck["architecture"].get("name", "NoduleRanker3D") + " (3D CNN, 8 ordinal heads)",
        "module": "backend.models.nodule_ranker_3d.model_3d.NoduleRanker3D",
        "n_features": int(ck["n_features"]), "attribute_names": targets,
        "checkpoint": str(ck_path), "checkpoint_epoch": ck.get("epoch"),
    }

    loaded = []
    for rec in records_in:
        p, arr = load_patch_array(patch_root, rec)
        validate_patch(rec, arr, expected_shape)
        loaded.append((rec, p, arr))

    prediction_records: list[dict[str, Any]] = []
    for s in range(0, len(loaded), args.batch_size):
        chunk = loaded[s:s + args.batch_size]
        ratings = infer_batch(model, mm, [c[2] for c in chunk], device)
        for (rec, p, _), r in zip(chunk, ratings):
            attrs = build_attributes(r, targets, scales)
            prediction_records.append(build_record(rec, p, attrs, ck_meta))
            print(f"Characterized patch_{int(rec['patch_id']):03d}: candidate={rec['candidate_id']} " +
                  " ".join(f"{n[:4]}={attrs[n]['rating_original_scale']:.2f}" for n in targets))

    metadata = {
        "schema_version": "03.2", "stage": "03", "module": "Nodule Attribute Characterization (3D CNN)",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "patient_id": args.patient_id,
        "coordinate_order": "ZYX", "coordinate_system": "DICOM_LPS",
        "source_space": "stage01_final_cropped_ct", "prediction_space": "patch_local_ct",
        "inputs": {"stage02_directory": str(stage02_dir),
                   "patch_manifest": str(patch_root / "patches_manifest.json"),
                   "patch_arrays": str(patch_root / "arrays")},
        "model": {**ck_meta, "input_shape": ck["input_shape"], "architecture_config": ck["architecture"],
                  "seed": ck.get("seed"), "validation_aggregate": (ck.get("validation_metrics") or {}).get("aggregate"),
                  "pretraining": None if not ck.get("pretraining") else {
                      "objective": ck["pretraining"]["config"].get("objective"),
                      "epochs_completed": ck["pretraining"]["epochs_completed"]}},
        "device": str(device),
        "preprocessing": {**pre, "spatial_shape_requirement": list(expected_shape),
                          "augmentation": "none (training and inference)"},
        "attribute_scales": scales,
        "prediction_semantics": {
            "rating_original_scale": "Expected LIDC-scale rating under the ordinal head: 1 + sum_k P(y>k). "
                                     "Continuous; comparable to the reader-mean rating.",
            "score_0_1": "(rating - min) / (max - min)",
            "rating_rounded": "floor(rating + 0.5) clipped to the scale",
            "malignancy": "Not predicted by this stage (reserved for a downstream stage).",
            "auxiliary_score_0_1": "Always null; retained only for output-contract compatibility.",
        },
        "patch_provenance": {
            "center_authority": "Stage 02 center remains unchanged; Stage 03 never re-centers the patch.",
            "anatomical_location": "Copied from Stage 02 without reinterpretation.",
            "coordinate_mapping": "p_LPS_mm = patch_origin_lps_mm + direction_lps @ ([z,y,x] * spacing_zyx_mm).",
        },
        "outputs": {"predictions_json": str(stage03_dir / "predictions.json"),
                    "predictions_csv": str(stage03_dir / "predictions.csv"),
                    "classification_metadata_json": str(stage03_dir / "classification_metadata.json")},
        "prediction_count": len(prediction_records),
    }
    save_json(stage03_dir / "predictions.json", prediction_records)
    write_csv(stage03_dir / "predictions.csv", prediction_records, targets)
    save_json(stage03_dir / "classification_metadata.json", metadata)
    print("\n" + "=" * 78 + "\nSTAGE 03 COMPLETE\n" + "=" * 78)
    print(f"Characterized patches : {len(prediction_records)}\nOutput                : {stage03_dir}")
    return stage03_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LungInsight Stage 03: 3D nodule attribute characterization.")
    p.add_argument("patient_id", help="e.g. LIDC-IDRI-0001")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                   help="directory containing model_3d.py")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                   help="trained 3D checkpoint (best.pth). Required to exist; nothing is guessed.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="auto")
    return p


if __name__ == "__main__":
    run_pipeline(build_parser().parse_args())
