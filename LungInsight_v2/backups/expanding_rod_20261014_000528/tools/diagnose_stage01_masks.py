#!/usr/bin/env python3
"""Create non-destructive Stage 01 lung/heart mask QA artifacts.

This is a diagnostic only. It does not alter the CT, masks, schema, candidates,
or any inference outputs. CT/mask arrays and candidate centers must all be in
Stage 01's final cropped CT voxel frame (ZYX).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON at {path}:{exc.lineno}:{exc.colno}: {exc.msg}") from exc


def records_from_payload(payload: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        records = payload.get("candidates", payload.get("predictions", payload.get("detections")))
        if records is None:
            raise ValueError(f"{path}: expected a list or an object with candidates/predictions/detections")
    else:
        raise ValueError(f"{path}: expected JSON array or object")
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        raise ValueError(f"{path}: candidate collection must be a list of objects")
    return [dict(r) for r in records]


def source_name(row: dict[str, Any], fallback: str) -> str:
    raw = str(row.get("detection_source", row.get("source", fallback)))
    low = raw.lower()
    if "log" in low or "blob" in low:
        return "LoG"
    if "vit" in low or "detr" in low:
        return "ViTDet3D"
    return fallback


def center_zyx(row: dict[str, Any]) -> np.ndarray | None:
    value = row.get("center_zyx")
    if value is None:
        value = row.get("center_voxel_zyx")
    try:
        center = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if center.shape != (3,) or not np.isfinite(center).all():
        return None
    return center


def sample(mask: np.ndarray, center: np.ndarray | None) -> tuple[bool | None, tuple[int, int, int] | None, str | None]:
    if center is None:
        return None, None, "missing_or_invalid_center_zyx"
    idx_arr = np.rint(center).astype(np.int64)
    idx = tuple(int(x) for x in idx_arr)
    if any(idx[a] < 0 or idx[a] >= mask.shape[a] for a in range(3)):
        return False, idx, "center_outside_ct_volume"
    return bool(mask[idx]), idx, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-dir", required=True, type=Path,
                        help="Stage 01 patient output folder with volume_hu_cropped.npy, lung_mask.npy, heart_mask.npy, meta.json")
    parser.add_argument("--vitdet-candidates", type=Path, default=None,
                        help="Optional Stage 02 candidates.json, for center-location diagnostics only")
    parser.add_argument("--log-candidates", type=Path, default=None,
                        help="Optional Stage 02 log_candidates_diagnostic.json; remains diagnostic-only")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-slices", type=int, default=15,
                        help="Maximum montage slice count (defaults to 15)")
    parser.add_argument("--window-low", type=float, default=-1000.0,
                        help="Display-only CT window low HU (does not affect the pipeline)")
    parser.add_argument("--window-high", type=float, default=400.0,
                        help="Display-only CT window high HU (does not affect the pipeline)")
    args = parser.parse_args()

    patient_dir = args.patient_dir.expanduser().resolve()
    output_dir = (args.output_dir or (patient_dir / "01_mask_qc")).expanduser().resolve()
    if args.max_slices < 1:
        parser.error("--max-slices must be >= 1")
    if not math.isfinite(args.window_low) or not math.isfinite(args.window_high) or args.window_high <= args.window_low:
        parser.error("window-high must be finite and greater than window-low")

    paths = {
        "ct": patient_dir / "volume_hu_cropped.npy",
        "lung": patient_dir / "lung_mask.npy",
        "heart": patient_dir / "heart_mask.npy",
        "meta": patient_dir / "meta.json",
    }
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required Stage 01 {key} file missing: {path}")

    ct = np.load(paths["ct"], mmap_mode="r", allow_pickle=False)
    lung = np.asarray(np.load(paths["lung"], mmap_mode="r", allow_pickle=False), dtype=bool)
    heart = np.asarray(np.load(paths["heart"], mmap_mode="r", allow_pickle=False), dtype=bool)
    meta = read_json(paths["meta"])
    if ct.ndim != 3 or lung.shape != ct.shape or heart.shape != ct.shape:
        raise ValueError(f"CT/mask geometry mismatch: CT={ct.shape}, lung={lung.shape}, heart={heart.shape}")
    if not isinstance(meta, dict):
        raise ValueError("meta.json must be a JSON object")

    input_groups: list[tuple[str, list[dict[str, Any]], Path]] = []
    for source, path in (("ViTDet3D", args.vitdet_candidates), ("LoG", args.log_candidates)):
        if path is not None:
            resolved = path.expanduser().resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"Candidate input missing: {resolved}")
            rows = records_from_payload(read_json(resolved), resolved)
            input_groups.append((source, rows, resolved))

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "slice_mask_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["z", "lung_voxels", "heart_voxels", "lung_fraction", "heart_fraction", "ct_min_hu", "ct_mean_hu", "ct_max_hu"])
        writer.writeheader()
        for z in range(ct.shape[0]):
            sl = np.asarray(ct[z])
            writer.writerow({
                "z": z,
                "lung_voxels": int(lung[z].sum()),
                "heart_voxels": int(heart[z].sum()),
                "lung_fraction": float(lung[z].mean()),
                "heart_fraction": float(heart[z].mean()),
                "ct_min_hu": float(np.min(sl)),
                "ct_mean_hu": float(np.mean(sl)),
                "ct_max_hu": float(np.max(sl)),
            })

    membership_rows: list[dict[str, Any]] = []
    all_plot_candidates: list[tuple[str, np.ndarray, str]] = []
    source_counts: dict[str, dict[str, int]] = {}
    vit_z: list[int] = []
    for group_fallback, rows, path in input_groups:
        counts = {"input": len(rows), "valid_centers": 0, "center_in_lung_mask": 0, "center_in_heart_mask": 0, "center_outside_lung_mask": 0, "center_outside_ct": 0, "invalid_center": 0}
        for idx0, row in enumerate(rows):
            source = source_name(row, group_fallback)
            center = center_zyx(row)
            lung_at, index, reason = sample(lung, center)
            heart_at, _heart_index, heart_reason = sample(heart, center)
            valid_center = center is not None
            if valid_center:
                counts["valid_centers"] += 1
            if reason == "center_outside_ct_volume":
                counts["center_outside_ct"] += 1
            elif reason == "missing_or_invalid_center_zyx":
                counts["invalid_center"] += 1
            elif lung_at is True:
                counts["center_in_lung_mask"] += 1
            elif lung_at is False:
                counts["center_outside_lung_mask"] += 1
            if heart_at is True:
                counts["center_in_heart_mask"] += 1
            if center is not None:
                all_plot_candidates.append((source, center, str(row.get("candidate_id", row.get("diagnostic_id", row.get("id", idx0))))))
                if source == "ViTDet3D":
                    zround = int(np.rint(center[0]))
                    if 0 <= zround < ct.shape[0]:
                        vit_z.append(zround)
            hu = float(ct[index]) if index is not None and all(0 <= index[a] < ct.shape[a] for a in range(3)) else None
            membership_rows.append({
                "input_file": str(path),
                "input_group": group_fallback,
                "detected_source": source,
                "candidate_id": row.get("candidate_id", row.get("diagnostic_id", row.get("id", idx0))),
                "center_z": None if center is None else float(center[0]),
                "center_y": None if center is None else float(center[1]),
                "center_x": None if center is None else float(center[2]),
                "rounded_z": None if index is None else index[0],
                "rounded_y": None if index is None else index[1],
                "rounded_x": None if index is None else index[2],
                "center_hu": hu,
                "center_in_lung_mask": lung_at,
                "center_in_heart_mask": heart_at,
                "membership_note": reason or heart_reason or ("heart mask is empty; heart absence is unverified" if not heart.any() else ""),
            })
        source_counts[group_fallback] = counts

    membership_path = output_dir / "candidate_mask_membership.csv"
    fields = ["input_file", "input_group", "detected_source", "candidate_id", "center_z", "center_y", "center_x", "rounded_z", "rounded_y", "rounded_x", "center_hu", "center_in_lung_mask", "center_in_heart_mask", "membership_note"]
    with membership_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(membership_rows)

    # Choose representative axial slices and include slices containing the
    # authoritative detector centers. No candidate files are modified.
    if ct.shape[0] <= args.max_slices:
        slice_indices = set(range(ct.shape[0]))
    else:
        slice_indices = set(np.linspace(0, ct.shape[0] - 1, args.max_slices, dtype=int).tolist())
    slice_indices.update(vit_z)
    if len(slice_indices) > args.max_slices:
        selected_base = sorted(set(np.linspace(0, ct.shape[0] - 1, max(1, args.max_slices - min(len(vit_z), args.max_slices - 1)), dtype=int).tolist()))
        slice_indices = set(selected_base) | set(vit_z[:args.max_slices])
    slice_indices = sorted(slice_indices)[:max(args.max_slices, min(len(vit_z), args.max_slices))]
    ncols = 3
    nrows = max(1, math.ceil(len(slice_indices) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.8 * nrows), squeeze=False)
    for ax, z in zip(axes.flat, slice_indices):
        ax.imshow(np.asarray(ct[z]), cmap="gray", vmin=args.window_low, vmax=args.window_high)
        if lung[z].any() and not lung[z].all():
            ax.contour(lung[z].astype(np.uint8), levels=[0.5], colors=["lime"], linewidths=0.65)
        if heart[z].any() and not heart[z].all():
            ax.contour(heart[z].astype(np.uint8), levels=[0.5], colors=["red"], linewidths=0.8)
        # Plot every available ViTDet3D center on its rounded axial slice.
        for source, center, _cid in all_plot_candidates:
            if int(np.rint(center[0])) != z:
                continue
            if source == "ViTDet3D":
                ax.scatter(center[2], center[1], marker="x", s=55, linewidths=1.3, color="red")
        # LoG is diagnostic-only. Display a deterministic subset to avoid
        # obscuring mask boundaries with thousands of points.
        log_points = [(center[2], center[1]) for source, center, _cid in all_plot_candidates if source == "LoG" and int(np.rint(center[0])) == z]
        if len(log_points) > 80:
            step = math.ceil(len(log_points) / 80)
            log_points = log_points[::step]
        if log_points:
            xs, ys = zip(*log_points)
            ax.scatter(xs, ys, marker=".", s=9, alpha=0.5, color="cyan")
        ax.set_title(f"z={z} | lung={int(lung[z].sum()):,} vox | heart={int(heart[z].sum()):,} vox")
        ax.set_axis_off()
    for ax in list(axes.flat)[len(slice_indices):]:
        ax.set_axis_off()
    handles = [
        plt.Line2D([0], [0], color="lime", lw=2, label="lung mask contour"),
        plt.Line2D([0], [0], color="red", lw=2, label="heart mask contour / ViTDet3D center (x)"),
        plt.Line2D([0], [0], marker=".", color="cyan", lw=0, label="sampled LoG diagnostic centers"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3)
    warning = "WARNING: heart_mask is EMPTY; cardiac exclusion is UNVERIFIED" if not heart.any() else ""
    fig.suptitle(f"Stage 01 Mask QC — {patient_dir.name}\n{warning}", color="red" if warning else "black", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    montage_path = output_dir / "mask_qc_montage.png"
    fig.savefig(montage_path, dpi=150)
    plt.close(fig)

    summary = {
        "patient_id": patient_dir.name,
        "patient_dir": str(patient_dir),
        "meta_schema_version": meta.get("meta_schema_version", meta.get("schema_version")),
        "ct_shape_zyx": list(ct.shape),
        "spacing_zyx_mm": meta.get("spacing_zyx_mm"),
        "origin_mm": meta.get("origin_mm"),
        "coordinate_system": meta.get("coordinate_system"),
        "lung_mask_voxels": int(lung.sum()),
        "lung_mask_fraction": float(lung.mean()),
        "heart_mask_voxels": int(heart.sum()),
        "heart_mask_fraction": float(heart.mean()),
        "heart_mask_found": bool(heart.any()),
        "heart_exclusion_verified": bool(heart.any()),
        "candidate_inputs": source_counts,
        "candidate_file_policy": "LoG remains diagnostic-only unless the authoritative project schema explicitly changes that policy.",
        "status": "BLOCKED_EMPTY_HEART_MASK" if not heart.any() else "MASKS_REQUIRE_VISUAL_QA",
        "outputs": {
            "montage_png": str(montage_path),
            "slice_metrics_csv": str(metrics_path),
            "candidate_mask_membership_csv": str(membership_path),
            "summary_json": str(output_dir / "mask_qc_summary.json"),
        },
        "interpretation": "An empty heart mask is not evidence that the heart is absent. Review the montage and per-candidate membership before enabling downstream inference.",
    }
    (output_dir / "mask_qc_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    # Non-zero is intentional when no heart voxels were identified.
    return 2 if not heart.any() else 0


if __name__ == "__main__":
    raise SystemExit(main())
