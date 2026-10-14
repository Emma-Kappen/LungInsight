#!/usr/bin/env python3
"""Create an all-slice visual/audit bundle for LungInsight Stage 01 masks.

This tool is read-only: it does not edit the CT, masks, schema, or pipeline.
It renders every axial slice in small contact sheets and collects existing
mask/candidate diagnostics so a reviewer can distinguish lung-mask geometry
from an empty dedicated heart-mask heuristic.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def display_hu(image: np.ndarray, level: float, width: float) -> np.ndarray:
    low, high = level - width / 2.0, level + width / 2.0
    return np.clip((image.astype(np.float32) - low) / width, 0.0, 1.0)


def render_sheets(ct: np.ndarray, lung: np.ndarray, heart: np.ndarray,
                  output: Path, slices_per_sheet: int, columns: int,
                  level: float, width: float) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    total = ct.shape[0]
    for start in range(0, total, slices_per_sheet):
        end = min(total, start + slices_per_sheet)
        count = end - start
        rows = math.ceil(count / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.0, rows * 3.1), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for offset, z in enumerate(range(start, end)):
            ax = axes.ravel()[offset]
            ax.imshow(display_hu(ct[z], level, width), cmap="gray", vmin=0, vmax=1, origin="upper")
            if lung[z].any():
                ax.contour(lung[z].astype(np.uint8), levels=[0.5], colors=["lime"], linewidths=0.55)
            if heart[z].any():
                ax.contour(heart[z].astype(np.uint8), levels=[0.5], colors=["red"], linewidths=0.8)
            ax.set_title(f"z={z} | lung={int(lung[z].sum()):,} | heart={int(heart[z].sum()):,}", fontsize=7)
            ax.axis("off")
        first = start
        last = end - 1
        fig.suptitle(f"LungInsight Stage 01 mask review | slices {first}–{last} of {total-1}\nGreen = lung_mask; red = heart_mask", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        path = output / f"mask_review_slices_{first:03d}_{last:03d}.png"
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def copy_existing(source: Path, dest_dir: Path, required: bool = False) -> Path | None:
    if not source.is_file():
        if required:
            raise FileNotFoundError(f"Required review artifact not found: {source}")
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / source.name
    target.write_bytes(source.read_bytes())
    return target


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--patient-dir", required=True, type=Path)
    p.add_argument("--schema", required=True, type=Path)
    p.add_argument("--stage01-source", required=True, type=Path)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--slices-per-sheet", type=int, default=28)
    p.add_argument("--columns", type=int, default=4)
    p.add_argument("--window-level", type=float, default=-600.0)
    p.add_argument("--window-width", type=float, default=1500.0)
    args = p.parse_args()

    patient_dir = args.patient_dir.resolve()
    schema_path = args.schema.resolve()
    stage01_path = args.stage01_source.resolve()
    out_dir = (args.output_dir or (patient_dir / "01_mask_qc" / "full_slice_review")).resolve()
    if args.slices_per_sheet < 1 or args.columns < 1:
        p.error("--slices-per-sheet and --columns must be positive")

    ct = np.load(patient_dir / "volume_hu_cropped.npy", mmap_mode="r")
    lung = np.load(patient_dir / "lung_mask.npy", mmap_mode="r")
    heart = np.load(patient_dir / "heart_mask.npy", mmap_mode="r")
    if ct.ndim != 3 or lung.shape != ct.shape or heart.shape != ct.shape:
        raise ValueError(f"CT/lung/heart shape mismatch: ct={ct.shape}, lung={lung.shape}, heart={heart.shape}")
    if not np.isfinite(np.asarray(ct[::max(1, ct.shape[0] // 10)], dtype=np.float32)).all():
        raise ValueError("CT sample contains non-finite values")

    out_dir.mkdir(parents=True, exist_ok=True)
    sheet_paths = render_sheets(ct, lung, heart, out_dir, args.slices_per_sheet,
                                args.columns, args.window_level, args.window_width)

    patient_id = patient_dir.name
    qc_dir = patient_dir / "01_mask_qc"
    fusion_dir = patient_dir / "02_candidate_fusion_audit"
    files_to_collect = [
        (qc_dir / "mask_qc_summary.json", True),
        (qc_dir / "slice_mask_metrics.csv", True),
        (qc_dir / "candidate_mask_membership.csv", True),
        (qc_dir / "mask_qc_montage.png", False),
        (fusion_dir / "merge_filter_report.json", False),
        (patient_dir / "meta.json", True),
        (schema_path, True),
        (stage01_path, True),
    ]
    collected_dir = out_dir / "context"
    copied = []
    for src, required in files_to_collect:
        target = copy_existing(src, collected_dir, required=required)
        if target is not None:
            copied.append(target)

    meta = read_json(patient_dir / "meta.json")
    summary = read_json(qc_dir / "mask_qc_summary.json")
    report = {
        "patient_id": patient_id,
        "ct_shape_zyx": list(ct.shape),
        "lung_mask_shape_zyx": list(lung.shape),
        "heart_mask_shape_zyx": list(heart.shape),
        "lung_mask_voxels": int(np.count_nonzero(lung)),
        "heart_mask_voxels": int(np.count_nonzero(heart)),
        "heart_mask_nonempty": bool(np.any(heart)),
        "ct_meta_schema_version": meta.get("meta_schema_version", meta.get("schema_version")),
        "diagnostic_status": summary.get("status"),
        "sheets": [x.name for x in sheet_paths],
        "notes": [
            "This is a read-only review bundle; no pipeline files or arrays were modified.",
            "Every axial slice is shown in exactly one contact sheet.",
            "An empty heart_mask is a failed/absent heuristic output, not evidence that the heart is absent.",
        ],
    }
    report_path = out_dir / "full_slice_review_summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    zip_path = out_dir / f"{patient_id}_mask_review_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in [*sheet_paths, *copied, report_path]:
            zf.write(path, arcname=path.relative_to(out_dir).as_posix())
    print(json.dumps({
        "status": "completed_read_only_mask_review_bundle",
        "patient_id": patient_id,
        "ct_shape_zyx": list(ct.shape),
        "lung_mask_voxels": int(np.count_nonzero(lung)),
        "heart_mask_voxels": int(np.count_nonzero(heart)),
        "sheets": [str(x) for x in sheet_paths],
        "bundle_zip": str(zip_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
