from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pipeline_schema import get_required, load_schema, check_meta_schema_version, validate_stage_version

SCHEMA = load_schema(__file__)
STAGE06_SCHEMA = get_required(SCHEMA, "stage_06_full_ct_gradcam", expected_type=dict)
STAGE02_SCHEMA = get_required(SCHEMA, "stage_02_detect_and_patch", expected_type=dict)
STAGE03_SCHEMA = get_required(SCHEMA, "stage_03_nodule_classification", expected_type=dict)
STAGE01_SCHEMA = get_required(SCHEMA, "stage_01_hu_and_crop", expected_type=dict)
STAGE02_PATHS = get_required(SCHEMA, "paths.stage_02", expected_type=dict)
STAGE02_PATCH_OUTPUTS = get_required(SCHEMA, "stage_02_detect_and_patch.patch_outputs", expected_type=dict)
STAGE05_PATHS = get_required(SCHEMA, "paths.stage_05", expected_type=dict)
STAGE06_PATHS = get_required(SCHEMA, "paths.stage_06", expected_type=dict)
VISUALIZATION_SCHEMA = get_required(SCHEMA, "visualization", expected_type=dict)
V2_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = V2_ROOT / get_required(SCHEMA, "paths.output_root")
ATTRIBUTES = tuple(get_required(SCHEMA, str(STAGE06_SCHEMA["attributes_ref"]), expected_type=list))
PALETTE = get_required(SCHEMA, str(STAGE06_SCHEMA["palette_ref"]), expected_type=list)
DISPLAY_WINDOW = get_required(SCHEMA, str(STAGE06_SCHEMA["display_window_ref"]), expected_type=dict)
PATCH_SHAPE = tuple(get_required(STAGE03_SCHEMA, "inputs.patch_shape_zyx", expected_type=list))

RAINBOW = LinearSegmentedColormap.from_list(
    "lunginsight_rainbow",
    [(float(item["position"]), item["hex"]) for item in PALETTE],
    N=256,
)

PATCH_EDGE_COLOR = str(VISUALIZATION_SCHEMA["patch_outline_hex"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project finalized Rainbow Grad-CAM patch arrays into the Stage 01 "
            "processed CT coordinate frame and render full-CT axial overlays."
        )
    )
    parser.add_argument(
        "--patient-id",
        default=str(get_required(SCHEMA, "defaults.patient_id")),
        help="Patient output folder name (default: LIDC-IDRI-0001).",
    )
    parser.add_argument(
        "--ct-volume",
        default=None,
        help=f"Stage 01 {STAGE01_SCHEMA['filenames']['volume_hu_cropped.npy']}; defaults under patient output folder.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help=f"Stage 02 {Path(str(STAGE02_PATCH_OUTPUTS['manifest'])).name}; defaults under patient output folder.",
    )
    parser.add_argument(
        "--cam-root",
        default=None,
        help="Folder containing 05_gradcam.py Rainbow outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory; defaults to <patient>/06_full_ct_gradcam_rainbow.",
    )
    parser.add_argument(
        "--attribute",
        choices=("all",) + ATTRIBUTES,
        default="all",
        help="Attribute to project, or all six finalized Rainbow attributes.",
    )
    parser.add_argument(
        "--alpha", type=float, default=float(STAGE06_SCHEMA["default_alpha"]),
        help="Heatmap overlay opacity from 0 to 1 (default: 0.48).",
    )
    parser.add_argument(
        "--window-level", type=float, default=float(DISPLAY_WINDOW["level"]),
        help="HU display window level (default: -600).",
    )
    parser.add_argument(
        "--window-width", type=float, default=float(DISPLAY_WINDOW["width"]),
        help="HU display window width (default: 1500).",
    )
    parser.add_argument(
        "--require-all-patches", action="store_true",
        help="Fail if any manifest patch has no CAM for any selected attribute.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            value = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {path} at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
    except OSError as exc:
        raise OSError(f"Could not read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _cam_filenames(attribute: str, kind: str) -> list[str]:
    if kind == "scaled":
        return [name.format(attribute=attribute) for name in STAGE06_SCHEMA["cam_scaled_filename_patterns"]]
    return [name.format(attribute=attribute) for name in STAGE06_SCHEMA["cam_raw_filename_patterns"]]


def find_patch_cam(
    cam_root: Path, patch_id: int, attribute: str
) -> tuple[Path | None, str | None]:
    """Find one patch's CAM without ever reusing patch 000 for another patch."""
    pid3 = f"{patch_id:03d}"
    patch_dirs = [cam_root / pattern.format(patch_id=patch_id) for pattern in STAGE06_SCHEMA["patch_directory_patterns"]]

    # Preferred layout: cam-root/patch_000/gradcam_<attribute>_rainbow_scaled.npy
    for folder in patch_dirs:
        for name in _cam_filenames(attribute, "scaled"):
            candidate = folder / name
            if candidate.is_file():
                return candidate, "scaled"
        for name in _cam_filenames(attribute, "raw"):
            candidate = folder / name
            if candidate.is_file():
                return candidate, "raw"

    # Alternate layout: all per-patch CAMs stored side-by-side with patch ID in name.
    side_by_side = [
        cam_root / pattern.format(attribute=attribute, patch_id=patch_id)
        for pattern in STAGE06_SCHEMA["cam_side_by_side_filename_patterns"]
    ]
    for candidate in side_by_side:
        if candidate.is_file():
            return candidate, "scaled" if "rainbow_scaled" in candidate.name else "raw"

    # Only patch_id == 0 may use unqualified CAM filenames, per meta_schema.json.
    if patch_id == int(STAGE06_SCHEMA["unqualified_patch_id"]):
        for name in _cam_filenames(attribute, "scaled"):
            candidate = cam_root / name
            if candidate.is_file():
                return candidate, "scaled"
        for name in _cam_filenames(attribute, "raw"):
            candidate = cam_root / name
            if candidate.is_file():
                return candidate, "raw"

    return None, None


def normalize_raw_cam(raw: np.ndarray) -> np.ndarray:
    """Apply the patch-level Rainbow min-max convention to a raw signed CAM."""
    finite = np.isfinite(raw)
    if not finite.all():
        raise ValueError("CAM contains NaN or Inf values.")
    lo = float(raw.min())
    hi = float(raw.max())
    if hi - lo < 1e-12:
        return np.full(raw.shape, 0.5, dtype=np.float32)
    return np.clip((raw - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def prepare_records(manifest_data: dict[str, Any]) -> list[dict[str, Any]]:
    records = manifest_data.get("patches")
    if not isinstance(records, list):
        raise ValueError("Manifest is missing a top-level 'patches' list.")

    cleaned = []
    seen_ids = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Encountered a non-object patch record in manifest.")
        if "patch_id" not in record or "crop_start_zyx" not in record:
            raise ValueError(
                "Each patch record must include patch_id and crop_start_zyx."
            )
        pid = int(record["patch_id"])
        if pid in seen_ids:
            raise ValueError(f"Duplicate patch_id {pid} in manifest.")
        seen_ids.add(pid)

        crop_start_raw = np.asarray(record["crop_start_zyx"], dtype=np.float64)
        if crop_start_raw.shape != (3,) or not np.isfinite(crop_start_raw).all():
            raise ValueError(f"Invalid crop_start_zyx for patch {pid}.")
        crop_start_round = np.rint(crop_start_raw).astype(np.int64)
        if not np.allclose(crop_start_raw, crop_start_round, atol=1e-5):
            raise ValueError(
                f"crop_start_zyx for patch {pid} is not integer voxel coordinates: "
                f"{crop_start_raw.tolist()}"
            )

        rec = dict(record)
        rec["patch_id"] = pid
        rec["crop_start_zyx"] = crop_start_round
        cleaned.append(rec)

    return sorted(cleaned, key=lambda r: r["patch_id"])


BBOX_EDGE_COLOR = "#FF00FF"  # detector box; deliberately outside the Rainbow palette


def add_patch_outline(ax, record: dict[str, Any], z: int, cam_shape: tuple[int, ...]) -> None:
    # Faint dashed footprint of the whole 64^3 patch (not a nodule mask).
    start = np.asarray(record["crop_start_zyx"], dtype=np.int64)
    end = start + np.asarray(cam_shape, dtype=np.int64)
    if start[0] <= z < end[0]:
        y0, x0 = max(0, int(start[1])), max(0, int(start[2]))
        ax.add_patch(
            Rectangle(
                (x0 - 0.5, y0 - 0.5),
                max(0, int(end[2]) - x0),
                max(0, int(end[1]) - y0),
                linewidth=0.8,
                edgecolor=PATCH_EDGE_COLOR,
                facecolor="none",
                linestyle="--",
                alpha=0.45,
            )
        )

    # Detector box in source-CT voxel coordinates: start inclusive, end exclusive.
    bbox_start = record.get("bbox_start_zyx")
    bbox_end = record.get("bbox_end_zyx")
    if bbox_start is None or bbox_end is None:
        return
    bs = np.floor(np.asarray(bbox_start, dtype=np.float64) + 1e-6).astype(np.int64)
    be = np.ceil(np.asarray(bbox_end, dtype=np.float64) - 1e-6).astype(np.int64)
    if bs[0] <= z < be[0]:
        ax.add_patch(
            Rectangle(
                (bs[2] - 0.5, bs[1] - 0.5),
                int(be[2] - bs[2]),
                int(be[1] - bs[1]),
                linewidth=1.8,
                edgecolor=BBOX_EDGE_COLOR,
                facecolor="none",
            )
        )


def raw_signed_summary(cam_path: Path, attribute: str) -> dict[str, float] | None:
    # Raw signed CAM stored next to the loaded one, if present.
    raw_path = cam_path.parent / f"gradcam_{attribute}_raw_signed.npy"
    if not raw_path.is_file():
        return None
    raw = np.load(raw_path)
    lo, hi = float(raw.min()), float(raw.max())
    zero = (0.0 - lo) / (hi - lo) if hi - lo > 1e-12 else float("nan")
    return {
        "raw_min": lo,
        "raw_max": hi,
        "zero_position": zero,
        "fraction_positive": float((raw > 0).mean()),
    }


def render_slice(
    ct_volume: np.ndarray,
    cam_volume: np.ndarray,
    coverage_count: np.ndarray,
    z: int,
    attr: str,
    out_path: Path,
    records_with_cams: list[tuple[dict[str, Any], tuple[int, int, int]]],
    alpha: float,
    window_level: float,
    window_width: float,
    signed_info: dict[int, dict[str, float] | None] | None = None,
) -> None:
    ct_slice = ct_volume[z]
    cam_slice = cam_volume[z]
    valid = coverage_count[z] > 0

    rgba = RAINBOW(Normalize(vmin=0.0, vmax=1.0)(np.nan_to_num(cam_slice, nan=0.0)))
    rgba[..., 3] = valid.astype(np.float32) * alpha

    fig, ax = plt.subplots(figsize=(8.2, 8.0))
    vmin = window_level - window_width / 2.0
    vmax = window_level + window_width / 2.0
    # origin="upper": array row 0 (anterior) at the top, the usual radiological orientation.
    ax.imshow(ct_slice, cmap="gray", vmin=vmin, vmax=vmax, origin="upper", interpolation="nearest")
    ax.imshow(rgba, origin="upper", interpolation="nearest")

    sm = ScalarMappable(norm=Normalize(0.0, 1.0), cmap=RAINBOW)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Relative patch CAM: blue low \u2192 red high")

    notes = []
    for record, shape in records_with_cams:
        add_patch_outline(ax, record, z, shape)
        crop_z = int(record["crop_start_zyx"][0])
        if not (crop_z <= z < crop_z + int(shape[0])):
            continue
        line = f"patch {record['patch_id']} / candidate {record.get('candidate_id', '?')}"
        info = (signed_info or {}).get(int(record["patch_id"]))
        if info is not None:
            zero = info["zero_position"]
            line += f": raw CAM {info['raw_min']:+.2e} to {info['raw_max']:+.2e}"
            if math.isnan(zero):
                line += ", constant map"
            elif zero < 0.0:
                line += ", all raw values positive (zero is below the scale)"
            elif zero > 1.0:
                line += ", all raw values negative (zero is above the scale)"
            else:
                line += f", raw zero at {zero:.2f} of the scale (black/white bar on colorbar)"
                cbar.ax.axhline(zero, color="black", linewidth=3.0)
                cbar.ax.axhline(zero, color="white", linewidth=1.2)
        notes.append(line)

    ax.set_title(f"{attr.replace('_', ' ').title()} \u2014 full processed CT, axial Z={z}")
    ax.set_xlabel("CT X voxel index")
    ax.set_ylabel("CT Y voxel index")
    ax.set_xlim(-0.5, ct_volume.shape[2] - 0.5)
    ax.set_ylim(ct_volume.shape[1] - 0.5, -0.5)

    footer = (
        f"Dashed yellow = full {PATCH_SHAPE}-voxel patch footprint; magenta box = detector box, drawn only on slices "
        "inside its Z interval. Colors are relative within each patch map, not voxelwise attribute scores."
    )
    if notes:
        footer += "\n" + "\n".join(notes)
    fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=7)
    fig.tight_layout(rect=(0, 0.05 + 0.02 * len(notes), 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def create_montage(slice_paths: list[Path], out_path: Path, columns: int = 5) -> None:
    if not slice_paths:
        return

    tile_w, tile_h = 340, 340
    rows = math.ceil(len(slice_paths) / columns)
    montage = Image.new("RGB", (columns * tile_w, rows * tile_h), "black")

    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for index, path in enumerate(slice_paths):
        with Image.open(path) as source:
            thumb = source.convert("RGB")
            thumb.thumbnail((tile_w - 12, tile_h - 36), Image.Resampling.LANCZOS)

        tile = Image.new("RGB", (tile_w, tile_h), "black")
        draw = ImageDraw.Draw(tile)
        label = path.stem.replace("full_ct_gradcam_", "")
        draw.text((6, 5), label, fill="white", font=font)
        paste_x = (tile_w - thumb.width) // 2
        paste_y = 28 + (tile_h - 28 - thumb.height) // 2
        tile.paste(thumb, (paste_x, paste_y))
        montage.paste(tile, ((index % columns) * tile_w, (index // columns) * tile_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(out_path, quality=92)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0,1].")
    if args.window_width <= 0:
        raise ValueError("--window-width must be positive.")

    patient_root = DEFAULT_OUTPUT_ROOT / args.patient_id
    ct_path = Path(args.ct_volume).expanduser() if args.ct_volume else DEFAULT_OUTPUT_ROOT / args.patient_id / STAGE01_SCHEMA["filenames"]["volume_hu_cropped.npy"]
    manifest_path = (
        Path(args.manifest).expanduser()
        if args.manifest
        else DEFAULT_OUTPUT_ROOT / args.patient_id / STAGE02_PATCH_OUTPUTS["manifest"]
    )
    cam_root = (
        Path(args.cam_root).expanduser()
        if args.cam_root
        else DEFAULT_OUTPUT_ROOT / args.patient_id / STAGE05_PATHS["directory"]
    )
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else DEFAULT_OUTPUT_ROOT / args.patient_id / STAGE06_PATHS["directory"]
    )

    for path in (ct_path, manifest_path, cam_root):
        if not path.exists():
            raise FileNotFoundError(f"Required input path does not exist: {path}")

    try:
        ct_volume = np.load(ct_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Could not load Stage 01 CT NPY {ct_path}: {exc}. Re-run Stage 01 to regenerate a valid CT volume."
        ) from exc
    if ct_volume.ndim != 3:
        raise ValueError(f"CT volume must be 3D ZYX, got shape {ct_volume.shape}.")
    if any(int(dim) <= 0 for dim in ct_volume.shape):
        raise ValueError(f"CT volume has an invalid empty dimension: {ct_volume.shape}.")
    ct_shape = tuple(int(v) for v in ct_volume.shape)
    manifest_data = read_json(manifest_path)
    check_meta_schema_version(manifest_data, SCHEMA, context=str(manifest_path), require=False)
    validate_stage_version(manifest_data.get("schema_version"), STAGE02_SCHEMA["manifest_schema_version"], context="Stage 02 patch manifest")
    records = prepare_records(manifest_data)
    if not records:
        raise ValueError("No patch records were found in the manifest.")

    selected_attributes = ATTRIBUTES if args.attribute == "all" else (args.attribute,)
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "patient_id": args.patient_id,
        "meta_schema_version": SCHEMA["schema_version"],
        "coordinate_frame": SCHEMA["coordinate_format"]["ct_frame_space"],
        "array_axes": list(SCHEMA["array_axes"]),
        "ct_volume": str(ct_path),
        "ct_shape_zyx": list(ct_shape),
        "patch_manifest": str(manifest_path),
        "cam_root": str(cam_root),
        "output_dir": str(output_dir),
        "mapping": STAGE06_SCHEMA["mapping_rule"],
        "overlap_rule": STAGE06_SCHEMA["overlap_rule"],
        "display_palette_low_to_high": [item["name"] for item in PALETTE],
        "attributes": {},
    }

    print(f"Patient:  {args.patient_id}")
    print(f"CT:       {ct_path} shape={ct_shape}")
    print(f"Manifest: {manifest_path} patches={len(records)}")
    print(f"CAM root: {cam_root}")
    print(f"Output:   {output_dir}")
    print("Mapping patch coordinates using Stage 02 crop_start_zyx in the Stage 01 CT voxel frame.")

    for attribute in selected_attributes:
        print(f"\n[{attribute}] Loading per-patch CAMs...")
        sum_cam = np.zeros(ct_shape, dtype=np.float32)
        coverage_count = np.zeros(ct_shape, dtype=np.uint16)
        loaded_records: list[tuple[dict[str, Any], tuple[int, int, int]]] = []
        missing_patch_ids: list[int] = []
        used_patch_data: list[dict[str, Any]] = []
        signed_info: dict[int, dict[str, float] | None] = {}

        for record in records:
            pid = int(record["patch_id"])
            cam_path, cam_kind = find_patch_cam(cam_root, pid, attribute)
            if cam_path is None:
                missing_patch_ids.append(pid)
                print(f"  [SKIP] patch {pid}: no attribute CAM found.")
                continue

            cam = np.load(cam_path).astype(np.float32)
            if cam.ndim != 3:
                raise ValueError(f"CAM must be 3D: {cam_path} shape={cam.shape}")
            if not np.isfinite(cam).all():
                raise ValueError(f"CAM contains NaN/Inf: {cam_path}")
            if cam_kind == "raw":
                cam = normalize_raw_cam(cam)
            else:
                min_value, max_value = float(cam.min()), float(cam.max())
                if min_value < -1e-4 or max_value > 1.0001:
                    raise ValueError(
                        f"Rainbow-scaled CAM should be in [0,1], got [{min_value},{max_value}]: {cam_path}"
                    )
                cam = np.clip(cam, 0.0, 1.0)

            expected_size = record.get("patch_size_vox_zyx")
            if expected_size is not None:
                expected = tuple(int(v) for v in expected_size)
                if tuple(cam.shape) != expected:
                    raise ValueError(
                        f"CAM shape for patch {pid} is {cam.shape}; manifest patch_size_vox_zyx={expected}."
                    )

            crop_start = np.asarray(record["crop_start_zyx"], dtype=np.int64)
            crop_end_manifest = record.get("crop_end_zyx")
            if crop_end_manifest is not None:
                extent = np.rint(np.asarray(crop_end_manifest, dtype=np.float64)).astype(np.int64) - crop_start
                if extent.tolist() != [int(v) for v in cam.shape]:
                    raise ValueError(
                        f"patch {pid}: crop_end - crop_start = {extent.tolist()} but CAM shape is {list(cam.shape)}."
                    )
            signed_info[pid] = raw_signed_summary(cam_path, attribute)
            patch_shape = np.asarray(cam.shape, dtype=np.int64)
            crop_end = crop_start + patch_shape
            ct_start = np.maximum(crop_start, 0)
            ct_end = np.minimum(crop_end, np.asarray(ct_shape))

            if np.any(ct_end <= ct_start):
                print(f"  [SKIP] patch {pid}: its crop does not intersect the CT volume.")
                missing_patch_ids.append(pid)
                continue

            patch_start = ct_start - crop_start
            patch_end = patch_start + (ct_end - ct_start)
            ct_slices = tuple(slice(int(a), int(b)) for a, b in zip(ct_start, ct_end))
            patch_slices = tuple(slice(int(a), int(b)) for a, b in zip(patch_start, patch_end))

            sum_cam[ct_slices] += cam[patch_slices]
            coverage_count[ct_slices] += 1
            loaded_records.append((record, tuple(int(v) for v in cam.shape)))
            used_patch_data.append({
                "patch_id": pid,
                "candidate_id": record.get("candidate_id"),
                "cam_file": str(cam_path),
                "cam_kind": cam_kind,
                "crop_start_zyx": crop_start.tolist(),
                "cam_shape_zyx": list(cam.shape),
                "mapped_ct_start_zyx": ct_start.tolist(),
                "mapped_ct_end_zyx_exclusive": ct_end.tolist(),
                "raw_signed_summary": signed_info.get(pid),
            })
            print(
                f"  [OK] patch {pid}: {cam_path.name} -> CT "
                f"{ct_start.tolist()}:{ct_end.tolist()}"
            )

        if args.require_all_patches and missing_patch_ids:
            raise RuntimeError(
                f"{attribute}: CAMs missing for patches {missing_patch_ids}. "
                "No partial output was rendered for this attribute."
            )
        if not loaded_records:
            print(f"  [WARN] No usable CAM patches found for {attribute}; skipping output.")
            report["attributes"][attribute] = {
                "status": "skipped_no_cams",
                "missing_patch_ids": missing_patch_ids,
                "used_patches": [],
            }
            continue

        covered = coverage_count > 0
        sum_cam[covered] /= coverage_count[covered].astype(np.float32)
        sum_cam[~covered] = np.nan

        volume_path = output_dir / STAGE06_SCHEMA["volume_filename_pattern"].format(attribute=attribute)
        np.savez_compressed(
            volume_path,
            cam=sum_cam,
            coverage_count=coverage_count,
        )

        z_indices = np.flatnonzero(covered.any(axis=(1, 2))).astype(int).tolist()
        slice_dir = output_dir / "slices" / attribute
        slice_paths: list[Path] = []
        for z in z_indices:
            image_path = slice_dir / STAGE06_SCHEMA["slice_filename_pattern"].format(attribute=attribute, z=z)
            render_slice(
                ct_volume=ct_volume,
                cam_volume=sum_cam,
                coverage_count=coverage_count,
                z=z,
                attr=attribute,
                out_path=image_path,
                records_with_cams=loaded_records,
                signed_info=signed_info,
                alpha=args.alpha,
                window_level=args.window_level,
                window_width=args.window_width,
            )
            slice_paths.append(image_path)

        montage_path = output_dir / STAGE06_SCHEMA["montage_filename_pattern"].format(attribute=attribute)
        create_montage(slice_paths, montage_path)

        report["attributes"][attribute] = {
            "status": "complete",
            "cam_volume_npz": str(volume_path),
            "montage_png": str(montage_path),
            "slice_count": len(z_indices),
            "z_indices_with_cam": z_indices,
            "missing_patch_ids": missing_patch_ids,
            "used_patches": used_patch_data,
            "overlap_rule": STAGE06_SCHEMA["overlap_rule"],
            "cam_values_outside_coverage": "NaN in cam array; coverage_count=0",
        }
        print(f"  Saved mapped volume: {volume_path}")
        print(f"  Saved axial montage: {montage_path}")
        print(f"  Rendered {len(slice_paths)} full-CT axial slices.")

        del sum_cam, coverage_count

    report_path = output_dir / STAGE06_SCHEMA["report_filename"]
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport: {report_path}")
    print("Done. The script only read the CT, manifest and CAM arrays; it did not alter model or Stage 01/02 outputs.")


if __name__ == "__main__":
    main()
