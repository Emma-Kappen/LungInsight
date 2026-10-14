from pathlib import Path
import argparse
import json
import sys

IMAGING_DIR = Path(__file__).resolve().parent
if IMAGING_DIR.name in {"stages", "inference", "visualization", "validation"}:
    IMAGING_DIR = IMAGING_DIR.parent
if str(IMAGING_DIR) not in sys.path:
    sys.path.insert(0, str(IMAGING_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle

import numpy as np
import torch
import torch.nn.functional as F

V2_ROOT = IMAGING_DIR.parent.parent
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))
from pipeline_schema import get_required, load_schema, check_meta_schema_version, validate_stage_version

SCHEMA = load_schema(__file__)
STAGE05_SCHEMA = get_required(SCHEMA, "stage_05_gradcam", expected_type=dict)
STAGE03_SCHEMA = get_required(SCHEMA, "stage_03_nodule_classification", expected_type=dict)
STAGE03_MODEL_SCHEMA = get_required(STAGE03_SCHEMA, "model", expected_type=dict)
VISUALIZATION_SCHEMA = get_required(SCHEMA, "visualization", expected_type=dict)
STAGE02_PATHS = get_required(SCHEMA, "paths.stage_02", expected_type=dict)
STAGE02_SCHEMA = get_required(SCHEMA, "stage_02_detect_and_patch", expected_type=dict)
STAGE05_PATHS = get_required(SCHEMA, "paths.stage_05", expected_type=dict)
STAGE02_PATCH_OUTPUTS = get_required(SCHEMA, "stage_02_detect_and_patch.patch_outputs", expected_type=dict)
STAGE02_FILES = get_required(SCHEMA, "stage_02_detect_and_patch.filenames", expected_type=dict)
DEFAULT_OUTPUT_ROOT = V2_ROOT / get_required(SCHEMA, "paths.output_root")

try:
    from backend.models.nodule_ranker_3d_distributional_v1.spatial_v1.model_objective import (
        MODEL_TARGETS,
        SpatialObjectiveModel,
        preprocess_hu,
    )
except ImportError as exc:
    raise ImportError(
        "Stage 05 requires the existing inference model definition at "
        "backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/model_objective.py. "
        "Restore that file from the trained-model project; do not retrain or modify checkpoints."
    ) from exc


def validate_target_contract(model_targets, required_attributes):
    """Require every requested Grad-CAM attribute to exist in the frozen model.

    The shared spatial-objective model exposes eight targets, whereas this
    pipeline's Stage 03 schema intentionally renders only six. Exact tuple
    equality is therefore incorrect; a subset check preserves the checkpoint's
    target order and allows Grad-CAM to select the matching output index.
    """
    target_names = tuple(str(name) for name in model_targets)
    required_names = tuple(str(name) for name in required_attributes)
    if len(target_names) != len(set(target_names)):
        raise RuntimeError(
            f"Grad-CAM model target list contains duplicate names: {target_names}. "
            "Restore the matching existing model definition; do not modify checkpoints."
        )
    missing = [name for name in required_names if name not in target_names]
    if missing:
        raise RuntimeError(
            "Stage 05 requests attributes that are absent from the frozen Grad-CAM model: "
            f"{missing}. Requested attributes={required_names}; model targets={target_names}. "
            "Restore a compatible existing model definition/checkpoint; do not retrain."
        )
    return {name: target_names.index(name) for name in required_names}


ATTRIBUTES = tuple(get_required(SCHEMA, str(STAGE05_SCHEMA["attributes_ref"]), expected_type=list))
MODEL_TARGET_INDEX = validate_target_contract(MODEL_TARGETS, ATTRIBUTES)
CHECKPOINTS = dict(get_required(STAGE05_SCHEMA, "checkpoint_by_attribute", expected_type=dict))
if set(CHECKPOINTS) != set(ATTRIBUTES):
    raise RuntimeError("Stage 05 checkpoint mapping keys must exactly match the shared Stage 03 attribute list in meta_schema.json.")

# The patch shape is owned by meta_schema.json and must match the frozen Stage 03 model contract.
_patch_shape_raw = get_required(STAGE05_SCHEMA, "patch_shape_zyx", expected_type=list)
if len(_patch_shape_raw) != 3 or any(type(size) is not int or size <= 0 for size in _patch_shape_raw):
    raise RuntimeError(
        "meta_schema.json stage_05_gradcam.patch_shape_zyx must contain three positive integers."
    )
PATCH_SHAPE = tuple(_patch_shape_raw)
STAGE03_PATCH_SHAPE = tuple(get_required(STAGE03_SCHEMA, "inputs.patch_shape_zyx", expected_type=list))
if PATCH_SHAPE != STAGE03_PATCH_SHAPE:
    raise RuntimeError(
        "Grad-CAM patch shape disagrees with Stage 03's frozen model input contract: "
        f"Stage 05={PATCH_SHAPE}, Stage 03={STAGE03_PATCH_SHAPE}. "
        "Correct the canonical metadata schema; do not resize patches or alter checkpoints."
    )

TARGET_LAYER = str(get_required(STAGE05_SCHEMA, "target_layer"))
PALETTE = get_required(SCHEMA, str(STAGE05_SCHEMA["palette_ref"]), expected_type=list)
PATCH_EDGE_COLOR = str(VISUALIZATION_SCHEMA["patch_outline_hex"])

# Low -> high relative CAM value.
RAINBOW = LinearSegmentedColormap.from_list(
    "blue_green_yellow_orange_red",
    [(i / (len(PALETTE) - 1), item["hex"]) for i, item in enumerate(PALETTE)],
    N=256,
)

# PATCH_EDGE_COLOR is sourced from meta_schema.json; this outlines the whole patch, not a segmentation.


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate rainbow Grad-CAM maps for the six Stage 03 attributes. "
            "The original model, checkpoint and saved CAM files are not changed."
        )
    )
    parser.add_argument(
        "--patient-id",
        default=str(get_required(SCHEMA, "defaults.patient_id")),
        help="Patient ID used for schema-derived default patch/output paths.",
    )
    parser.add_argument(
        "--patch",
        default=None,
        help="Path to a raw-HU patch; defaults to Stage 02 patch_000 for --patient-id.",
    )
    parser.add_argument(
        "--checkpoint-root",
        default=(
            str(V2_ROOT / STAGE03_MODEL_SCHEMA["checkpoint_root"])
        ),
        help="Directory containing the per-attribute checkpoint folders.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="New output directory for images, arrays and metadata.",
    )
    parser.add_argument(
        "--attribute",
        choices=("all",) + ATTRIBUTES,
        default="all",
        help="Attribute to render, or all six (default: all).",
    )
    parser.add_argument(
        "--manifest", default=None,
        help="Optional Stage 02 manifest for batch Grad-CAM; writes one patch_<id> subdirectory per record.",
    )
    parser.add_argument(
        "--batch-limit", type=int, default=0,
        help="Maximum records from --manifest to process; 0 processes every record in that manifest.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=float(STAGE05_SCHEMA["default_alpha"]),
        help="Heatmap opacity from 0 to 1 (default from meta_schema.json).",
    )
    return parser.parse_args()


def load_model(checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Malformed checkpoint {checkpoint_path}: expected a dictionary.")
    if not isinstance(checkpoint.get("architecture"), dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
        raise ValueError(
            f"Checkpoint {checkpoint_path} lacks 'architecture' or 'model_state_dict'. "
            "Restore a checkpoint produced by the existing spatial_v1 objective training run; do not retrain or edit it."
        )
    arch = checkpoint["architecture"]
    try:
        model = SpatialObjectiveModel(
            arch["objective"], arch["attention_dim"], arch["head_hidden"], arch["dropout"],
        ).to(device)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Checkpoint architecture metadata is incompatible in {checkpoint_path}: {exc}") from exc

    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"Checkpoint weights are incompatible with the existing SpatialObjectiveModel: {checkpoint_path}: {exc}") from exc
    model.eval()
    return model


def calculate_cam(model, patch_hu, attribute, device):
    """Return raw signed Grad-CAM, expected rating and target-layer metadata."""
    modules = dict(model.named_modules())
    if TARGET_LAYER not in modules:
        raise KeyError(f"Target layer not found: {TARGET_LAYER}")

    layer = modules[TARGET_LAYER]
    captured = {}

    def capture_activation(module, inputs, output):
        captured["activation"] = output

    def capture_gradient(module, grad_input, grad_output):
        captured["gradient"] = grad_output[0]

    forward_handle = layer.register_forward_hook(capture_activation)
    backward_handle = layer.register_full_backward_hook(capture_gradient)

    try:
        x = torch.from_numpy(patch_hu)[None, None].to(device)
        model.zero_grad(set_to_none=True)

        outputs = model(preprocess_hu(x))
        score = model.expected_ratings(outputs)[
            0, MODEL_TARGET_INDEX[attribute]
        ]

        score.backward()

        activation = captured["activation"].detach()
        gradient = captured["gradient"].detach()

        # Grad-CAM: average gradients spatially to obtain channel weights.
        weights = gradient.mean(dim=(2, 3, 4), keepdim=True)

        low_resolution_cam = (weights * activation).sum(
            dim=1, keepdim=True
        )

        raw_cam = F.interpolate(
            low_resolution_cam,
            size=PATCH_SHAPE,
            mode="trilinear",
            align_corners=False,
        )[0, 0].cpu().numpy()

        if raw_cam.shape != PATCH_SHAPE:
            raise ValueError(f"Unexpected CAM shape: {raw_cam.shape}")
        if not np.isfinite(raw_cam).all():
            raise ValueError(f"CAM contains NaN or Inf for {attribute}")

        score_value = float(score.detach().cpu())
        layer_shape = tuple(activation.shape)

        return raw_cam, score_value, layer_shape

    finally:
        forward_handle.remove()
        backward_handle.remove()


def normalize_for_display(raw_cam):
    """
    Map the lowest raw CAM value to blue and the highest to red.

    This is a within-map relative scale. It is not a voxelwise prediction
    of the attribute and does not preserve the sign as a separate color.
    """
    lower = float(np.min(raw_cam))
    upper = float(np.max(raw_cam))

    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError("Cannot visualize non-finite CAM values.")

    if upper - lower < 1e-12:
        return np.full(raw_cam.shape, 0.5, dtype=np.float32)

    result = (raw_cam - lower) / (upper - lower)
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def add_patch_outline(ax):
    ax.add_patch(
        Rectangle(
            (-0.5, -0.5),
            PATCH_SHAPE[2],
            PATCH_SHAPE[1],
            linewidth=2.0,
            edgecolor=PATCH_EDGE_COLOR,
            facecolor="none",
        )
    )


def draw_attribute(
    patch_hu,
    raw_cam,
    display_cam,
    attribute,
    score,
    layer_shape,
    output_dir,
    alpha,
    checkpoint_path,
):
    # Arrays are ordered Z,Y,X. All displayed slices are central slices.
    z = patch_hu.shape[0] // 2
    y = patch_hu.shape[1] // 2
    x = patch_hu.shape[2] // 2

    planes = [
        (
            "Axial",
            patch_hu[z, :, :],
            display_cam[z, :, :],
            f"Z = {z}",
        ),
        (
            "Coronal",
            patch_hu[:, y, :],
            display_cam[:, y, :],
            f"Y = {y}",
        ),
        (
            "Sagittal",
            patch_hu[:, :, x],
            display_cam[:, :, x],
            f"X = {x}",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    heatmap_mappable = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=RAINBOW)
    heatmap_mappable.set_array([])

    for ax, (plane_name, image, heatmap, coordinate) in zip(axes, planes):
        ax.imshow(
            image,
            cmap="gray",
            vmin=-1000,
            vmax=400,
            origin="lower",
            interpolation="nearest",
        )
        ax.imshow(
            heatmap,
            cmap=RAINBOW,
            norm=Normalize(vmin=0, vmax=1),
            alpha=alpha,
            origin="lower",
            interpolation="bilinear",
        )
        add_patch_outline(ax)
        ax.set_title(f"{plane_name} ({coordinate})")
        ax.set_xlabel("Patch pixels")
        ax.set_ylabel("Patch pixels")
        ax.set_xlim(-0.5, 63.5)
        ax.set_ylim(-0.5, 63.5)

    fig.suptitle(
        f"{attribute.replace('_', ' ').title()} Grad-CAM\n"
        f"Expected ordinal rating: {score:.3f}",
        fontsize=14,
    )

    colorbar = fig.colorbar(
        heatmap_mappable,
        ax=list(axes),
        fraction=0.025,
        pad=0.025,
        ticks=[0, 0.25, 0.5, 0.75, 1],
    )
    colorbar.ax.set_yticklabels(
        ["Lowest", "Low-mid", "Middle", "High-mid", "Highest"]
    )
    colorbar.set_label("Relative CAM value within this attribute map")

    fig.text(
        0.5,
        0.015,
        "Yellow outline = entire 64x64 patch; not a nodule segmentation. "
        "Colors show relative CAM values, not voxelwise attribute ratings.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.05, 0.95, 0.90))

    png_path = output_dir / STAGE05_SCHEMA["filenames"]["image_pattern"].format(attribute=attribute)
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Preserve the raw signed CAM for numerical inspection and future plots.
    raw_path = output_dir / STAGE05_SCHEMA["filenames"]["raw_cam_pattern"].format(attribute=attribute)
    display_path = output_dir / STAGE05_SCHEMA["filenames"]["scaled_cam_pattern"].format(attribute=attribute)
    np.save(raw_path, raw_cam.astype(np.float32))
    np.save(display_path, display_cam.astype(np.float32))

    metadata = {
        "meta_schema_version": SCHEMA["schema_version"],
        "attribute": attribute,
        "target_output_index": int(MODEL_TARGET_INDEX[attribute]),
        "target_scalar_definition": STAGE05_SCHEMA["target_scalar_definition"],
        "target_rating_range": get_required(
            SCHEMA, f"stage_03_nodule_classification.attribute_scales.{attribute}.rating_range", expected_type=list
        ),
        "raw_cam_formula": STAGE05_SCHEMA["raw_cam_formula"],
        "gradient_polarity_semantics": STAGE05_SCHEMA["gradient_polarity_semantics"],
        "attribute_specific_polarity_inversion": False,
        "expected_ordinal_rating": score,
        "checkpoint": str(checkpoint_path),
        "target_layer": TARGET_LAYER,
        "activation_shape": list(layer_shape),
        "patch_shape_zyx": list(patch_hu.shape),
        "raw_cam_min": float(raw_cam.min()),
        "raw_cam_max": float(raw_cam.max()),
        "raw_cam_mean": float(raw_cam.mean()),
        "raw_cam_std": float(raw_cam.std()),
        "display_scaling": "per-attribute min-max to [0,1]",
        "palette_low_to_high": [item["name"] for item in PALETTE],
        "heatmap_alpha": alpha,
        "interpretation": (
            "Colors represent relative Grad-CAM values within this patch. "
            "They are not voxelwise predictions of the attribute. "
            "The raw signed array is saved separately."
        ),
        "files": {
            "image": png_path.name,
            "raw_signed_cam": raw_path.name,
            "rainbow_scaled_cam": display_path.name,
        },
    }

    metadata_path = output_dir / STAGE05_SCHEMA["filenames"]["metadata_pattern"].format(attribute=attribute)
    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(
        f"[OK] {attribute}: rating={score:.4f}, "
        f"raw CAM range=[{raw_cam.min():.6g}, {raw_cam.max():.6g}]"
    )
    print(f"     Image: {png_path}")


def _resolve_manifest_patch(record: dict, manifest_path: Path) -> Path:
    try:
        patch_id = int(record["patch_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Malformed Stage 02 manifest record missing integer patch_id: {record!r}") from exc
    candidates: list[Path] = []
    raw = record.get("array_path")
    if raw:
        raw_path = Path(str(raw)).expanduser()
        candidates.append(raw_path if raw_path.is_absolute() else manifest_path.parent / raw_path)
    filename = f"patch_{patch_id:03d}.npy"
    candidates.extend((manifest_path.parent / "arrays" / filename, manifest_path.parent.parent / "arrays" / filename))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Patch array for manifest patch_id={patch_id} was not found. Tried: "
        + "; ".join(str(path) for path in candidates)
    )


def _generate_for_patch(
    patch_path: Path, output_dir: Path, checkpoint_root: Path, selected: tuple[str, ...],
    alpha: float, device: torch.device, model_cache: dict[Path, torch.nn.Module],
) -> None:
    try:
        patch_hu = np.load(patch_path, allow_pickle=False).astype(np.float32)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load patch NPY {patch_path}: {exc}") from exc
    if patch_hu.shape != PATCH_SHAPE:
        raise ValueError(f"Expected raw-HU patch shape {PATCH_SHAPE}, got {patch_hu.shape}: {patch_path}")
    if not np.isfinite(patch_hu).all():
        raise ValueError(f"Patch contains NaN or Inf: {patch_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nPatch: {patch_path}\nOutput: {output_dir}")
    for attribute in selected:
        checkpoint_path = (checkpoint_root / CHECKPOINTS[attribute]).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint for attribute {attribute!r} not found: {checkpoint_path}. "
                "Restore the existing trained checkpoint folder or pass --checkpoint-root; no retraining is performed."
            )
        model = model_cache.get(checkpoint_path)
        if model is None:
            model = load_model(checkpoint_path, device)
            model_cache[checkpoint_path] = model
        print(f"Generating {attribute}...")
        raw_cam, score, layer_shape = calculate_cam(model, patch_hu, attribute, device)
        display_cam = normalize_for_display(raw_cam)
        draw_attribute(
            patch_hu=patch_hu,
            raw_cam=raw_cam,
            display_cam=display_cam,
            attribute=attribute,
            score=score,
            layer_shape=layer_shape,
            output_dir=output_dir,
            alpha=alpha,
            checkpoint_path=checkpoint_path,
        )


def main():
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1.")
    if args.batch_limit < 0:
        raise ValueError("--batch-limit must be >= 0.")
    if args.manifest and args.patch:
        raise ValueError("Use either --manifest (batch mode) or --patch (single-patch mode), not both.")

    default_patch = (
        DEFAULT_OUTPUT_ROOT / args.patient_id / STAGE02_PATCH_OUTPUTS["array_directory"]
        / STAGE02_FILES["patch_array_pattern"].format(patch_id=0)
    )
    patch_path = Path(args.patch).expanduser().resolve() if args.patch else default_patch.resolve()
    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    default_output = DEFAULT_OUTPUT_ROOT / args.patient_id / STAGE05_PATHS["directory"]
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output.resolve()
    device = torch.device("cpu")
    selected = ATTRIBUTES if args.attribute == "all" else (args.attribute,)
    model_cache: dict[Path, torch.nn.Module] = {}

    extra_targets = tuple(name for name in MODEL_TARGETS if name not in ATTRIBUTES)
    if extra_targets:
        print("[info] The frozen model also exposes targets outside the Stage 05 attribute contract; rendering only: " + ", ".join(ATTRIBUTES))
    print("Device: CPU")
    print("Targets: " + ", ".join(selected))
    print("Palette: " + " -> ".join(str(item["name"]) for item in PALETTE))

    if args.manifest:
        manifest_path = Path(args.manifest).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Stage 02 manifest not found: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed Stage 02 manifest {manifest_path}: {exc}") from exc
        if not isinstance(manifest, dict) or not isinstance(manifest.get("patches"), list):
            raise ValueError(f"{manifest_path}: expected an object with a 'patches' list.")
        check_meta_schema_version(manifest, SCHEMA, context=str(manifest_path), require=False)
        validate_stage_version(
            manifest.get("schema_version"), STAGE02_SCHEMA["manifest_schema_version"],
            context="Stage 02 diagnostic patch manifest",
        )
        patches = manifest["patches"]
        ids = [item.get("patch_id") for item in patches if isinstance(item, dict)]
        if len(ids) != len(patches) or any(not isinstance(pid, int) for pid in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"{manifest_path}: each patch must be an object with a unique integer patch_id.")
        patches = sorted(patches, key=lambda item: int(item["patch_id"]))
        if args.batch_limit:
            patches = patches[:args.batch_limit]
        if not patches:
            raise ValueError(f"No patch records selected from {manifest_path}.")
        print(f"Batch manifest: {manifest_path}; processing {len(patches)} patch(es).")
        for index, record in enumerate(patches, start=1):
            record_path = _resolve_manifest_patch(record, manifest_path)
            patch_id = int(record["patch_id"])
            patch_dir_pattern = str(get_required(STAGE05_SCHEMA, "batch_processing.patch_output_directory_pattern"))
            per_patch_output = output_dir / patch_dir_pattern.format(patch_id=patch_id)
            print(f"\n[LoG Grad-CAM] {index}/{len(patches)} patch_id={patch_id}", flush=True)
            _generate_for_patch(record_path, per_patch_output, checkpoint_root, selected, args.alpha, device, model_cache)
        print(f"\nBatch Grad-CAM complete: {len(patches)} patch(es); loaded {len(model_cache)} unique checkpoint file(s).")
    else:
        if not patch_path.is_file():
            raise FileNotFoundError(f"Patch not found: {patch_path}")
        _generate_for_patch(patch_path, output_dir, checkpoint_root, selected, args.alpha, device, model_cache)
        print("\nAll requested Grad-CAM maps generated.")
    print("No checkpoint, H5, split, or existing CAM output was modified.")


if __name__ == "__main__":
    main()
