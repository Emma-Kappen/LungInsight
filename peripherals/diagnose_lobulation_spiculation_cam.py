from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from backend.models.nodule_ranker_3d_distributional_v1.spatial_v1.model_objective import (
    MODEL_TARGETS,
    SpatialObjectiveModel,
    preprocess_hu,
)

ROOT = Path(
    "backend/output/LIDC-IDRI-0001/05_explainability/"
    "gradcam_signed_multiepsilon"
)
PATCH_PATH = Path(
    "backend/output/LIDC-IDRI-0001/02_detect_and_patch/"
    "patches/arrays/patch_000.npy"
)
CHECKPOINT = Path(
    "backend/models/nodule_ranker_3d_distributional_v1/"
    "spatial_v1/checkpoints/hybrid/lobulation_spiculation/best.pth"
)
ATTRIBUTES = ("lobulation", "spiculation")
LAYER_NAME = "encoder.stage3.1.conv2"


def correlation(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def cosine(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 1e-15 else float("nan")


def relative_difference(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-15))


def describe(name, values):
    values = np.asarray(values)
    q = np.percentile(values, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    print(f"\n{name}")
    print(f"  shape={values.shape}")
    print(f"  min={values.min():.8g}, max={values.max():.8g}")
    print(f"  mean={values.mean():.8g}, std={values.std():.8g}")
    print("  percentiles [0,1,5,25,50,75,95,99,100]:")
    print(" ", np.array2string(q, precision=7))


def main():
    for path in (ROOT, PATCH_PATH, CHECKPOINT):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    device = torch.device("cpu")
    checkpoint = torch.load(
        CHECKPOINT, map_location=device, weights_only=False
    )
    arch = checkpoint["architecture"]

    model = SpatialObjectiveModel(
        arch["objective"],
        arch["attention_dim"],
        arch["head_hidden"],
        arch["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    patch = np.load(PATCH_PATH).astype(np.float32)
    if patch.shape != (64, 64, 64):
        raise ValueError(f"Expected patch shape (64,64,64), got {patch.shape}")

    x = torch.from_numpy(patch)[None, None].to(device)
    layer = dict(model.named_modules()).get(LAYER_NAME)
    if layer is None:
        raise KeyError(f"Layer not found: {LAYER_NAME}")

    results = {}

    for attribute in ATTRIBUTES:
        captured = {}
        hooks = [
            layer.register_forward_hook(
                lambda module, inputs, output:
                    captured.__setitem__("activation", output)
            ),
            layer.register_full_backward_hook(
                lambda module, grad_input, grad_output:
                    captured.__setitem__("gradient", grad_output[0])
            ),
        ]

        try:
            model.zero_grad(set_to_none=True)
            outputs = model(preprocess_hu(x))
            score = model.expected_ratings(outputs)[
                0, MODEL_TARGETS.index(attribute)
            ]
            score.backward()

            activation = captured["activation"].detach()
            gradient = captured["gradient"].detach()

            # Standard Grad-CAM channel weights: spatial mean of gradients.
            weights = gradient.mean(dim=(2, 3, 4)).squeeze(0)

            # Raw CAM before per-map max-absolute normalization.
            low_resolution_cam = (
                weights[None, :, None, None, None] * activation
            ).sum(dim=1, keepdim=True)

            raw_cam = F.interpolate(
                low_resolution_cam,
                size=(64, 64, 64),
                mode="trilinear",
                align_corners=False,
            )[0, 0].cpu().numpy()

            if not np.isfinite(raw_cam).all():
                raise RuntimeError(f"Non-finite raw CAM for {attribute}")

            results[attribute] = {
                "score": float(score.detach()),
                "weights": weights.cpu().numpy().copy(),
                "raw_cam": raw_cam,
                "activation_shape": tuple(activation.shape),
            }
        finally:
            for hook in hooks:
                hook.remove()

    lob = results["lobulation"]
    spi = results["spiculation"]

    print("=" * 72)
    print("READ-ONLY LOBULATION / SPICULATION GRAD-CAM DIAGNOSTICS")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Patch:      {PATCH_PATH}")
    print(f"Layer:      {LAYER_NAME}")
    print(f"Activation shape: {lob['activation_shape']}")
    print("=" * 72)

    # Diagnostic 1: raw, unnormalized CAM arrays.
    print("\nDIAGNOSTIC 1 — RAW CAMS BEFORE NORMALIZATION")
    for attribute in ATTRIBUTES:
        result = results[attribute]
        raw = result["raw_cam"]
        describe(f"{attribute}: raw CAM", raw)
        print(f"  expected rating={result['score']:.8f}")

        saved_path = ROOT / f"gradcam_{attribute}.npy"
        if saved_path.exists():
            saved = np.load(saved_path)
            print(
                f"  raw CAM vs saved CAM correlation="
                f"{correlation(raw, saved):.8f}"
            )
            print(
                "  NOTE: saved CAM is normalized, so compare correlation "
                "and spatial pattern, not absolute scale."
            )

    a, b = lob["raw_cam"], spi["raw_cam"]
    print("\nRaw CAM pair comparison:")
    print(f"  Pearson correlation: {correlation(a, b):.8f}")
    print(f"  Cosine similarity:   {cosine(a, b):.8f}")
    print(f"  Relative difference: {relative_difference(a, b):.8f}")

    # Diagnostic 2: channel weights before multiplication by activations.
    print("\nDIAGNOSTIC 2 — GRAD-CAM CHANNEL WEIGHTS")
    wa, wb = lob["weights"], spi["weights"]
    describe("lobulation channel weights", wa)
    describe("spiculation channel weights", wb)
    print(f"\nChannel-weight vector length: {wa.size}")
    print(f"  Pearson correlation: {correlation(wa, wb):.8f}")
    print(f"  Cosine similarity:   {cosine(wa, wb):.8f}")
    print(f"  Relative difference: {relative_difference(wa, wb):.8f}")

    # Diagnostic 3: spatial distribution of the difference, not just one
    # global correlation. Values are descriptive diagnostics, not thresholds.
    print("\nDIAGNOSTIC 3 — WHERE DO THE MAPS DIFFER?")
    difference = a - b
    shared = 0.5 * (a + b)
    abs_difference = np.abs(difference)
    abs_shared = np.abs(shared)

    describe("lobulation minus spiculation (raw CAM difference)", difference)
    print(f"\n  Difference norm: {np.linalg.norm(difference):.8g}")
    print(f"  Shared-map norm: {np.linalg.norm(shared):.8g}")
    print(
        "  Difference/shared norm ratio: "
        f"{np.linalg.norm(difference) / max(np.linalg.norm(shared), 1e-15):.8f}"
    )

    # Summarize absolute difference by three spatial axes. This reveals
    # whether differences are diffuse or concentrated in particular slices.
    for axis, axis_name in enumerate(("z", "y", "x")):
        reduce_axes = tuple(i for i in range(3) if i != axis)
        profile = abs_difference.mean(axis=reduce_axes)
        top = np.argsort(profile)[-5:][::-1]
        print(
            f"  Top {axis_name}-slices by mean absolute difference: "
            + ", ".join(f"{int(i)} ({profile[i]:.6g})" for i in top)
        )

    # Compare the strongest difference voxels against the full-volume
    # distribution. This is descriptive; it does not assume a nodule mask.
    threshold = np.percentile(abs_difference, 95)
    top_region = abs_difference >= threshold
    print(
        f"\n  Top 5% difference voxels: {top_region.mean() * 100:.2f}% "
        f"of patch"
    )
    print(
        "  Mean |difference| in top 5%: "
        f"{abs_difference[top_region].mean():.8g}"
    )
    print(
        "  Mean |difference| elsewhere: "
        f"{abs_difference[~top_region].mean():.8g}"
    )

    print("\nCompleted. No model, checkpoint, H5, split, or saved CAM was changed.")


if __name__ == "__main__":
    main()
