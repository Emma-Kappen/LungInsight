from pathlib import Path
import numpy as np

root = Path(
    r"backend/output/LIDC-IDRI-0001/05_explainability/gradcam_signed_multiepsilon"
)
names = [
    "subtlety", "sphericity", "margin",
    "lobulation", "spiculation", "texture",
]
cams = {}

# Existing patch-000 detector box in patch-local ZYX coordinates,
# using half-open bounds. This is not a segmentation mask.
z0, y0, x0 = 27, 19, 19
z1, y1, x1 = 36, 45, 45

for name in names:
    p = root / f"gradcam_{name}.npy"
    if not p.exists():
        print(f"MISSING: {p}")
        continue

    cam = np.load(p).astype(np.float64)
    if cam.shape != (64, 64, 64) or not np.isfinite(cam).all():
        print(f"INVALID: {name}, shape={cam.shape}")
        continue

    cams[name] = cam
    abs_cam = np.abs(cam)
    flat = abs_cam.ravel()
    total = flat.sum()
    k = max(1, int(np.ceil(flat.size * 0.10)))
    top_mass = np.partition(flat, flat.size-k)[-k:].sum() / max(total, 1e-12)

    roi = abs_cam[z0:z1, y0:y1, x0:x1]
    roi_mass = roi.sum() / max(total, 1e-12)

    print(
        f"{name:12s} "
        f"min={cam.min(): .4f} max={cam.max(): .4f} "
        f"mean={cam.mean(): .4f} std={cam.std():.4f} "
        f"positive={(cam > 1e-6).mean():.3f} "
        f"negative={(cam < -1e-6).mean():.3f} "
        f"top10_abs_mass={top_mass:.3f} "
        f"bbox_abs_mass={roi_mass:.3f}"
    )

print("\nPairwise Pearson correlation of signed voxel maps:")
for i, a in enumerate(names):
    for b in names[i+1:]:
        if a not in cams or b not in cams:
            continue
        corr = np.corrcoef(cams[a].ravel(), cams[b].ravel())[0, 1]
        print(f"{a:12s} vs {b:12s}: {corr:.4f}")
