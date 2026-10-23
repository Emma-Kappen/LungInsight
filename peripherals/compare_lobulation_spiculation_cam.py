from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from backend.models.nodule_ranker_3d_distributional_v1.spatial_v1.model_objective import (
    MODEL_TARGETS, SpatialObjectiveModel, preprocess_hu
)

root = Path("backend/output/LIDC-IDRI-0001/05_explainability/gradcam_signed_multiepsilon")
patch_path = Path(
    "backend/output/LIDC-IDRI-0001/02_detect_and_patch/patches/arrays/patch_000.npy"
)
checkpoint = Path(
    "backend/models/nodule_ranker_3d_distributional_v1/"
    "spatial_v1/checkpoints/hybrid/lobulation_spiculation/best.pth"
)
device = torch.device("cpu")

ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
arch = ckpt["architecture"]
model = SpatialObjectiveModel(
    arch["objective"], arch["attention_dim"],
    arch["head_hidden"], arch["dropout"]
).to(device)
model.load_state_dict(ckpt["model_state_dict"], strict=True)
model.eval()

patch = np.load(patch_path).astype(np.float32)
assert patch.shape == (64, 64, 64), patch.shape
x = torch.from_numpy(patch)[None, None].to(device)
layer = dict(model.named_modules())["encoder.stage3.1.conv2"]

def calculate_cam(attribute):
    captured = {}
    h1 = layer.register_forward_hook(
        lambda module, inputs, output: captured.__setitem__("activation", output)
    )
    h2 = layer.register_full_backward_hook(
        lambda module, grad_input, grad_output:
            captured.__setitem__("gradient", grad_output[0])
    )
    try:
        model.zero_grad(set_to_none=True)
        outputs = model(preprocess_hu(x))
        score = model.expected_ratings(outputs)[0, MODEL_TARGETS.index(attribute)]
        score.backward()

        activation = captured["activation"].detach()
        gradient = captured["gradient"].detach()
        weights = gradient.mean(dim=(2, 3, 4), keepdim=True)
        raw = F.interpolate(
            (weights * activation).sum(dim=1, keepdim=True),
            size=(64, 64, 64),
            mode="trilinear",
            align_corners=False,
        )[0, 0].cpu().numpy()

        scale = np.abs(raw).max()
        if not np.isfinite(raw).all() or scale < 1e-12:
            raise RuntimeError(f"Invalid CAM for {attribute}")
        return raw / scale, float(score.detach())
    finally:
        h1.remove()
        h2.remove()

maps = {}
for attribute in ("lobulation", "spiculation"):
    cam, score = calculate_cam(attribute)
    maps[attribute] = cam
    saved_path = root / f"gradcam_{attribute}.npy"
    if saved_path.exists():
        saved = np.load(saved_path)
        print(
            f"{attribute}: recomputed-vs-saved correlation="
            f"{np.corrcoef(cam.ravel(), saved.ravel())[0,1]:.8f}"
        )
    print(
        f"{attribute}: score={score:.6f}, min={cam.min():.5f}, "
        f"max={cam.max():.5f}, mean={cam.mean():.5f}, std={cam.std():.5f}"
    )

a, b = maps["lobulation"].ravel(), maps["spiculation"].ravel()
print("\nIndependent CAM comparison")
print("Pearson correlation:", float(np.corrcoef(a, b)[0, 1]))
print("Cosine similarity:", float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b))))
print("Relative difference:", float(np.linalg.norm(a-b) / max(np.linalg.norm(a), 1e-12)))
