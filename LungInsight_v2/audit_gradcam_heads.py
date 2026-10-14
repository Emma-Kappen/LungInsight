import numpy as np
import torch

from backend.models.nodule_ranker_3d_distributional_v1.spatial_v1.model_objective import (
    MODEL_TARGETS,
    SpatialObjectiveModel,
    preprocess_hu,
)

checkpoint = (
    "backend/models/nodule_ranker_3d_distributional_v1/"
    "spatial_v1/checkpoints/hybrid/lobulation_spiculation/best.pth"
)
patch_path = (
    "backend/output/LIDC-IDRI-0001/"
    "02_detect_and_patch/patches/arrays/patch_000.npy"
)
layer_name = "encoder.stage3.1.conv2"

ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
arch = ckpt["architecture"]
model = SpatialObjectiveModel(
    arch["objective"], arch["attention_dim"],
    arch["head_hidden"], arch["dropout"],
)
model.load_state_dict(ckpt["model_state_dict"], strict=True)
model.eval()

patch = np.load(patch_path).astype(np.float32)
assert patch.shape == (64, 64, 64), patch.shape
x = torch.from_numpy(patch)[None, None].requires_grad_(True)

modules = dict(model.named_modules())
assert layer_name in modules, f"Missing layer: {layer_name}"
captured = {}

def forward_hook(module, inputs, output):
    captured["activation"] = output

def backward_hook(module, grad_input, grad_output):
    captured["gradient"] = grad_output[0]

h1 = modules[layer_name].register_forward_hook(forward_hook)
h2 = modules[layer_name].register_full_backward_hook(backward_hook)

results = {}
try:
    print("MODEL_TARGETS:", MODEL_TARGETS)

    for attribute in ("lobulation", "spiculation"):
        captured.clear()
        model.zero_grad(set_to_none=True)

        outputs = model(preprocess_hu(x))
        ratings = model.expected_ratings(outputs)
        score = ratings[0, MODEL_TARGETS.index(attribute)]
        score.backward()

        activation = captured["activation"].detach()
        gradient = captured["gradient"].detach().float()
        grad_np = gradient.cpu().numpy().copy()
        results[attribute] = {
            "score": float(score.detach()),
            "gradient": grad_np,
        }

        print(
            f"\n{attribute}: expected_rating={float(score.detach()):.6f}"
            f"\n  activation_shape={tuple(activation.shape)}"
            f"\n  gradient_shape={tuple(gradient.shape)}"
            f"\n  gradient_min={gradient.min().item():.8g}"
            f"\n  gradient_max={gradient.max().item():.8g}"
            f"\n  gradient_std={gradient.std().item():.8g}"
            f"\n  gradient_abs_mean={gradient.abs().mean().item():.8g}"
        )

    ga = results["lobulation"]["gradient"].ravel()
    gb = results["spiculation"]["gradient"].ravel()

    corr = np.corrcoef(ga, gb)[0, 1]
    relative_difference = (
        np.linalg.norm(ga - gb) / max(np.linalg.norm(ga), 1e-12)
    )
    print(f"\nLayer-gradient Pearson correlation: {corr:.8f}")
    print(f"Layer-gradient relative difference: {relative_difference:.8g}")
finally:
    h1.remove()
    h2.remove()
