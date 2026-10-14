import torch

from backend.models.nodule_ranker_3d_distributional_v1.spatial_v1.model_objective import (
    SpatialObjectiveModel,
)

checkpoint = (
    "backend/models/nodule_ranker_3d_distributional_v1/"
    "spatial_v1/checkpoints/hybrid/lobulation_spiculation/best.pth"
)

ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
arch = ckpt["architecture"]

model = SpatialObjectiveModel(
    arch["objective"],
    arch["attention_dim"],
    arch["head_hidden"],
    arch["dropout"],
)
model.load_state_dict(ckpt["model_state_dict"], strict=True)
model.eval()

print("Architecture:", arch)
print("\nNamed modules containing encoder/stage/conv:")
for name, module in model.named_modules():
    low = name.lower()
    if any(token in low for token in ("encoder", "stage", "conv")):
        print(f"{name:65s} {type(module).__name__}")

print("\nTop-level children:")
for name, module in model.named_children():
    print(f"{name:40s} {type(module).__name__}")
