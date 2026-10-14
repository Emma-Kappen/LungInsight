"""Adapter for existing spatial-hybrid ordinal_reader checkpoints."""
import torch
import torch.nn as nn
from .model_objective import MODEL_TARGETS, SpatialObjectiveModel, preprocess_hu

ATTRIBUTES=("subtlety","sphericity","margin","lobulation","spiculation","texture")
CHECKPOINT_GROUP={"subtlety":"subtlety","sphericity":"sphericity","margin":"margin",
                  "lobulation":"lobulation_spiculation","spiculation":"lobulation_spiculation","texture":"texture"}

class GradCAMAttributeModel(nn.Module):
    def __init__(self, core, attribute):
        super().__init__()
        if attribute not in ATTRIBUTES: raise ValueError(f"Unsupported attribute: {attribute}")
        self.core=core; self.attribute=attribute; self.target_index=MODEL_TARGETS.index(attribute)
    def forward(self, raw_hu):
        outputs=self.core(preprocess_hu(raw_hu))
        ratings=self.core.expected_ratings(outputs)
        # The six ordered LIDC attributes use increasing ratings for more of the named property.
        return {self.attribute: ratings[:,self.target_index]}

def load_gradcam_model(checkpoint_path, device, attribute):
    if attribute not in ATTRIBUTES: raise ValueError(f"Unsupported attribute: {attribute}")
    ckpt=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
    arch=ckpt["architecture"]
    if arch.get("objective")!="ordinal_reader": raise ValueError("Expected ordinal_reader checkpoint")
    model=SpatialObjectiveModel(arch["objective"],arch["attention_dim"],arch["head_hidden"],arch["dropout"])
    model.load_state_dict(ckpt["model_state_dict"],strict=True)
    return GradCAMAttributeModel(model.eval(),attribute).to(device).eval()

def load_gradcam_model_for_attribute(checkpoint_path, device, attribute):
    return load_gradcam_model(checkpoint_path,device,attribute)
