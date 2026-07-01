# LungInsight — Explainable Lung Cancer Diagnosis Pipeline

This repository contains tools to preprocess the LIDC‑IDRI chest CT dataset, train a multi‑head 3D SE‑ResNet50 model for radiological characteristic prediction, and produce per‑head 3D Grad‑CAM explanations.

Quick highlights

- CPU preprocessing and extraction: `preprocess_cpu.py` and `cir_multihead_pipeline.py`
- 3D SE-ResNet50 backbone: `se_resnet3d.py` — a from-scratch 3D reimplementation (no pretrained weights; ImageNet RGB weights have no valid initialization for single-channel volumetric CT input)
- Local inference + explainability: `inference_cpu.py` (produces per‑head confidence scores and 3D Grad‑CAM heatmaps via `pytorch-grad-cam`)
- Colab GPU training/validation: `train_colab_gpu.ipynb`, `validate_colab_gpu.ipynb` (fully self-contained; all training/eval code is inlined in the notebook cells)
- Model architecture: 3D SE‑ResNet50 backbone ([3, 4, 6, 3] SEBottleneck layout, matching `moskomule/senet.pytorch`'s `se_resnet50`) with 10 independent sigmoid confidence heads
- Labels: each of the 10 radiological characteristics is a continuous confidence score in [0, 1] (min-max normalized from pylidc's raw annotation ratings), not a binarized class
- Loss balancing: GradNorm integration (`lucidrains/gradnorm-pytorch`) for multi-task training

Files of interest

- [cir_multihead_pipeline.py](cir_multihead_pipeline.py) — canonical pipeline: manifest generation with [0,1] confidence scores, `LIDCPatchDataset` (the single dataset class used project-wide), model constructor, 3D Grad-CAM generator.
- [se_resnet3d.py](se_resnet3d.py) — the 3D SE-ResNet50 backbone and multi-head wrapper.
- [preprocess_cpu.py](preprocess_cpu.py) — patient-aware split and patch archive helper for CPU environments. Re-exports `LIDCPatchDataset` as `ExtendedCirDataset` for backward compatibility; it is not a separate dataset implementation.
- [inference_cpu.py](inference_cpu.py) — single-patch inference and GradCAM++ explainability on CPU.
- [train_colab_gpu.ipynb](train_colab_gpu.ipynb) — Colab notebook to train on GPU (mount Drive, install deps, run training loop).
- [validate_colab_gpu.ipynb](validate_colab_gpu.ipynb) — Colab notebook to evaluate a saved checkpoint and print per-head regression metrics (MAE, RMSE, Pearson r).

Dependencies

See `requirements.txt`:

- torch, torchvision
- numpy, pandas
- scikit-learn
- pylidc
- gradnorm-pytorch
- grad-cam (imports as `pytorch_grad_cam`)

The 3D SE-ResNet50 backbone is implemented directly in this repo (`se_resnet3d.py`) — no external `senet` package is required or installed.

Quick start

1. Prepare environment (local CPU for preprocessing):

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

2. Extract patches and create patient splits (local machine):

```bash
python preprocess_cpu.py --manifest path/to/extended_cir_manifest.csv --output-dir cpu_split --val-fraction 0.2
```

This produces `cpu_split/train_split.csv`, `cpu_split/val_split.csv`, and `cpu_split/npy_patches.zip`.

3. Upload `cpu_split/npy_patches.zip` and CSVs to Google Drive, along with this repository's `.py` files (to `/content/LungInsight`), open `train_colab_gpu.ipynb` in Colab, mount Drive, and run the notebook cells to train on GPU.

4. Run local explainability on a saved patch and checkpoint:

```bash
python inference_cpu.py --patch path/to/nodule_0001_001.npy --checkpoint path/to/best_model_gpu.pth
```

Notes and caveats

- The model and Grad-CAM utilities expect 3D patches in shape `(64, 64, 64)` saved as `.npy` arrays.
- `pylidc` must be configured to point to the LIDC-IDRI DICOM archives for extraction to work.
- Colab notebooks install dependencies at runtime; ensure GPU accelerator is enabled in Colab.
- The `se_resnet3d.py` backbone has not been validated against a real LIDC-IDRI training run in this environment (no GPU/torch available here) — verify shapes and a short training run end-to-end before relying on it for real experiments.