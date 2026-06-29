# LungInsight — Explainable Lung Cancer Diagnosis Pipeline

This repository contains tools to preprocess the LIDC‑IDRI chest CT dataset, train a multi‑head 3D SE‑ResNet50 model for radiological characteristic prediction, and produce per‑head 3D Grad‑CAM explanations.

Quick highlights

- CPU preprocessing and extraction: `preprocess_cpu.py` and `cir_multihead_pipeline.py`
- Local inference + explainability: `inference_cpu.py` (produces per‑head probabilities and 3D Grad‑CAM heatmaps)
- Colab GPU training/validation: `train_colab_gpu.ipynb`, `validate_colab_gpu.ipynb`
- Model architecture: 3D SE‑ResNet50 backbone with 10 independent binary heads
- Loss balancing: GradNorm integration for multi‑task training

Files of interest

- [Preprocess utilities](lidc-preprocessing/README.md) — project submodule README with extraction details and pylidc usage.
- [cir_multihead_pipeline.py](cir_multihead_pipeline.py) — shared pipeline: manifest generation, `LIDCPatchDataset`, model constructor, native 3D Grad‑CAM generator.
- [preprocess_cpu.py](preprocess_cpu.py) — patient‑aware split and patch archive helper for CPU environments.
- [inference_cpu.py](inference_cpu.py) — single‑patch inference and GradCAM++ explainability on CPU.
- [train_colab_gpu.ipynb](train_colab_gpu.ipynb) — Colab notebook to train on GPU (mount Drive, install deps, run training loop).
- [validate_colab_gpu.ipynb](validate_colab_gpu.ipynb) — Colab notebook to evaluate saved checkpoint and print per‑head metrics.

Dependencies

A minimal set of Python packages used by the pipeline (see `requirements.txt` for details):

- torch, torchvision
- numpy, pandas
- scikit-learn
- pylidc
- gradnorm-pytorch
- pytorch-grad-cam

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

3. Upload `cpu_split/npy_patches.zip` and CSVs to Google Drive, open `train_colab_gpu.ipynb` in Colab, mount Drive, and run the notebook cells to train on GPU.

4. Run local explainability on a saved patch and checkpoint:

```bash
python inference_cpu.py --patch path/to/nodule_0001_001.npy --checkpoint path/to/best_model_gpu.pth
```

Notes and caveats

- The model and Grad‑CAM utilities expect 3D patches in shape `(64, 64, 64)` saved as `.npy` arrays.
- `pylidc` must be configured to point to the LIDC‑IDRI DICOM archives for extraction to work.
- Colab notebooks install dependencies at runtime; ensure GPU accelerator is enabled in Colab.

Next actions I can take

- Run a quick import/test to validate `cir_multihead_pipeline` and model creation.
- Add a CLI extractor script wrapping `aggregate_and_extract` for convenience.
- Add a short CONTRIBUTING.md and license summary.

If you'd like one of those next steps, tell me which and I'll proceed.