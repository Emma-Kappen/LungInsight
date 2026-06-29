# lidc-preprocessing
This repository contains code to pre-process the LIDC-IDRI dataset of CT-scans with pulmonary nodules into an explainable AI preprocessing pipeline.


## Overview

The workflow consists of a few steps

1. use the pylidc library to process image annotations and grouped nodule clusters across patients
2. resample to 1mm x 1mm x 1mm and normalize to Hounsfield Units (HU)
3. extract fixed-size 3D volumetric crops of 64x64x64 voxels centered on each nodule
4. export each crop as an individual binary NumPy file and build a manifest with continuous normalized confidence labels for nine semantic characteristics


## Download scans

Download the original scans using the steps from this website: https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI



## Setup python environment

1. download anaconda 3
2. create a new environment (e.g. conda create --name lidc)
3. install some packages

(note we need scikit-image version 0.13 since replacement of measure.marching_cubes with measure.marching_cubes_lewiner in version 0.14 breaks compatibility with pylidc (as of yet)

`conda install jupyter numpy pandas feather-format scikit-image=0.13`

`pip install pylidc pypng`

4. configure pylidc to know where the scans are located, follow these steps: https://pylidc.github.io/install.html

## Follow the notebook

Pre processing: `lidc-preprocessing.ipynb`

Modeling example:

- keras + tf CNN 3D: `CNN_keras_3D.ipynb`
- keras + tf CNN 2D: `CNN_keras_2D.ipynb`

## New extraction pipeline

The current pipeline supports:

- `extract_nodule_patches_and_manifest(...)` in `utils-preprocessing.py`
- output of 3D patches as `nodule_[PatientID]_[NoduleIndex].npy`
- a manifest file named `dataset_manifest.csv`
- nine normalized confidence label columns:
  - `subtlety_confidence`
  - `internalStructure_confidence`
  - `calcification_confidence`
  - `sphericity_confidence`
  - `margin_confidence`
  - `lobulation_confidence`
  - `spiculation_confidence`
  - `texture_confidence`
  - `malignancy_confidence`

## CPU / Colab integration

This folder now includes end-to-end CPU preprocessing and Colab GPU training/validation scripts for the multi-head 3D SE-ResNet50 pipeline.

- `preprocess_cpu.py` - patient-aware train/validation CSV split creation and `.npy` patch archive creation.
- `inference_cpu.py` - single-patch CPU inference with per-head probabilities and GradCAM++ heatmaps.
- `train_colab_gpu.py` - GPU training using GradNorm loss balancing for the 10 independent binary heads.
- `validate_colab_gpu.py` - checkpoint validation that reports ROC AUC, precision, recall and F1 for each feature.
- `cir_multihead_pipeline.py` - common manifest extraction, dataset, model creation, and native 3D Grad-CAM utilities.
- `requirements.txt` - environment dependencies for the combined CPU/Colab workflow.

## Usage examples

1. Prepare the manifest and patches locally:

```bash
python preprocess_cpu.py --manifest path/to/extended_cir_manifest.csv --output-dir cpu_split --val-fraction 0.2
```

2. Upload the generated zip and CSVs to Colab, then train:

```bash
python train_colab_gpu.py --train-csv cpu_split/train_split.csv --val-csv cpu_split/val_split.csv --drive-dir /content/drive/MyDrive/lunginsight --batch-size 4 --epochs 10 --lr 1e-4
```

3. Validate the final checkpoint:

```bash
python validate_colab_gpu.py --val-csv cpu_split/val_split.csv --checkpoint /content/drive/MyDrive/lunginsight/best_model_gpu.pth
```

4. Run local explainability on a saved patch:

```bash
python inference_cpu.py --patch path/to/nodule_xxx.npy --checkpoint path/to/best_model_gpu.pth
```

## Issues

Currently, the code uses the pylidc function `cluster_annotations` twice: once to create a DataFrame of annotations, and a second time to export the images. Since this function takes some time, this could be made more efficient.

This is by no means an 'optimal' approach in the sense that I have not experimented with pre-processing hyperparameters like:

- resampling size
- label normalization methods
- output size
- number of 2D slices
- modeling architecture modifications

The current code is intended to provide a clean explainable pipeline for extracting 3D nodule patches with continuous radiologist confidence scores.



