"""
preprocess_cpu.py

Local CPU-only dataset setup and manifest split utilities for the Explainable Lung Cancer Diagnosis AI.
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
IMAGING_ROOT = SCRIPT_DIR.parent
for candidate in (SCRIPT_DIR, IMAGING_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from path_setup import ensure_project_paths

ensure_project_paths(__file__)

import argparse
import os
import random
import zipfile
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cir_multihead_pipeline import FEATURE_NAMES


class ExtendedCirDataset(Dataset):
    """Dataset for loading 3D patches and 10 binary characteristic labels."""

    def __init__(self, manifest_csv: str, transform=None, device: str = 'cpu'):
        if not os.path.isfile(manifest_csv):
            raise FileNotFoundError(f'Manifest file not found: {manifest_csv}')
        self.df = pd.read_csv(manifest_csv)
        self.transform = transform
        self.device = torch.device(device)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patch_path = row['file_path']
        if not os.path.isfile(patch_path):
            raise FileNotFoundError(f'Patch file not found: {patch_path}')

        patch = np.load(patch_path)
        if patch.ndim != 3 or patch.shape != (64, 64, 64):
            raise ValueError(f'Invalid patch shape {patch.shape} in {patch_path}, expected (64,64,64)')

        patch_tensor = torch.from_numpy(patch.astype(np.float32)).unsqueeze(0)
        if self.transform is not None:
            patch_tensor = self.transform(patch_tensor)

        labels = {
            feat: torch.tensor(int(row[f'{feat}_label']), dtype=torch.long)
            for feat in FEATURE_NAMES
        }
        return patch_tensor.to(self.device), labels


def split_patients_by_manifest(manifest_csv: str, val_fraction: float = 0.2, seed: int = 42):
    """Split manifest into train/validation ensuring PatientID leakage is avoided."""
    df = pd.read_csv(manifest_csv)
    if 'patient_id' not in df.columns:
        raise ValueError('Manifest must include a patient_id column')

    patient_ids = sorted(df['patient_id'].unique())
    random.Random(seed).shuffle(patient_ids)
    num_val = max(1, int(len(patient_ids) * val_fraction))
    val_ids = set(patient_ids[:num_val])
    train_ids = set(patient_ids[num_val:])

    train_df = df[df['patient_id'].isin(train_ids)].reset_index(drop=True)
    val_df = df[df['patient_id'].isin(val_ids)].reset_index(drop=True)
    return train_df, val_df


def save_split_csvs(manifest_csv: str, output_dir: str, val_fraction: float = 0.2, seed: int = 42):
    os.makedirs(output_dir, exist_ok=True)
    train_df, val_df = split_patients_by_manifest(manifest_csv, val_fraction=val_fraction, seed=seed)

    train_path = os.path.join(output_dir, 'train_split.csv')
    val_path = os.path.join(output_dir, 'val_split.csv')
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    return train_path, val_path


def zip_patch_files(manifest_csv: str, output_zip: str):
    df = pd.read_csv(manifest_csv)
    patch_paths = df['file_path'].tolist()

    with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for patch_path in patch_paths:
            if not os.path.isfile(patch_path):
                raise FileNotFoundError(f'Patch missing: {patch_path}')
            archive.write(patch_path, arcname=os.path.basename(patch_path))
    return output_zip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Prepare CPU dataset splits and patch archive for Colab upload.')
    parser.add_argument('--manifest', required=True, help='Path to extended_cir_manifest.csv')
    parser.add_argument('--output-dir', default='cpu_split', help='Output directory for train/val CSVs and archive')
    parser.add_argument('--val-fraction', type=float, default=0.2, help='Validation fraction of patients')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for patient splitting')
    parser.add_argument('--zip-name', default='npy_patches.zip', help='Name of the patch archive file')
    return parser.parse_args()


def main():
    args = parse_args()
    train_csv, val_csv = save_split_csvs(
        manifest_csv=args.manifest,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    zip_path = os.path.join(args.output_dir, args.zip_name)
    zip_patch_files(args.manifest, zip_path)
    print(f'Saved train split: {train_csv}')
    print(f'Saved val split: {val_csv}')
    print(f'Created patch archive: {zip_path}')


if __name__ == '__main__':
    main()
