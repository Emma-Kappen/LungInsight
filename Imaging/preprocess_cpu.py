"""
preprocess_cpu.py

Local CPU-only dataset setup and manifest split utilities for the Explainable
Lung Cancer Diagnosis AI.

Note: dataset loading is handled by LIDCPatchDataset in
cir_multihead_pipeline.py (the canonical dataset class for this project).
This module only handles patient-aware splitting and patch archiving.
"""
import argparse
import os
import random
import zipfile

import pandas as pd

# Re-exported for convenience so existing imports of
# `from preprocess_cpu import ExtendedCirDataset` keep working; this is just
# an alias for the canonical dataset class, not a separate implementation.
from cir_multihead_pipeline import LIDCPatchDataset as ExtendedCirDataset  # noqa: F401


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