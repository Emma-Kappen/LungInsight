"""
diag_segment.py

Diagnostic for segment_lungs() on a real pylidc scan. This script imports and
runs the shared segmentation function from detect_candidates_cpu.py so the
report reflects the implementation actually used by the detector.

Usage:
    python diag_segment.py --patient-id LIDC-IDRI-0141
"""
import argparse

import numpy as np

import configparser
if not hasattr(configparser, 'SafeConfigParser'):
    configparser.SafeConfigParser = configparser.ConfigParser
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'bool'):
    np.bool = bool

import pylidc as pl
from detect_candidates_cpu import load_volume_hu, segment_lungs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patient-id', required=True)
    args = ap.parse_args()

    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == args.patient_id).first()
    if scan is None:
        raise ValueError(f'No scan found for {args.patient_id}')

    print(f'Loading volume for {args.patient_id} ...')
    volume_hu = load_volume_hu(scan)
    lung_mask = segment_lungs(volume_hu)
    n_slices = volume_hu.shape[0]
    print(f'Volume shape: {volume_hu.shape}')
    print(f'Total lung voxels: {int(lung_mask.sum())}\n')

    sample_fracs = [0.15, 0.30, 0.50, 0.70, 0.85]
    sample_z = sorted(set(int(f * n_slices) for f in sample_fracs))

    for z in sample_z:
        n_lung_px = int(lung_mask[z].sum())
        print(f'slice z={z}: lung pixels = {n_lung_px}')


if __name__ == '__main__':
    main()