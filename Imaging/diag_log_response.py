"""
diag_log_response.py

Now that load_volume_hu() produces correct HU, the old LOG_THRESHOLD=0.015
(tuned against double-offset HU data) is catching almost every voxel in the
lung field instead of just nodule-like blobs. Rather than guess a new
constant blind, this script:

  1. Prints the percentile distribution of raw LoG responses within the lung
     mask, at a few representative scales, so you can pick a threshold from
     actual data instead of a guess.
  2. Specifically prints the response value AT each real annotated nodule's
     location, at every scale -- this tells you directly what threshold
     range would actually capture real nodules, which is the number that
     matters (a threshold that's merely "sparse" isn't good enough if it's
     still below the response at real nodules).

Usage:
    python diag_log_response.py --patient-id LIDC-IDRI-0007
"""
import argparse
import numpy as np
from scipy.ndimage import gaussian_laplace

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
from detect_candidates_cpu import (
    load_volume_hu, segment_lungs, get_spacing_mm, _sigma_range,
    _resolve_detection_mask, MIN_DIAM_MM, MAX_DIAM_MM, N_SCALES,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patient-id', required=True)
    args = ap.parse_args()

    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == args.patient_id).first()
    if scan is None:
        raise ValueError(f'No scan found for {args.patient_id}')

    print(f'Loading {args.patient_id} ...')
    volume_hu = load_volume_hu(scan)
    spacing_mm = get_spacing_mm(scan)
    lung_mask = segment_lungs(volume_hu)
    lung_mask = _resolve_detection_mask(lung_mask, volume_hu.shape)
    print(f'Lung mask fraction: {lung_mask.sum() / lung_mask.size:.3f}')

    HU_MIN, HU_MAX = -1000.0, 400.0
    vol_norm = np.clip(volume_hu, HU_MIN, HU_MAX).astype(np.float32)
    vol_norm = (vol_norm - HU_MIN) / (HU_MAX - HU_MIN)

    # Get real annotation centroids in (z, y, x) voxel space, same convention
    # used elsewhere in this file (annotation loading reorders pylidc's
    # native (i,j,k) into (k,i,j) = (z,y,x)).
    ann_centroids = []
    for ann in scan.annotations:
        c = ann.centroid
        ann_centroids.append((float(c[2]), float(c[0]), float(c[1])))
    print(f'{len(ann_centroids)} annotation(s) at (z,y,x): {ann_centroids}\n')

    sigmas_mm = _sigma_range(MIN_DIAM_MM, MAX_DIAM_MM, N_SCALES)

    for i, sigma_mm in enumerate(sigmas_mm):
        sigma_vox = tuple(sigma_mm / s for s in spacing_mm)
        resp = (-sigma_mm ** 2) * gaussian_laplace(vol_norm, sigma=sigma_vox, mode='reflect')

        in_mask = resp[lung_mask]
        pcts = np.percentile(in_mask, [50, 90, 99, 99.9, 99.99])
        diam_mm = 2 * np.sqrt(3) * sigma_mm
        print(f'scale {i+1:2d}/{N_SCALES}  sigma={sigma_mm:.2f}mm  diam~{diam_mm:.1f}mm')
        print(f'  in-mask response percentiles [50, 90, 99, 99.9, 99.99]: '
              f'{[f"{p:.4f}" for p in pcts]}')

        for j, (z, y, x) in enumerate(ann_centroids):
            zi, yi, xi = int(round(z)), int(round(y)), int(round(x))
            if (0 <= zi < resp.shape[0] and 0 <= yi < resp.shape[1] and 0 <= xi < resp.shape[2]):
                val = resp[zi, yi, xi]
                # also report the max in a small neighborhood, since the
                # centroid voxel itself may not be the exact response peak
                z_lo, z_hi = max(0, zi - 2), min(resp.shape[0], zi + 3)
                y_lo, y_hi = max(0, yi - 2), min(resp.shape[1], yi + 3)
                x_lo, x_hi = max(0, xi - 2), min(resp.shape[2], xi + 3)
                local_max = resp[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi].max()
                print(f'    ann {j:02d} at ({zi},{yi},{xi}): '
                      f'response={val:.4f}  local_max(5x5x5)={local_max:.4f}')
        print()


if __name__ == '__main__':
    main()