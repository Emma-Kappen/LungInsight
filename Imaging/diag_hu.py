"""
diag_hu.py

Checks whether load_volume_hu() is producing sane Hounsfield Units, or
whether the intercept is being applied twice (or some other rescale bug),
by comparing raw stored pixel values against the computed HU volume.

Typical sane chest CT HU landmarks:
    air            ~ -1000
    lung parenchyma  ~ -700 to -600
    fat            ~ -100 to -50
    soft tissue    ~   0 to  80
    bone           ~ 400 and above

If min HU is far below ~-1100 or the soft-tissue/bone range is missing
entirely, the rescale is almost certainly being double-applied or the
RescaleIntercept/RescaleSlope read for this scan is wrong.

Usage:
    python diag_hu.py --patient-id LIDC-IDRI-0141
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patient-id', required=True)
    args = ap.parse_args()

    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == args.patient_id).first()
    if scan is None:
        raise ValueError(f'No scan found for {args.patient_id}')

    imgs = scan.load_all_dicom_images(verbose=False)
    dcm0 = imgs[0]

    print(f'Number of DICOM slices: {len(imgs)}')
    print(f'RescaleIntercept (slice 0): {getattr(dcm0, "RescaleIntercept", "MISSING")}')
    print(f'RescaleSlope     (slice 0): {getattr(dcm0, "RescaleSlope", "MISSING")}')
    print(f'BitsStored: {getattr(dcm0, "BitsStored", "?")}  '
          f'PixelRepresentation: {getattr(dcm0, "PixelRepresentation", "?")}  '
          f'PhotometricInterpretation: {getattr(dcm0, "PhotometricInterpretation", "?")}')

    # Check whether intercept/slope are consistent across ALL slices --
    # a per-slice varying intercept would break a single-value approach.
    intercepts = set()
    slopes = set()
    for img in imgs:
        intercepts.add(float(getattr(img, 'RescaleIntercept', float('nan'))))
        slopes.add(float(getattr(img, 'RescaleSlope', float('nan'))))
    print(f'\nDistinct RescaleIntercept values across all slices: {intercepts if len(intercepts) <= 5 else f"{len(intercepts)} distinct values"}')
    print(f'Distinct RescaleSlope values across all slices: {slopes if len(slopes) <= 5 else f"{len(slopes)} distinct values"}')

    raw0 = dcm0.pixel_array.astype(np.float64)
    print(f'\nRaw stored pixel_array (slice 0): min={raw0.min():.1f}  max={raw0.max():.1f}  '
          f'mean={raw0.mean():.1f}')

    intercept = float(dcm0.RescaleIntercept)
    slope = float(dcm0.RescaleSlope)
    hu0_manual = raw0 * slope + intercept
    print(f'Manually rescaled slice 0 (raw*slope+intercept): '
          f'min={hu0_manual.min():.1f}  max={hu0_manual.max():.1f}  mean={hu0_manual.mean():.1f}')

    # Now compare against what load_volume_hu actually produces for this slice
    from detect_candidates_cpu import load_volume_hu
    volume_hu = load_volume_hu(scan)
    hu0_pipeline = volume_hu[0]
    print(f'load_volume_hu() slice 0:                      '
          f'min={hu0_pipeline.min():.1f}  max={hu0_pipeline.max():.1f}  mean={hu0_pipeline.mean():.1f}')

    diff = hu0_pipeline.astype(np.float64) - hu0_manual
    print(f'\nDifference (pipeline - manual): min={diff.min():.1f}  max={diff.max():.1f}  '
          f'mean={diff.mean():.1f}  (should be ~0 if consistent)')

    # HU histogram bins for a mid-volume slice, to sanity-check tissue distribution
    mid = volume_hu.shape[0] // 2
    mid_slice = volume_hu[mid]
    bins = [(-3000, -1100, 'below-air (suspicious)'),
            (-1100, -400, 'air'),
            (-400, -150, 'lung parenchyma'),
            (-150, -20, 'fat'),
            (-20, 150, 'soft tissue'),
            (150, 4000, 'bone/dense')]
    print(f'\nHU distribution, mid-volume slice (z={mid}), {mid_slice.size} pixels total:')
    for lo, hi, label in bins:
        count = int(((mid_slice >= lo) & (mid_slice < hi)).sum())
        pct = 100.0 * count / mid_slice.size
        print(f'  [{lo:6d}, {hi:6d})  {label:<28s} {count:7d} px  ({pct:5.1f}%)')


if __name__ == '__main__':
    main()