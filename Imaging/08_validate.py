"""
08_validate.py

STEP 8 (validation) in the pipeline:
    01_dicom_to_hu.py               -> DICOM -> HU volume
    02_mask_and_crop.py             -> lung segmentation + non-lung blanking + Z-crop
    03_visualize.py                 -> QA montage
    04_detect_candidates.py         -> nodule candidate detection (ViTDet3D)
    05_extract_candidate_patches.py -> crop 64^3 classifier patches
    06_xai.py                       -> classify + Grad-CAM++ / saliency per patch
    07_visualize.py                 -> overlay heatmaps on the full CT volume
    08_validate.py                  <- this file: sanity-check 06_xai.py's
                                        per-candidate metrics and Grad-CAM/
                                        saliency maps before trusting them
                                        downstream (e.g. before handing
                                        results to 07_visualize.py or to a
                                        clinician-facing report)

This does NOT re-run the model. It is a pure data-integrity/sanity pass
over what 06_xai.py already wrote (xai_manifest.csv + meta.json +
<candidate_id>_xai.npz), catching the kinds of bugs that "the script
didn't crash" doesn't catch: a broken sigmoid application, a dead
gradient through --target-layer, a manifest that drifted from the npz
files after a partial re-run, or a checkpoint whose scores are silently
out of range.

Checks performed, per candidate:

  METRICS validity (per head):
    - the head's *_score column exists in xai_manifest.csv and the
      matching {head}_score key exists in the candidate's .npz
    - the score is finite (no NaN/Inf)
    - the score lies in [0, 1] -- 06_xai.py applies sigmoid to the raw
      logit exactly once, so anything outside [0, 1] means sigmoid was
      applied twice, skipped, or the checkpoint/head is mismatched
    - the manifest's score matches the score stored inside the npz
      itself, within --score-tolerance (catches manifest/npz drift from
      an interrupted or partially re-run 06_xai.py)

  GRAD-CAM validity (per head), and SALIENCY validity if
  meta.json says saliency was computed:
    - array shape is exactly (patch_size, patch_size, patch_size)
    - values are finite and lie in [0, 1] -- 06_xai.py min-max
      normalizes both maps to this range
    - FAIL if the map is missing entirely when it should be present
    - WARN (not fail) if the map is uniformly zero. compute_gradcampp()
      in 06_xai.py legitimately zeroes out a degenerate map when
      max-min <= EPS, so a handful of these is not necessarily a bug --
      but if it happens across most/all candidates for a head, gradients
      through --target-layer are effectively dead (bad target-layer
      path, frozen backbone, or a head with no learned signal), and
      that's surfaced in the summary as a distinct high-degenerate-rate
      warning
    - WARN if the map is nearly spatially uniform (high normalized
      entropy) -- a real localization failure. This is a heuristic, not
      a hard rule about model quality, so it stays a WARN

  PATCH sanity:
    - the stored patch array has no NaN/Inf voxels

  MANIFEST/FILE consistency:
    - every candidate_id in xai_manifest.csv has a corresponding
      <candidate_id>_xai.npz on disk, and vice versa

Usage:
    python 08_validate.py output/LIDC-IDRI-0001_xai

Outputs (written to --out-dir, default: the xai_dir itself):
    validation_report.csv  -> one row per (candidate_id, head, check)
    validation_summary.json -> aggregate PASS/WARN/FAIL counts

Exit code is nonzero if any HARD FAILURE was found (so this can gate a
pipeline/CI run); warnings alone exit 0.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

EPS = 1e-8


def load_run(xai_dir: str):
    manifest_path = os.path.join(xai_dir, "xai_manifest.csv")
    meta_path = os.path.join(xai_dir, "meta.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"'{manifest_path}' not found. Run 06_xai.py first, and pass "
            f"its --out-dir here."
        )
    manifest = pd.read_csv(manifest_path)
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return manifest, meta


def normalized_entropy(arr: np.ndarray) -> float:
    """Spatial-uniformity measure for a nonnegative map, in [0, 1].

    Treats arr (already >= 0, e.g. a [0,1]-normalized Grad-CAM/saliency
    map) as an unnormalized distribution over voxels and computes its
    Shannon entropy relative to the maximum possible entropy for that
    many voxels (the uniform distribution). Values near 1.0 mean the map
    is close to spatially uniform (no real localization); values near
    0.0 mean the map's mass is concentrated in a small region.
    Returns 0.0 for an all-zero map (nothing to localize, handled
    separately as a degenerate-map check).
    """
    flat = arr.astype(np.float64).ravel()
    total = flat.sum()
    if total <= EPS:
        return 0.0
    p = flat / total
    nz = p[p > 0]
    entropy = -np.sum(nz * np.log(nz))
    max_entropy = np.log(flat.size)
    if max_entropy <= EPS:
        return 0.0
    return float(entropy / max_entropy)


def check_array(name: str, arr, expected_shape, allow_missing: bool):
    """Shape/range/finiteness checks shared by gradcam/saliency/patch
    arrays. Returns a list of (check, status, detail) tuples.
    """
    rows = []
    if arr is None:
        status = "FAIL" if not allow_missing else "WARN"
        rows.append((f"{name}_present", status, "array missing from npz"))
        return rows

    rows.append((f"{name}_present", "PASS", ""))

    if tuple(arr.shape) != tuple(expected_shape):
        rows.append((
            f"{name}_shape", "FAIL",
            f"expected {tuple(expected_shape)}, got {tuple(arr.shape)}",
        ))
        return rows  # further checks assume the right shape
    rows.append((f"{name}_shape", "PASS", ""))

    finite = np.isfinite(arr)
    if not finite.all():
        n_bad = int((~finite).sum())
        rows.append((f"{name}_finite", "FAIL", f"{n_bad} NaN/Inf voxel(s)"))
        return rows
    rows.append((f"{name}_finite", "PASS", ""))

    return rows


def check_map_range_and_degeneracy(name: str, arr: np.ndarray,
                                    diffuse_entropy_ratio: float):
    rows = []
    arr_min, arr_max = float(arr.min()), float(arr.max())
    if arr_min < -EPS or arr_max > 1.0 + EPS:
        rows.append((
            f"{name}_range", "FAIL",
            f"values outside [0,1]: min={arr_min:.4f}, max={arr_max:.4f}",
        ))
    else:
        rows.append((f"{name}_range", "PASS", ""))

    if arr_max - arr_min <= EPS:
        rows.append((
            f"{name}_degenerate", "WARN",
            "map is uniformly zero (dead/zero gradient through the "
            "target layer for this candidate+head)",
        ))
        return rows  # entropy is meaningless on an all-zero map
    rows.append((f"{name}_degenerate", "PASS", ""))

    ratio = normalized_entropy(arr)
    if ratio >= diffuse_entropy_ratio:
        rows.append((
            f"{name}_localization", "WARN",
            f"map is nearly spatially uniform (entropy ratio={ratio:.3f} "
            f">= {diffuse_entropy_ratio}); weak/no localization signal",
        ))
    else:
        rows.append((f"{name}_localization", "PASS", f"entropy ratio={ratio:.3f}"))

    return rows


def validate_candidate(candidate_id: str, npz_path: str, manifest_row: pd.Series,
                        head_names, patch_size: int, saliency_expected: bool,
                        score_tolerance: float, diffuse_entropy_ratio: float):
    rows = []  # (candidate_id, head, check, status, detail)

    if not os.path.isfile(npz_path):
        rows.append((candidate_id, "", "npz_file_present", "FAIL",
                     f"missing file: {npz_path}"))
        return rows
    rows.append((candidate_id, "", "npz_file_present", "PASS", ""))

    data = np.load(npz_path, allow_pickle=True)
    expected_shape = (patch_size, patch_size, patch_size)

    # Patch sanity (not per-head)
    patch = data["patch"] if "patch" in data else None
    for check, status, detail in check_array("patch", patch, expected_shape, allow_missing=False):
        rows.append((candidate_id, "", check, status, detail))

    for head in head_names:
        score_key = f"{head}_score"
        gradcam_key = f"{head}_gradcam"
        saliency_key = f"{head}_saliency"

        # --- score ---
        if score_key not in data:
            rows.append((candidate_id, head, "score_present", "FAIL",
                         f"'{score_key}' missing from npz"))
        else:
            score = float(np.asarray(data[score_key]).item())
            if not np.isfinite(score):
                rows.append((candidate_id, head, "score_finite", "FAIL",
                             f"score is {score}"))
            elif score < -EPS or score > 1.0 + EPS:
                rows.append((candidate_id, head, "score_range", "FAIL",
                             f"score={score:.4f} outside [0,1] -- check for "
                             f"double/missing sigmoid application"))
            else:
                rows.append((candidate_id, head, "score_range", "PASS", ""))

            manifest_col = f"{head}_score"
            if manifest_col in manifest_row and pd.notna(manifest_row[manifest_col]):
                manifest_score = float(manifest_row[manifest_col])
                if abs(manifest_score - score) > score_tolerance:
                    rows.append((candidate_id, head, "score_manifest_consistency", "FAIL",
                                 f"manifest={manifest_score:.4f} vs npz={score:.4f}"))
                else:
                    rows.append((candidate_id, head, "score_manifest_consistency", "PASS", ""))
            else:
                rows.append((candidate_id, head, "score_manifest_consistency", "WARN",
                             f"'{manifest_col}' missing from xai_manifest.csv row"))

        # --- gradcam ---
        gradcam = data[gradcam_key] if gradcam_key in data else None
        array_rows = check_array("gradcam", gradcam, expected_shape, allow_missing=False)
        rows.extend((candidate_id, head, c, s, d) for c, s, d in array_rows)
        if gradcam is not None and tuple(gradcam.shape) == expected_shape and np.isfinite(gradcam).all():
            for c, s, d in check_map_range_and_degeneracy("gradcam", gradcam, diffuse_entropy_ratio):
                rows.append((candidate_id, head, c, s, d))

        # --- saliency (only checked if 06_xai.py's meta.json says it was computed) ---
        if saliency_expected:
            saliency = data[saliency_key] if saliency_key in data else None
            array_rows = check_array("saliency", saliency, expected_shape, allow_missing=False)
            rows.extend((candidate_id, head, c, s, d) for c, s, d in array_rows)
            if saliency is not None and tuple(saliency.shape) == expected_shape and np.isfinite(saliency).all():
                for c, s, d in check_map_range_and_degeneracy("saliency", saliency, diffuse_entropy_ratio):
                    rows.append((candidate_id, head, c, s, d))

    return rows


def validate_run(xai_dir: str, out_dir: str, patch_size: int = 64,
                  score_tolerance: float = 1e-4, diffuse_entropy_ratio: float = 0.98):
    manifest, meta = load_run(xai_dir)

    head_names = meta.get("head_names")
    if not head_names:
        # Fall back to inferring heads from manifest columns.
        head_names = sorted({
            c[:-len("_score")] for c in manifest.columns
            if c.endswith("_score") and c not in ("candidate_score",)
        })
    saliency_expected = bool(meta.get("saliency_computed", True))

    print(f"[info] Validating {len(manifest)} candidate(s), heads={head_names}, "
          f"saliency_expected={saliency_expected}")

    all_rows = []

    # Manifest <-> file consistency, both directions.
    manifest_ids = set(manifest["candidate_id"].astype(str))
    npz_files_on_disk = {
        fn[:-len("_xai.npz")] for fn in os.listdir(xai_dir) if fn.endswith("_xai.npz")
    }
    for extra_id in npz_files_on_disk - manifest_ids:
        all_rows.append((extra_id, "", "orphaned_npz_file", "WARN",
                          "*_xai.npz on disk has no row in xai_manifest.csv"))

    for _, row in manifest.iterrows():
        candidate_id = str(row["candidate_id"])
        npz_path = row.get("xai_result_path")
        if not isinstance(npz_path, str) or not npz_path:
            npz_path = os.path.join(xai_dir, f"{candidate_id}_xai.npz")
        candidate_rows = validate_candidate(
            candidate_id, npz_path, row, head_names, patch_size,
            saliency_expected, score_tolerance, diffuse_entropy_ratio,
        )
        all_rows.extend(candidate_rows)

    report_df = pd.DataFrame(
        all_rows, columns=["candidate_id", "head", "check", "status", "detail"]
    )

    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "validation_report.csv")
    report_df.to_csv(report_path, index=False)

    status_counts = report_df["status"].value_counts().to_dict()
    n_fail_candidates = report_df.loc[report_df["status"] == "FAIL", "candidate_id"].nunique()
    n_warn_candidates = report_df.loc[
        (report_df["status"] == "WARN") & (~report_df["candidate_id"].isin(
            report_df.loc[report_df["status"] == "FAIL", "candidate_id"]
        )), "candidate_id"
    ].nunique()
    n_total_candidates = manifest["candidate_id"].nunique()

    # Degenerate-Grad-CAM rate per head, to flag a systemic dead-gradient
    # problem (see module docstring) rather than just noisy individual cases.
    degenerate_rates = {}
    for head in head_names:
        head_gradcam_checks = report_df[
            (report_df["head"] == head) & (report_df["check"] == "gradcam_degenerate")
        ]
        if len(head_gradcam_checks) == 0:
            continue
        rate = float((head_gradcam_checks["status"] == "WARN").mean())
        degenerate_rates[head] = rate
        if rate >= 0.5:
            print(f"[warn] head '{head}': {rate * 100:.0f}% of candidates have an "
                  f"all-zero Grad-CAM++ map -- gradients through the "
                  f"--target-layer used by 06_xai.py may be dead for this head.")

    summary = {
        "xai_dir": os.path.abspath(xai_dir),
        "num_candidates": int(n_total_candidates),
        "head_names": head_names,
        "saliency_expected": saliency_expected,
        "check_status_counts": {k: int(v) for k, v in status_counts.items()},
        "candidates_with_failures": int(n_fail_candidates),
        "candidates_with_only_warnings": int(n_warn_candidates),
        "gradcam_degenerate_rate_by_head": degenerate_rates,
    }
    summary_path = os.path.join(out_dir, "validation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] {n_total_candidates} candidate(s): "
          f"{n_total_candidates - n_fail_candidates - n_warn_candidates} clean, "
          f"{n_warn_candidates} with warnings only, {n_fail_candidates} with failures.")
    print(f"[done] Wrote validation_report.csv + validation_summary.json -> '{out_dir}'")

    return report_df, summary, n_fail_candidates


def parse_args():
    parser = argparse.ArgumentParser(
        description="STEP 8: Sanity-check 06_xai.py's per-candidate scores "
        "and Grad-CAM++/saliency maps for validity (not model quality)."
    )
    parser.add_argument(
        "xai_dir",
        help="Directory containing xai_manifest.csv + *_xai.npz (the "
        "--out-dir from 06_xai.py).",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Directory to write validation_report.csv / "
        "validation_summary.json (default: same as xai_dir).",
    )
    parser.add_argument(
        "--patch-size", type=int, default=64,
        help="Expected cubic patch side length in voxels (default: 64, "
        "matching 05_extract_candidate_patches.py / se_resnet3d.py).",
    )
    parser.add_argument(
        "--score-tolerance", type=float, default=1e-4,
        help="Max allowed absolute difference between a manifest score "
        "and the same score stored in its .npz before flagging a FAIL "
        "(default: 1e-4).",
    )
    parser.add_argument(
        "--diffuse-entropy-ratio", type=float, default=0.98,
        help="Normalized-entropy threshold (0-1) above which a Grad-CAM/"
        "saliency map is flagged WARN as 'nearly spatially uniform, weak "
        "localization' (default: 0.98).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or args.xai_dir
    _, _, n_fail_candidates = validate_run(
        args.xai_dir, out_dir,
        patch_size=args.patch_size,
        score_tolerance=args.score_tolerance,
        diffuse_entropy_ratio=args.diffuse_entropy_ratio,
    )
    if n_fail_candidates > 0:
        print(f"[fail] {n_fail_candidates} candidate(s) had hard validation "
              f"failures -- see validation_report.csv.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
