"""
07_visualize_gradcam.py

STEP 7: places each candidate's patch-level Grad-CAM heatmap (from
06_run_inference_xai.py) back into the FULL masked CT volume at its
real anatomical location, and provides a scrollable viewer -- so you
can see WHERE in the actual lung the model's attention is, in context,
rather than only the isolated 64^3 patch views 06 already produces.

    01_dicom_to_hu.py           -> DICOM -> HU volume
    02_mask_and_crop.py         -> lung segmentation + non-lung blanking + Z-crop
    03_visualize.py             -> viewing (raw volume + air/lung/heart masks)
    04_detect_and_patch.py      -> multi-scale 3D LoG candidate detection
    05_shape_filter_and_grow.py -> reject tubular candidates, grow + crop
    06_run_inference_xai.py     -> classify + Grad-CAM each nodule patch
    07_visualize_gradcam.py     <- this file: place heatmaps back into
                                    the full scan + interactive viewer

=== Why this needs its own placement step (not just pasting the patch back) ===

Each candidate's heatmap in 06's saved .npz is a FIXED (patch_shape,
patch_shape, patch_shape) array -- e.g. 64x64x64 -- because that's the
uniform input size the classifier requires. But 05_shape_filter_and_grow.py
crops each candidate from a DIFFERENT-sized region of the original
volume (the grown nodule's own bounding box + margin, adaptively sized
per candidate -- see that script's own docstring for why), then
resamples it to the fixed patch size. So placing a heatmap back
correctly means reversing that resample: stretch/shrink the 64^3
heatmap back to whatever physical box it actually came from, THEN
paste it into the full volume at that box's real location. Pasting a
patch-shaped block at a fixed size back into the volume, without
reversing this, would place it at the wrong physical scale for any
candidate that wasn't exactly patch_shape voxels in its original crop
(nearly always, since almost no real nodule's grown+margin box happens
to be exactly patch_shape voxels across).

Two ways this script determines the real placement box, tried in order:

  1. EXACT: if --nodules-dir is given and 05's nodules.json has an
     explicit per-candidate bounding box field (checked under a few
     plausible key names, since the exact field name wasn't confirmed
     against your installed 05_shape_filter_and_grow.py version:
     'bbox_zyx', 'crop_bbox_zyx', 'grown_bbox_zyx', 'bbox'), that box
     is used directly -- pixel-accurate placement.
  2. APPROXIMATE (fallback, always available): a cube centered on the
     candidate's voxel center, sized from
     nodules.json/ranked_candidates.csv's own equivalent_diameter_mm
     plus --placement-margin-mm on each side. This won't be pixel-
     identical to 05's original (possibly non-cubic, asymmetric) grown
     region, but is anatomically reasonable and always available since
     it only needs numbers this pipeline already reports. Which path
     was used is always printed per candidate, never silently assumed.

=== Overlap handling ===

If two candidates' placed regions overlap in the full volume (rare,
but possible for adjacent nodules), the OVERLAP takes the ELEMENTWISE
MAXIMUM heatmap value rather than the later candidate overwriting the
earlier one -- so a strong attention signal from either candidate
still shows, instead of depending on paste order.

=== What this script displays ===

The background is volume_hu_masked.npy (non-lung tissue already
blanked by 02) -- matching what you asked for ("masked lung CT
scan") -- rendered on a standard lung window. The Grad-CAM heatmap is
overlaid ONLY where a candidate's placed region actually has signal
(pure grayscale everywhere else, so you're never looking at spurious
color from float noise near zero).

Usage:
    # Exact placement using 05's own bounding boxes
    python 07_visualize_gradcam.py \
        output/LIDC-IDRI-0141_masked output/LIDC-IDRI-0141_xai \
        --nodules-dir output/LIDC-IDRI-0141_candidates_nodules \
        --head malignancy

    # Approximate placement only (no --nodules-dir)
    python 07_visualize_gradcam.py \
        output/LIDC-IDRI-0141_masked output/LIDC-IDRI-0141_xai \
        --head malignancy --placement-margin-mm 8

    # Headless: save a montage of the top-K candidates in full-lung context
    python 07_visualize_gradcam.py \
        output/LIDC-IDRI-0141_masked output/LIDC-IDRI-0141_xai \
        --save montage.png --top-k 8

Interactive viewer controls:
    mouse wheel / Left-Right arrows : change slice
    n / p                           : jump to next / previous candidate
                                       (sorted by --head score, descending)
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
except ImportError:
    print(
        "matplotlib is required. Install it with:\n"
        "    pip install matplotlib --break-system-packages",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from scipy import ndimage
except ImportError:
    print(
        "scipy is required. Install it with:\n"
        "    pip install scipy --break-system-packages",
        file=sys.stderr,
    )
    sys.exit(1)


LUNG_WINDOW_CENTER = -600
LUNG_WINDOW_WIDTH = 1500
HEATMAP_DISPLAY_MIN = 0.15  # heatmap values below this are treated as "no signal" (pure grayscale)


def apply_window(hu_slice, center=LUNG_WINDOW_CENTER, width=LUNG_WINDOW_WIDTH):
    low = center - width / 2.0
    high = center + width / 2.0
    windowed = np.clip(hu_slice, low, high)
    return (windowed - low) / (high - low)


def get_slice(volume, plane, index):
    if plane == "axial":
        return volume[index, :, :]
    elif plane == "coronal":
        return volume[:, index, :]
    elif plane == "sagittal":
        return volume[:, :, index]
    raise ValueError(f"Unknown plane '{plane}'")


def plane_length(volume, plane):
    if plane == "axial":
        return volume.shape[0]
    elif plane == "coronal":
        return volume.shape[1]
    elif plane == "sagittal":
        return volume.shape[2]
    raise ValueError(f"Unknown plane '{plane}'")


def load_meta(masked_dir):
    with open(os.path.join(masked_dir, "meta.json")) as f:
        return json.load(f)


def load_manifest(xai_dir):
    """Load candidate_results_manifest.csv (written by 06's
    save_candidate_results) -- has result_path + center_z/y/x + every
    head's score for every classified candidate."""
    path = os.path.join(xai_dir, "candidate_results_manifest.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"'{path}' not found. Pass 06_run_inference_xai.py's --out-dir "
            f"as xai_dir."
        )
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def load_ranked_extra(xai_dir):
    """Optionally load ranked_candidates.csv for equivalent_diameter_mm
    (used for the fallback placement box). Keyed by candidate_id
    (the bare int, as written by 06) -- candidate_results_manifest.csv
    uses a compound string id ('<patient>_nodule_XXXX'), so we match by
    parsing the trailing XXXX out of that string instead of assuming
    the two files share a literal key."""
    path = os.path.join(xai_dir, "ranked_candidates.csv")
    extra = {}
    if not os.path.exists(path):
        return extra
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            extra[int(row["candidate_id"])] = row
    return extra


def parse_trailing_candidate_id(compound_id: str):
    """'<patient>_nodule_0007' -> 7. Returns None if it doesn't match
    that pattern (never raises -- fallback placement doesn't strictly
    need this to succeed)."""
    try:
        return int(compound_id.rsplit("_", 1)[-1])
    except (ValueError, IndexError):
        return None


def load_nodule_bboxes(nodules_dir):
    """
    Try to load an EXACT per-candidate voxel bounding box from 05's
    nodules.json, checking a few plausible field names since the exact
    schema wasn't confirmed against your installed 05 version. Returns
    {candidate_id: (z0,z1,y0,y1,x0,x1)} for whichever candidates have
    one of these fields present; candidates without a usable field are
    simply absent from the returned dict (caller falls back to the
    approximate cube for those).
    """
    if not nodules_dir:
        return {}
    path = os.path.join(nodules_dir, "nodules.json")
    if not os.path.exists(path):
        print(f"[warn] --nodules-dir given but '{path}' not found -- "
              f"falling back to approximate placement for every candidate.")
        return {}

    with open(path) as f:
        data = json.load(f)
    nodules = data.get("nodules", data) if isinstance(data, dict) else data

    candidate_keys = ("bbox_zyx", "crop_bbox_zyx", "grown_bbox_zyx", "bbox")
    bboxes = {}
    for row in nodules:
        cid = row.get("candidate_id")
        if cid is None:
            continue
        for key in candidate_keys:
            if key in row and row[key]:
                val = row[key]
                # Accept either a flat 6-tuple/list or a
                # {"z":[lo,hi],"y":[lo,hi],"x":[lo,hi]} dict.
                if isinstance(val, dict) and all(k in val for k in ("z", "y", "x")):
                    z0, z1 = val["z"]
                    y0, y1 = val["y"]
                    x0, x1 = val["x"]
                elif isinstance(val, (list, tuple)) and len(val) == 6:
                    z0, z1, y0, y1, x0, x1 = val
                else:
                    continue
                bboxes[cid] = (int(z0), int(z1), int(y0), int(y1), int(x0), int(x1))
                break

    if not bboxes:
        print(f"[info] '{path}' exists but none of {candidate_keys} were found "
              f"in its rows -- falling back to approximate placement for "
              f"every candidate.")
    return bboxes


def smooth_full_heatmap(full_heatmap, spacing_zyx, sigma_mm):
    """
    Gaussian-blur the placed heatmap in PHYSICAL mm (anisotropic voxel
    sigma from spacing_zyx, same pattern used throughout this
    pipeline) so each candidate's hard placement-box edge feathers
    smoothly into the surrounding lung instead of cutting off abruptly
    -- that hard rectangular edge is exactly what a bare box-paste
    produces, since outside the box the value is exactly 0 with no
    transition. Renormalizes afterward so the peak value matches the
    pre-smoothing peak (blurring alone would otherwise dim the hottest
    point, since it's averaged with neighboring zeros).
    """
    if sigma_mm <= 0 or not full_heatmap.any():
        return full_heatmap
    sigma_vox = [sigma_mm / s for s in spacing_zyx]
    smoothed = ndimage.gaussian_filter(full_heatmap, sigma=sigma_vox)
    original_peak = full_heatmap.max()
    smoothed_peak = smoothed.max()
    if smoothed_peak > 0:
        smoothed = smoothed * (original_peak / smoothed_peak)
    return smoothed


def resample_to_shape(array, target_shape):
    """Resize a 3D array to target_shape via linear interpolation
    (order=1) -- same tool used throughout this pipeline for
    patch<->native-voxel resampling."""
    if array.shape == tuple(target_shape):
        return array
    zoom_factors = [t / s for t, s in zip(target_shape, array.shape)]
    return ndimage.zoom(array, zoom_factors, order=1, mode="nearest")


def compute_fallback_bbox(center_zyx, equivalent_diameter_mm, spacing_zyx, margin_mm, volume_shape):
    """Approximate placement box: a cube centered on center_zyx, sized
    from equivalent_diameter_mm (+ margin_mm on every side), converted
    to voxels via the volume's own physical spacing. Used only when no
    exact bbox is available (see load_nodule_bboxes)."""
    slice_spacing_mm, y_spacing_mm, x_spacing_mm = spacing_zyx
    diameter_mm = equivalent_diameter_mm if equivalent_diameter_mm else 10.0
    half_size_mm = diameter_mm / 2.0 + margin_mm

    half_vox = (
        int(np.ceil(half_size_mm / slice_spacing_mm)),
        int(np.ceil(half_size_mm / y_spacing_mm)),
        int(np.ceil(half_size_mm / x_spacing_mm)),
    )
    cz, cy, cx = center_zyx
    z0 = max(0, cz - half_vox[0]); z1 = min(volume_shape[0] - 1, cz + half_vox[0])
    y0 = max(0, cy - half_vox[1]); y1 = min(volume_shape[1] - 1, cy + half_vox[1])
    x0 = max(0, cx - half_vox[2]); x1 = min(volume_shape[2] - 1, cx + half_vox[2])
    return (z0, z1, y0, y1, x0, x1)


def build_full_volume_heatmap(volume_shape, candidates, head, exact_bboxes,
                               spacing_zyx, placement_margin_mm):
    """
    Build a (Z,Y,X) float32 heatmap over the full volume by placing
    each candidate's patch-level heatmap into its real location (see
    module docstring for the exact-vs-approximate placement logic),
    combining overlaps with elementwise max.

    Returns (full_heatmap, placement_info) where placement_info is a
    list of dicts (one per candidate) recording which placement method
    was used and the final bbox, for printing/debugging.
    """
    full_heatmap = np.zeros(volume_shape, dtype=np.float32)
    placement_info = []

    for cand in candidates:
        cid = cand["candidate_id"]
        heatmap_patch = cand["heatmap"]
        center_zyx = cand["center_zyx"]

        if cid in exact_bboxes:
            bbox = exact_bboxes[cid]
            method = "exact (05 bbox)"
        else:
            bbox = compute_fallback_bbox(
                center_zyx, cand.get("equivalent_diameter_mm"),
                spacing_zyx, placement_margin_mm, volume_shape,
            )
            method = "approximate (diameter + margin)"

        z0, z1, y0, y1, x0, x1 = bbox
        target_shape = (max(1, z1 - z0 + 1), max(1, y1 - y0 + 1), max(1, x1 - x0 + 1))
        resized = resample_to_shape(heatmap_patch, target_shape)

        region = full_heatmap[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
        full_heatmap[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1] = np.maximum(region, resized)

        placement_info.append({
            "candidate_id": cid, "method": method, "bbox": bbox,
            "center_zyx": center_zyx, "score": cand.get("score"),
        })
        print(f"[info] Candidate {cid}: placed via {method}, "
              f"bbox z[{z0}:{z1}] y[{y0}:{y1}] x[{x0}:{x1}], "
              f"{head}={cand.get('score')}")

    return full_heatmap, placement_info


def load_candidates(masked_dir, xai_dir, nodules_dir, head):
    meta = load_meta(masked_dir)
    manifest_rows = load_manifest(xai_dir)
    ranked_extra = load_ranked_extra(xai_dir)
    exact_bboxes = load_nodule_bboxes(nodules_dir)

    score_col = f"{head}_score"
    heatmap_key = f"{head}_heatmap"

    candidates = []
    for row in manifest_rows:
        result_path = row["result_path"]
        if not os.path.isabs(result_path):
            # result_path was written relative to the CWD 06 was run
            # from, which may not be this script's CWD -- try as-is
            # first, then relative to xai_dir.
            if not os.path.exists(result_path):
                candidate_path = os.path.join(xai_dir, os.path.basename(result_path))
                if os.path.exists(candidate_path):
                    result_path = candidate_path
        if not os.path.exists(result_path):
            print(f"[warn] Could not find result file for candidate "
                  f"'{row['candidate_id']}' (tried '{row['result_path']}') -- skipping.")
            continue

        data = np.load(result_path)
        if heatmap_key not in data:
            print(f"[warn] '{heatmap_key}' not found in {result_path} "
                  f"(available: {[k for k in data.files if k.endswith('_heatmap')]}) "
                  f"-- skipping this candidate.")
            continue

        compound_id = row["candidate_id"]
        trailing_id = parse_trailing_candidate_id(compound_id)
        extra = ranked_extra.get(trailing_id, {})

        candidates.append({
            "candidate_id": trailing_id if trailing_id is not None else compound_id,
            "compound_id": compound_id,
            "center_zyx": (int(float(row["center_z"])), int(float(row["center_y"])), int(float(row["center_x"]))),
            "heatmap": data[heatmap_key].astype(np.float32),
            "score": float(row[score_col]) if score_col in row and row[score_col] not in (None, "") else None,
            "equivalent_diameter_mm": float(extra["equivalent_diameter_mm"]) if extra.get("equivalent_diameter_mm") else None,
        })

    candidates.sort(key=lambda c: (c["score"] if c["score"] is not None else -1), reverse=True)
    return meta, candidates, exact_bboxes


def render_panels(volume_hu_masked, full_heatmap, plane, i):
    """Return (ct_only_rgb, ct_with_overlay_rgb) for one slice -- the
    same plain-CT panel and heatmap-overlay panel used by both the
    interactive viewer and the saved montage, so the two views are
    guaranteed to render identically."""
    ct_slice = get_slice(volume_hu_masked, plane, i)
    heat_slice = get_slice(full_heatmap, plane, i)
    gray = apply_window(ct_slice)
    ct_only = np.stack([gray, gray, gray], axis=-1)

    overlay = ct_only.copy()
    mask = heat_slice >= HEATMAP_DISPLAY_MIN
    if mask.any():
        cmap = matplotlib.colormaps.get_cmap("jet")
        heat_rgb = cmap(heat_slice)[..., :3]
        alpha = np.clip((heat_slice - HEATMAP_DISPLAY_MIN) / (1.0 - HEATMAP_DISPLAY_MIN), 0, 1) * 0.7
        alpha = alpha[..., None]
        overlay = np.where(mask[..., None], (1 - alpha) * overlay + alpha * heat_rgb, overlay)

    return np.clip(ct_only, 0, 1), np.clip(overlay, 0, 1)


def interactive_viewer(volume_hu_masked, full_heatmap, candidates, meta, head,
                        plane="axial", start_index=None):
    n = plane_length(volume_hu_masked, plane)
    if start_index is None:
        start_index = candidates[0]["center_zyx"][0] if candidates else n // 2
    idx = max(0, min(start_index, n - 1))
    current_candidate_idx = [0]

    fig, (ax_ct, ax_overlay) = plt.subplots(1, 2, figsize=(14, 7))
    plt.subplots_adjust(bottom=0.15)

    ct_only, overlay = render_panels(volume_hu_masked, full_heatmap, plane, idx)
    im_ct = ax_ct.imshow(ct_only)
    ax_ct.axis("off")
    ax_ct.set_title("Masked lung (CT)")
    im_overlay = ax_overlay.imshow(overlay)
    ax_overlay.axis("off")
    ax_overlay.set_title(f"Grad-CAM overlay ({head})")

    patient_id = meta.get("patient_id", "unknown")

    def nearby_candidates_text(i, tol=3):
        nearby = [c for c in candidates if abs(c["center_zyx"][0] - i) <= tol] if plane == "axial" else []
        if not nearby:
            return ""
        parts = [f"id={c['candidate_id']} {head}={c['score']:.3f}" for c in nearby if c["score"] is not None]
        return " | near: " + ", ".join(parts) if parts else ""

    def update_title(i):
        fig.suptitle(f"Patient {patient_id} -- {plane} slice {i + 1}/{n}{nearby_candidates_text(i)}", fontsize=11)

    update_title(idx)

    ax_slider = plt.axes([0.15, 0.03, 0.7, 0.03])
    slider = Slider(ax_slider, "Slice", 0, n - 1, valinit=idx, valstep=1)

    def on_slider(val):
        i = int(slider.val)
        ct_only, overlay = render_panels(volume_hu_masked, full_heatmap, plane, i)
        im_ct.set_data(ct_only)
        im_overlay.set_data(overlay)
        update_title(i)
        fig.canvas.draw_idle()

    slider.on_changed(on_slider)

    def on_scroll(event):
        step = 1 if event.button == "up" else -1
        slider.set_val(int(np.clip(slider.val + step, 0, n - 1)))

    def on_key(event):
        if event.key == "right":
            slider.set_val(int(np.clip(slider.val + 1, 0, n - 1)))
        elif event.key == "left":
            slider.set_val(int(np.clip(slider.val - 1, 0, n - 1)))
        elif event.key in ("n", "p") and candidates:
            direction = 1 if event.key == "n" else -1
            current_candidate_idx[0] = (current_candidate_idx[0] + direction) % len(candidates)
            target = candidates[current_candidate_idx[0]]
            target_index = target["center_zyx"][0] if plane == "axial" else \
                            target["center_zyx"][1] if plane == "coronal" else target["center_zyx"][2]
            print(f"[info] Jumped to candidate {target['candidate_id']} "
                  f"({head}={target['score']}), slice {target_index}")
            slider.set_val(int(np.clip(target_index, 0, n - 1)))

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)

    print(f"[info] Interactive viewer: scroll/arrows to change slice, "
          f"'n'/'p' to jump between candidates (sorted by {head}, "
          f"strongest first). {len(candidates)} candidate(s) loaded.")
    plt.show()


def save_montage(volume_hu_masked, full_heatmap, candidates, meta, out_path, head, top_k):
    to_render = candidates[:top_k]
    if not to_render:
        print("[info] No candidates to render.")
        return

    rows = len(to_render)
    fig, axes = plt.subplots(rows, 2, figsize=(9, 4.5 * rows))
    axes = np.atleast_2d(axes)
    if axes.shape[1] != 2:  # single-row case: plt.subplots gives a 1D array
        axes = axes.reshape(rows, 2)

    for row_idx, cand in enumerate(to_render):
        z = cand["center_zyx"][0]
        ct_only, overlay = render_panels(volume_hu_masked, full_heatmap, "axial", z)
        score = cand["score"]
        score_str = f"{score:.3f}" if score is not None else "n/a"

        ax_ct, ax_overlay = axes[row_idx]
        ax_ct.imshow(ct_only)
        ax_ct.set_title(f"id={cand['candidate_id']} -- masked lung (slice {z})", fontsize=9)
        ax_ct.axis("off")

        ax_overlay.imshow(overlay)
        ax_overlay.set_title(f"id={cand['candidate_id']} -- {head}={score_str}", fontsize=9)
        ax_overlay.axis("off")

    patient_id = meta.get("patient_id", "unknown")
    fig.suptitle(f"Patient {patient_id} -- top {len(to_render)} candidates by {head} (full-lung context)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] Saved montage to '{out_path}'")


def parse_args():
    parser = argparse.ArgumentParser(
        description="STEP 7: place per-candidate Grad-CAM heatmaps back "
        "into the full masked CT volume and view them in anatomical context."
    )
    parser.add_argument("masked_dir", help="02_mask_and_crop.py --out-dir.")
    parser.add_argument("xai_dir", help="06_run_inference_xai.py --out-dir.")
    parser.add_argument(
        "--nodules-dir", default=None,
        help="05_shape_filter_and_grow.py --out-dir. If given and its "
        "nodules.json has an explicit bounding box per candidate, that's "
        "used for EXACT placement; otherwise placement is approximated "
        "from equivalent_diameter_mm + --placement-margin-mm.",
    )
    parser.add_argument("--head", default="malignancy", help="Which head's heatmap to place/view (default: 'malignancy').")
    parser.add_argument("--placement-margin-mm", type=float, default=10.0, help="Margin added around equivalent_diameter_mm for the fallback placement box (default: 10.0mm). Ignored for candidates with an exact bbox available.")
    parser.add_argument("--plane", choices=["axial", "coronal", "sagittal"], default="axial")
    parser.add_argument("--slice", type=int, default=None, help="Starting slice (default: strongest candidate's own slice).")
    parser.add_argument("--save", default=None, help="Save a static top-K montage PNG instead of the interactive viewer.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of candidates in the saved montage (default: 8).")
    parser.add_argument(
        "--heatmap-smooth-mm", type=float, default=6.0,
        help="Gaussian-blur sigma in mm applied to the placed heatmap "
        "(default: 6.0mm) so each candidate's placement-box edge "
        "feathers smoothly into the surrounding lung instead of cutting "
        "off abruptly. Pass 0 to disable and see the raw hard-edged box.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    meta, candidates, exact_bboxes = load_candidates(args.masked_dir, args.xai_dir, args.nodules_dir, args.head)

    if not candidates:
        print("[info] No candidates with a usable heatmap were found -- nothing to show.")
        return

    print(f"[info] Loading masked volume from '{args.masked_dir}'...")
    volume_hu_masked = np.load(os.path.join(args.masked_dir, "volume_hu_masked.npy"))

    pixel_spacing_mm = meta.get("pixel_spacing_mm", [1.0, 1.0])
    slice_spacing_mm = meta.get("slice_spacing_mm", 1.0) or 1.0
    spacing_zyx = (slice_spacing_mm, pixel_spacing_mm[0], pixel_spacing_mm[1])

    full_heatmap, _ = build_full_volume_heatmap(
        volume_hu_masked.shape, candidates, args.head, exact_bboxes,
        spacing_zyx, args.placement_margin_mm,
    )
    full_heatmap = smooth_full_heatmap(full_heatmap, spacing_zyx, args.heatmap_smooth_mm)

    if args.save:
        save_montage(volume_hu_masked, full_heatmap, candidates, meta, args.save, args.head, args.top_k)
    else:
        interactive_viewer(volume_hu_masked, full_heatmap, candidates, meta, args.head,
                            plane=args.plane, start_index=args.slice)


if __name__ == "__main__":
    main()