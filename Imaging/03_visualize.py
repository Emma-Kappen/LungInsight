"""
03_visualize.py

STEP 3 of 3 in the pipeline:
    01_dicom_to_hu.py       -> DICOM -> HU volume
    02_mask_and_crop.py     -> lung segmentation + non-lung blanking + Z-crop
    03_visualize.py         <- this file: viewing

Visualizes the output of 02_mask_and_crop.py: a CT volume (HU) together
with a mask. By default this shows the LUNG mask (lung_mask.npy) -- the
full lung region (parenchyma + vessel/airway walls + trachea/bronchi +
a pleural wall margin), not the raw physical air mask (air_mask.npy).
Pass --mask air if you specifically want to see the raw physical air
mask instead.

Note: 02_mask_and_crop.py already Z-crops volume_hu.npy down to lung-
containing slices by default (using a 3D-connectivity-verified lung
component, not a per-slice symmetry heuristic -- see that script's
docstring), so there usually isn't much non-lung anatomy left to look
at here. The view is ALSO auto-cropped to the lung mask's XY bounding
box (with a margin) for the same reason. Because lung_mask itself is
already anatomically verified in 3D (abdominal gas pockets are
excluded at the mask level, not just hidden by cropping), this XY crop
is a plain bounding box around every True voxel -- no per-slice
heuristics needed here. Pass --no-crop to see the full uncropped slice.

Two ways to look at it:

  1. Interactive viewer (default): scroll through slices with the
     mouse wheel / arrow keys. Left panel shows the raw HU slice on a
     standard lung window; middle panel shows the same slice with the
     mask region highlighted in an overlay color; right panel shows the
     already-blanked volume_hu_masked.npy so you can see exactly what
     "applying the mask" produced.

  2. Static montage (--save): renders a grid of evenly-spaced slices
     to a single PNG, useful for quick inspection or sharing/embedding
     in notes without needing an interactive window.

Usage:
    # Interactive (requires a display / X11 / VS Code interactive window)
    python 03_visualize.py output/LIDC-IDRI-0001_masked

    # Static montage saved to disk (works headless)
    python 03_visualize.py output/LIDC-IDRI-0001_masked \
        --save montage.png --num-slices 12

    # Pick a specific plane and starting slice
    python 03_visualize.py output/LIDC-IDRI-0001_masked --plane coronal

    # See the raw physical air mask instead of the lung mask
    python 03_visualize.py output/LIDC-IDRI-0001_masked --mask air

    # See the full, uncropped extent instead of the lung-cropped view
    python 03_visualize.py output/LIDC-IDRI-0001_masked --no-crop
"""

import argparse
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
        "scipy is required for the enhanced display mode. Install it with:\n"
        "    pip install scipy --break-system-packages",
        file=sys.stderr,
    )
    sys.exit(1)


# Standard "lung window" for CT display: center -600 HU, width 1500 HU.
LUNG_WINDOW_CENTER = -600
LUNG_WINDOW_WIDTH = 1500

# Standard "soft tissue" CT window: center 40 HU, width 400 HU -- shows
# mediastinum/heart/vessel detail but crushes the lungs to solid black.
SOFT_TISSUE_WINDOW_CENTER = 40
SOFT_TISSUE_WINDOW_WIDTH = 400

# Wide band used as the base for the "enhanced" display mode: covers
# both lung (~-950 to -500 HU) and soft tissue (~-100 to +80 HU) in one
# window, deliberately excluding bone (>+300 HU) so bone doesn't wash
# out the local-contrast stretch applied on top of it.
ENHANCED_DISPLAY_CENTER = -100
ENHANCED_DISPLAY_WIDTH = 1300

# Local-contrast (CLAHE-style) parameters.
LOCAL_CONTRAST_SIGMA_PX = 25
LOCAL_CONTRAST_CLIP_STD = 3.0
LOCAL_CONTRAST_BLEND = 0.6  # 0 = pure fixed window, 1 = pure local-contrast

# Unsharp-mask parameters for edge emphasis (vessel/airway walls,
# nodule margins).
UNSHARP_SIGMA_PX = 1.5
UNSHARP_AMOUNT = 1.2


def find_masked_dirs(search_root: str, max_results: int = 5):
    """Walk search_root looking for directories that contain a
    lung_mask.npy, to suggest likely candidates when the path the user
    gave doesn't resolve to anything."""
    matches = []
    for root, _dirs, files in os.walk(search_root):
        if "lung_mask.npy" in files:
            matches.append(root)
            if len(matches) >= max_results:
                break
    return matches


def resolve_masked_dir(masked_dir: str):
    """
    Try to locate the processed directory even if the path the caller
    gave doesn't resolve as-is (handles the common Windows leading-
    slash pitfall where '/output/...' anchors to the drive root
    instead of the current folder).

    Returns the resolved absolute directory path, or None if nothing
    was found.
    """
    candidates = [masked_dir]

    if os.path.isabs(masked_dir):
        stripped = masked_dir.lstrip("\\/")
        candidates.append(os.path.join(os.getcwd(), stripped))
    else:
        drive = os.path.splitdrive(os.getcwd())[0] or "C:"
        candidates.append(os.path.join(drive + os.sep, masked_dir))

    for candidate in candidates:
        if os.path.isdir(candidate) and os.path.exists(
            os.path.join(candidate, "lung_mask.npy")
        ):
            return os.path.abspath(candidate)

    return None


def load_volume(masked_dir: str):
    """Load volume_hu.npy, volume_hu_masked.npy, air_mask.npy,
    lung_mask.npy, heart_mask.npy (if present), and meta.json from a
    directory produced by 02_mask_and_crop.py."""
    resolved_dir = resolve_masked_dir(masked_dir)

    if resolved_dir is None:
        suggestions = find_masked_dirs(os.getcwd())
        msg_lines = [
            f"Could not find lung_mask.npy under '{masked_dir}'.",
            f"  Given path resolved to: {os.path.abspath(masked_dir)}",
            f"  Current working directory: {os.getcwd()}",
        ]
        if suggestions:
            msg_lines.append("Found these masked folders instead — did you mean one of these?")
            msg_lines.extend(f"    {s}" for s in suggestions)
        else:
            msg_lines.append(
                "No masked folders (containing lung_mask.npy) were found "
                "under the current directory at all. Run 01_dicom_to_hu.py "
                "then 02_mask_and_crop.py first."
            )
        raise FileNotFoundError("\n".join(msg_lines))

    if resolved_dir != os.path.abspath(masked_dir):
        print(f"[info] '{masked_dir}' not found as given; using '{resolved_dir}' instead.")

    masked_dir = resolved_dir
    vol_path = os.path.join(masked_dir, "volume_hu.npy")
    vol_masked_path = os.path.join(masked_dir, "volume_hu_masked.npy")
    air_mask_path = os.path.join(masked_dir, "air_mask.npy")
    lung_mask_path = os.path.join(masked_dir, "lung_mask.npy")
    heart_mask_path = os.path.join(masked_dir, "heart_mask.npy")
    meta_path = os.path.join(masked_dir, "meta.json")

    volume_hu = np.load(vol_path)
    air_mask = np.load(air_mask_path)
    lung_mask = np.load(lung_mask_path)

    if os.path.exists(heart_mask_path):
        heart_mask = np.load(heart_mask_path)
    else:
        print(
            "[warn] heart_mask.npy not found (older 02_mask_and_crop.py "
            "output) -- re-run it to get a dedicated heart mask. Using an "
            "empty heart mask for this view."
        )
        heart_mask = np.zeros_like(lung_mask)

    if os.path.exists(vol_masked_path):
        volume_hu_masked = np.load(vol_masked_path)
    else:
        print(
            "[warn] volume_hu_masked.npy not found -- re-run "
            "02_mask_and_crop.py to generate it. Falling back to computing "
            "it on the fly for this view."
        )
        volume_hu_masked = volume_hu.copy()
        volume_hu_masked[~lung_mask] = -1000.0

    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    for name, arr in [("air_mask", air_mask), ("lung_mask", lung_mask),
                       ("heart_mask", heart_mask), ("volume_hu_masked", volume_hu_masked)]:
        if arr.shape != volume_hu.shape:
            raise ValueError(
                f"volume_hu shape {volume_hu.shape} != {name} shape {arr.shape}."
            )

    return volume_hu, volume_hu_masked, air_mask, lung_mask, heart_mask, meta


def compute_crop_bbox(mask: np.ndarray, padding_frac: float = 0.12):
    """
    Bounding box (as slice objects for Z, Y, X) around the True region
    of `mask`, expanded by `padding_frac` of each dimension's extent so
    the crop isn't drawn flush against the lungs. Returns None if the
    mask is empty.

    This is a plain bounding box over every True voxel -- no per-slice
    heuristics are needed here, because lung_mask itself is already an
    anatomically-verified 3D structure by the time it reaches this
    script (02_mask_and_crop.py excludes abdominal gas pockets etc. at
    the mask level via 3D connected-component identity, not just via
    Z-cropping). See that script's docstring for details.
    """
    if not mask.any():
        return None

    zs, ys, xs = np.where(mask)
    bounds = []
    for coords, dim_size in zip((zs, ys, xs), mask.shape):
        lo, hi = int(coords.min()), int(coords.max())
        pad = int((hi - lo) * padding_frac)
        lo = max(0, lo - pad)
        hi = min(dim_size - 1, hi + pad)
        bounds.append(slice(lo, hi + 1))

    return tuple(bounds)


def crop_volume(volume: np.ndarray, bbox):
    """Apply a (z_slice, y_slice, x_slice) bounding box to a volume."""
    if bbox is None:
        return volume
    return volume[bbox]


def apply_window(hu_slice: np.ndarray, center=LUNG_WINDOW_CENTER, width=LUNG_WINDOW_WIDTH):
    """Map an HU slice to [0, 1] display intensities using a CT window."""
    low = center - width / 2.0
    high = center + width / 2.0
    windowed = np.clip(hu_slice, low, high)
    return (windowed - low) / (high - low)


def local_contrast_normalize(
    image01: np.ndarray, sigma_px=LOCAL_CONTRAST_SIGMA_PX, clip_std=LOCAL_CONTRAST_CLIP_STD,
):
    """
    Adaptive local-contrast stretch (a CLAHE-style enhancement, built
    only from scipy.ndimage.uniform_filter so it needs no extra
    dependency beyond what this script already requires).
    """
    local_mean = ndimage.uniform_filter(image01, size=sigma_px)
    local_sqmean = ndimage.uniform_filter(image01 ** 2, size=sigma_px)
    local_var = np.clip(local_sqmean - local_mean ** 2, 1e-8, None)
    local_std = np.sqrt(local_var)
    normalized = (image01 - local_mean) / (local_std + 1e-3)
    normalized = np.clip(normalized, -clip_std, clip_std)
    return (normalized + clip_std) / (2 * clip_std)


def unsharp_mask(image01: np.ndarray, sigma_px=UNSHARP_SIGMA_PX, amount=UNSHARP_AMOUNT):
    """
    Sharpen vessel walls, airway walls, and nodule margins by boosting
    high-frequency detail.
    """
    blurred = ndimage.gaussian_filter(image01, sigma=sigma_px)
    sharpened = image01 + amount * (image01 - blurred)
    return np.clip(sharpened, 0.0, 1.0)


def apply_enhanced_window(hu_slice: np.ndarray, blend=LOCAL_CONTRAST_BLEND):
    """
    Display windowing that fixes the two things a plain fixed lung
    window does badly: soft tissue reads as a near-featureless
    light-gray blob, and edges look soft. Fix: start from a window
    wide enough to cover both lung and soft tissue, blend in the
    adaptive local-contrast stretch, then unsharp-mask the result.
    """
    base = apply_window(hu_slice, ENHANCED_DISPLAY_CENTER, ENHANCED_DISPLAY_WIDTH)
    local = local_contrast_normalize(base)
    blended = (1 - blend) * base + blend * local
    return unsharp_mask(blended)


def render_display(hu_slice: np.ndarray, mode: str = "enhanced"):
    """Dispatch to the selected display mode (see --display CLI flag)."""
    if mode == "lung":
        return apply_window(hu_slice, LUNG_WINDOW_CENTER, LUNG_WINDOW_WIDTH)
    elif mode == "soft-tissue":
        return apply_window(hu_slice, SOFT_TISSUE_WINDOW_CENTER, SOFT_TISSUE_WINDOW_WIDTH)
    elif mode == "enhanced":
        return apply_enhanced_window(hu_slice)
    else:
        raise ValueError(f"Unknown display mode '{mode}'")


def render_masked_display(hu_slice: np.ndarray, mask_slice: np.ndarray, mode: str = "enhanced"):
    """
    Render a slice that has had non-lung voxels blanked to a flat HU
    value (volume_hu_masked.npy), forcing the blanked region to pure
    black regardless of display mode.

    Why this is needed: 'enhanced' mode's local-contrast step rescales
    each pixel relative to the mean/std of its own neighborhood so
    low-contrast anatomy doesn't get crushed. That's exactly wrong for
    a blanked region, which is perfectly FLAT (every voxel is the same
    HU) -- for a flat patch, (pixel - local_mean) / (local_std + eps)
    is ~0/eps, which maps to the MIDDLE of the display range (mid-
    gray), not the bottom. The plain windowing step alone puts blanked
    voxels at black correctly, but the local-contrast blend on top
    pulls them back toward gray. Re-blackening the blanked region after
    rendering sidesteps that artifact instead of relying on the
    enhancement math to behave on a region it was never meant to act on.
    """
    rendered = render_display(hu_slice, mode)
    rendered = rendered.copy()
    rendered[~mask_slice] = 0.0
    return rendered


def get_slice(volume: np.ndarray, plane: str, index: int) -> np.ndarray:
    """Extract a 2D slice from a (Z, Y, X) volume along the given plane."""
    if plane == "axial":
        return volume[index, :, :]
    elif plane == "coronal":
        return volume[:, index, :]
    elif plane == "sagittal":
        return volume[:, :, index]
    else:
        raise ValueError(f"Unknown plane '{plane}'")


def plane_length(volume: np.ndarray, plane: str) -> int:
    if plane == "axial":
        return volume.shape[0]
    elif plane == "coronal":
        return volume.shape[1]
    elif plane == "sagittal":
        return volume.shape[2]
    else:
        raise ValueError(f"Unknown plane '{plane}'")


def build_overlay_rgb(
    hu_slice: np.ndarray, mask_layers, display_mode: str = "enhanced",
):
    """
    Grayscale HU slice (rendered per `display_mode`) with one or more
    mask regions each tinted their own translucent color, so mask
    boundaries are easy to see against anatomy.

    `mask_layers` is a list of (mask_slice, color) pairs, applied in
    order (later layers drawn on top). A single mask can still be
    passed as [(mask_slice, color)].
    """
    gray = render_display(hu_slice, display_mode)
    rgb = np.stack([gray, gray, gray], axis=-1)

    alpha = 0.45
    for mask_slice, color in mask_layers:
        color_arr = np.array(color)
        rgb[mask_slice] = (1 - alpha) * rgb[mask_slice] + alpha * color_arr
    return np.clip(rgb, 0, 1)


def interactive_viewer(
    volume_hu, volume_hu_masked, mask_layers, blank_mask, overlay_title, meta,
    plane="axial", start_index=None, display_mode="enhanced",
):
    """
    `mask_layers` is a list of (full_volume_mask, color) pairs to draw
    in the overlay panel. `blank_mask` is always the true lung_mask --
    i.e. what volume_hu_masked was actually blanked against -- so panel
    3 renders correctly regardless of which mask(s) the user chose to
    visualize in panel 2.
    """
    n = plane_length(volume_hu, plane)
    idx = start_index if start_index is not None else n // 2
    idx = max(0, min(idx, n - 1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    plt.subplots_adjust(bottom=0.15)

    hu_slice = get_slice(volume_hu, plane, idx)
    masked_slice = get_slice(volume_hu_masked, plane, idx)
    blank_mask_slice = get_slice(blank_mask, plane, idx)
    layer_slices = [(get_slice(m, plane, idx), c) for m, c in mask_layers]

    im0 = axes[0].imshow(render_display(hu_slice, display_mode), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"CT ({display_mode} display)")
    axes[0].axis("off")

    im1 = axes[1].imshow(build_overlay_rgb(hu_slice, layer_slices, display_mode))
    axes[1].set_title(f"{overlay_title} overlay")
    axes[1].axis("off")

    im2 = axes[2].imshow(render_masked_display(masked_slice, blank_mask_slice, display_mode), cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Mask applied\n(non-lung blanked)")
    axes[2].axis("off")

    patient_id = meta.get("patient_id", "unknown")
    fig.suptitle(f"Patient {patient_id} — {plane} slice {idx + 1}/{n}")

    ax_slider = plt.axes([0.2, 0.03, 0.6, 0.03])
    slider = Slider(ax_slider, "Slice", 0, n - 1, valinit=idx, valstep=1)

    def update(val):
        i = int(slider.val)
        hu_s = get_slice(volume_hu, plane, i)
        masked_s = get_slice(volume_hu_masked, plane, i)
        blank_s = get_slice(blank_mask, plane, i)
        layer_s = [(get_slice(m, plane, i), c) for m, c in mask_layers]

        im0.set_data(render_display(hu_s, display_mode))
        im1.set_data(build_overlay_rgb(hu_s, layer_s, display_mode))
        im2.set_data(render_masked_display(masked_s, blank_s, display_mode))
        fig.suptitle(f"Patient {patient_id} — {plane} slice {i + 1}/{n}")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    def on_scroll(event):
        step = 1 if event.button == "up" else -1
        slider.set_val(int(np.clip(slider.val + step, 0, n - 1)))

    def on_key(event):
        if event.key == "right":
            slider.set_val(int(np.clip(slider.val + 1, 0, n - 1)))
        elif event.key == "left":
            slider.set_val(int(np.clip(slider.val - 1, 0, n - 1)))

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)

    print("[info] Interactive viewer: scroll wheel or Left/Right arrows to change slice.")
    plt.show()


def save_montage(
    volume_hu, mask_layers, overlay_title, meta, out_path, plane="axial", num_slices=12,
    display_mode="enhanced",
):
    n = plane_length(volume_hu, plane)
    num_slices = min(num_slices, n)
    indices = np.linspace(0, n - 1, num_slices, dtype=int)

    cols = min(4, num_slices)
    rows = int(np.ceil(num_slices / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, i in zip(axes, indices):
        hu_slice = get_slice(volume_hu, plane, i)
        layer_slices = [(get_slice(m, plane, i), c) for m, c in mask_layers]
        ax.imshow(build_overlay_rgb(hu_slice, layer_slices, display_mode))
        ax.set_title(f"{plane} slice {i}", fontsize=9)
        ax.axis("off")

    for ax in axes[len(indices):]:
        ax.axis("off")

    patient_id = meta.get("patient_id", "unknown")
    fig.suptitle(f"Patient {patient_id} — {overlay_title} overlay ({plane})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] Saved montage to '{out_path}'")


def parse_args():
    parser = argparse.ArgumentParser(
        description="STEP 3/3: Visualize a CT volume with its mask "
        "applied, using output from 02_mask_and_crop.py."
    )
    parser.add_argument(
        "masked_dir",
        help="Directory containing volume_hu.npy / volume_hu_masked.npy / "
        "air_mask.npy / lung_mask.npy / meta.json (the --out-dir from "
        "02_mask_and_crop.py).",
    )
    parser.add_argument(
        "--plane",
        choices=["axial", "coronal", "sagittal"],
        default="axial",
        help="Anatomical plane to slice through (default: axial).",
    )
    parser.add_argument(
        "--slice",
        type=int,
        default=None,
        help="Starting slice index for the interactive viewer "
        "(default: middle slice).",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="If set, skip the interactive viewer and save a static "
        "montage PNG to this path instead (works headless/no display).",
    )
    parser.add_argument(
        "--num-slices",
        type=int,
        default=12,
        help="Number of slices to include in the montage (default: 12). "
        "Only used with --save.",
    )
    parser.add_argument(
        "--mask",
        choices=["lung", "air", "heart", "lung+heart"],
        default="lung+heart",
        help="Which mask to visualize: 'lung' is the segmented lung "
        "region (heart already excluded); 'air' is the raw physical-air "
        "threshold; 'heart' is the dedicated heart/great-vessel mask; "
        "'lung+heart' (default) overlays both at once (green = lung, "
        "red = heart) so you can see the boundary between them.",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Show the full, uncropped extent instead of auto-cropping to "
        "the lung mask's bounding box.",
    )
    parser.add_argument(
        "--display",
        choices=["enhanced", "lung", "soft-tissue"],
        default="enhanced",
        help="How to render the grayscale CT background (default: "
        "'enhanced').",
    )
    return parser.parse_args()


LUNG_COLOR = (0.15, 0.75, 0.45)   # green
AIR_COLOR = (0.15, 0.55, 1.0)     # blue
HEART_COLOR = (0.85, 0.2, 0.2)    # red


def main():
    args = parse_args()
    volume_hu, volume_hu_masked, air_mask, lung_mask, heart_mask, meta = load_volume(args.masked_dir)

    print(f"[info] Loaded volume {volume_hu.shape} (Z, Y, X) for patient "
          f"{meta.get('patient_id', 'unknown')}.")

    if args.mask == "lung":
        mask_layers = [(lung_mask, LUNG_COLOR)]
        overlay_title = "Lung region mask"
    elif args.mask == "air":
        mask_layers = [(air_mask, AIR_COLOR)]
        overlay_title = "Air mask"
    elif args.mask == "heart":
        mask_layers = [(heart_mask, HEART_COLOR)]
        overlay_title = "Heart mask"
        if not heart_mask.any():
            print("[warn] heart_mask is empty for this scan (no qualifying "
                  "candidate found, or an older 02_mask_and_crop.py output "
                  "without heart_mask.npy).")
    else:  # "lung+heart"
        mask_layers = [(lung_mask, LUNG_COLOR), (heart_mask, HEART_COLOR)]
        overlay_title = "Lung (green) + heart (red)"

    crop_bbox = None
    if not args.no_crop:
        # Always crop to the lung mask's extent (even when displaying
        # the air mask) so the view stays anchored to the anatomy of
        # interest instead of jumping around with --mask air.
        crop_bbox = compute_crop_bbox(lung_mask)
        if crop_bbox is None:
            print("[warn] Lung mask is empty; showing the full uncropped volume.")

    if crop_bbox is not None:
        volume_hu = crop_volume(volume_hu, crop_bbox)
        volume_hu_masked = crop_volume(volume_hu_masked, crop_bbox)
        lung_mask = crop_volume(lung_mask, crop_bbox)
        mask_layers = [(crop_volume(m, crop_bbox), c) for m, c in mask_layers]
        print(f"[info] Cropped to lung bounding box (+ margin): "
              f"Z {crop_bbox[0].start}:{crop_bbox[0].stop}, "
              f"Y {crop_bbox[1].start}:{crop_bbox[1].stop}, "
              f"X {crop_bbox[2].start}:{crop_bbox[2].stop}")

    if args.save:
        save_montage(
            volume_hu, mask_layers, overlay_title, meta, args.save,
            plane=args.plane, num_slices=args.num_slices, display_mode=args.display,
        )
    else:
        interactive_viewer(
            volume_hu, volume_hu_masked, mask_layers, lung_mask, overlay_title, meta,
            plane=args.plane, start_index=args.slice, display_mode=args.display,
        )


if __name__ == "__main__":
    main()