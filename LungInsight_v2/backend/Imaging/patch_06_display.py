from pathlib import Path

path = Path(r"backend\Imaging\06_full_ct_gradcam.py")
src = path.read_text(encoding="utf-8")

def once(text, old):
    assert text.count(old) == 1, f"expected exactly one match for {old[:70]!r}, found {text.count(old)}"

def swap_block(text, start, end, new):
    once(text, start)
    once(text, end)
    i, j = text.index(start), text.index(end)
    assert i < j
    return text[:i] + new + text[j:]

def replace_once(text, old, new):
    once(text, old)
    return text.replace(old, new)

NEW_OUTLINE = r'''BBOX_EDGE_COLOR = "#FF00FF"  # detector box; deliberately outside the Rainbow palette


def add_patch_outline(ax, record: dict[str, Any], z: int, cam_shape: tuple[int, ...]) -> None:
    # Faint dashed footprint of the whole 64^3 patch (not a nodule mask).
    start = np.asarray(record["crop_start_zyx"], dtype=np.int64)
    end = start + np.asarray(cam_shape, dtype=np.int64)
    if start[0] <= z < end[0]:
        y0, x0 = max(0, int(start[1])), max(0, int(start[2]))
        ax.add_patch(
            Rectangle(
                (x0 - 0.5, y0 - 0.5),
                max(0, int(end[2]) - x0),
                max(0, int(end[1]) - y0),
                linewidth=0.8,
                edgecolor=PATCH_EDGE_COLOR,
                facecolor="none",
                linestyle="--",
                alpha=0.45,
            )
        )

    # Detector box in source-CT voxel coordinates: start inclusive, end exclusive.
    bbox_start = record.get("bbox_start_zyx")
    bbox_end = record.get("bbox_end_zyx")
    if bbox_start is None or bbox_end is None:
        return
    bs = np.floor(np.asarray(bbox_start, dtype=np.float64) + 1e-6).astype(np.int64)
    be = np.ceil(np.asarray(bbox_end, dtype=np.float64) - 1e-6).astype(np.int64)
    if bs[0] <= z < be[0]:
        ax.add_patch(
            Rectangle(
                (bs[2] - 0.5, bs[1] - 0.5),
                int(be[2] - bs[2]),
                int(be[1] - bs[1]),
                linewidth=1.8,
                edgecolor=BBOX_EDGE_COLOR,
                facecolor="none",
            )
        )


'''

NEW_RENDER = r'''def raw_signed_summary(cam_path: Path, attribute: str) -> dict[str, float] | None:
    # Raw signed CAM stored next to the loaded one, if present.
    raw_path = cam_path.parent / f"gradcam_{attribute}_raw_signed.npy"
    if not raw_path.is_file():
        return None
    raw = np.load(raw_path)
    lo, hi = float(raw.min()), float(raw.max())
    zero = (0.0 - lo) / (hi - lo) if hi - lo > 1e-12 else float("nan")
    return {
        "raw_min": lo,
        "raw_max": hi,
        "zero_position": zero,
        "fraction_positive": float((raw > 0).mean()),
    }


def render_slice(
    ct_volume: np.ndarray,
    cam_volume: np.ndarray,
    coverage_count: np.ndarray,
    z: int,
    attr: str,
    out_path: Path,
    records_with_cams: list[tuple[dict[str, Any], tuple[int, int, int]]],
    alpha: float,
    window_level: float,
    window_width: float,
    signed_info: dict[int, dict[str, float] | None] | None = None,
) -> None:
    ct_slice = ct_volume[z]
    cam_slice = cam_volume[z]
    valid = coverage_count[z] > 0

    rgba = RAINBOW(Normalize(vmin=0.0, vmax=1.0)(np.nan_to_num(cam_slice, nan=0.0)))
    rgba[..., 3] = valid.astype(np.float32) * alpha

    fig, ax = plt.subplots(figsize=(8.2, 8.0))
    vmin = window_level - window_width / 2.0
    vmax = window_level + window_width / 2.0
    # origin="upper": array row 0 (anterior) at the top, the usual radiological orientation.
    ax.imshow(ct_slice, cmap="gray", vmin=vmin, vmax=vmax, origin="upper", interpolation="nearest")
    ax.imshow(rgba, origin="upper", interpolation="nearest")

    sm = ScalarMappable(norm=Normalize(0.0, 1.0), cmap=RAINBOW)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Relative patch CAM: blue low \u2192 red high")

    notes = []
    for record, shape in records_with_cams:
        add_patch_outline(ax, record, z, shape)
        crop_z = int(record["crop_start_zyx"][0])
        if not (crop_z <= z < crop_z + int(shape[0])):
            continue
        line = f"patch {record['patch_id']} / candidate {record.get('candidate_id', '?')}"
        info = (signed_info or {}).get(int(record["patch_id"]))
        if info is not None:
            zero = info["zero_position"]
            line += f": raw CAM {info['raw_min']:+.2e} to {info['raw_max']:+.2e}"
            if math.isnan(zero):
                line += ", constant map"
            elif zero < 0.0:
                line += ", all raw values positive (zero is below the scale)"
            elif zero > 1.0:
                line += ", all raw values negative (zero is above the scale)"
            else:
                line += f", raw zero at {zero:.2f} of the scale (black/white bar on colorbar)"
                cbar.ax.axhline(zero, color="black", linewidth=3.0)
                cbar.ax.axhline(zero, color="white", linewidth=1.2)
        notes.append(line)

    ax.set_title(f"{attr.replace('_', ' ').title()} \u2014 full processed CT, axial Z={z}")
    ax.set_xlabel("CT X voxel index")
    ax.set_ylabel("CT Y voxel index")
    ax.set_xlim(-0.5, ct_volume.shape[2] - 0.5)
    ax.set_ylim(ct_volume.shape[1] - 0.5, -0.5)

    footer = (
        "Dashed yellow = full 64-voxel patch footprint; magenta box = detector box, drawn only on slices "
        "inside its Z interval. Colors are relative within each patch map, not voxelwise attribute scores."
    )
    if notes:
        footer += "\n" + "\n".join(notes)
    fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=7)
    fig.tight_layout(rect=(0, 0.05 + 0.02 * len(notes), 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


'''

src = swap_block(src, "def add_patch_outline(", "def render_slice(", NEW_OUTLINE)
src = swap_block(src, "def render_slice(", "def create_montage(", NEW_RENDER)

src = replace_once(
    src,
    "used_patch_data: list[dict[str, Any]] = []",
    "used_patch_data: list[dict[str, Any]] = []\n        signed_info: dict[int, dict[str, float] | None] = {}",
)

CROP_OLD = 'crop_start = np.asarray(record["crop_start_zyx"], dtype=np.int64)'
CROP_NEW = CROP_OLD + r'''
            crop_end_manifest = record.get("crop_end_zyx")
            if crop_end_manifest is not None:
                extent = np.rint(np.asarray(crop_end_manifest, dtype=np.float64)).astype(np.int64) - crop_start
                if extent.tolist() != [int(v) for v in cam.shape]:
                    raise ValueError(
                        f"patch {pid}: crop_end - crop_start = {extent.tolist()} but CAM shape is {list(cam.shape)}."
                    )
            signed_info[pid] = raw_signed_summary(cam_path, attribute)'''
src = replace_once(src, CROP_OLD, CROP_NEW)

src = replace_once(
    src,
    '"mapped_ct_end_zyx_exclusive": ct_end.tolist(),',
    '"mapped_ct_end_zyx_exclusive": ct_end.tolist(),\n                "raw_signed_summary": signed_info.get(pid),',
)
src = replace_once(
    src,
    "records_with_cams=loaded_records,",
    "records_with_cams=loaded_records,\n                signed_info=signed_info,",
)

path.write_text(src, encoding="utf-8")
print("06_full_ct_gradcam.py patched.")
