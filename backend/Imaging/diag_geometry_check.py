import hashlib, json
from itertools import combinations
from pathlib import Path
import numpy as np

root = Path("backend/output/LIDC-IDRI-0001")
rd = lambda p: json.loads(p.read_text(encoding="utf-8-sig"))
ct = np.load(root / "volume_hu_cropped.npy", mmap_mode="r")
meta = rd(root / "meta.json")
man = rd(root / "02_detect_and_patch/patches/patches_manifest.json")
cam_root = root / "05_explainability/gradcam_rainbow_final"
print("CT shape ZYX:", ct.shape)
print("spacing_zyx_mm:", meta.get("spacing_zyx_mm"))
print("direction_lps:", meta.get("direction_lps"))
print("meta keys:", sorted(meta.keys()))

for r in man["patches"]:
    pid = int(r["patch_id"])
    patch = np.load(root / f"02_detect_and_patch/patches/arrays/patch_{pid:03d}.npy")
    cs = np.array(r["crop_start_zyx"]); ce = np.array(r["crop_end_zyx"])
    print(f"\n=== patch {pid} / candidate {r.get('candidate_id')} ===")
    print("patch shape:", patch.shape, "| crop extent:", (ce - cs).tolist(), "| crop_start:", cs.tolist())
    a = np.maximum(cs, 0); b = np.minimum(ce, np.array(ct.shape))
    sl_ct = tuple(slice(int(i), int(j)) for i, j in zip(a, b))
    sl_pt = tuple(slice(int(i - c), int(j - c)) for i, j, c in zip(a, b, cs))
    same = np.array_equal(np.asarray(ct[sl_ct]), patch[sl_pt])
    pad = np.ones(patch.shape, bool); pad[sl_pt] = False
    print("patch == CT[crop clip]:", same)
    print("padded voxels:", int(pad.sum()), "| all -1000:", bool((patch[pad] == -1000).all()) if pad.any() else "n/a")
    bs = np.array(r["bbox_start_zyx"]); be = np.array(r["bbox_end_zyx"])
    print("bbox [start,end) ZYX:", bs.tolist(), be.tolist(), "| Z slices:", int(be[0] - bs[0]))
    print("center_zyx:", r["center_zyx"], "| center - crop_start:", (np.array(r["center_zyx"]) - cs).tolist(),
          "| patch_center_local:", r["patch_center_local_zyx"])

by_dir = {}
for f in sorted(cam_root.rglob("gradcam_*_raw_signed.npy")):
    c = np.load(f)
    lo, hi = float(c.min()), float(c.max())
    zero_pos = (0 - lo) / (hi - lo) if hi > lo else float("nan")
    am = np.unravel_index(int(c.argmax()), c.shape)
    h = hashlib.sha1(c.tobytes()).hexdigest()[:10]
    print(f"{f.relative_to(cam_root)} shape={c.shape} sha1={h} min={lo:.4g} max={hi:.4g} "
          f"frac>0={(c > 0).mean():.2f} zero->{zero_pos:.2f} argmaxZYX={am}")
    by_dir.setdefault(f.parent, {})[f.name] = c
for d, items in by_dir.items():
    for (n1, c1), (n2, c2) in combinations(items.items(), 2):
        print(f"corr [{d.name}] {n1[8:-15]} vs {n2[8:-15]} = {np.corrcoef(c1.ravel(), c2.ravel())[0, 1]:.3f}")
