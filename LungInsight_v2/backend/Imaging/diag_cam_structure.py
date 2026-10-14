import json
from pathlib import Path
import numpy as np

root = Path("backend/output/LIDC-IDRI-0001")
rd = lambda p: json.loads(p.read_text(encoding="utf-8-sig"))
meta = rd(root / "meta.json")
man = rd(root / "02_detect_and_patch/patches/patches_manifest.json")
cam_root = root / "05_explainability/gradcam_rainbow_final"
sp = np.array(meta["spacing_zyx_mm"], float)
D = np.array(meta["direction_lps"], float)

print("== A. In-plane axis orientation ==")
for k in ("origin_mm", "voxel_000_lps_mm", "last_voxel_index_zyx", "last_voxel_lps_mm",
          "crop_offset_zyx", "_z_step_sign", "final_shape_zyx"):
    print(f"{k}: {meta.get(k)}")
last = np.array(meta["last_voxel_index_zyx"], float)
print("origin + D@(last*sp):", (np.array(meta["origin_mm"]) + D @ (last * sp)).round(2).tolist())
lung = np.load(root / "lung_mask.npy", mmap_mode="r")
ext = {}
for i, (name, axes) in enumerate((("z", (1, 2)), ("y", (0, 2)), ("x", (0, 1)))):
    idx = np.flatnonzero(np.asarray(lung.any(axis=axes)))
    ext[name] = (int(idx[0]), int(idx[-1]), round(float((idx[-1] - idx[0] + 1) * sp[i]), 1))
print("lung extent [first,last,mm]:", ext, "(lungs are normally wider left-right than front-back)")
for r in man["patches"]:
    al = r.get("anatomical_location") or {}
    print("patch", r["patch_id"], "center_zyx", np.round(r["center_zyx"], 1).tolist(),
          {k: al.get(k) for k in ("primary", "region", "laterality", "compartment")})

print("\n== B. CAM structure vs patch geometry ==")
zz, yy, xx = np.meshgrid(np.arange(64), np.arange(64), np.arange(64), indexing="ij")
for r in man["patches"]:
    pid = int(r["patch_id"]); d = cam_root / f"patch_{pid:03d}"
    patch = np.clip(np.load(root / f"02_detect_and_patch/patches/arrays/patch_{pid:03d}.npy"), -1000, 400)
    c0 = np.array(r["patch_center_local_zyx"], float)
    rad = np.sqrt(((zz - c0[0]) * sp[0]) ** 2 + ((yy - c0[1]) * sp[1]) ** 2 + ((xx - c0[2]) * sp[2]) ** 2)
    bs = np.clip(np.floor(r["bbox_local_start_zyx"]).astype(int), 0, 64)
    be = np.clip(np.ceil(r["bbox_local_end_zyx"]).astype(int), 0, 64)
    box = np.zeros((64, 64, 64), bool); box[bs[0]:be[0], bs[1]:be[1], bs[2]:be[2]] = True
    inside = lambda p: bool(box[tuple(p)])
    print(f"\npatch {pid}: bbox_local {bs.tolist()}..{be.tolist()} volume fraction={box.mean():.4f}")
    for a in ("subtlety", "sphericity", "margin", "lobulation", "spiculation", "texture"):
        c = np.load(d / f"gradcam_{a}_raw_signed.npy")
        m = rd(d / f"gradcam_{a}_metadata.json")
        pk = np.array(np.unravel_index(int(c.argmax()), c.shape))
        tr = np.array(np.unravel_index(int(c.argmin()), c.shape))
        rel = c - c.min(); ab = np.abs(c)
        enr = lambda w: float(w[box].sum() / w.sum() / box.mean())
        print(f"{a:12s} act_shape={m.get('activation_shape')} peak={pk.tolist()} "
              f"d_mm={((pk - c0) * sp).round(1).tolist()} in_box={inside(pk)} | "
              f"trough={tr.tolist()} in_box={inside(tr)} | "
              f"box enrichment rel={enr(rel):.2f} abs={enr(ab):.2f} | "
              f"corr(-radius)={np.corrcoef(c.ravel(), -rad.ravel())[0, 1]:.2f} "
              f"corr(HU)={np.corrcoef(c.ravel(), patch.ravel())[0, 1]:.2f}")
