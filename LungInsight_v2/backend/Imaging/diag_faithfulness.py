import importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch

repo = Path.cwd()
sys.path.insert(0, str(repo))
spec = importlib.util.spec_from_file_location("gradcam05", repo / "backend" / "Imaging" / "05_gradcam.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)

root = repo / "backend" / "output" / "LIDC-IDRI-0001"
cam_root = root / "05_explainability" / "gradcam_rainbow_final"
ck_root = repo / "backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/checkpoints/hybrid"
rd = lambda p: json.loads(p.read_text(encoding="utf-8-sig"))
man = rd(root / "02_detect_and_patch/patches/patches_manifest.json")
ct = np.load(root / "volume_hu_cropped.npy", mmap_mode="r")
dev = torch.device("cpu")
rng = np.random.default_rng(0)
models = {}

def get_model(a):
    k = g.CHECKPOINTS[a]
    if k not in models:
        models[k] = g.load_model(ck_root / k, dev)
    return models[k]

def rating(a, patch):
    m = get_model(a)
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(patch, dtype=np.float32))[None, None]
        return float(m.expected_ratings(m(g.preprocess_hu(x)))[0, g.MODEL_TARGETS.index(a)])

def block(patch, lo, size, fill=-1000.0):
    q = patch.copy()
    q[lo[0]:lo[0]+size[0], lo[1]:lo[1]+size[1], lo[2]:lo[2]+size[2]] = fill
    return q

def centered(c, size):
    return np.clip(np.round(np.asarray(c) - size / 2).astype(int), 0, 64 - size)

def overlaps(a, b, size):
    return bool(np.all(a < b + size) and np.all(b < a + size))

cc = lambda u, v: float(np.corrcoef(u.ravel(), v.ravel())[0, 1])

print("== 1. Occlusion (fill -1000 HU): d = change in expected rating ==")
for r in man["patches"]:
    pid = int(r["patch_id"])
    patch = np.load(root / f"02_detect_and_patch/patches/arrays/patch_{pid:03d}.npy").astype(np.float32)
    bs = np.clip(np.floor(r["bbox_local_start_zyx"]).astype(int), 0, 64)
    be = np.clip(np.ceil(r["bbox_local_end_zyx"]).astype(int), 0, 64)
    size = be - bs
    print(f"\npatch {pid}: detector box {bs.tolist()}..{be.tolist()} size {size.tolist()}")
    for a in g.ATTRIBUTES:
        raw = np.load(cam_root / f"patch_{pid:03d}" / f"gradcam_{a}_raw_signed.npy")
        base = rating(a, patch)
        pk = np.array(np.unravel_index(int(raw.argmax()), raw.shape))
        am = np.array(np.unravel_index(int(np.abs(raw).argmax()), raw.shape))
        d_det = rating(a, block(patch, bs, size)) - base
        d_pk = rating(a, block(patch, centered(pk, size), size)) - base
        d_am = rating(a, block(patch, centered(am, size), size)) - base
        rand = []
        while len(rand) < 24:
            lo = rng.integers(0, 64 - size + 1)
            if not overlaps(lo, bs, size):
                rand.append(rating(a, block(patch, lo, size)) - base)
        rand = np.array(rand)
        pct = lambda d: float((np.abs(rand) < abs(d)).mean())
        print(f"{a:11s} base={base:.3f} d_det={d_det:+.4f}(pct {pct(d_det):.2f}) "
              f"d_peak={d_pk:+.4f}(pct {pct(d_pk):.2f}) d_absmax={d_am:+.4f}(pct {pct(d_am):.2f}) "
              f"| random mean={rand.mean():+.4f} sd={rand.std():.4f}")

print("\n== 2. Blank-patch control (all -1000 HU) ==")
blank = np.full((64, 64, 64), -1000.0, dtype=np.float32)
for a in g.ATTRIBUTES:
    c, s, _ = g.calculate_cam(get_model(a), blank, a, dev)
    pk = tuple(int(v) for v in np.unravel_index(int(c.argmax()), c.shape))
    tr = tuple(int(v) for v in np.unravel_index(int(c.argmin()), c.shape))
    cors = [round(cc(c, np.load(cam_root / f"patch_{int(r['patch_id']):03d}" / f"gradcam_{a}_raw_signed.npy")), 2)
            for r in man["patches"]]
    print(f"{a:11s} rating={s:.3f} range=[{c.min():.3g},{c.max():.3g}] peak={pk} trough={tr} corr_with_patches={cors}")

print("\n== 3. Shift control: crop moved dz=-8, dx=+16 (whole cells) ==")
s = np.array([-8, 0, 16])
for r in man["patches"]:
    pid = int(r["patch_id"])
    cs = np.array(r["crop_start_zyx"], int) + s
    assert np.all(cs >= 0) and np.all(cs + 64 <= np.array(ct.shape)), "shifted crop leaves the CT"
    shifted = np.asarray(ct[cs[0]:cs[0]+64, cs[1]:cs[1]+64, cs[2]:cs[2]+64], dtype=np.float32)
    print(f"\npatch {pid}: content should move by {(-s).tolist()} voxels if the CAM follows content")
    for a in g.ATTRIBUTES:
        o = np.load(cam_root / f"patch_{pid:03d}" / f"gradcam_{a}_raw_signed.npy")
        c, _, _ = g.calculate_cam(get_model(a), shifted, a, dev)
        po = np.array(np.unravel_index(int(o.argmax()), o.shape))
        ps = np.array(np.unravel_index(int(c.argmax()), c.shape))
        print(f"{a:11s} corr aligned={cc(o[0:56, :, 16:64], c[8:64, :, 0:48]):+.2f} "
              f"same-index={cc(o[0:56, :, 16:64], c[0:56, :, 16:64]):+.2f} | "
              f"peak orig={po.tolist()} shifted={ps.tolist()} content-driven would give {(po - s).tolist()}")
