import importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch
from scipy.ndimage import map_coordinates

repo = Path.cwd()
sys.path.insert(0, str(repo))
spec = importlib.util.spec_from_file_location("gradcam05", repo / "backend" / "Imaging" / "05_gradcam.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)

root = repo / "backend" / "output" / "LIDC-IDRI-0001"
cam_root = root / "05_explainability" / "gradcam_rainbow_final"
ck_root = repo / "backend/models/nodule_ranker_3d_distributional_v1/spatial_v1/checkpoints/hybrid"
rd = lambda p: json.loads(p.read_text(encoding="utf-8-sig"))
meta = rd(root / "meta.json")
man = rd(root / "02_detect_and_patch/patches/patches_manifest.json")
ct = np.array(np.load(root / "volume_hu_cropped.npy"), dtype=np.float32)
lung = np.load(root / "lung_mask.npy")
sp = np.array(meta["spacing_zyx_mm"], float)
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
        x = torch.from_numpy(np.array(patch, dtype=np.float32))[None, None]
        return float(m.expected_ratings(m(g.preprocess_hu(x)))[0, g.MODEL_TARGETS.index(a)])

cc = lambda u, v: float(np.corrcoef(np.ravel(u), np.ravel(v))[0, 1])
patches = {int(r["patch_id"]): np.load(root / f"02_detect_and_patch/patches/arrays/patch_{int(r['patch_id']):03d}.npy").astype(np.float32)
           for r in man["patches"]}

print("== A. Rating spread across very different inputs (expected rating, 1-5 scale) ==")
fixed_inputs = {"blank_-1000": np.full((64, 64, 64), -1000.0, np.float32),
                "uniform_-800": np.full((64, 64, 64), -800.0, np.float32),
                "uniform_+40": np.full((64, 64, 64), 40.0, np.float32)}
lung_vox = np.argwhere(lung)
crops = []
for i in rng.choice(len(lung_vox), 40, replace=False):
    s = np.clip(lung_vox[i] - 32, 0, np.array(ct.shape) - 64)
    crops.append(ct[s[0]:s[0]+64, s[1]:s[1]+64, s[2]:s[2]+64].copy())
for a in g.ATTRIBUTES:
    vals = np.array([rating(a, c) for c in crops])
    named = {**{k: rating(a, v) for k, v in fixed_inputs.items()},
             **{f"patch{p}": rating(a, v) for p, v in patches.items()}}
    print(f"{a:11s} 40 lung crops: mean={vals.mean():.3f} sd={vals.std():.4f} min={vals.min():.3f} max={vals.max():.3f} | "
          + " ".join(f"{k}={v:.3f}" for k, v in named.items()))

print("\n== B. Same candidate resampled to isotropic voxels (trilinear, from the Stage 01 CT) ==")
def iso_patch(center, step_mm):
    offs = (np.arange(64) - 32) * step_mm
    zs = center[0] + offs / sp[0]; ys = center[1] + offs / sp[1]; xs = center[2] + offs / sp[2]
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    return map_coordinates(ct, [Z, Y, X], order=1, mode="constant", cval=-1000.0).astype(np.float32)

blank_cam = {a: g.calculate_cam(get_model(a), fixed_inputs["blank_-1000"], a, dev)[0] for a in g.ATTRIBUTES}
for r in man["patches"]:
    pid = int(r["patch_id"]); c = np.array(r["center_zyx"], float)
    variants = {"native": (patches[pid], sp),
                "iso1.0mm": (iso_patch(c, 1.0), np.ones(3)),
                "iso0.70mm": (iso_patch(c, 0.703125), np.full(3, 0.703125))}
    print(f"\npatch {pid}")
    for a in g.ATTRIBUTES:
        parts = []
        for name, (p, s) in variants.items():
            cam, score, _ = g.calculate_cam(get_model(a), p, a, dev)
            pk = np.array(np.unravel_index(int(cam.argmax()), cam.shape))
            parts.append(f"{name}: rating={score:.3f} peak={pk.tolist()} d={float(np.linalg.norm((pk - 32) * s)):.0f}mm corr_blank={cc(cam, blank_cam[a]):+.2f}")
        print(f"{a:11s} " + " | ".join(parts))

print("\n== C. Stage 03 stored predictions vs ratings computed by 05_gradcam.py ==")
files = sorted(root.glob("03*/predictions.json"))
print("found:", [str(f.relative_to(root)) for f in files] or "none")
def walk(o, pid=None, path=""):
    if isinstance(o, dict):
        pid = o.get("patch_id", pid)
        for k, v in o.items():
            if k in g.ATTRIBUTES:
                print(f"  patch {pid} {path}/{k}: {json.dumps(v)[:170]}")
            else:
                walk(v, pid, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk(v, pid, f"{path}[{i}]")
for f in files:
    print(f.relative_to(root))
    walk(json.loads(f.read_text(encoding="utf-8-sig")))
print("05_gradcam ratings:", {p: {a: round(rating(a, v), 3) for a in g.ATTRIBUTES} for p, v in patches.items()})

print("\n== D. Occlusion with a surround-matched fill instead of -1000 HU ==")
W = np.array([2, 4, 4])
def fill_value(patch, lo, size):
    lo2 = np.maximum(lo - W, 0); hi2 = np.minimum(lo + size + W, 64)
    sub = patch[lo2[0]:hi2[0], lo2[1]:hi2[1], lo2[2]:hi2[2]]
    keep = np.ones(sub.shape, bool)
    a = lo - lo2; b = a + size
    keep[a[0]:b[0], a[1]:b[1], a[2]:b[2]] = False
    return float(np.median(sub[keep]))
def block(patch, lo, size):
    q = patch.copy()
    q[lo[0]:lo[0]+size[0], lo[1]:lo[1]+size[1], lo[2]:lo[2]+size[2]] = fill_value(patch, lo, size)
    return q
def centered(c, size):
    return np.clip(np.round(np.asarray(c) - size / 2).astype(int), 0, 64 - size)
def overlaps(a, b, size):
    return bool(np.all(a < b + size) and np.all(b < a + size))
for r in man["patches"]:
    pid = int(r["patch_id"]); patch = patches[pid]
    bs = np.clip(np.floor(r["bbox_local_start_zyx"]).astype(int), 0, 64)
    be = np.clip(np.ceil(r["bbox_local_end_zyx"]).astype(int), 0, 64)
    size = be - bs
    print(f"\npatch {pid}: detector box {bs.tolist()}..{be.tolist()}")
    for a in ("lobulation", "spiculation"):
        raw = np.load(cam_root / f"patch_{pid:03d}" / f"gradcam_{a}_raw_signed.npy")
        base = rating(a, patch)
        pk = np.array(np.unravel_index(int(raw.argmax()), raw.shape))
        tr = np.array(np.unravel_index(int(raw.argmin()), raw.shape))
        rand = []
        while len(rand) < 24:
            lo = rng.integers(0, 64 - size + 1)
            if not overlaps(lo, bs, size):
                rand.append(rating(a, block(patch, lo, size)) - base)
        rand = np.array(rand)
        text = []
        for name, lo in {"detector": bs, "CAM peak": centered(pk, size), "CAM trough": centered(tr, size)}.items():
            d = rating(a, block(patch, lo, size)) - base
            text.append(f"{name} d={d:+.4f} (pct {float((np.abs(rand) < abs(d)).mean()):.2f})")
        print(f"{a:11s} base={base:.3f} " + " | ".join(text) + f" | random mean={rand.mean():+.4f} sd={rand.std():.4f}")
