#!/usr/bin/env python3
"""
LungInsight v2 -- distributional_v1 3D nodule-attribute training.

This is a new experiment and does not overwrite the historical ordinal trainer.
It uses the fixed H5 and fixed patient/sample split, with no augmentation and no
malignancy neural-network target.

Commands:
  check-h5   read-only H5/schema/target/split checks
  sanity     tiny overfit on REAL TRAIN nodules only
  train      supervised train on train + validation
  evaluate   one-shot test evaluation after model selection
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

sys.path.insert(0, str(HERE))
from model_3d import (  # noqa: E402
    MODEL_IDENTITY,
    MODEL_TARGETS,
    PREPROCESSING,
    RATING_MAX,
    READER_TO_MODEL,
    DEFAULT_ENCODER_CFG,
    INPUT_SHAPE,
    MaskedReconstructionModel,
    architecture_config,
    build_model,
    count_params,
    preprocess_hu,
)

DEFAULT_H5 = ROOT / "datasets" / "LIDC" / "lidc_patches.h5"
DEFAULT_SPLIT = ROOT / "backend" / "models" / "nodule_ranker_3d" / "checkpoints_model2_final" / "split.json"
DEFAULT_OUT = HERE / "checkpoints"

EXPECTED = {
    "samples": 2651,
    "patients": 875,
    "studies": 883,
    "annotations": 6859,
    "patients_split": (613, 131, 131),
    "nodules_split": (1859, 437, 355),
}
KS = [RATING_MAX[t] for t in MODEL_TARGETS]
NH = len(MODEL_TARGETS)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.generic):
        return clean(o.item())
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(obj), indent=2), encoding="utf-8")


def set_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_device(arg):
    return (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if arg == "auto"
        else torch.device(arg)
    )


class Amp:
    def __init__(self, mode, device):
        self.enabled = device.type == "cuda" and mode != "off"
        self.dtype = torch.float16 if mode == "fp16" else torch.bfloat16
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.enabled and mode == "fp16"
        )
        self.device_type = device.type
        self.mode = mode if self.enabled else "off"

    def ctx(self):
        return torch.autocast(
            device_type=self.device_type,
            dtype=self.dtype,
            enabled=self.enabled,
        )


def env_info(device):
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "argv": sys.argv,
    }


# -----------------------------------------------------------------------------
# H5 / target construction
# -----------------------------------------------------------------------------

def _strs(ds):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in ds[:]])


def load_meta(h5_path):
    with h5py.File(h5_path, "r") as f:
        return {
            "patient_id": _strs(f["patient_id"]),  # type: ignore
            "nodule_id": _strs(f["nodule_id"]),  # type: ignore
            "study_uid": _strs(f["study_instance_uid"]),  # type: ignore
            "reader_labels": f["reader_labels"][:],  # type: ignore
            "reader_mask": f["reader_mask"][:].astype(bool),  # type: ignore
            "n": int(f["patches"].shape[0]),  # type: ignore
        }


def read_patches(h5_path, idx):
    """Read H5 patch rows into one host float32 tensor, preserving index order."""
    idx = np.asarray(idx, dtype=np.int64)
    order = np.argsort(idx)
    sidx = idx[order]
    out = np.empty((len(idx), *INPUT_SHAPE), np.float32)
    with h5py.File(h5_path, "r") as f:
        ds = f["patches"]  # type: ignore
        for s in range(0, ds.shape[0], 128):  # type: ignore
            lo, hi = np.searchsorted(sidx, [s, s + 128])
            if hi > lo:
                out[order[lo:hi]] = ds[s:s + 128][sidx[lo:hi] - s]  # type: ignore
    return torch.from_numpy(out)


def build_targets(reader_labels, reader_mask):
    """
    Build empirical reader distributions for all eight attributes.

    Returns:
      distribution : (N, 8, 6), padded after each head's true K
      mean         : (N, 8), identical to sum(rating * distribution)
      count        : (N, 8), valid reader count per attribute
      valid        : (N, 8)
    """
    lab = reader_labels[:, :, READER_TO_MODEL]
    ok = reader_mask[:, :, None] & np.isfinite(lab)
    n = lab.shape[0]
    kmax = max(KS)
    distribution = np.zeros((n, NH, kmax), dtype=np.float32)
    count = ok.sum(1).astype(np.float32)
    denom = np.maximum(count, 1.0)

    for h, k in enumerate(KS):
        vh = ok[:, :, h]
        values = lab[:, :, h]
        for rating in range(1, k + 1):
            distribution[:, h, rating - 1] = (
                ((values == rating) & vh).sum(1) / denom[:, h]
            ).astype(np.float32)

    values_all = np.arange(1, kmax + 1, dtype=np.float32)
    mean = (distribution * values_all[None, None, :]).sum(-1).astype(np.float32)

    for h, k in enumerate(KS):
        bad = ok[:, :, h] & ((lab[:, :, h] < 1) | (lab[:, :, h] > k))
        if bad.any():
            raise ValueError(f"{MODEL_TARGETS[h]}: rating outside 1..{k}")
        valid_nodes = count[:, h] > 0
        valid_sums = distribution[valid_nodes, h, :k].sum(-1)
        if not np.allclose(valid_sums, 1.0, atol=1e-6):
            raise ValueError(f"{MODEL_TARGETS[h]}: valid target distributions do not sum to 1")
    return {
        "distribution": distribution,
        "mean": mean,
        "count": count,
        "valid": count > 0,
    }


def subset_targets(targets, idx):
    return {
        k: torch.from_numpy(np.ascontiguousarray(v[idx]))
        for k, v in targets.items()
    }


# -----------------------------------------------------------------------------
# Fixed split -- consumes the canonical split directly
# -----------------------------------------------------------------------------

def _entry_value(d, aliases):
    for a in aliases:
        if a in d:
            return d[a]
    return None


def parse_split_file(path, meta):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        raise ValueError(f"{path}: expected JSON object")

    pat_set = set(meta["patient_id"])
    out = {}

    for name in ("train", "val", "test"):
        entry = d.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: '{name}' must be an object")

        sample_indices = _entry_value(entry, ("sample_indices", "nodule_indices", "indices"))
        patients = _entry_value(entry, ("patients", "patient_ids"))
        if sample_indices is None:
            raise ValueError(f"{path}: '{name}' has no sample_indices/nodule_indices/indices")
        if not isinstance(sample_indices, list) or not all(isinstance(x, int) for x in sample_indices):
            raise ValueError(f"{path}: '{name}.sample_indices' must be a list of integers")

        arr = np.asarray(sample_indices, dtype=np.int64)
        if len(arr) != len(set(arr.tolist())):
            raise ValueError(f"{path}: duplicate sample index in {name}")
        if np.any(arr < 0) or np.any(arr >= meta["n"]):
            raise ValueError(f"{path}: out-of-range sample index in {name}")
        out[name] = np.sort(arr)

        if patients is not None:
            if not isinstance(patients, list) or not all(isinstance(x, str) for x in patients):
                raise ValueError(f"{path}: '{name}.patients' must be a list of strings")
            unknown = sorted(set(patients) - pat_set)
            if unknown:
                raise ValueError(f"{path}: unknown patient IDs in {name}: {unknown[:5]}")
            actual = set(meta["patient_id"][out[name]])
            if actual != set(patients):
                raise ValueError(
                    f"{path}: {name} patient list does not match its sample_indices "
                    f"(listed={len(patients)}, selected={len(actual)})"
                )

        declared = entry.get("sample_count", entry.get("nodule_count"))
        if declared is not None and int(declared) != len(out[name]):
            raise ValueError(
                f"{path}: {name} declares {declared} samples but contains {len(out[name])}"
            )

    all_idx = np.concatenate([out["train"], out["val"], out["test"]])
    if len(all_idx) != meta["n"] or len(np.unique(all_idx)) != meta["n"]:
        raise ValueError(
            f"{path}: split must partition all H5 samples exactly once "
            f"(split={len(all_idx)}, unique={len(np.unique(all_idx))}, H5={meta['n']})"
        )

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        pa = set(meta["patient_id"][out[a]])
        pb = set(meta["patient_id"][out[b]])
        if pa & pb:
            raise ValueError(f"{path}: patient overlap between {a} and {b}")

    got_p = tuple(len(set(meta["patient_id"][out[k]])) for k in ("train", "val", "test"))
    got_n = tuple(len(out[k]) for k in ("train", "val", "test"))
    if got_p != EXPECTED["patients_split"] or got_n != EXPECTED["nodules_split"]:
        raise ValueError(
            f"{path}: split sizes {got_p} patients / {got_n} nodules "
            f"do not match authoritative {EXPECTED['patients_split']} / {EXPECTED['nodules_split']}"
        )
    return out


def resolve_split(args, meta):
    sf = Path(args.split_file) if args.split_file else DEFAULT_SPLIT
    if not sf.is_file():
        raise FileNotFoundError(
            f"Authoritative split not found: {sf}. "
            "The distributional experiment refuses to silently reproduce a different split."
        )
    idx = parse_split_file(sf, meta)
    pats = {
        k: sorted(set(meta["patient_id"][v]))
        for k, v in idx.items()
    }
    split = {
        "source": str(sf.resolve()),
        "seed": 42,
        "patients": pats,
        "nodule_indices": {k: v.tolist() for k, v in idx.items()},
        "counts": {
            "patients": tuple(len(pats[k]) for k in ("train", "val", "test")),
            "nodules": tuple(len(idx[k]) for k in ("train", "val", "test")),
        },
    }
    return idx, split


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

class PatchDataset(Dataset):
    def __init__(self, patches, targets=None):
        self.patches = patches
        self.targets = targets or {}

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, idx):
        i = torch.as_tensor(idx, dtype=torch.long)
        out = {"x": self.patches[i]}
        out.update({k: v[i] for k, v in self.targets.items()})
        return out


def make_loader(ds, bs, shuffle, seed, device, workers, drop_last=False):
    g = torch.Generator()
    g.manual_seed(seed)
    sampler = RandomSampler(ds, generator=g) if shuffle else SequentialSampler(ds)
    return DataLoader(
        ds,
        batch_size=None,
        sampler=BatchSampler(sampler, bs, drop_last),
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )


def to_dev(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


# -----------------------------------------------------------------------------
# Distributional loss and validation metrics
# -----------------------------------------------------------------------------

def distribution_loss(model, logits, b, lambda_mean=0.2):
    """
    Ordinal-aware CDF/EMD loss plus a small expected-rating Huber term.

    Reader count is retained in the batch/metrics but does not weight the loss:
    every valid nodule-attribute contributes equally in this first replacement.
    """
    probs = model.probabilities(logits)
    target = b["distribution"].float()
    valid = b["valid"].float()
    total_num = torch.zeros(NH, device=target.device)
    total_den = valid.sum(0)
    expected_parts = []

    for h, (p, k) in enumerate(zip(probs, KS)):
        q = target[:, h, :k]
        cdf_p = torch.cumsum(p, dim=-1)
        cdf_q = torch.cumsum(q, dim=-1)
        emd = torch.abs(cdf_p - cdf_q).mean(-1)

        expected = (p * torch.arange(1, k + 1, device=p.device, dtype=p.dtype)).sum(-1)
        expected_parts.append(expected)
        mean_target = b["mean"][:, h].float()
        mean_pred_norm = (expected - 1.0) / float(k - 1)
        mean_true_norm = (mean_target - 1.0) / float(k - 1)
        mean_term = F.smooth_l1_loss(
            mean_pred_norm,
            mean_true_norm,
            beta=0.05,
            reduction="none",
        )

        per_sample = emd + float(lambda_mean) * mean_term
        total_num[h] = (per_sample * valid[:, h]).sum()

    expected_all = torch.stack(expected_parts, dim=1)
    return total_num, total_den, expected_all


def _corr(fn, p, t):
    if len(p) < 3 or np.std(p) < 1e-9 or np.std(t) < 1e-9:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(fn(p, t)[0])


def rounded(x):
    return np.floor(np.asarray(x) + 0.5)


def compute_metrics(pred, mean, count, valid, min_readers=1):
    per, pool = {}, {"err": [], "sq": [], "ex": [], "w1": []}
    for h, name in enumerate(MODEL_TARGETS):
        v = valid[:, h] & (count[:, h] >= min_readers)
        p = pred[v, h].astype(np.float64)
        t = mean[v, h].astype(np.float64)
        K = KS[h]
        if len(p) == 0:
            per[name] = {
                "n": 0, "mae": float("nan"), "rmse": float("nan"),
                "exact_acc": float("nan"), "within1_acc": float("nan"),
                "pearson": float("nan"), "spearman": float("nan"),
                "pred_std": float("nan"), "true_std": float("nan"),
                "std_ratio": float("nan"), "collapsed": True,
            }
            continue
        pr, tr = np.clip(rounded(p), 1, K), rounded(t)
        e = p - t
        per[name] = {
            "n": int(v.sum()),
            "mae": float(np.abs(e).mean()),
            "rmse": float(np.sqrt((e ** 2).mean())),
            "exact_acc": float((pr == tr).mean()),
            "within1_acc": float((np.abs(pr - tr) <= 1).mean()),
            "pearson": _corr(stats.pearsonr, p, t),
            "spearman": _corr(stats.spearmanr, p, t),
            "pred_std": float(p.std()),
            "true_std": float(t.std()),
            "std_ratio": float(p.std() / max(t.std(), 1e-9)),
            "collapsed": bool(p.std() < 0.25 * t.std()),
        }
        pool["err"].append(np.abs(e))
        pool["sq"].append(e ** 2)
        pool["ex"].append(pr == tr)
        pool["w1"].append(np.abs(pr - tr) <= 1)

    cat = {k: np.concatenate(v) for k, v in pool.items() if v}
    sp = [per[n]["spearman"] for n in MODEL_TARGETS]
    pe = [per[n]["pearson"] for n in MODEL_TARGETS]
    agg = {
        "n_pairs": int(len(cat["err"])),
        "mae": float(cat["err"].mean()),
        "rmse": float(np.sqrt(cat["sq"].mean())),
        "exact_acc": float(cat["ex"].mean()),
        "within1_acc": float(cat["w1"].mean()),
        "mean_pearson": float(np.nanmean(pe)) if not np.all(np.isnan(pe)) else None,
        "mean_spearman": float(np.nanmean(sp)) if not np.all(np.isnan(sp)) else None,
        "mean_std_ratio": float(np.nanmean([per[n]["std_ratio"] for n in MODEL_TARGETS])),
        "n_collapsed_heads": int(sum(per[n]["collapsed"] for n in MODEL_TARGETS)),
    }
    return {"aggregate": agg, "per_head": per, "min_readers": min_readers}


def selection_score(metrics, kind):
    if kind == "loss":
        raise ValueError("loss selection is disabled for distributional_v1; use Spearman")
    sp = [x["spearman"] for x in metrics["per_head"].values()]
    vals = [0.0 if np.isnan(s) else s for s in sp]
    return float(np.mean(vals))


def print_metrics(m, title):
    a = m["aggregate"]
    print(f"\n== {title} ==")
    print(
        f"  aggregate: MAE {a['mae']:.4f} RMSE {a['rmse']:.4f} "
        f"exact {a['exact_acc']:.4f} within1 {a['within1_acc']:.4f} "
        f"Pearson {a['mean_pearson']} Spearman {a['mean_spearman']} "
        f"collapsed heads {a['n_collapsed_heads']}/8"
    )
    print(
        f"  {'head':<19}{'n':>5}{'MAE':>7}{'RMSE':>7}{'exact':>7}"
        f"{'w1':>7}{'pear':>7}{'spear':>7}{'predSD':>8}{'trueSD':>8}"
    )
    for n, h in m["per_head"].items():
        def f(x):
            return "    nan" if x is None or np.isnan(x) else f"{x:7.3f}"
        print(
            f"  {n:<19}{h['n']:>5}{h['mae']:7.3f}{h['rmse']:7.3f}"
            f"{h['exact_acc']:7.3f}{h['within1_acc']:7.3f}"
            f"{f(h['pearson'])}{f(h['spearman'])}"
            f"{h['pred_std']:8.3f}{h['true_std']:8.3f}"
            f"{'  <-- COLLAPSED' if h['collapsed'] else ''}"
        )


@torch.no_grad()
def predict(model, loader, device, amp, lambda_mean=0.2):
    model.eval()
    preds, nums, dens = [], torch.zeros(NH, device=device), torch.zeros(NH, device=device)
    for batch in loader:
        b = to_dev(batch, device)
        with amp.ctx():
            logits = model(preprocess_hu(b["x"]).unsqueeze(1))
        n, d, e = distribution_loss(model, logits, b, lambda_mean)
        nums += n
        dens += d
        preds.append(e.cpu())
    return torch.cat(preds).numpy(), float((nums / dens.clamp(min=1e-8)).mean())


# -----------------------------------------------------------------------------
# Optimisation
# -----------------------------------------------------------------------------

def make_optimizer(model, lr, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.999),
    )


def make_scheduler(opt, total, warmup, floor=0.01):
    def f(s):
        if s < warmup:
            return (s + 1) / max(1, warmup)
        p = (s - warmup) / max(1, total - warmup)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def backward_step(loss, model, opt, sched, amp, clip):
    opt.zero_grad(set_to_none=True)
    amp.scaler.scale(loss).backward()
    amp.scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    amp.scaler.step(opt)
    amp.scaler.update()
    sched.step()


def supervised_epoch(model, loader, opt, sched, amp, device, args):
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        b = to_dev(batch, device)
        with amp.ctx():
            logits = model(preprocess_hu(b["x"]).unsqueeze(1))
        num, den, _ = distribution_loss(model, logits, b, args.lambda_mean)
        loss = (num / den.clamp(min=1e-8)).mean()
        backward_step(loss, model, opt, sched, amp, args.grad_clip)
        total += float(loss.detach()) * len(b["x"])
        count += len(b["x"])
    return total / max(count, 1)


def load_pretrained_encoder(model, path, train_idx):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("encoder_config") != DEFAULT_ENCODER_CFG:
        raise ValueError("pretrained encoder config mismatch")
    leaked = set(ck.get("train_nodule_indices", [])) - set(train_idx.tolist())
    if leaked:
        raise RuntimeError(
            f"pretrained encoder includes {len(leaked)} nodule indices outside the current TRAIN split"
        )
    model.encoder.load_state_dict(ck["encoder_state_dict"], strict=True)
    return {
        "path": str(path),
        "config": ck.get("pretrain_config"),
        "epochs_completed": ck.get("epochs_completed"),
        "complete": ck.get("complete"),
        "n_pretrain_nodules": len(ck.get("train_nodule_indices", [])),
        "final_loss": ck["history"][-1]["loss"] if ck.get("history") else None,
    }


# -----------------------------------------------------------------------------
# Checks
# -----------------------------------------------------------------------------

def cmd_check_h5(args):
    p = Path(args.h5)
    ok = True

    def chk(cond, msg):
        nonlocal ok
        print(("  [ OK ] " if cond else "  [FAIL] ") + msg)
        ok &= bool(cond)

    print(f"HDF5: {p}")
    with h5py.File(p, "r") as f:
        N = f["patches"].shape[0]  # type: ignore
        spec = {
            "patches": ((N, 64, 64, 64), "float32"),
            "attribute_mean": ((N, 8), "float32"),
            "attribute_std": ((N, 8), "float32"),
            "attribute_count": ((N, 8), "int32"),
            "malignancy_distribution": ((N, 5), "float32"),
            "malignancy_mean": ((N,), "float32"),
            "malignancy_std": ((N,), "float32"),
            "malignancy_count": ((N,), "int32"),
            "reader_labels": ((N, 8, 9), "float32"),
            "reader_mask": ((N, 8), "bool"),
        }
        for k, (shape, dt) in spec.items():
            chk(
                k in f and f[k].shape == shape and (  # type: ignore
                    dt == "bool" or f[k].dtype == np.dtype(dt)  # type: ignore
                ),
                f"{k} {f[k].shape if k in f else 'MISSING'} "  # type: ignore
                f"{f[k].dtype if k in f else ''}",  # type: ignore
            )

        for k, v in (
            ("coordinate_order", "ZYX"),
            ("patch_size", 64),
            ("patch_mm", 64.0),
            ("hu_min", -1000.0),
            ("hu_max", 400.0),
        ):
            chk(k in f.attrs and f.attrs[k] == v, f"attr {k} == {v} (got {f.attrs.get(k)})")

        bad_nan = flat = 0
        lo, hi = np.inf, -np.inf
        for s in range(0, N, 128):
            blk = f["patches"][s:s + 128]  # type: ignore
            bad_nan += int((~np.isfinite(blk)).any(axis=(1, 2, 3)).sum())  # type: ignore
            lo, hi = min(lo, float(blk.min())), max(hi, float(blk.max()))  # type: ignore
            flat += int((blk.reshape(len(blk), -1).std(1) < 1e-3).sum())  # type: ignore
        chk(bad_nan == 0, f"patches finite (non-finite patches: {bad_nan})")
        chk(lo >= -1000.001 and hi <= 400.001, f"HU range [{lo:.1f}, {hi:.1f}] within [-1000, 400]")
        chk(flat == 0, f"no constant patches (found {flat})")

    meta = load_meta(p)
    nread = meta["reader_mask"].sum(1)
    chk(N == EXPECTED["samples"], f"samples {N} (expected {EXPECTED['samples']})")
    chk(len(set(meta["patient_id"])) == EXPECTED["patients"], f"patients {len(set(meta['patient_id']))}")
    chk(len(set(meta["study_uid"])) == EXPECTED["studies"], f"studies {len(set(meta['study_uid']))}")
    chk(int(nread.sum()) == EXPECTED["annotations"], f"reader annotations {int(nread.sum())}")
    chk(nread.min() >= 1 and nread.max() <= 8, f"readers/nodule in [{nread.min()}, {nread.max()}], mean {nread.mean():.3f}")

    try:
        t = build_targets(meta["reader_labels"], meta["reader_mask"])
        chk(True, "all valid ratings inside their per-head scale")
        for h, n in enumerate(MODEL_TARGETS):
            k = KS[h]
            d = t["distribution"][:, h, :k]
            v = t["valid"][:, h]
            chk(np.allclose(d[v].sum(-1), 1.0, atol=1e-6), f"{n}: distributions sum to 1")
            vals = meta["reader_labels"][:, :, READER_TO_MODEL[h]]
            okv = meta["reader_mask"] & np.isfinite(vals)
            got = np.unique(vals[okv]).astype(int).tolist()
            chk(all(1 <= x <= k for x in got), f"{n}: observed ratings within 1..{k}: {got}")
            target_mean = (d * np.arange(1, k + 1, dtype=np.float32)).sum(-1)
            chk(np.allclose(target_mean[v], t["mean"][v, h], atol=1e-6), f"{n}: expected rating equals reader mean")
    except ValueError as e:
        chk(False, str(e))

    try:
        idx, split = resolve_split(args, meta)
        chk(True, f"authoritative split resolved: {[len(idx[k]) for k in ('train','val','test')]}")
        chk(split["counts"]["patients"] == EXPECTED["patients_split"], "patient split counts exact")
        chk(split["counts"]["nodules"] == EXPECTED["nodules_split"], "nodule split counts exact")
    except Exception as e:
        chk(False, f"split: {e}")

    print("\nH5 CHECK", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def cmd_sanity(args):
    device = get_device(args.device)
    set_seed(args.seed, args.deterministic)
    amp = Amp(args.amp, device)
    meta = load_meta(args.h5)
    idx, _ = resolve_split(args, meta)
    targets = build_targets(meta["reader_labels"], meta["reader_mask"])

    if args.n_nodules < 16:
        raise ValueError("Use at least 16 TRAIN nodules for the required tiny-overfit check.")
    sub = np.sort(
        np.random.RandomState(args.seed).choice(
            idx["train"], args.n_nodules, replace=False
        )
    )
    ds = PatchDataset(read_patches(args.h5, sub), subset_targets(targets, sub))
    cfg = architecture_config(dropout=0.0)
    model = build_model(cfg).to(device)

    print(
        f"Sanity: {len(sub)} TRAIN nodules, params {count_params(model)/1e6:.2f}M, "
        f"device {device}, amp {amp.mode}"
    )
    args.lambda_mean = 0.2
    args.grad_clip = 1.0
    loader = make_loader(ds, args.batch_size, True, args.seed, device, 0)
    eval_loader = make_loader(ds, args.batch_size, False, 0, device, 0)
    opt = make_optimizer(model, args.lr, 0.0)
    sched = make_scheduler(opt, args.epochs * len(loader), min(10, args.epochs * len(loader)), 0.1)

    p0, l0 = predict(model, eval_loader, device, amp, args.lambda_mean)
    t = {k: v.numpy() for k, v in ds.targets.items()}
    m0 = compute_metrics(p0, t["mean"], t["count"], t["valid"].astype(bool))
    print(f"  epoch   0: loss {l0:.4f}  MAE {m0['aggregate']['mae']:.3f}")

    m = m0
    for ep in range(1, args.epochs + 1):
        tl = supervised_epoch(model, loader, opt, sched, amp, device, args)
        if ep % max(1, args.print_every) == 0 or ep == args.epochs:
            p, l = predict(model, eval_loader, device, amp, args.lambda_mean)
            m = compute_metrics(p, t["mean"], t["count"], t["valid"].astype(bool))
            print(
                f"  epoch {ep:3d}: train {tl:.4f} loss {l:.4f} "
                f"MAE {m['aggregate']['mae']:.3f} "
                f"mean std-ratio {m['aggregate']['mean_std_ratio']:.2f}"
            )

    passed = (
        m["aggregate"]["mae"] <= 0.40
        and m["aggregate"]["mae"] <= 0.65 * m0["aggregate"]["mae"]
        and m["aggregate"]["mean_std_ratio"] >= 0.50
    )
    print_metrics(m, "sanity subset (TRAIN only; not a scientific evaluation)")
    print(
        f"\nSANITY {'PASSED' if passed else 'FAILED'}: "
        f"MAE {m0['aggregate']['mae']:.3f} -> {m['aggregate']['mae']:.3f}"
    )
    return 0 if passed else 1


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def make_ckpt(
    model, opt, sched, epoch, args, split, arch, val_metrics,
    pretraining, val_loss, device, score
):
    return {
        "model_identity": MODEL_IDENTITY,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "scheduler_state_dict": sched.state_dict(),
        "epoch": epoch,
        "targets": list(MODEL_TARGETS),
        "n_features": NH,
        "architecture": arch,
        "input_shape": [1, *INPUT_SHAPE],
        "preprocessing": PREPROCESSING,
        "rating_scales": {t: {"min": 1, "max": RATING_MAX[t]} for t in MODEL_TARGETS},
        "class_cardinalities": {t: RATING_MAX[t] for t in MODEL_TARGETS},
        "seed": args.seed,
        "split": split,
        "validation_metrics": val_metrics,
        "validation_loss": val_loss,
        "selection": {"metric": "mean_spearman", "score": score},
        "test_metrics": None,
        "pretraining": pretraining,
        "training_config": vars(args),
        "environment": env_info(device),
        "no_augmentation": True,
        "malignancy_target": False,
        "loss_config": {
            "distribution_loss": "mean absolute CDF distance (ordinal EMD-style)",
            "lambda_mean": args.lambda_mean,
            "mean_loss": "SmoothL1 on [0,1]-normalized expected rating",
            "reader_count_weighting": "none; valid nodule-attribute samples are equally weighted",
        },
    }


def cmd_train(args):
    device = get_device(args.device)
    set_seed(args.seed, args.deterministic)
    amp = Amp(args.amp, device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = load_meta(args.h5)
    idx, split = resolve_split(args, meta)
    tg = build_targets(meta["reader_labels"], meta["reader_mask"])

    dsets = {
        k: PatchDataset(read_patches(args.h5, idx[k]), subset_targets(tg, idx[k]))
        for k in ("train", "val")
    }

    arch = architecture_config(dropout=args.dropout)
    model = build_model(arch).to(device)

    pre = None
    if args.pretrained and args.pretrained.lower() != "none":
        pre = load_pretrained_encoder(model, args.pretrained, idx["train"])
        print(
            f"Loaded TRAIN-only pretrained encoder from {args.pretrained} "
            f"({pre['epochs_completed']} epochs). Distribution heads are fresh."
        )
    else:
        print("Training distributional model from scratch.")

    print(
        f"model={MODEL_IDENTITY} params={count_params(model)/1e6:.2f}M | "
        f"train={len(idx['train'])} val={len(idx['val'])} nodules | amp={amp.mode}"
    )

    tl = make_loader(dsets["train"], args.batch_size, True, args.seed, device, args.num_workers)
    vl = make_loader(dsets["val"], args.batch_size, False, 0, device, args.num_workers)

    opt = make_optimizer(model, args.lr, args.wd)
    sched = make_scheduler(
        opt,
        args.epochs * len(tl),
        args.warmup * len(tl),
    )

    vt = {k: v.numpy() for k, v in dsets["val"].targets.items()}
    best = -float("inf")
    bad = 0
    history = []
    t0 = time.time()

    save_json(
        out / "training_config.json",
        {
            "model_identity": MODEL_IDENTITY,
            "args": vars(args),
            "split": split,
            "targets": MODEL_TARGETS,
            "rating_max": RATING_MAX,
        },
    )

    for ep in range(1, args.epochs + 1):
        trl = supervised_epoch(model, tl, opt, sched, amp, device, args)
        pv, vloss = predict(model, vl, device, amp, args.lambda_mean)
        vm = compute_metrics(pv, vt["mean"], vt["count"], vt["valid"].astype(bool))
        score = selection_score(vm, "spearman")
        a = vm["aggregate"]

        row = {
            "epoch": ep,
            "train_loss": trl,
            "val_loss": vloss,
            "val_mae": a["mae"],
            "val_rmse": a["rmse"],
            "val_exact": a["exact_acc"],
            "val_within1": a["within1_acc"],
            "val_mean_spearman": a["mean_spearman"],
            "val_mean_pearson": a["mean_pearson"],
            "val_mean_std_ratio": a["mean_std_ratio"],
            "val_collapsed_heads": a["n_collapsed_heads"],
            "selection_score": score,
            "lr": opt.param_groups[0]["lr"],
        }
        history.append(row)

        improved = score > best + 1e-6
        ck = make_ckpt(
            model, opt, sched, ep, args, split, arch, vm,
            pre, vloss, device, score
        )
        if improved:
            best, bad = score, 0
            torch.save(ck, out / "best.pth")
            save_json(out / "validation_metrics.json", {"epoch": ep, **vm, "val_loss": vloss})
        else:
            bad += 1
        torch.save(ck, out / "last.pth")
        save_json(out / "history.json", history)

        eta = (time.time() - t0) / ep * (args.epochs - ep) / 60
        print(
            f"ep {ep:3d} | train {trl:.4f} val {vloss:.4f} | "
            f"MAE {a['mae']:.3f} spear {a['mean_spearman'] if a['mean_spearman'] is not None else float('nan'):.3f} "
            f"SD-ratio {a['mean_std_ratio']:.2f} collapsed {a['n_collapsed_heads']} "
            f"| sel {score:.4f}{' *' if improved else ''} "
            f"| bad {bad}/{args.patience} | ETA {eta:.1f}m"
        )
        if bad >= args.patience:
            print(f"Early stopping at epoch {ep}.")
            break

    if not (out / "best.pth").is_file():
        raise RuntimeError("No best checkpoint was produced.")
    best_ck = torch.load(out / "best.pth", map_location="cpu", weights_only=False)
    print(
        f"\nBest epoch {best_ck['epoch']} "
        f"(mean validation Spearman = {best_ck['selection']['score']:.4f})."
    )
    print_metrics(best_ck["validation_metrics"], "VALIDATION @ best checkpoint")
    print("\nTest set NOT touched. Next: `evaluate`.")
    return 0


# -----------------------------------------------------------------------------
# Test evaluation -- explicitly separate from train
# -----------------------------------------------------------------------------

def cmd_evaluate(args):
    device = get_device(args.device)
    set_seed(args.seed, args.deterministic)
    amp = Amp(args.amp, device)
    ck_path = Path(args.checkpoint) if args.checkpoint else Path(args.out_dir) / "best.pth"
    model, ck = __import__("model_3d").load_trained_model(ck_path, device)

    if ck.get("test_metrics") is not None and not args.force:
        raise SystemExit(
            f"{ck_path} already has test metrics. Use --force only if you deliberately "
            "intend to re-run test evaluation."
        )

    meta = load_meta(args.h5)
    split = ck["split"]["nodule_indices"]
    te = np.array(split["test"], dtype=np.int64)
    va = np.array(split["val"], dtype=np.int64)
    tr = np.array(split["train"], dtype=np.int64)
    if set(te) & (set(tr) | set(va)):
        raise RuntimeError("test overlaps train/val in checkpoint split")

    tg = build_targets(meta["reader_labels"], meta["reader_mask"])
    ds = PatchDataset(read_patches(args.h5, te), subset_targets(tg, te))
    loader = make_loader(ds, args.batch_size, False, 0, device, args.num_workers)
    pred, loss = predict(model, loader, device, amp, args.lambda_mean)

    t = {k: v.numpy() for k, v in ds.targets.items()}
    tm = compute_metrics(pred, t["mean"], t["count"], t["valid"].astype(bool))
    tm2 = compute_metrics(
        pred, t["mean"], t["count"], t["valid"].astype(bool), min_readers=2
    )
    n_pat = len(set(meta["patient_id"][te]))

    res = {
        "checkpoint": str(ck_path),
        "model_identity": MODEL_IDENTITY,
        "epoch": ck["epoch"],
        "n_test_nodules": int(len(te)),
        "n_test_patients": int(n_pat),
        "test_loss": loss,
        "ground_truth": "empirical reader-mean rating per nodule",
        "all_nodules": tm,
        "nodules_with_at_least_2_readers": tm2,
        "validation_metrics_at_selected_checkpoint": ck["validation_metrics"],
    }
    print_metrics(ck["validation_metrics"], "VALIDATION (selected checkpoint)")
    print_metrics(tm, "TEST (all nodules)")
    print_metrics(tm2, "TEST (nodules with >=2 readers)")
    save_json(Path(ck_path).parent / "test_metrics.json", res)
    ck["test_metrics"] = clean(res)
    torch.save(ck, ck_path)
    return 0


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--h5", default=str(DEFAULT_H5))
    common.add_argument("--split-file", default=str(DEFAULT_SPLIT))
    common.add_argument("--out-dir", default=str(DEFAULT_OUT))
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--device", default="auto")
    common.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16")
    common.add_argument("--num-workers", type=int, default=0)
    common.add_argument("--batch-size", type=int, default=16)
    common.add_argument("--deterministic", action="store_true")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check-h5", parents=[common])

    s = sub.add_parser("sanity", parents=[common])
    s.add_argument("--n-nodules", type=int, default=16)
    s.add_argument("--epochs", type=int, default=40)
    s.add_argument("--lr", type=float, default=5e-4)
    s.add_argument("--print-every", type=int, default=5)

    t = sub.add_parser("train", parents=[common])
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--wd", type=float, default=1e-2)
    t.add_argument("--warmup", type=int, default=5)
    t.add_argument("--patience", type=int, default=20)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--lambda-mean", type=float, default=0.2)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--pretrained", default="none")

    e = sub.add_parser("evaluate", parents=[common])
    e.add_argument("--checkpoint", default=None)
    e.add_argument("--force", action="store_true")
    e.add_argument("--lambda-mean", type=float, default=0.2)

    args = ap.parse_args()
    fn = {
        "check-h5": cmd_check_h5,
        "sanity": cmd_sanity,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
    }[args.cmd]
    return fn(args)


if __name__ == "__main__":
    sys.exit(main())
