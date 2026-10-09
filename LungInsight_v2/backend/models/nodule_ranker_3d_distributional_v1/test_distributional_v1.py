#!/usr/bin/env python3
"""CPU-only implementation checks for LungInsight distributional_v1.

These tests do not require the H5.  They deliberately do NOT substitute for the
required real-H5 tiny-overfit check.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import time
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("dist_model", HERE / "model_3d.py")
mm = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mm)

spec = importlib.util.spec_from_file_location("dist_train", HERE / "train_nodule_ranker_3d.py")
tt = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tt)


def target_sanity():
    labels = np.full((3, 8, 9), np.nan, np.float32)
    mask = np.zeros((3, 8), bool)

    # subtlety: [2,2,3,4] -> [0,.5,.25,.25,0], mean 2.75
    for j, v in enumerate([2, 2, 3, 4]):
        labels[0, j, 7] = v
        mask[0, j] = True

    # calcification: [1,3,6] exercises its K=6 scale.
    for j, v in enumerate([1, 3, 6]):
        labels[0, j, 0] = v
        mask[0, j] = True

    targets = tt.build_targets(labels, mask)
    d = targets["distribution"]

    assert np.isclose(targets["mean"][0, 0], 2.75)
    assert np.allclose(d[0, 2, :6], [1/3, 0, 1/3, 0, 0, 1/3], atol=1e-6)
    assert np.isclose(targets["mean"][0, 2], 10/3)

    for h, k in enumerate(tt.KS):
        v = targets["valid"][:, h]
        assert np.allclose(d[v, h, :k].sum(-1), 1.0, atol=1e-6)
        expected = (d[:, h, :k] * np.arange(1, k + 1, dtype=np.float32)).sum(-1)
        assert np.allclose(expected[v], targets["mean"][v, h], atol=1e-6)
    print("[PASS] target distribution sanity")


def forward_test():
    torch.manual_seed(42)
    torch.set_num_threads(4)
    model = mm.build_model(mm.architecture_config(dropout=0.0))
    x = torch.randn(2, 1, 64, 64, 64)
    with torch.no_grad():
        logits = model(x)
        probs = model.probabilities(logits)
        expected = model.expected_rating(logits)

    assert len(logits) == 8
    assert [z.shape[-1] for z in logits] == tt.KS
    assert all(torch.allclose(p.sum(-1), torch.ones(2), atol=1e-6) for p in probs)
    assert torch.isfinite(expected).all()
    for h, k in enumerate(tt.KS):
        assert torch.all((expected[:, h] >= 1) & (expected[:, h] <= k))
    print("[PASS] forward: 8 heads, correct K, normalized probabilities, bounded E[Y]")


def backward_test():
    torch.manual_seed(1)
    torch.set_num_threads(4)
    model = mm.build_model(mm.architecture_config(dropout=0.0))
    opt = tt.make_optimizer(model, 3e-4, 1e-2)
    x = torch.randn(2, 1, 64, 64, 64)

    dist = torch.zeros(2, 8, 6)
    base = torch.tensor([0.2, 0.3, 0.2, 0.2, 0.1, 0.0])
    for b in range(2):
        for h, k in enumerate(tt.KS):
            dist[b, h, :k] = base[:k]
            dist[b, h, :k] /= dist[b, h, :k].sum()
    mean = torch.zeros(2, 8)
    for h, k in enumerate(tt.KS):
        mean[:, h] = (dist[:, h, :k] * torch.arange(1, k + 1)).sum(-1)

    batch = {
        "distribution": dist,
        "mean": mean,
        "valid": torch.ones(2, 8),
        "count": torch.ones(2, 8),
    }
    before = next(model.parameters()).detach().clone()

    logits = model(x)
    num, den, _ = tt.distribution_loss(model, logits, batch, lambda_mean=0.2)
    loss = (num / den.clamp(min=1e-8)).mean()
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    opt.step()
    after = next(model.parameters()).detach().clone()
    assert not torch.equal(before, after)
    print(f"[PASS] backward: finite loss={float(loss.detach()):.6f}, parameters updated")


if __name__ == "__main__":
    t0 = time.time()
    target_sanity()
    forward_test()
    backward_test()
    print(f"CPU implementation checks complete in {time.time()-t0:.2f}s")
    print("[NOT RUN] real-H5 tiny-overfit: H5 is not available in this handoff")
