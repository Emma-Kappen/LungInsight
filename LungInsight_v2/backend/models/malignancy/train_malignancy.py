#!/usr/bin/env python3
"""Train the radiologist malignancy suspicion fusion model."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader, TensorDataset

from malignancy_fusion import MalignancyFusion, ordinal_reader_loss


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def load_rows(path: Path, input_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        features = np.asarray(data["X"], dtype=np.float32)[:, :input_dim]
        ratings = np.asarray(data["y"], dtype=np.float32)
        valid = np.asarray(data["y_mask"], dtype=bool) if "y_mask" in data else np.isfinite(ratings)
        patients = np.asarray(data["patient_ids"]).astype(str) if "patient_ids" in data else np.arange(len(features)).astype(str)
    valid &= np.isfinite(ratings) & (ratings >= 1) & (ratings <= 5)
    row_ids, reader_ids = np.nonzero(valid)
    if len(row_ids) == 0:
        raise ValueError(f"No valid reader-level malignancy ratings in {path}")
    return features[row_ids], ratings[row_ids, reader_ids], patients[row_ids], row_ids


def grouped_validation(features: np.ndarray, ratings: np.ndarray, patients: np.ndarray,
                       seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    groups = np.unique(patients)
    if len(groups) < 2:
        raise ValueError("At least two patients are needed for a grouped validation split")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_val = max(1, min(len(groups) - 1, int(round(len(groups) * 0.2))))
    val_patients = set(groups[:n_val])
    val = np.asarray([patient in val_patients for patient in patients])
    return features[~val], ratings[~val], features[val], ratings[val]


def metrics(model: MalignancyFusion, features: torch.Tensor, ratings: torch.Tensor,
            device: torch.device) -> dict[str, float]:
    model.eval()
    with torch.inference_mode():
        predictions = model.predict(features.to(device))
        expected = predictions["malignancy_expected"].cpu().numpy()
        distribution = predictions["class_probabilities"].cpu().numpy()
        predicted_class = distribution.argmax(axis=1) + 1
        truth = ratings.cpu().numpy().astype(np.int64)
    qwk = cohen_kappa_score(truth, predicted_class, weights="quadratic", labels=[1, 2, 3, 4, 5])
    rho = spearmanr(truth, expected).statistic if len(np.unique(truth)) > 1 and len(np.unique(expected)) > 1 else float("nan")
    return {"qwk": float(qwk) if np.isfinite(qwk) else float("nan"),
            "spearman": float(rho) if np.isfinite(rho) else float("nan")}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--features", required=True, help="Training NPZ, normally train_oof.npz")
    parser.add_argument("--val-features", default=None, help="Held-out val.npz; grouped split is used if omitted")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    return parser


def main(args: argparse.Namespace, input_dim: int = 12) -> int:
    seed_all(args.seed)
    device = choose_device(args.device)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    train_x, train_y, train_groups, _ = load_rows(Path(args.features).resolve(), input_dim)
    if args.val_features:
        val_x, val_y, _, _ = load_rows(Path(args.val_features).resolve(), input_dim)
    else:
        train_x, train_y, val_x, val_y = grouped_validation(train_x, train_y, train_groups, args.seed)

    feature_mean = train_x.mean(axis=0)
    feature_std = train_x.std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0
    train_x = (train_x - feature_mean) / feature_std
    val_x = (val_x - feature_mean) / feature_std
    if not np.isfinite(train_x).all() or not np.isfinite(val_x).all():
        raise ValueError("Feature arrays contain non-finite values")

    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y.astype(np.float32)))
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed))
    val_features = torch.from_numpy(val_x)
    val_ratings = torch.from_numpy(val_y.astype(np.float32))
    model = MalignancyFusion(input_dim=input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_qwk = -float("inf")
    best_spearman = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    history_path = out_dir / "history.csv"
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = ordinal_reader_loss(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_metrics = metrics(model, val_features, val_ratings, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), **val_metrics}
        history.append(row)
        qwk = val_metrics["qwk"]
        rho = val_metrics["spearman"]
        improved = np.isfinite(qwk) and (qwk > best_qwk or (qwk == best_qwk and rho > best_spearman))
        if improved:
            best_qwk, best_spearman, best_epoch = qwk, rho, epoch
            stale_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "architecture": {"name": "MalignancyFusion", "input_dim": input_dim,
                                 "hidden_dims": [32, 16], "dropout": 0.2, "ordinal_classes": 5},
                "feature_mean": feature_mean.astype(np.float32),
                "feature_std": feature_std.astype(np.float32),
                "best_epoch": epoch,
                "best_qwk": best_qwk,
                "best_spearman": best_spearman,
                "seed": args.seed,
                "training_config": vars(args),
                "interpretation": "Radiologist malignancy suspicion score, not a cancer diagnosis.",
            }, out_dir / "best.pth")
        else:
            stale_epochs += 1
        print(f"epoch={epoch:03d} train_loss={row['train_loss']:.4f} "
              f"val_qwk={qwk:.4f} val_spearman={rho:.4f} best_epoch={best_epoch}", flush=True)
        if stale_epochs >= args.patience:
            break

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "qwk", "spearman"])
        writer.writeheader()
        writer.writerows(history)
    config = {**vars(args), "input_dim": input_dim, "best_epoch": best_epoch,
              "best_qwk": best_qwk, "best_spearman": best_spearman,
              "rating_interpretation": {"1": "very unlikely", "2": "unlikely",
                  "3": "indeterminate", "4": "suspicious", "5": "highly suspicious"}}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Saved {out_dir / 'best.pth'} (validation QWK={best_qwk:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(build_parser().parse_args()))