#!/usr/bin/env python3
"""Compare matched multi-task and single-task validation checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import torch

ATTRIBUTES = ["subtlety", "sphericity", "margin", "lobulation", "spiculation", "texture"]


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("training_config", {})
    architecture = checkpoint.get("architecture", {})
    attribute = config.get("attribute", checkpoint.get("attribute", "all"))
    if attribute not in {"all", *ATTRIBUTES}:
        raise ValueError(f"{path}: unsupported checkpoint attribute {attribute!r}")

    ranking_target = config.get("ranking_target", checkpoint.get("ranking_target", "mean"))
    config_key = {
        "objective": config.get("objective", architecture.get("objective")),
        "ranking_target": ranking_target,
        "rank_loss_weight": float(config.get("rank_loss_weight", 0.0)),
        "ranking_pair_threshold": float(config.get("ranking_pair_threshold", 0.5)),
        "ranking_min_confidence": (
            float(config.get("ranking_min_confidence", 0.10))
            if ranking_target == "reader_pairwise" else None
        ),
        "architecture": json.dumps(architecture, sort_keys=True),
    }
    seed = int(config.get("seed", checkpoint.get("seed", -1)))
    validation = checkpoint.get("validation_metrics", {})
    return {
        "path": path,
        "attribute": attribute,
        "seed": seed,
        "config": config_key,
        "validation": validation,
    }


def checkpoint_index(root: Path) -> dict[tuple[tuple[str, str], str], dict[int, dict[str, Any]]]:
    index: dict[tuple[tuple[str, str], str], dict[int, dict[str, Any]]] = {}
    paths = sorted(root.rglob("best.pth"))
    if not paths:
        raise FileNotFoundError(f"No best.pth checkpoints found under {root}")

    for path in paths:
        record = load_checkpoint(path)
        config_items = tuple(sorted(record["config"].items()))
        key = (config_items, record["attribute"])
        by_seed = index.setdefault(key, {})
        seed = record["seed"]
        if seed in by_seed:
            raise ValueError(
                f"Duplicate {record['attribute']} checkpoint for seed {seed} and config "
                f"{dict(config_items)}:\n  {by_seed[seed]['path']}\n  {path}"
            )
        by_seed[seed] = record
    return index


def config_matches(config: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.objective is not None and config["objective"] != args.objective:
        return False
    if args.ranking_target is not None and config["ranking_target"] != args.ranking_target:
        return False
    if args.rank_loss_weight is not None and not math.isclose(config["rank_loss_weight"], args.rank_loss_weight):
        return False
    if args.ranking_pair_threshold is not None and not math.isclose(
        config["ranking_pair_threshold"], args.ranking_pair_threshold
    ):
        return False
    if args.ranking_min_confidence is not None and (
        config["ranking_min_confidence"] is None
        or not math.isclose(config["ranking_min_confidence"], args.ranking_min_confidence)
    ):
        return False
    return True


def metric_value(record: dict[str, Any], attribute: str, metric: str) -> float:
    try:
        value = float(record["validation"][attribute][metric])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{record['path']}: missing validation_metrics[{attribute!r}][{metric!r}]") from exc
    return value


def main(args: argparse.Namespace) -> int:
    root = Path(args.checkpoint_dir).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    index = checkpoint_index(root)

    multitask_by_config: dict[tuple[tuple[str, str], ...], dict[int, dict[str, Any]]] = {}
    singletask_by_config: dict[tuple[tuple[str, str], ...], dict[str, dict[int, dict[str, Any]]]] = {}
    for (config_items, attribute), by_seed in index.items():
        config = dict(config_items)
        if not config_matches(config, args):
            continue
        if attribute == "all":
            multitask_by_config[config_items] = by_seed
        else:
            singletask_by_config.setdefault(config_items, {})[attribute] = by_seed

    candidates: list[tuple[tuple[tuple[str, str], ...], dict[str, set[int]]]] = []
    for config_items, multitask in multitask_by_config.items():
        single = singletask_by_config.get(config_items, {})
        matched: dict[str, set[int]] = {}
        for attribute in ATTRIBUTES:
            single_seeds = single.get(attribute, {})
            common = set(multitask) & set(single_seeds)
            if not common:
                break
            matched[attribute] = common
        if len(matched) == len(ATTRIBUTES):
            candidates.append((config_items, matched))

    if not candidates:
        raise ValueError(
            "No complete matched experiment found. Need multi-task and all six single-task "
            "checkpoints with overlapping seeds and identical objective/ranking settings."
        )
    if len(candidates) > 1:
        choices = "\n".join(str(dict(config)) for config, _ in candidates)
        raise ValueError(
            "More than one matching experiment is present. Narrow it with --objective, "
            "--ranking-target, --rank-loss-weight, --ranking-pair-threshold, and/or "
            "--ranking-min-confidence:\n" + choices
        )

    config_items, matched = candidates[0]
    config = dict(config_items)
    multitask = multitask_by_config[config_items]
    singletask = singletask_by_config[config_items]
    rows: list[dict[str, Any]] = []
    for attribute in ATTRIBUTES:
        seeds = sorted(matched[attribute])
        multitask_scores = [metric_value(multitask[seed], attribute, "spearman_ge3") for seed in seeds]
        singletask_records = singletask[attribute]
        singletask_scores = [metric_value(singletask_records[seed], attribute, "spearman_ge3") for seed in seeds]
        multitask_ratios = [metric_value(multitask[seed], attribute, "sd_ratio") for seed in seeds]
        singletask_ratios = [metric_value(singletask_records[seed], attribute, "sd_ratio") for seed in seeds]
        if not all(math.isfinite(value) for value in multitask_scores + singletask_scores):
            raise ValueError(f"{attribute}: non-finite spearman_ge3 in a matched checkpoint; inspect collapsed runs")
        valid_ratio_pairs = [
            (left, right)
            for left, right in zip(multitask_ratios, singletask_ratios)
            if math.isfinite(left) and math.isfinite(right)
        ]
        if not valid_ratio_pairs:
            raise ValueError(f"{attribute}: no finite matched sd_ratio values")

        multitask_best = mean(multitask_scores)
        singletask_best = mean(singletask_scores)
        rows.append({
            "attribute": attribute,
            "multitask_best": multitask_best,
            "singletask_best": singletask_best,
            "multitask_sd_ratio": mean(left for left, _ in valid_ratio_pairs),
            "singletask_sd_ratio": mean(right for _, right in valid_ratio_pairs),
            "improvement": singletask_best - multitask_best,
        })
        print(f"{attribute}: matched seeds={seeds}")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "attribute",
        "multitask_best",
        "singletask_best",
        "multitask_sd_ratio",
        "singletask_sd_ratio",
        "improvement",
    ]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Compared selected-checkpoint mean validation spearman_ge3 for config {config}")
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--objective", choices=["ordinal_reader", "mean_regression"], default=None)
    parser.add_argument("--ranking-target", choices=["mean", "median", "mean_median_blend", "reader_pairwise"], default=None)
    parser.add_argument("--rank-loss-weight", type=float, default=None)
    parser.add_argument("--ranking-pair-threshold", type=float, default=None)
    parser.add_argument("--ranking-min-confidence", type=float, default=None)
    raise SystemExit(main(parser.parse_args()))