"""
validate_colab_gpu.py

Colab-compatible GPU validation script for 10-head 3D SE-ResNet50 outputs.
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
IMAGING_ROOT = SCRIPT_DIR.parent
for candidate in (SCRIPT_DIR, IMAGING_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from path_setup import ensure_project_paths

ensure_project_paths(__file__)

import argparse
import os
from typing import List, Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

from cir_multihead_pipeline import FEATURE_NAMES, create_multihead_model
from preprocess_cpu import ExtendedCirDataset


def load_model(checkpoint_path: str, device: torch.device, head_names: Optional[List[str]] = None):
    head_names = head_names or FEATURE_NAMES
    model = create_multihead_model(head_names=head_names, device=device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, device: torch.device):
    y_true = {name: [] for name in FEATURE_NAMES}
    y_scores = {name: [] for name in FEATURE_NAMES}

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            for name in FEATURE_NAMES:
                logits = outputs[name]
                if logits.dim() > 1 and logits.size(1) == 1:
                    logits = logits.squeeze(1)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                labels = targets[name].numpy()
                y_scores[name].extend(probs.tolist())
                y_true[name].extend(labels.tolist())

    metrics = {}
    for name in FEATURE_NAMES:
        try:
            auc = roc_auc_score(y_true[name], y_scores[name])
        except ValueError:
            auc = float('nan')
        preds = [1 if p >= 0.5 else 0 for p in y_scores[name]]
        precision = precision_score(y_true[name], preds, zero_division=0)
        recall = recall_score(y_true[name], preds, zero_division=0)
        f1 = f1_score(y_true[name], preds, zero_division=0)
        metrics[name] = {
            'roc_auc': float(auc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        }
    return metrics


def print_metrics(metrics):
    print('# Validation Metrics')
    for name, vals in metrics.items():
        print(f'## {name}')
        print(f'- ROC AUC: {vals["roc_auc"]:.4f}')
        print(f'- Precision: {vals["precision"]:.4f}')
        print(f'- Recall: {vals["recall"]:.4f}')
        print(f'- F1: {vals["f1"]:.4f}')
        print('')


def parse_args():
    parser = argparse.ArgumentParser(description='Validate multi-head 3D SE-ResNet50 on Colab GPU')
    parser.add_argument('--val-csv', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    val_dataset = ExtendedCirDataset(args.val_csv, device='cpu')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = load_model(args.checkpoint, device=device)
    metrics = evaluate_model(model, val_loader, device)
    print_metrics(metrics)


if __name__ == '__main__':
    main()
