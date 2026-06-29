"""
train_colab_gpu.py

Colab-compatible GPU training script for multi-head 3D SE-ResNet50 with GradNorm.
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

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from gradnorm_pytorch import GradNorm
from cir_multihead_pipeline import FEATURE_NAMES, create_multihead_model
from preprocess_cpu import ExtendedCirDataset


def build_gradnorm_model(device: torch.device, alpha: float = 1.0, head_names: Optional[List[str]] = None):
    head_names = head_names or FEATURE_NAMES
    model = create_multihead_model(head_names=head_names, device=device)
    last_shared_layer = model.layer4
    gradnorm = GradNorm(model, num_losses=len(head_names), alpha=alpha, last_shared_layer=last_shared_layer)
    return model, gradnorm


def _extract_gradnorm_weights(gradnorm: GradNorm, head_names: List[str]):
    if hasattr(gradnorm, 'weights'):
        weights = gradnorm.weights
    elif hasattr(gradnorm, 'loss_weights'):
        weights = gradnorm.loss_weights
    else:
        weights = None

    if isinstance(weights, torch.Tensor):
        return {name: float(weights[i].detach().cpu().item()) for i, name in enumerate(head_names)}
    if isinstance(weights, (list, tuple)):
        return {name: float(weights[i]) for i, name in enumerate(head_names)}
    return {name: 1.0 for name in head_names}


def train_one_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    gradnorm: GradNorm,
    optimizer: Optimizer,
    device: torch.device,
    head_names: Optional[List[str]] = None,
    log_interval: int = 10,
):
    head_names = head_names or FEATURE_NAMES
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    model.train()
    running_loss = {name: 0.0 for name in head_names}

    for batch_idx, (inputs, targets) in enumerate(dataloader, 1):
        inputs = inputs.to(device)
        targets = {name: targets[name].to(device).float() for name in head_names}

        optimizer.zero_grad()
        outputs = model(inputs)
        losses = []

        for name in head_names:
            logits = outputs[name]
            if logits.dim() > 1 and logits.size(1) == 1:
                logits = logits.squeeze(1)
            loss = criterion(logits, targets[name])
            losses.append(loss)
            running_loss[name] += loss.item()

        losses_tensor = torch.stack(losses)
        gradnorm(inputs, losses_tensor)
        optimizer.step()

        if batch_idx % log_interval == 0:
            current_weights = _extract_gradnorm_weights(gradnorm, head_names)
            loss_str = ', '.join(f'{name}: {losses[i].item():.4f}' for i, name in enumerate(head_names))
            weight_str = ', '.join(f'{name}: {current_weights[name]:.4f}' for name in head_names)
            print(f'Batch {batch_idx}/{len(dataloader)} | Losses: {loss_str} | Weights: {weight_str}')

    avg_loss = {name: running_loss[name] / len(dataloader) for name in head_names}
    return avg_loss


def validate_one_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    device: torch.device,
    head_names: Optional[List[str]] = None,
):
    head_names = head_names or FEATURE_NAMES
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    model.eval()
    val_loss = {name: 0.0 for name in head_names}

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = {name: targets[name].to(device).float() for name in head_names}
            outputs = model(inputs)
            for name in head_names:
                logits = outputs[name]
                if logits.dim() > 1 and logits.size(1) == 1:
                    logits = logits.squeeze(1)
                loss = criterion(logits, targets[name])
                val_loss[name] += loss.item()

    avg_loss = {name: val_loss[name] / len(dataloader) for name in head_names}
    return avg_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Train multi-head 3D SE-ResNet50 on Colab GPU using GradNorm')
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--val-csv', required=True)
    parser.add_argument('--drive-dir', required=True, help='Mounted Google Drive output directory')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--num-workers', type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    train_dataset = ExtendedCirDataset(args.train_csv, device='cpu')
    val_dataset = ExtendedCirDataset(args.val_csv, device='cpu')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model, gradnorm = build_gradnorm_model(device=device, alpha=args.alpha)
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    best_model_path = os.path.join(args.drive_dir, 'best_model_gpu.pth')
    os.makedirs(args.drive_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f'=== Epoch {epoch}/{args.epochs} ===')
        train_loss = train_one_epoch(train_loader, model, gradnorm, optimizer, device)
        val_loss = validate_one_epoch(val_loader, model, device)

        print('Training losses:', train_loss)
        print('Validation losses:', val_loss)

        avg_val = sum(val_loss.values()) / len(val_loss)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved best model to {best_model_path} (avg val loss {best_val_loss:.4f})')


if __name__ == '__main__':
    main()
