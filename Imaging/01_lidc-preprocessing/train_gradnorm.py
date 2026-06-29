"""
train_gradnorm.py

Training utilities for a multi-head 3D SE-ResNet50 with dynamic loss balancing via GradNorm.
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
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from gradnorm_pytorch import GradNorm
from cir_multihead_pipeline import LIDCPatchDataset, create_multihead_model, FEATURE_NAMES


def build_gradnorm_model(device: str = 'cpu', alpha: float = 1.0, head_names: Optional[List[str]] = None):
    head_names = head_names or FEATURE_NAMES
    model = create_multihead_model(head_names=head_names, device=device)

    # Attempt to create GradNorm wrapper with the last shared backbone layer.
    # If the package expects a direct module or a string name, both are handled.
    last_shared_layer = model.layer4
    try:
        gradnorm = GradNorm(model, num_losses=len(head_names), alpha=alpha, last_shared_layer=last_shared_layer)
    except TypeError:
        gradnorm = GradNorm(model, num_losses=len(head_names), alpha=alpha, last_shared_layer='layer4')

    return model, gradnorm


def _extract_gradnorm_weights(gradnorm: GradNorm, head_names: List[str]) -> Dict[str, float]:
    weight_attrs = ['weights', 'loss_weights', 'task_weights', 'alpha', 'w']
    weights = None
    for attr in weight_attrs:
        if hasattr(gradnorm, attr):
            candidate = getattr(gradnorm, attr)
            if isinstance(candidate, torch.Tensor):
                weights = candidate.detach().cpu().tolist()
                break
            if isinstance(candidate, (list, tuple)) and len(candidate) == len(head_names):
                weights = list(candidate)
                break
    if weights is None:
        # fallback if the wrapper stores weights inside a submodule
        if hasattr(gradnorm, 'gradnorm'):  # hypothetical internal wrapper
            internal = getattr(gradnorm, 'gradnorm')
            for attr in weight_attrs:
                if hasattr(internal, attr):
                    candidate = getattr(internal, attr)
                    if isinstance(candidate, torch.Tensor):
                        weights = candidate.detach().cpu().tolist()
                        break
                    if isinstance(candidate, (list, tuple)) and len(candidate) == len(head_names):
                        weights = list(candidate)
                        break
    if weights is None:
        weights = [1.0] * len(head_names)
    return {name: float(weights[i]) for i, name in enumerate(head_names)}


def train_gradnorm_epoch(
    model: nn.Module,
    gradnorm: GradNorm,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: str = 'cpu',
    head_names: Optional[List[str]] = None,
    log_interval: int = 10,
) -> Dict[str, float]:
    head_names = head_names or FEATURE_NAMES
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    model.train()

    running_loss = {name: 0.0 for name in head_names}
    running_weight = {name: 0.0 for name in head_names}
    batch_count = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader, start=1):
        inputs = inputs.to(device)
        targets = {name: targets[name].to(device).float() for name in head_names}

        optimizer.zero_grad()

        outputs = model(inputs)
        if not isinstance(outputs, dict):
            raise RuntimeError('Model must return a dict of head_name->logits')

        individual_losses = []
        raw_losses = {}
        for name in head_names:
            logits = outputs[name]
            if logits.dim() > 1 and logits.size(1) == 1:
                logits = logits.squeeze(1)
            loss = criterion(logits, targets[name])
            individual_losses.append(loss)
            raw_losses[name] = loss.item()

        losses_tensor = torch.stack(individual_losses)

        # GradNorm expects the wrapped model and list/tensor of scalar losses.
        try:
            result = gradnorm(inputs, losses_tensor)
        except TypeError:
            result = gradnorm(losses_tensor)

        # If GradNorm returns a scalar loss, backward it.
        if isinstance(result, torch.Tensor):
            result.backward()
            optimizer.step()
        else:
            optimizer.step()

        batch_count += 1
        for name in head_names:
            running_loss[name] += raw_losses[name]

        if batch_idx % log_interval == 0 or batch_idx == len(dataloader):
            weights = _extract_gradnorm_weights(gradnorm, head_names)
            loss_summary = ', '.join(f'{name}: {raw_losses[name]:.4f}' for name in head_names)
            weight_summary = ', '.join(f'{name}: {weights[name]:.4f}' for name in head_names)
            print(f'Batch {batch_idx}/{len(dataloader)} | Losses: {loss_summary} | Weights: {weight_summary}')

    avg_loss = {name: running_loss[name] / batch_count for name in head_names}
    return avg_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train 3D SE-ResNet50 with GradNorm for multi-head LIDC targets')
    parser.add_argument('--manifest', type=str, required=True, help='Path to extended_cir_manifest.csv')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num-workers', type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device

    dataset = LIDCPatchDataset(args.manifest)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model, gradnorm = build_gradnorm_model(device=device, alpha=args.alpha)
    optimizer = Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        print(f'=== Epoch {epoch}/{args.epochs} ===')
        avg_loss = train_gradnorm_epoch(
            model,
            gradnorm,
            dataloader,
            optimizer,
            device=device,
            head_names=FEATURE_NAMES,
            log_interval=10,
        )
        avg_summary = ', '.join(f'{name}: {avg_loss[name]:.4f}' for name in FEATURE_NAMES)
        print(f'Epoch {epoch} average losses: {avg_summary}')


if __name__ == '__main__':
    main()
