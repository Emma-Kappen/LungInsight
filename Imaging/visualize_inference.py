"""
visualize_inference.py

Visualize the output of inference_cpu.py for a single nodule.

Usage:
    python visualize_inference.py <path_to_inference_results.npz>

Produces a PNG alongside the .npz showing:
  - Top row: raw CT central axial slice per head (with confidence score title)
  - Bottom row: Grad-CAM heatmap overlaid on CT for each head
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # no display required; works headless and on Windows
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        print('Usage: python visualize_inference.py <path_to_inference_results.npz>')
        sys.exit(1)

    npz_path = sys.argv[1]
    if not os.path.isfile(npz_path):
        print(f'File not found: {npz_path}')
        sys.exit(1)

    data = np.load(npz_path)

    # Patch: stored as (1, 1, D, H, W) by inference_cpu.py
    patch = data['patch']
    if patch.ndim == 5:
        patch = patch[0, 0]   # -> (D, H, W)
    elif patch.ndim == 4:
        patch = patch[0]      # -> (D, H, W) if stored as (1, D, H, W)
    elif patch.ndim == 3:
        pass
    else:
        raise ValueError(f'Unexpected patch shape: {patch.shape}')

    mid = patch.shape[0] // 2  # central axial slice index

    heads = sorted(
        [k.replace('_heatmap', '') for k in data.files if k.endswith('_heatmap')]
    )
    if not heads:
        print('No heatmap keys found in npz. Keys present:', list(data.files))
        sys.exit(1)

    n = len(heads)
    fig, axes = plt.subplots(2, n, figsize=(n * 3, 6))
    if n == 1:
        axes = axes.reshape(2, 1)

    # Normalize CT slice for display
    ct_slice = patch[mid].astype(np.float32)
    p1, p99 = np.percentile(ct_slice, [1, 99])
    ct_norm = np.clip((ct_slice - p1) / (p99 - p1 + 1e-8), 0.0, 1.0)

    for i, head in enumerate(heads):
        heatmap_key = f'{head}_heatmap'
        prob_key    = f'{head}_prob'

        heatmap = data[heatmap_key]
        # Heatmap may be (D, H, W) or (1, D, H, W)
        if heatmap.ndim == 4:
            heatmap = heatmap[0]
        heatmap_slice = heatmap[mid]

        prob = float(data[prob_key]) if prob_key in data.files else float('nan')

        # Top row: raw CT slice
        axes[0, i].imshow(ct_norm, cmap='gray', vmin=0, vmax=1)
        axes[0, i].set_title(f'{head}\n{prob:.3f}', fontsize=8)
        axes[0, i].axis('off')

        # Bottom row: Grad-CAM overlay
        axes[1, i].imshow(ct_norm, cmap='gray', vmin=0, vmax=1)
        axes[1, i].imshow(heatmap_slice, cmap='jet', alpha=0.45, vmin=0, vmax=1)
        axes[1, i].axis('off')

    nodule_name = os.path.basename(npz_path).replace('_inference_results.npz', '')
    fig.suptitle(nodule_name, fontsize=10, fontweight='bold')
    axes[0, 0].set_ylabel('CT slice', fontsize=8)
    axes[1, 0].set_ylabel('Grad-CAM', fontsize=8)

    plt.tight_layout()

    out_path = npz_path.replace('.npz', '_vis.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()