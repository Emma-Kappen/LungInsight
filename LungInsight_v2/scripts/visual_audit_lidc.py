#!/usr/bin/env python3
"""
Visual audit of prepared LIDC 64^3 nodule patches.

Produces an 8-column x 3-row montage:
    row 1: central axial slice
    row 2: central coronal slice
    row 3: central sagittal slice

Samples are deliberately selected from low, middle, and high
center-vs-shell intensity contrast so that patch centering can be
visually inspected.

The output also prints the eight reader-mean attributes for every
displayed sample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


ATTRIBUTE_NAMES = [
    "subtlety",
    "internalStructure",
    "calcification",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
]

READER_TO_MODEL = [7, 1, 0, 5, 4, 2, 6, 8]


def decode(v):
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return str(v)


def reader_means(labels, mask):
    labels = labels[:, :, READER_TO_MODEL].transpose(0, 2, 1)

    out = []
    for a in range(8):
        y = labels[:, a, :]
        valid = mask & np.isfinite(y)
        den = valid.sum(axis=1).clip(min=1)
        num = np.where(valid, y, 0.0).sum(axis=1)
        out.append(num / den)

    return np.stack(out, axis=1)


def select_samples(patches, count=8):
    n = len(patches)
    contrast = np.empty(n, dtype=np.float32)

    for i in range(n):
        x = np.asarray(patches[i], dtype=np.float32)
        center = x[28:36, 28:36, 28:36]
        shell = x.copy()
        shell[27:37, 27:37, 27:37] = np.nan
        contrast[i] = (
            np.nanmean(center) -
            np.nanmean(shell)
        )

    order = np.argsort(contrast)

    # Spread selection over the full contrast range.
    positions = np.linspace(
        0,
        n - 1,
        count,
    ).round().astype(int)

    indices = order[positions]
    return indices, contrast


def window(x):
    # HU display window suitable for lung / nodule inspection.
    return np.clip(x, -1000.0, 400.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--count", type=int, default=8)
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as h5:
        patches = h5["patches"]
        labels = np.asarray(
            h5["reader_labels"],
            dtype=np.float32,
        )
        mask = np.asarray(
            h5["reader_mask"],
            dtype=bool,
        )
        patient_ids = np.asarray(
            h5["patient_id"]
        )
        nodule_ids = np.asarray(
            h5["nodule_id"]
        )

        means = reader_means(
            labels,
            mask,
        )

        indices, contrasts = select_samples(
            patches,
            count=args.count,
        )

        fig, axes = plt.subplots(
            3,
            len(indices),
            figsize=(3.2 * len(indices), 10),
            squeeze=False,
        )

        fig.suptitle(
            "LIDC HDF5 Patch Centering / Visual Audit",
            fontsize=16,
        )

        rows = [
            "Axial z=32",
            "Coronal y=32",
            "Sagittal x=32",
        ]

        print("=" * 100)
        print("SELECTED PATCHES")
        print("=" * 100)

        for col, idx in enumerate(indices):
            x = np.asarray(
                patches[int(idx)],
                dtype=np.float32,
            )

            # Axial: z, y, x
            axial = x[32, :, :]
            coronal = x[:, 32, :]
            sagittal = x[:, :, 32]

            images = [
                axial,
                coronal,
                sagittal,
            ]

            for row, image in enumerate(images):
                ax = axes[row, col]
                ax.imshow(
                    window(image),
                    cmap="gray",
                    vmin=-1000,
                    vmax=400,
                )
                ax.set_xticks([])
                ax.set_yticks([])

                if col == 0:
                    ax.set_ylabel(
                        rows[row],
                        fontsize=10,
                    )

            patient = decode(
                patient_ids[int(idx)]
            )
            nodule = decode(
                nodule_ids[int(idx)]
            )

            title = (
                f"{int(idx)}\n"
                f"{patient}\n"
                f"{nodule}\n"
                f"Δcenter-shell={contrasts[int(idx)]:.1f}"
            )

            axes[0, col].set_title(
                title,
                fontsize=8,
            )

            print(
                f"index={int(idx):4d} "
                f"contrast={contrasts[int(idx)]:8.2f} "
                f"patient={patient} "
                f"nodule={nodule}"
            )
            print(
                "  "
                + " | ".join(
                    f"{a}={means[int(idx), j]:.2f}"
                    for j, a in enumerate(ATTRIBUTE_NAMES)
                )
            )

        plt.tight_layout(
            rect=(0, 0, 1, 0.96),
        )

        output = Path(args.output)
        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output,
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

        print()
        print("Saved:", output)


if __name__ == "__main__":
    main()
