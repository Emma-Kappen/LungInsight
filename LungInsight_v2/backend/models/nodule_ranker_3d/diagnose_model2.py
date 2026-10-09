import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import the exact implementation used for Model 2.
from train_model2 import (
    H5Dataset,
    Net,
    H5_ORDER,
    H5_TO_MODEL,
    ATTRIBUTE_NAMES,
    RANGES,
    GROUPS,
    seed_all,
)


CHECKPOINT = Path(
    r".\LungInsight_v2\backend\models\nodule_ranker\checkpoints_model2_final\best.pth"
)

H5_PATH = Path(
    r".\datasets\LIDC\lidc_patches.h5"
)

SEED = 42
BATCH_SIZE = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    seed_all(SEED)

    print("Device:", DEVICE)
    print("Checkpoint:", CHECKPOINT)
    print("HDF5:", H5_PATH)

    # ------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------
    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
    )

    print("\nCheckpoint:")
    print("  epoch:", checkpoint["epoch"])
    print("  val_mae:", checkpoint["val_mae"])
    print("  model_config:", checkpoint["model_config"])

    target_mean = np.asarray(
        checkpoint["target_mean"],
        dtype=np.float32,
    )

    target_std = np.asarray(
        checkpoint["target_std"],
        dtype=np.float32,
    )

    # ------------------------------------------------------------
    # Load HDF5
    # ------------------------------------------------------------
    with h5py.File(H5_PATH, "r") as f:

        patches = np.asarray(
            f["patches"][:],
            dtype=np.float32,
        )

        patients = np.asarray(
            f["patient_id"][:],
            dtype=str,
        )

        y0 = np.asarray(
            f["attribute_mean"][:],
            dtype=np.float32,
        )

        raw_names = f.attrs["attribute_names"]

        if isinstance(raw_names, bytes):
            raw_names = raw_names.decode("utf-8")

        names = json.loads(str(raw_names))

    assert names == H5_ORDER

    # Convert HDF5 ordering -> required model ordering.
    y = y0[:, H5_TO_MODEL]

    # ------------------------------------------------------------
    # Reproduce EXACT patient split used during training.
    # ------------------------------------------------------------
    pids = np.unique(patients).tolist()

    import random

    random.Random(SEED).shuffle(pids)

    n = len(pids)

    nt = max(1, round(0.15 * n))
    nv = max(1, round(0.15 * n))

    test_patients = set(pids[:nt])
    val_patients = set(pids[nt:nt + nv])
    train_patients = set(pids[nt + nv:])

    test_idx = np.where(
        np.isin(patients, list(test_patients))
    )[0]

    print("\nSplit:")
    print("  train patients:", len(train_patients))
    print("  val patients:", len(val_patients))
    print("  test patients:", len(test_patients))
    print("  test nodules:", len(test_idx))

    assert len(test_idx) == 355

    # ------------------------------------------------------------
    # Normalize targets exactly as training did.
    # ------------------------------------------------------------
    train_idx = np.where(
        np.isin(patients, list(train_patients))
    )[0]

    train_mean = y[train_idx].mean(axis=0)
    train_std = y[train_idx].std(axis=0)
    train_std = np.maximum(train_std, 0.10)

    np.testing.assert_allclose(
        train_mean,
        target_mean,
        atol=1e-5,
    )

    np.testing.assert_allclose(
        train_std,
        target_std,
        atol=1e-5,
    )

    y_norm = (
        y - target_mean
    ) / target_std

    # ------------------------------------------------------------
    # Dataset / loader.
    #
    # augment=False is critical because this is evaluation.
    # ------------------------------------------------------------
    dataset = H5Dataset(
        patches,
        y_norm,
        test_idx,
        augment=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # ------------------------------------------------------------
    # Build exact model.
    # ------------------------------------------------------------
    base = checkpoint["model_config"]["base"]

    model = Net(base=base)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    # ------------------------------------------------------------
    # Predict.
    # ------------------------------------------------------------
    predictions = []

    with torch.no_grad():

        for x, _ in loader:

            x = x.to(
                DEVICE,
                non_blocking=True,
            )

            outputs = model(x)

            pred = torch.stack(
                outputs,
                dim=1,
            )

            pred = (
                pred
                * torch.from_numpy(target_std).to(DEVICE)
                + torch.from_numpy(target_mean).to(DEVICE)
            )

            predictions.append(
                pred.cpu().numpy()
            )

    predictions = np.concatenate(
        predictions,
        axis=0,
    )

    ground_truth = y[test_idx]

    # ------------------------------------------------------------
    # Diagnostic table.
    # ------------------------------------------------------------
    print("\n")
    print(
        f"{'ATTRIBUTE':<20}"
        f"{'GT_MEAN':>10}"
        f"{'GT_STD':>10}"
        f"{'PRED_MEAN':>12}"
        f"{'PRED_STD':>11}"
        f"{'PRED_MIN':>11}"
        f"{'PRED_MAX':>11}"
    )

    print("-" * 85)

    for j, name in enumerate(ATTRIBUTE_NAMES):

        gt = ground_truth[:, j]
        pred = predictions[:, j]

        print(
            f"{name:<20}"
            f"{gt.mean():10.3f}"
            f"{gt.std():10.3f}"
            f"{pred.mean():12.3f}"
            f"{pred.std():11.3f}"
            f"{pred.min():11.3f}"
            f"{pred.max():11.3f}"
        )

    # ------------------------------------------------------------
    # Prediction-vs-GT standard deviation ratio.
    # ------------------------------------------------------------
    print("\nPrediction/GT STD ratio:")
    print("-" * 50)

    for j, name in enumerate(ATTRIBUTE_NAMES):

        gt_std = ground_truth[:, j].std()
        pred_std = predictions[:, j].std()

        ratio = pred_std / gt_std if gt_std > 0 else np.nan

        print(
            f"{name:<20} "
            f"ratio={ratio:.3f}"
        )


if __name__ == "__main__":
    main()