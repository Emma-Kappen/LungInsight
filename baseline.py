import sys, json, h5py, numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, "/home/jl_fs/LungInsight/LungInsight_v2/backend/models/nodule_ranker")
import LungInsight_v2.backend.models.nodule_ranker.train as train

H5 = "/home/jl_fs/LungInsight/datasets/LIDC_h5/lidc_patches.h5"
OUT = "/home/jl_fs/LungInsight/LungInsight_v2/backend/models/nodule_ranker/checkpoints_h5_final"

split = json.load(open(OUT + "/split.json"))

train_patients = set(split["train_patients"])
test_patients = set(split["test_patients"])

with h5py.File(H5, "r") as f:
    patients = np.asarray(f["patient_id"][:], dtype=str) # type: ignore
    Y = np.asarray(f["attribute_mean"][:], dtype=np.float32)[:, train.H5_TO_MODEL] # type: ignore

train_idx = np.where(np.isin(patients, list(train_patients)))[0]
test_idx  = np.where(np.isin(patients, list(test_patients)))[0]

Ytr = Y[train_idx]
Yte = Y[test_idx]

print("CONSTANT TRAIN-MEAN BASELINE")
print("train samples:", len(Ytr))
print("test samples :", len(Yte))
print()

metrics = []

for j, a in enumerate(train.ATTRIBUTE_NAMES):
    pred = np.full(len(Yte), Ytr[:, j].mean(), dtype=np.float32)
    truth = Yte[:, j]

    lo, hi = train.RANGES[a]

    mae = np.mean(np.abs(pred - truth))
    rmse = np.sqrt(np.mean((pred - truth) ** 2))
    pearson = pearsonr(truth, pred).statistic # type: ignore
    spearman = spearmanr(truth, pred).statistic # type: ignore
    exact = np.mean(np.clip(np.rint(pred), lo, hi) == np.rint(truth))
    within1 = np.mean(np.abs(np.clip(np.rint(pred), lo, hi) - np.rint(truth)) <= 1)

    print(
        f"{a:20s} "
        f"MAE={mae:.4f} "
        f"RMSE={rmse:.4f} "
        f"Pearson={pearson:.4f} "
        f"Spearman={spearman:.4f} "
        f"Exact={exact:.4f} "
        f"Within1={within1:.4f}"
    )

    metrics.append([mae, rmse, pearson, spearman, exact, within1])

metrics = np.asarray(metrics)

print()
print(
    "AGGREGATE          "
    f"MAE={metrics[:,0].mean():.4f} "
    f"RMSE={metrics[:,1].mean():.4f} "
    f"Pearson={metrics[:,2].mean():.4f} "
    f"Spearman={metrics[:,3].mean():.4f} "
    f"Exact={metrics[:,4].mean():.4f} "
    f"Within1={metrics[:,5].mean():.4f}"
)