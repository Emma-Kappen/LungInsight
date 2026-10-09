import sys, json, h5py, numpy as np, torch
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent / "backend" / "models" / "nodule_ranker"),
)
import LungInsight_v2.backend.models.nodule_ranker.train as train

H5 = "/home/jl_fs/LungInsight/datasets/LIDC_h5/lidc_patches.h5"
OUT = "/home/jl_fs/LungInsight/LungInsight_v2/backend/models/nodule_ranker/checkpoints_h5_final"

ck = torch.load(OUT + "/best.pth", map_location="cpu")
model = train.Net(ck["model_config"]["base"])
model.load_state_dict(ck["model_state_dict"])
model.eval()

split = json.load(open(OUT + "/split.json"))
test_patients = set(split["test_patients"])

with h5py.File(H5, "r") as f:
    patients = np.asarray(f["patient_id"][:], dtype=str) # type: ignore
    idx = np.where(np.isin(patients, list(test_patients)))[0]
    X = np.asarray(f["patches"][idx], dtype=np.float32) # type: ignore
    Y = np.asarray(f["attribute_mean"][idx], dtype=np.float32)[:, train.H5_TO_MODEL] # type: ignore

loader = torch.utils.data.DataLoader(
    train.H5Dataset(X, Y, np.arange(len(idx))),
    batch_size=32,
    shuffle=False
)

preds = []
truth = []

with torch.no_grad():
    for x, y in loader:
        out = model(x)
        p = np.stack(
            [train.decode(out[j], a).numpy()
             for j, a in enumerate(train.ATTRIBUTE_NAMES)],
            axis=1
        )
        preds.append(p)
        truth.append(y.numpy())

P = np.concatenate(preds)
Y = np.concatenate(truth)

print("TEST SAMPLES:", len(Y))
print()
print("ATTRIBUTE              GT_MEAN GT_STD   PRED_MEAN PRED_STD  PRED_MIN PRED_MAX")

for j, a in enumerate(train.ATTRIBUTE_NAMES):
    print(
        f"{a:20s} "
        f"{Y[:,j].mean():7.3f} {Y[:,j].std():7.3f} "
        f"{P[:,j].mean():9.3f} {P[:,j].std():8.3f} "
        f"{P[:,j].min():8.3f} {P[:,j].max():8.3f}"
    )