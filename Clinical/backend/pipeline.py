"""Multimodal clinical + CT inference orchestration."""
from __future__ import annotations
import base64, io, logging, tempfile
from pathlib import Path
from typing import Any
import joblib, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import *
from .imaging_processor import load_ct, resample_isotropic, preview_slice

log = logging.getLogger(__name__)

class ClinicalPipeline:
    def __init__(self, model_dir: str | Path | None = None):
        self.model_dir = Path(model_dir or MODEL_DIR)
        self.preprocessor = self._load("clinical_preprocessor.joblib", required=True)
        self.stage_bundle = self._load("stage_model_bundle.joblib", required=False)
        self.stage_model = self._unwrap_model(self.stage_bundle, ["model", "classifier", "estimator"]) or self._load("stage_rf_model.joblib", required=True)
        self.hist_bundle = self._load("histology_model_bundle.joblib", required=False)
        self.hist_model = self._unwrap_model(self.hist_bundle, ["model", "classifier", "estimator"]) or self._load("histology_rf_model.joblib", required=False)
        self.survival_model = self._load("survival_coxph_model.joblib", required=False)
        self.fusion_model = self._load("multimodal_fusion_model.joblib", required=False)
        self.imaging_extractor = self._load("imaging_feature_extractor.pt", required=False) or self._load("imaging_model.pt", required=False)
        self.stage_threshold = float(self._bundle_get(self.stage_bundle, "threshold", 0.535))
        log.info("ClinicalPipeline loaded from %s", self.model_dir)

    def _load(self, name, required=False):
        p = self.model_dir / name
        if not p.exists():
            if required: raise FileNotFoundError(f"Required model artifact missing: {p}")
            return None
        if p.suffix == ".joblib": return joblib.load(p)
        return __import__("torch").load(p, map_location="cpu")

    @staticmethod
    def _bundle_get(bundle, key, default=None):
        if isinstance(bundle, dict): return bundle.get(key, default)
        return getattr(bundle, key, default)

    @staticmethod
    def _unwrap_model(bundle, names):
        if bundle is None: return None
        if hasattr(bundle, "predict"): return bundle
        if isinstance(bundle, dict):
            for n in names:
                if hasattr(bundle.get(n), "predict"): return bundle[n]
        for n in names:
            obj = getattr(bundle, n, None)
            if hasattr(obj, "predict"): return obj
        return None

    def validate(self, data: dict) -> dict:
        out = {}
        for k, (lo, hi) in NUMERIC_RANGES.items():
            v = data.get(k, np.nan)
            if v in ("", None, "Unknown / Missing"): v = np.nan
            else: v = float(v)
            if not np.isnan(v) and not (lo <= v <= hi):
                raise ValueError(f"{k} must be between {lo} and {hi}.")
            out[k] = v
        for k, allowed in CATEGORIES.items():
            v = data.get(k, "Unknown / Missing" if k == "ever_pdl1_positive" else np.nan)
            if k == "ever_pdl1_positive" and v == "Unknown / Missing": v = np.nan
            if not pd.isna(v) and v not in allowed: raise ValueError(f"Invalid {k}.")
            out[k] = v
        out["missing_ever_pdl1_positive"] = int(pd.isna(out["ever_pdl1_positive"]))
        out["missing_eastern_cancer_oncology_group"] = int(pd.isna(out["eastern_cancer_oncology_group"]))
        out["missing_karnofsky_performance_score"] = int(pd.isna(out["karnofsky_performance_score"]))
        out["SOURCE_DATASET"] = data.get("SOURCE_DATASET", "Web Upload")
        return out

    def _tabular_frame(self, data):
        d = self.validate(data)
        return pd.DataFrame([d]), d

    def _stage(self, x):
        probs = None
        if hasattr(self.stage_model, "predict_proba"):
            probs = np.asarray(self.stage_model.predict_proba(x))[0]
            classes = list(getattr(self.stage_model, "classes_", range(len(probs))))
            positive = "Stage 4" if "Stage 4" in classes else classes[-1]
            pi = classes.index(positive)
            label = positive if probs[pi] >= self.stage_threshold else ("Stage 1-3" if positive == "Stage 4" else classes[0])
            return label, dict(zip(map(str, classes), probs.tolist())), float(probs[pi])
        pred = self.stage_model.predict(x)[0]
        return str(pred), {}, None

    def _histology(self, x):
        if self.hist_model is None: return {"label": None, "probabilities": {}, "available": False}
        probs = {}
        if hasattr(self.hist_model, "predict_proba"):
            p = np.asarray(self.hist_model.predict_proba(x))[0]
            cls = list(getattr(self.hist_model, "classes_", range(len(p))))
            probs = dict(zip(map(str, cls), p.tolist()))
        return {"label": str(self.hist_model.predict(x)[0]), "probabilities": probs, "available": True}

    def _survival(self, x):
        if self.survival_model is None: return {"available": False}
        m = self.survival_model
        risk = float(np.asarray(m.predict(x)).reshape(-1)[0])
        result = {"available": True, "risk_score": risk}
        try:
            fn = m.predict_survival_function(x)[0]
            times = np.asarray(fn.x, dtype=float)
            surv = np.asarray(fn.y, dtype=float)
            result["times_months"] = times.tolist()
            result["survival_probability"] = surv.tolist()
            below = np.where(surv <= 0.5)[0]
            result["median_survival_months"] = float(times[below[0]]) if len(below) else None
        except Exception as e:
            log.warning("Survival curve unavailable: %s", e)
        return result

    def _imaging_features(self, volume):
        if self.imaging_extractor is None:
            raise RuntimeError("No imaging feature extractor artifact found. Add imaging_feature_extractor.pt or imaging_model.pt.")
        import torch
        model = self.imaging_extractor
        if isinstance(model, dict):
            raise RuntimeError("The imaging .pt is a state_dict/config artifact; provide the corresponding model architecture or an exported TorchScript module.")
        model.eval()
        arr = np.clip(volume, -1000, 400) / 1400.0
        t = torch.from_numpy(arr[None, None].astype(np.float32))
        with torch.no_grad():
            z = model(t)
        if isinstance(z, (tuple, list)): z = z[0]
        return np.asarray(z.detach().cpu()).reshape(1, -1)

    def _shap(self, x):
        try:
            import shap
            explainer = shap.TreeExplainer(self.stage_model)
            sv = explainer.shap_values(x)
            if isinstance(sv, list): sv = sv[-1]
            vals = np.asarray(sv)[0]
            names = list(getattr(self.preprocessor, "get_feature_names_out", lambda: [f"feature_{i}" for i in range(len(vals))])())
            pairs = sorted(zip(names, vals.tolist()), key=lambda z: z[1], reverse=True)
            pos, neg = pairs[:5], sorted(pairs, key=lambda z: z[1])[:5]
            fig, ax = plt.subplots(figsize=(8, 4.5))
            top = sorted(pos[:5] + neg[:5], key=lambda z: z[1])
            ax.barh([n for n, _ in top], [v for _, v in top])
            ax.axvline(0, linewidth=0.8)
            ax.set_title("Local Stage Model Feature Contributions")
            ax.set_xlabel("SHAP value")
            fig.tight_layout()
            b = io.BytesIO(); fig.savefig(b, format="png", dpi=150); plt.close(fig)
            return {"positive": pos, "negative": neg, "plot_base64": base64.b64encode(b.getvalue()).decode()}
        except Exception as e:
            log.exception("SHAP computation failed")
            return {"positive": [], "negative": [], "plot_base64": None, "error": str(e)}

    def predict(self, tabular_data: dict, ct_path: str | None = None) -> dict:
        raw, clean = self._tabular_frame(tabular_data)
        x = self.preprocessor.transform(raw)
        stage_label, stage_probs, stage_conf = self._stage(x)
        result = {
            "clinical_inputs": clean,
            "stage": {"label": stage_label, "probabilities": stage_probs, "positive_probability": stage_conf, "threshold": self.stage_threshold},
            "histology": self._histology(x),
            "survival": self._survival(x),
            "shap": self._shap(x),
            "imaging": {"available": False},
        }
        if ct_path:
            with tempfile.TemporaryDirectory(prefix="lunginsight_ct_") as td:
                ct = resample_isotropic(load_ct(ct_path, td))
                emb = self._imaging_features(ct.volume_hu)
                result["imaging"] = {
                    "available": True, "shape_zyx": list(ct.volume_hu.shape),
                    "spacing_zyx_mm": list(ct.spacing_zyx), "series_uid": ct.series_uid,
                    "scan_date": ct.scan_date, "representative_slice": preview_slice(ct.volume_hu, ct.volume_hu.shape[0] // 2),
                    "embedding_dimensions": int(emb.shape[1]),
                }
                if self.fusion_model is not None:
                    try:
                        fusion_x = np.hstack([np.asarray(x.toarray() if hasattr(x, "toarray") else x), emb])
                        fp = self.fusion_model.predict_proba(fusion_x)[0] if hasattr(self.fusion_model, "predict_proba") else None
                        result["fusion"] = {"available": True, "probabilities": fp.tolist() if fp is not None else {}, "label": str(self.fusion_model.predict(fusion_x)[0])}
                    except Exception as e:
                        result["fusion"] = {"available": False, "error": str(e)}
                else:
                    result["fusion"] = {"available": False, "error": "multimodal_fusion_model.joblib not found."}
        return result
