from __future__ import annotations

import base64
import io
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    'AGE', 'SEX', 'RACE', 'ETHNICITY', 'SMOKING_STATUS',
    'num_treatment_events', 'num_distinct_tumor_sites',
    'ever_pdl1_positive', 'eastern_cancer_oncology_group',
    'karnofsky_performance_score', 'SOURCE_DATASET',
    'missing_ever_pdl1_positive',
    'missing_eastern_cancer_oncology_group',
    'missing_karnofsky_performance_score',
]

NUMERIC_BOUNDS = {
    'AGE': (18.0, 100.0),
    'num_treatment_events': (0.0, 20.0),
    'num_distinct_tumor_sites': (1.0, 15.0),
    'eastern_cancer_oncology_group': (0.0, 4.0),
    'karnofsky_performance_score': (0.0, 100.0),
}
CATEGORICALS = {
    'SEX': ['Female', 'Male'],
    'RACE': ['White', 'Black or African American', 'Asian', 'Other'],
    'ETHNICITY': ['Non-Spanish; Non-Hispanic', 'Spanish; Hispanic', 'Unknown'],
    'SMOKING_STATUS': ['Former/Current Smoker', 'Never'],
    'ever_pdl1_positive': ['Yes', 'No', 'Unknown / Missing'],
    'SOURCE_DATASET': ['MSK-CHORD', 'TCGA-LUAD', 'TCGA-LUSC'],
}
MODEL_DEFAULT_SOURCE_DATASET = "MSK-CHORD"
DEFAULTS = {
    'AGE': 65.0, 'SEX': 'Female', 'RACE': 'White',
    'ETHNICITY': 'Non-Spanish; Non-Hispanic',
    'SMOKING_STATUS': 'Former/Current Smoker',
    'num_treatment_events': 2.0, 'num_distinct_tumor_sites': 1.0,
    'ever_pdl1_positive': 'Unknown / Missing',
    'eastern_cancer_oncology_group': 1.0,
    'karnofsky_performance_score': 80.0,
    'SOURCE_DATASET': 'MSK-CHORD',
}


class ClinicalPipeline:
    """Inference adapter for the exact artifacts produced by train_clinical.ipynb.

    Imaging is intentionally not reimplemented here. The class consumes an
    embedding/output directory produced by the existing Imaging/01..09 pipeline.
    """

    def __init__(self, model_dir: str | os.PathLike[str], imaging_root: str | os.PathLike[str] | None = None):
        self.model_dir = Path(model_dir)
        self.imaging_root = Path(imaging_root) if imaging_root else self.model_dir.parent.parent / 'Imaging'
        self.preprocessor = self._load_required('clinical_preprocessor.joblib')
        self.stage_bundle = self._load_optional('stage_model_bundle.joblib')
        self.stage_model = self._load_optional('stage_rf_model.joblib')
        self.hist_bundle = self._load_optional('histology_model_bundle.joblib')
        self.hist_model = self._load_optional('histology_rf_model.joblib')
        self.survival_bundle = self._load_optional('survival_coxph_model.joblib')
        self.stage_explainer = self._load_optional('stage_explainer.joblib')
        self.fusion_model = self._load_optional('multimodal_fusion_model.joblib')
        self._validate_loaded_contracts()

    def _load_required(self, name: str):
        p = self.model_dir / name
        if not p.exists():
            raise FileNotFoundError(f'Missing required clinical artifact: {p}')
        return joblib.load(p)

    def _load_optional(self, name: str):
        p = self.model_dir / name
        return joblib.load(p) if p.exists() else None

    def _validate_loaded_contracts(self):
        if self.stage_bundle is None and self.stage_model is None:
            warnings.warn('No stage model artifact found; stage prediction will be unavailable.')
        if self.hist_bundle is None and self.hist_model is None:
            warnings.warn('No histology model artifact found; histology prediction will be unavailable.')
        if self.survival_bundle is None:
            warnings.warn('No survival bundle found; survival prediction will be unavailable.')

    @staticmethod
    def validate_patient_data(patient_data: Dict[str, Any]) -> Dict[str, Any]:
        x = dict(DEFAULTS)
        incoming = {
            k: v for k, v in patient_data.items()
            if k != 'SOURCE_DATASET'
        }
        x.update({k: v for k, v in incoming.items() if v is not None and v != ''})
        # Keep the fitted preprocessor's training-time feature contract.
        x['SOURCE_DATASET'] = MODEL_DEFAULT_SOURCE_DATASET
        errors = []
        for col, (lo, hi) in NUMERIC_BOUNDS.items():
            value = x.get(col)
            if value is None or value == '':
                continue
            try:
                x[col] = float(value)
            except (TypeError, ValueError):
                errors.append(f'{col} must be numeric.')
                continue
            if not (lo <= x[col] <= hi):
                errors.append(f'{col} must be between {lo} and {hi}.')
        for col, allowed in CATEGORICALS.items():
            if x.get(col) is not None and x[col] not in allowed:
                errors.append(f'{col} must be one of: {allowed}.')
        if errors:
            raise ValueError(' '.join(errors))
        # Preserve actual missing numeric values if explicitly supplied as null.
        for col in NUMERIC_BOUNDS:
            if col in patient_data and patient_data[col] in (None, ''):
                x[col] = np.nan
        return x

    @staticmethod
    def engineer_features(patient: Dict[str, Any]) -> pd.DataFrame:
        row = dict(patient)
        pdl1_missing = pd.isna(row.get('ever_pdl1_positive')) or row.get('ever_pdl1_positive') == 'Unknown / Missing'
        ecog_missing = pd.isna(row.get('eastern_cancer_oncology_group'))
        kps_missing = pd.isna(row.get('karnofsky_performance_score'))
        # Notebook creates missing flags before preprocessing. Numeric NaN remains
        # available to the fitted median imputer, while categorical unknown/missing
        # is represented as NaN so the fitted categorical imputer handles it.
        if row.get('ever_pdl1_positive') == 'Unknown / Missing':
            row['ever_pdl1_positive'] = np.nan
        row['missing_ever_pdl1_positive'] = int(pdl1_missing)
        row['missing_eastern_cancer_oncology_group'] = int(ecog_missing)
        row['missing_karnofsky_performance_score'] = int(kps_missing)
        return pd.DataFrame([row], columns=FEATURE_COLUMNS)

    def _tabular_matrix(self, patient: Dict[str, Any]) -> pd.DataFrame:
        raw = self.engineer_features(patient)
        transformed = self.preprocessor.transform(raw)
        names = list(self.preprocessor.get_feature_names_out())
        return pd.DataFrame(transformed, columns=names, dtype=float)

    @staticmethod
    def _bundle_model(bundle, fallback):
        if isinstance(bundle, dict):
            return bundle.get('model') or fallback
        return bundle or fallback

    def _stage_predict(self, x: pd.DataFrame) -> Dict[str, Any]:
        model = self._bundle_model(self.stage_bundle, self.stage_model)
        if model is None:
            return {'label': 'Unavailable', 'confidence': None, 'probabilities': {}}
        if not hasattr(model, 'predict_proba'):
            label = str(model.predict(x)[0])
            return {'label': label, 'confidence': None, 'probabilities': {}}
        classes = list(model.classes_)
        proba = np.asarray(model.predict_proba(x)[0], dtype=float)
        bundle = self.stage_bundle if isinstance(self.stage_bundle, dict) else {}
        threshold = float(bundle.get('threshold', 0.535))
        positive = bundle.get('positive_label', 'Stage 4')
        if positive in classes:
            pi = classes.index(positive)
            label = positive if proba[pi] >= threshold else next((c for c in classes if c != positive), classes[0])
        else:
            label = classes[int(np.argmax(proba))]
        return {
            'label': str(label),
            'confidence': float(max(proba)),
            'stage4_probability': float(proba[classes.index(positive)]) if positive in classes else None,
            'threshold': threshold,
            'probabilities': {str(c): float(p) for c, p in zip(classes, proba)},
        }

    def _histology_predict(self, x: pd.DataFrame) -> Dict[str, Any]:
        model = self._bundle_model(self.hist_bundle, self.hist_model)
        if model is None:
            return {'label': 'Unavailable', 'confidence': None, 'probabilities': {}}
        proba = np.asarray(model.predict_proba(x)[0], dtype=float) if hasattr(model, 'predict_proba') else None
        if proba is not None:
            idx = int(np.argmax(proba)); label = model.classes_[idx]
            return {'label': str(label), 'confidence': float(proba[idx]), 'probabilities': {str(c): float(p) for c,p in zip(model.classes_,proba)}}
        return {'label': str(model.predict(x)[0]), 'confidence': None, 'probabilities': {}}

    def _survival_matrix(self, patient: Dict[str, Any]) -> pd.DataFrame:
        if not isinstance(self.survival_bundle, dict):
            return pd.DataFrame()
        cols = self.survival_bundle.get('feature_columns')
        if not cols:
            return pd.DataFrame()
        raw = self.engineer_features(patient)
        # Training excluded num_treatment_events because it is often post-baseline.
        raw = raw.drop(columns=['num_treatment_events'], errors='ignore')
        cat_cols = self.survival_bundle.get('categorical_cols', [])
        num_cols = self.survival_bundle.get('numeric_cols', [])
        ni = self.survival_bundle.get('num_imputer')
        ci = self.survival_bundle.get('cat_imputer')
        if num_cols:
            raw[num_cols] = ni.transform(raw[num_cols])
        if cat_cols:
            raw[cat_cols] = ci.transform(raw[cat_cols])
        out = pd.get_dummies(raw, columns=cat_cols, drop_first=True)
        return out.reindex(columns=cols, fill_value=0).astype(float)

    def _survival_predict(self, patient: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(self.survival_bundle, dict) or self.survival_bundle.get('model') is None:
            return {'risk_score': None, 'times_months': [], 'survival_probability': []}
        model = self.survival_bundle['model']
        x = self._survival_matrix(patient)
        risk = float(np.asarray(model.predict(x.astype(np.float64)))[0])
        times = np.linspace(1.0, 120.0, 60)
        # sksurv's baseline_survival_ is a StepFunction in fitted CoxPH models.
        try:
            base = model.baseline_survival_(times)
            surv = np.power(np.asarray(base, dtype=float), np.exp(risk))
        except Exception:
            # Fallback to direct baseline function evaluation at its native grid.
            base_fn = model.baseline_survival_
            native = np.asarray(getattr(base_fn, 'x', times), dtype=float)
            times = np.linspace(float(native.min()), float(native.max()), 60)
            base = np.asarray(base_fn(times), dtype=float)
            surv = np.power(base, np.exp(risk))
        return {'risk_score': risk, 'times_months': [float(t) for t in times], 'survival_probability': [float(np.clip(v,0,1)) for v in surv]}

    @staticmethod
    def _find_embedding(image_output_dir: Path) -> Optional[np.ndarray]:
        candidates = ['imaging_embedding.npy', 'embedding.npy', 'ct_embedding.npy', 'features.npy']
        for name in candidates:
            p = image_output_dir / name
            if p.exists():
                arr = np.asarray(np.load(p, allow_pickle=False), dtype=float)
                return arr.reshape(1, -1) if arr.ndim == 1 else arr
        for name in ['embedding.json', 'imaging_embedding.json']:
            p = image_output_dir / name
            if p.exists():
                obj = json.loads(p.read_text())
                arr = np.asarray(obj.get('embedding', obj), dtype=float)
                return arr.reshape(1, -1) if arr.ndim == 1 else arr
        return None

    def _fusion_predict(self, tabular: pd.DataFrame, image_output_dir: Optional[str]) -> Optional[Dict[str, Any]]:
        if self.fusion_model is None or not image_output_dir:
            return None
        emb = self._find_embedding(Path(image_output_dir))
        if emb is None:
            return None
        model = self.fusion_model.get('model') if isinstance(self.fusion_model, dict) else self.fusion_model
        if model is None:
            return None
        tab = tabular.to_numpy(dtype=float)
        try:
            fused = np.hstack([tab, emb])
            if hasattr(model, 'predict_proba'):
                p = np.asarray(model.predict_proba(fused)[0], dtype=float)
                classes = getattr(model, 'classes_', np.arange(len(p)))
                return {'used': True, 'probabilities': {str(c): float(v) for c,v in zip(classes,p)}, 'label': str(classes[int(np.argmax(p))]), 'confidence': float(np.max(p))}
            return {'used': True, 'label': str(model.predict(fused)[0]), 'confidence': None, 'probabilities': {}}
        except Exception as exc:
            warnings.warn(f'Multimodal fusion skipped because artifact input contract did not match: {exc}')
            return None

    def _shap(self, x: pd.DataFrame, stage_result: Dict[str, Any]) -> Dict[str, Any]:
        if self.stage_explainer is None:
            return {'positive': [], 'negative': [], 'waterfall_png_b64': None}
        explainer = self.stage_explainer
        try:
            raw = explainer.shap_values(x, check_additivity=False)
            if isinstance(raw, list):
                vals = np.stack(raw, axis=-1)
            else:
                vals = np.asarray(raw)
                if vals.ndim == 3 and vals.shape[1] == len(getattr(explainer, 'classes_', [])):
                    vals = np.moveaxis(vals, 1, -1)
            if vals.ndim == 3:
                classes = list(getattr(explainer, 'classes_', []))
                target = 'Stage 4'
                ci = classes.index(target) if target in classes else int(np.argmax(np.abs(vals[0]).sum(axis=0)))
                values = vals[0, :, ci]
                base = np.asarray(explainer.expected_value).reshape(-1)
                base_value = float(base[ci] if len(base) > 1 else base[0])
            else:
                values = vals[0]
                base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
            names = list(x.columns)
            pairs = sorted(zip(names, values), key=lambda z: float(z[1]), reverse=True)
            pos = [{'feature': str(k), 'value': float(v)} for k,v in pairs if v > 0][:5]
            neg = [{'feature': str(k), 'value': float(v)} for k,v in sorted(pairs, key=lambda z: float(z[1])) if v < 0][:5]
            import shap
            exp = shap.Explanation(values=np.asarray(values), base_values=base_value, data=x.iloc[0].to_numpy(), feature_names=names)
            fig = plt.figure(figsize=(10, 6))
            shap.plots.waterfall(exp, max_display=15, show=False)
            plt.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=160, bbox_inches='tight'); plt.close(fig)
            return {'positive': pos, 'negative': neg, 'waterfall_png_b64': base64.b64encode(buf.getvalue()).decode('ascii')}
        except Exception as exc:
            warnings.warn(f'SHAP explanation unavailable: {exc}')
            return {'positive': [], 'negative': [], 'waterfall_png_b64': None}

    def predict(self, patient_data: Dict[str, Any], ct_output_dir: Optional[str] = None) -> Dict[str, Any]:
        patient = self.validate_patient_data(patient_data)
        tabular = self._tabular_matrix(patient)
        stage = self._stage_predict(tabular)
        hist = self._histology_predict(tabular)
        survival = self._survival_predict(patient)
        fusion = self._fusion_predict(tabular, ct_output_dir)
        shap_result = self._shap(tabular, stage)
        return {
            'patient': {k: (None if pd.isna(v) else v) for k,v in patient.items()},
            'stage': stage,
            'histology': hist,
            'survival': survival,
            'multimodal_fusion': fusion or {'used': False},
            'shap': shap_result,
            'model_contract': {
                'preprocessed_features': int(tabular.shape[1]),
                'stage_threshold': stage.get('threshold', 0.535),
                'survival_excludes': ['num_treatment_events'],
            },
        }
