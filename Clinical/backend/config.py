"""Configuration and feature contracts for the multimodal inference service."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.getenv("LUNGINSIGHT_MODEL_DIR", ROOT / "Clinical" / "models"))
UPLOAD_DIR = Path(os.getenv("LUNGINSIGHT_UPLOAD_DIR", ROOT / "Clinical" / "uploads"))
MAX_UPLOAD_MB = int(os.getenv("LUNGINSIGHT_MAX_UPLOAD_MB", "512"))

NUMERIC_RANGES = {
    "AGE": (18.0, 100.0),
    "num_treatment_events": (0.0, 20.0),
    "num_distinct_tumor_sites": (1.0, 15.0),
    "eastern_cancer_oncology_group": (0.0, 4.0),
    "karnofsky_performance_score": (0.0, 100.0),
}
CATEGORIES = {
    "SEX": ["Female", "Male"],
    "RACE": ["White", "Black or African American", "Asian", "Other"],
    "ETHNICITY": ["Non-Spanish; Non-Hispanic", "Spanish; Hispanic", "Unknown"],
    "SMOKING_STATUS": ["Former/Current Smoker", "Never"],
    "ever_pdl1_positive": ["Yes", "No", "Unknown / Missing"],
}
MISSINGNESS = [
    "missing_ever_pdl1_positive",
    "missing_eastern_cancer_oncology_group",
    "missing_karnofsky_performance_score",
]
DEFAULTS = {
    "AGE": 65.0, "num_treatment_events": 2.0, "num_distinct_tumor_sites": 1.0,
    "eastern_cancer_oncology_group": 1.0, "karnofsky_performance_score": 80.0,
    "SEX": "Male", "RACE": "White", "ETHNICITY": "Non-Spanish; Non-Hispanic",
    "SMOKING_STATUS": "Former/Current Smoker", "ever_pdl1_positive": "Unknown / Missing",
}
