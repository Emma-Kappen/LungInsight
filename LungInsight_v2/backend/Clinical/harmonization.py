from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

PERFORMANCE_STATUS_MAPPING_VERSION = "ecog-kps-interval-v1"
ECOG_TO_KPS_INTERVAL: dict[int, tuple[int, int]] = {
    0: (90, 100),
    1: (70, 80),
    2: (50, 60),
    3: (30, 40),
    4: (10, 20),
    5: (0, 0),
}


class ClinicalValidationError(ValueError):
    """Raised when a clinical value violates its declared input contract."""


@dataclass(frozen=True)
class PerformanceStatus:
    kps_value: int | None
    kps_category: str | None
    provenance: str
    source_kps: int | None
    source_ecog: int | None
    mapping_version: str
    discrepancy: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _integer_or_none(value: Any, name: str, minimum: int, maximum: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ClinicalValidationError(f"{name} must be an integer.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ClinicalValidationError(f"{name} must be an integer.") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ClinicalValidationError(f"{name} must be an integer.")
    result = int(number)
    if not minimum <= result <= maximum:
        raise ClinicalValidationError(f"{name} must be between {minimum} and {maximum}.")
    return result


def _kps_category(value: int) -> str:
    return str(value)


def normalize_performance_status(
    *,
    kps: Any = None,
    ecog: Any = None,
) -> PerformanceStatus:
    """Prefer measured KPS; retain ECOG-derived KPS as an interval/category only."""
    direct_kps = _integer_or_none(kps, "KPS", 0, 100)
    source_ecog = _integer_or_none(ecog, "ECOG", 0, 5)
    if direct_kps is not None:
        discrepancy = False
        if source_ecog is not None:
            lower, upper = ECOG_TO_KPS_INTERVAL[source_ecog]
            discrepancy = not lower <= direct_kps <= upper
        return PerformanceStatus(
            kps_value=direct_kps,
            kps_category=_kps_category(direct_kps),
            provenance="direct_kps",
            source_kps=direct_kps,
            source_ecog=source_ecog,
            mapping_version=PERFORMANCE_STATUS_MAPPING_VERSION,
            discrepancy=discrepancy,
        )
    if source_ecog is not None:
        lower, upper = ECOG_TO_KPS_INTERVAL[source_ecog]
        category = "0" if lower == upper == 0 else f"{lower}-{upper}"
        return PerformanceStatus(
            kps_value=None,
            kps_category=category,
            provenance="approximate_ecog_interval",
            source_kps=None,
            source_ecog=source_ecog,
            mapping_version=PERFORMANCE_STATUS_MAPPING_VERSION,
            discrepancy=False,
        )
    return PerformanceStatus(
        kps_value=None,
        kps_category=None,
        provenance="missing",
        source_kps=None,
        source_ecog=None,
        mapping_version=PERFORMANCE_STATUS_MAPPING_VERSION,
        discrepancy=False,
    )


def normalize_sex(value: Any) -> str | None:
    if value is None or not str(value).strip():
        return None
    normalized = str(value).strip().casefold()
    if normalized in {"female", "f"}:
        return "female"
    if normalized in {"male", "m"}:
        return "male"
    if normalized in {"other", "unknown", "not reported", "not reported/unknown"}:
        return "other_or_unspecified"
    raise ClinicalValidationError("Unsupported source value for sex_at_baseline.")


def validate_clinician_inputs(values: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "age_at_baseline_years",
        "age_at_baseline_years_status",
        "sex_at_baseline",
        "race_at_baseline",
        "ethnicity_at_baseline",
        "smoking_status",
        "years_smoked",
        "years_smoked_status",
        "packs_per_day",
        "packs_per_day_status",
        "pack_years",
        "years_since_cessation",
        "family_history_lung_cancer",
        "medical_history_status",
        "medical_history_conditions",
        "symptoms_status",
        "symptoms",
        "kps",
        "ecog",
    }
    extra = set(values) - allowed
    if extra:
        raise ClinicalValidationError(f"Unsupported clinician fields: {sorted(extra)}.")
    result = dict(values)
    age = result.get("age_at_baseline_years")
    if age is not None:
        if isinstance(age, bool):
            raise ClinicalValidationError("age_at_baseline_years must be numeric.")
        try:
            age = float(age)
        except (TypeError, ValueError) as exc:
            raise ClinicalValidationError("age_at_baseline_years must be numeric.") from exc
        if not math.isfinite(age) or not 18 <= age <= 120:
            raise ClinicalValidationError("age_at_baseline_years must be between 18 and 120.")
        result["age_at_baseline_years"] = age
    age_status = result.get(
        "age_at_baseline_years_status",
        "reported" if age is not None else "not_reported",
    )
    if not isinstance(age_status, str) or age_status not in {"reported", "unknown", "not_reported"}:
        raise ClinicalValidationError("age_at_baseline_years_status is invalid.")
    if (age_status == "reported") != (age is not None):
        raise ClinicalValidationError("Age value and age status must agree.")
    result["age_at_baseline_years_status"] = age_status

    sex = result.get("sex_at_baseline")
    if sex is not None and (not isinstance(sex, str) or sex not in {
        "female", "male", "other_or_unspecified", "unknown", "not_reported"
    }):
        raise ClinicalValidationError("Unsupported sex_at_baseline category.")
    if sex == "unknown":
        result["sex_at_baseline_status"] = "unknown"
        result["sex_at_baseline"] = None
    elif sex == "not_reported":
        result["sex_at_baseline_status"] = "not_reported"
        result["sex_at_baseline"] = None
    else:
        result["sex_at_baseline_status"] = "reported" if sex is not None else "not_reported"

    categories = {
        "race_at_baseline": {
            "american_indian_or_alaska_native", "asian", "black_or_african_american",
            "native_hawaiian_or_pacific_islander", "white", "multiple", "other",
            "unknown", "not_reported", None,
        },
        "ethnicity_at_baseline": {
            "hispanic_or_latino", "not_hispanic_or_latino", "other",
            "unknown", "not_reported", None,
        },
    }
    for name, supported in categories.items():
        value = result.get(name)
        if value is not None and (not isinstance(value, str) or value not in supported):
            raise ClinicalValidationError(f"Unsupported {name} category.")

    result.setdefault("smoking_status", "not_reported")
    if not isinstance(result["smoking_status"], str) or result["smoking_status"] not in {"current", "former", "never", "unknown", "not_reported"}:
        raise ClinicalValidationError("Unsupported smoking_status category.")

    def nonnegative_number(name: str, maximum: float) -> float | None:
        value = result.get(name)
        if value is None:
            return None
        if isinstance(value, bool):
            raise ClinicalValidationError(f"{name} must be numeric.")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ClinicalValidationError(f"{name} must be numeric.") from exc
        if not math.isfinite(parsed) or not 0 <= parsed <= maximum:
            raise ClinicalValidationError(f"{name} must be between 0 and {maximum}.")
        result[name] = parsed
        return parsed

    years = nonnegative_number("years_smoked", 100)
    packs = nonnegative_number("packs_per_day", 20)
    for name, value in (("years_smoked", years), ("packs_per_day", packs)):
        status_key = f"{name}_status"
        status = result.get(status_key, "reported" if value is not None else "not_reported")
        if not isinstance(status, str) or status not in {"reported", "unknown", "not_reported"}:
            raise ClinicalValidationError(f"{status_key} is invalid.")
        if (status == "reported") != (value is not None):
            raise ClinicalValidationError(f"{name} value and status must agree.")
        result[status_key] = status
    if result["smoking_status"] == "never":
        if years not in (None, 0.0) or packs not in (None, 0.0):
            raise ClinicalValidationError("Never-smokers cannot have positive smoking duration or packs per day.")
        years = packs = 0.0
        result["years_smoked"] = 0.0
        result["packs_per_day"] = 0.0
        result["years_smoked_status"] = result["packs_per_day_status"] = "reported"
    cessation = nonnegative_number("years_since_cessation", 100)
    if result["smoking_status"] != "former" and cessation is not None:
        raise ClinicalValidationError("years_since_cessation is only valid for former smokers.")
    calculated_pack_years = round(years * packs, 2) if years is not None and packs is not None else None
    supplied_pack_years = result.get("pack_years")
    if supplied_pack_years is not None:
        if isinstance(supplied_pack_years, bool):
            raise ClinicalValidationError("pack_years must be numeric.")
        try:
            supplied_pack_years = float(supplied_pack_years)
        except (TypeError, ValueError) as exc:
            raise ClinicalValidationError("pack_years must be numeric.") from exc
        if not math.isfinite(supplied_pack_years) or supplied_pack_years < 0:
            raise ClinicalValidationError("pack_years must be a finite non-negative value.")
        if calculated_pack_years is None or not math.isclose(
            supplied_pack_years, calculated_pack_years, rel_tol=0, abs_tol=0.01
        ):
            raise ClinicalValidationError("pack_years must match years_smoked multiplied by packs_per_day.")
    result["pack_years"] = calculated_pack_years

    family_history = result.get("family_history_lung_cancer")
    if family_history is not None and (
        not isinstance(family_history, str)
        or family_history not in {"yes", "no", "unknown", "not_reported"}
    ):
        raise ClinicalValidationError("family_history_lung_cancer must be yes, no, unknown, or not_reported.")
    result.setdefault("family_history_lung_cancer", "not_reported")

    supported_conditions = {"emphysema", "copd", "asthma", "pulmonary_fibrosis", "prior_cancer", "other"}
    supported_symptoms = {"cough", "dyspnea", "hemoptysis", "chest_pain", "weight_loss", "fatigue", "other"}
    for status_key, list_key, supported in (
        ("medical_history_status", "medical_history_conditions", supported_conditions),
        ("symptoms_status", "symptoms", supported_symptoms),
    ):
        status = result.get(status_key, "reported" if list_key in result else "not_reported")
        items = result.get(list_key, [])
        if not isinstance(status, str) or status not in {"reported", "unknown", "not_reported"}:
            raise ClinicalValidationError(f"{status_key} is invalid.")
        if (
            not isinstance(items, list)
            or any(not isinstance(item, str) or item not in supported for item in items)
            or len(items) != len(set(items))
        ):
            raise ClinicalValidationError(f"{list_key} contains unsupported or duplicate values.")
        if status != "reported" and items:
            raise ClinicalValidationError(f"{list_key} must be empty when its status is {status}.")
        result[status_key] = status
        result[list_key] = items

    status = normalize_performance_status(kps=result.get("kps"), ecog=result.get("ecog"))
    result["performance_status"] = status.to_dict()
    result["kps_category"] = status.kps_category
    return result