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
        "sex_at_baseline",
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

    sex = result.get("sex_at_baseline")
    if sex is not None and sex not in {"female", "male", "other_or_unspecified"}:
        raise ClinicalValidationError("Unsupported sex_at_baseline category.")
    status = normalize_performance_status(kps=result.get("kps"), ecog=result.get("ecog"))
    result["performance_status"] = status.to_dict()
    result["kps_category"] = status.kps_category
    return result