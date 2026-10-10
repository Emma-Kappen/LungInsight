#!/usr/bin/env python3
"""Shared parsing, matching, split, and ordinal-model utilities."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
XML_ROOT_DEFAULT = REPO_ROOT / "datasets" / "LIDC" / "LIDC-XML-only" / "tcia-lidc-xml"
H5_DEFAULT = REPO_ROOT / "datasets" / "LIDC" / "lidc_patches.h5"
SPLIT_DEFAULT = HERE.parent / "nodule_ranker_3d" / "checkpoints_model2_final" / "split.json"
OUTPUT_DEFAULT = HERE.parents[2] / "backend" / "output" / "malignancy" / "xml_only_run"

RATING_FEATURES = (
    "subtlety", "internalStructure", "calcification", "sphericity",
    "margin", "lobulation", "spiculation", "texture",
)
RATING_LIMITS = {
    "subtlety": 5, "internalStructure": 4, "calcification": 6,
    "sphericity": 5, "margin": 5, "lobulation": 5, "spiculation": 5,
    "texture": 5,
}
XML_TO_H5_FEATURE = {name: name for name in RATING_FEATURES}
MALIGNANCY_MIN = 1
MALIGNANCY_MAX = 5
FEATURE_COLUMNS = tuple(
    column for name in RATING_FEATURES for column in (f"{name}_mean", f"{name}_std")
) + (
    "roi_count_mean", "roi_count_std", "contour_points_mean", "contour_points_std",
    "contour_bbox_width_mean", "contour_bbox_width_std", "contour_bbox_height_mean",
    "contour_bbox_height_std", "contour_equivalent_diameter_mean",
    "contour_equivalent_diameter_std",
)
TARGET_DEFINITION = "XML reader malignancy suspicion rating, ordinal categories 1-5; not a cancer diagnosis."
DISCLAIMER = "Estimate of radiologist-assessed malignancy suspicion, not a cancer diagnosis."
AUDITED_XML_TAGS = (
    "subtlety", "internalStructure", "calcification", "sphericity", "margin",
    "lobulation", "spiculation", "texture", "malignancy", "diameter", "roi",
    "edgeMap", "xCoord", "yCoord", "imageZposition", "imageSOP_UID",
    "servicingRadiologistID", "noduleID",
)
CANONICAL_SPLIT_COUNTS = {"train": 1859, "val": 437, "test": 355}
CANONICAL_SPLIT_HASHES = {
    "train": "1814147c1f1f0ccfbda73b56c5dbb392c14412b72c6b0fe316ee7a81957ad922",
    "val": "0831f0b8e0ba4b54de794c914677e783d21197bedc5c6f93c695eb07011051f7",
    "test": "e3ebe6f752237c58240e0888492679295ab068d6929cd75be1df83b821e4b56c",
}


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def first_text(element: ET.Element | None, tag_name: str) -> str | None:
    if element is None:
        return None
    for node in element.iter():
        if local_name(node.tag) == tag_name and node.text and node.text.strip():
            return node.text.strip()
    return None


def parse_integer_rating(value: Any, name: str, minimum: int = 1,
                         maximum: int | None = None) -> int | None:
    if value is None or maximum is None:
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not number.is_integer():
        return None
    integer = int(number)
    return integer if minimum <= integer <= maximum else None


def parse_float(value: Any) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def xml_tag_counts(root: ET.Element) -> dict[str, int]:
    counts = Counter()
    audited = set(AUDITED_XML_TAGS)
    for node in root.iter():
        name = local_name(node.tag)
        if name in audited:
            counts[name] += 1
    return {name: counts[name] for name in AUDITED_XML_TAGS}


def extract_geometry(nodule: ET.Element) -> dict[str, float | int | None]:
    roi_elements = [node for node in nodule.iter() if local_name(node.tag) == "roi"]
    point_count = 0
    widths: list[float] = []
    heights: list[float] = []
    diameters: list[float] = []
    for roi in roi_elements:
        coordinates: list[tuple[float, float]] = []
        for edge in roi.iter():
            if local_name(edge.tag) != "edgeMap":
                continue
            x = parse_float(first_text(edge, "xCoord"))
            y = parse_float(first_text(edge, "yCoord"))
            if x is not None and y is not None:
                coordinates.append((x, y))
        point_count += len(coordinates)
        if coordinates:
            width = max(point[0] for point in coordinates) - min(point[0] for point in coordinates)
            height = max(point[1] for point in coordinates) - min(point[1] for point in coordinates)
            widths.append(width)
            heights.append(height)
            diameters.append(math.sqrt(max(width, 0.0) * max(height, 0.0)))

    def mean(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    return {
        "roi_count": len(roi_elements),
        "contour_points": point_count,
        "contour_bbox_width": mean(widths),
        "contour_bbox_height": mean(heights),
        "contour_equivalent_diameter": mean(diameters),
    }


def parse_xml_file(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse reader-level unblinded nodule records and report file errors."""
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        return [], {"path": str(path), "status": "malformed_xml", "error": str(exc), "tag_counts": {}}

    header = next((node for node in root if local_name(node.tag) == "ResponseHeader"), None)
    tag_counts = xml_tag_counts(root)
    patient_id = first_text(root, "patientId")
    study_uid = first_text(header, "StudyInstanceUID")
    series_uid = first_text(header, "SeriesInstanceUid")
    if not study_uid or not series_uid:
        return [], {
            "path": str(path), "status": "missing_dicom_identity", "error": None,
            "tag_counts": tag_counts,
        }

    rows: list[dict[str, Any]] = []
    for session_index, session in enumerate(
        node for node in root if local_name(node.tag) == "readingSession"
    ):
        reader_id = first_text(session, "servicingRadiologistID")
        for nodule in (node for node in session if local_name(node.tag) == "unblindedReadNodule"):
            nodule_id = first_text(nodule, "noduleID")
            characteristics = next(
                (node for node in nodule if local_name(node.tag) == "characteristics"), None
            )
            feature_values: dict[str, int | None] = {}
            raw_values: dict[str, str | None] = {}
            for name in RATING_FEATURES:
                raw = first_text(characteristics, name)
                raw_values[name] = raw
                feature_values[name] = parse_integer_rating(raw, name, maximum=RATING_LIMITS[name])
            raw_target = first_text(characteristics, "malignancy")
            row = {
                "patient_id": patient_id,
                "study_instance_uid": study_uid,
                "series_instance_uid": series_uid,
                "annotation_file": str(path),
                "session_index": session_index,
                "reader_id": reader_id,
                "xml_nodule_id": nodule_id,
                "feature_values": feature_values,
                "raw_feature_values": raw_values,
                "malignancy_rating": parse_integer_rating(
                    raw_target, "malignancy", maximum=MALIGNANCY_MAX
                ),
                "raw_malignancy_rating": raw_target,
                "geometry": extract_geometry(nodule),
                "match_status": "unmatched",
                "match_reason": "not_evaluated",
            }
            rows.append(row)
    return rows, {"path": str(path), "status": "parsed", "error": None, "tag_counts": tag_counts}


def _rating_signature(values: dict[str, Any]) -> tuple[tuple[str, int], ...]:
    return tuple(
        (name, int(values[name])) for name in RATING_FEATURES if values.get(name) is not None
    )


def validate_canonical_split(
    h5_path: Path,
    split_path: Path,
    enforce_project_canonical: bool = False,
) -> tuple[dict[str, list[int]], dict[int, str], dict[str, Any]]:
    raw = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Canonical split must be a JSON object")
    split_indices: dict[str, list[int]] = {}
    index_to_split: dict[int, str] = {}
    patient_sets: dict[str, set[str]] = {}
    with h5py.File(h5_path, "r") as h5:
        sample_count = len(h5["patient_id"])
        for name in ("train", "val", "test"):
            entry = raw.get(name)
            if not isinstance(entry, dict) or not isinstance(entry.get("sample_indices"), list):
                raise ValueError(f"Canonical split has no {name}.sample_indices list")
            indices = entry["sample_indices"]
            if not all(isinstance(value, int) for value in indices):
                raise ValueError(f"{name} sample indices must be integers")
            if len(indices) != len(set(indices)):
                raise ValueError(f"{name} contains duplicate sample indices")
            if any(index < 0 or index >= sample_count for index in indices):
                raise ValueError(f"{name} contains out-of-range sample indices")
            index_hash = hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()
            if enforce_project_canonical:
                if len(indices) != CANONICAL_SPLIT_COUNTS[name]:
                    raise ValueError(
                        f"{name} canonical sample count mismatch: expected "
                        f"{CANONICAL_SPLIT_COUNTS[name]}, got {len(indices)}"
                    )
                if index_hash != CANONICAL_SPLIT_HASHES[name]:
                    raise ValueError(f"{name} canonical split index hash mismatch: {index_hash}")
            actual_patients = {
                _decode(h5["patient_id"][index]) for index in indices
            }
            if set(entry.get("patients", [])) != actual_patients:
                raise ValueError(f"{name} patient list does not match H5 sample indices")
            split_indices[name] = indices
            patient_sets[name] = actual_patients
            for index in indices:
                if index in index_to_split:
                    raise ValueError(f"H5 sample index {index} appears in multiple splits")
                index_to_split[index] = name
        if set(index_to_split) != set(range(sample_count)):
            raise ValueError("Canonical split does not partition the H5 samples exactly")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = patient_sets[left] & patient_sets[right]
        if overlap:
            raise ValueError(f"Patient leakage between {left} and {right}: {sorted(overlap)[:5]}")
    metadata = {
        "sample_count": sample_count,
        "split_sample_counts": {name: len(values) for name, values in split_indices.items()},
        "split_patient_counts": {name: len(values) for name, values in patient_sets.items()},
        "project_canonical_hashes_enforced": enforce_project_canonical,
        "patient_disjoint": True,
    }
    return split_indices, index_to_split, metadata


def _decode(value: Any) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def match_reader_records(rows: list[dict[str, Any]], h5_path: Path) -> dict[str, Any]:
    """Match only unique exact non-malignancy signatures inside exact DICOM series."""
    by_series_signature: dict[
        tuple[str, str, tuple[tuple[str, int], ...]],
        list[tuple[int, int, dict[str, Any], int | None]],
    ] = defaultdict(list)
    h5_sample_metadata: dict[int, dict[str, str]] = {}
    with h5py.File(h5_path, "r") as h5:
        label_names = json.loads(h5.attrs["reader_feature_names"])
        expected_non_malignancy = set(RATING_FEATURES)
        actual_non_malignancy = set(label_names) - {"malignancy"}
        if not expected_non_malignancy.issubset(actual_non_malignancy):
            raise ValueError("H5 reader label schema lacks required non-malignancy ratings")
        for sample_index in range(len(h5["patient_id"])):
            meta = {
                "patient_id": _decode(h5["patient_id"][sample_index]),
                "study_instance_uid": _decode(h5["study_instance_uid"][sample_index]),
                "series_instance_uid": _decode(h5["series_instance_uid"][sample_index]),
                "nodule_id": _decode(h5["nodule_id"][sample_index]),
            }
            h5_sample_metadata[sample_index] = meta
            for reader_slot in range(h5["reader_mask"].shape[1]):
                if not bool(h5["reader_mask"][sample_index, reader_slot]):
                    continue
                labels = h5["reader_labels"][sample_index, reader_slot]
                values = {
                    name: parse_integer_rating(
                        labels[label_names.index(name)], name, maximum=RATING_LIMITS[name]
                    ) for name in RATING_FEATURES
                }
                signature = _rating_signature(values)
                if len(signature) == len(RATING_FEATURES):
                    key = (meta["study_instance_uid"], meta["series_instance_uid"], signature)
                    target_value = parse_integer_rating(
                        labels[label_names.index("malignancy")],
                        "malignancy", maximum=MALIGNANCY_MAX,
                    )
                    by_series_signature[key].append((sample_index, reader_slot, values, target_value))

    xml_counts: Counter[tuple[str, str, tuple[tuple[str, int], ...]]] = Counter()
    row_keys: list[tuple[str, str, tuple[tuple[str, int], ...]] | None] = []
    for row in rows:
        signature = _rating_signature(row["feature_values"])
        if len(signature) != len(RATING_FEATURES):
            row["match_reason"] = "incomplete_non_malignancy_signature"
            row_keys.append(None)
            continue
        key = (row["study_instance_uid"], row["series_instance_uid"], signature)
        xml_counts[key] += 1
        row_keys.append(key)

    reverse_counts: Counter[tuple[int, int]] = Counter()
    candidates: list[list[tuple[int, int, dict[str, Any], int | None]]] = []
    for key in row_keys:
        current = by_series_signature.get(key, []) if key is not None else []
        candidates.append(current)
        if len(current) == 1 and key is not None and xml_counts[key] == 1:
            reverse_counts[(current[0][0], current[0][1])] += 1

    matched_count = 0
    for row, key, current in zip(rows, row_keys, candidates):
        if key is None:
            continue
        if len(current) == 0:
            row["match_reason"] = "no_h5_signature_match"
            continue
        if len(current) != 1 or xml_counts[key] != 1:
            row["match_reason"] = "ambiguous_signature"
            continue
        sample_index, reader_slot, h5_features, h5_target = current[0]
        if reverse_counts[(sample_index, reader_slot)] != 1:
            row["match_reason"] = "duplicate_xml_to_h5_slot"
            continue
        row["h5_sample_index"] = sample_index
        row["h5_reader_slot"] = reader_slot
        row["h5_nodule_id"] = h5_sample_metadata[sample_index]["nodule_id"]
        row["patient_id"] = h5_sample_metadata[sample_index]["patient_id"]
        row["match_status"] = "matched"
        row["match_reason"] = "unique_exact_non_malignancy_signature"
        row["target_h5_rating"] = h5_target
        row["target_crosscheck"] = (
            row["malignancy_rating"] is None
            or row["target_h5_rating"] is None
            or row["malignancy_rating"] == row["target_h5_rating"]
        )
        row["feature_values"] = h5_features
        matched_count += 1

    return {
        "matched_record_count": matched_count,
        "unmatched_reason_counts": dict(Counter(row["match_reason"] for row in rows if row["match_status"] != "matched")),
        "target_crosscheck_mismatch_count": sum(
            1 for row in rows if row["match_status"] == "matched" and not row["target_crosscheck"]
        ),
        "sample_metadata": h5_sample_metadata,
    }


def _aggregate(values: list[float | int | None]) -> tuple[float | None, float | None]:
    array = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=float)
    if not len(array):
        return None, None
    return float(array.mean()), float(array.std(ddof=0))


def aggregate_nodules(rows: list[dict[str, Any]], index_to_split: dict[int, str],
                      sample_metadata: dict[int, dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["match_status"] == "matched":
            grouped[row["h5_sample_index"]].append(row)
    output: list[dict[str, Any]] = []
    for sample_index, group in sorted(grouped.items()):
        features: dict[str, Any] = {}
        for name in RATING_FEATURES:
            mean, std = _aggregate([row["feature_values"].get(name) for row in group])
            features[f"{name}_mean"] = mean
            features[f"{name}_std"] = std
        for source, columns in (
            ("roi_count", ("roi_count_mean", "roi_count_std")),
            ("contour_points", ("contour_points_mean", "contour_points_std")),
            ("contour_bbox_width", ("contour_bbox_width_mean", "contour_bbox_width_std")),
            ("contour_bbox_height", ("contour_bbox_height_mean", "contour_bbox_height_std")),
            ("contour_equivalent_diameter", ("contour_equivalent_diameter_mean", "contour_equivalent_diameter_std")),
        ):
            mean, std = _aggregate([row["geometry"].get(source) for row in group])
            features[columns[0]], features[columns[1]] = mean, std
        target_counts = Counter(
            row["malignancy_rating"] for row in group if row["malignancy_rating"] is not None
        )
        target_ratings = sorted(
            rating for row in group
            if (rating := row["malignancy_rating"]) is not None
        )
        metadata = sample_metadata[sample_index]
        output.append({
            "stable_nodule_id": f"{metadata['study_instance_uid']}::{metadata['series_instance_uid']}::{metadata['nodule_id']}",
            "patient_id": metadata["patient_id"],
            "canonical_sample_index": sample_index,
            "canonical_nodule_id": metadata["nodule_id"],
            "split": index_to_split[sample_index],
            **features,
            "matched_reader_count": len(group),
            "target_reader_count": len(target_ratings),
            "target_rating_counts_json": json.dumps({str(key): target_counts.get(key, 0) for key in range(1, 6)}, sort_keys=True),
            "target_ratings_json": json.dumps(target_ratings),
            "target_mean": float(np.mean(target_ratings)) if target_ratings else None,
            "target_median": float(np.median(target_ratings)) if target_ratings else None,
        })
    return output


def feature_schema(rows: list[dict[str, Any]], all_reader_rows: list[dict[str, Any]],
                   parse_issues: list[dict[str, Any]],
                   source_tag_counts: dict[str, int] | None = None) -> dict[str, Any]:
    schema_features: dict[str, Any] = {}
    for name in RATING_FEATURES:
        for suffix in ("mean", "std"):
            column = f"{name}_{suffix}"
            values = [row.get(column) for row in rows if row.get(column) is not None]
            schema_features[column] = {
                "type": "numeric", "source": f"XML characteristics/{name}",
                "rating_range": [1, RATING_LIMITS[name]] if suffix == "mean" else [0, RATING_LIMITS[name] - 1],
                "missing_count": len(rows) - len(values),
                "missing_fraction": (len(rows) - len(values)) / len(rows) if rows else None,
                "observed_range": [min(values), max(values)] if values else None,
            }
    for column in FEATURE_COLUMNS[len(RATING_FEATURES) * 2:]:
        values = [row.get(column) for row in rows if row.get(column) is not None]
        schema_features[column] = {
            "type": "numeric", "source": "XML ROI/edgeMap contour geometry; pixel units, no scan coordinates",
            "missing_count": len(rows) - len(values),
            "missing_fraction": (len(rows) - len(values)) / len(rows) if rows else None,
            "observed_range": [min(values), max(values)] if values else None,
        }
    field_audit: dict[str, Any] = {}
    for name in (*RATING_FEATURES, "malignancy"):
        raw_values = [
            row["raw_feature_values"].get(name) if name in RATING_FEATURES else row.get("raw_malignancy_rating")
            for row in all_reader_rows
        ]
        maximum = RATING_LIMITS.get(name, MALIGNANCY_MAX)
        valid = [
            parsed for raw in raw_values
            if (parsed := parse_integer_rating(raw, name, maximum=maximum)) is not None
        ]
        field_audit[name] = {
            "present_count": sum(value is not None for value in raw_values),
            "missing_or_invalid_count": sum(value is None for value in raw_values),
            "valid_range": [min(valid), max(valid)] if valid else None,
            "source_range": [1, maximum],
            "predictor": name != "malignancy",
        }
    return {
        "schema_version": "1.0",
        "target_definition": TARGET_DEFINITION,
        "target_rating_scale": [1, 2, 3, 4, 5],
        "target_is_predictor": False,
        "excluded_predictive_fields": [
            "malignancy", "target_h5_rating", "target_rating_counts_json", "target_ratings_json",
            "target_mean", "target_median", "patient_id", "study_instance_uid", "series_instance_uid",
            "annotation_file", "reader_id", "xml_nodule_id", "canonical_sample_index",
            "canonical_nodule_id", "stable_nodule_id", "split",
        ],
        "feature_order": list(FEATURE_COLUMNS),
        "features": schema_features,
        "source_xml_field_audit": field_audit,
        "source_xml_tag_audit": {
            name: {
                "occurrence_count": int((source_tag_counts or {}).get(name, 0)),
                "present_in_corpus": int((source_tag_counts or {}).get(name, 0)) > 0,
                "predictor": name in RATING_FEATURES or name in {"roi", "edgeMap", "xCoord", "yCoord"},
            }
            for name in AUDITED_XML_TAGS
        },
        "observed_xml_annotation_records": len(all_reader_rows),
        "malformed_or_unidentified_xml_files": len(parse_issues),
        "matching_policy": "Unique exact signature over all 8 valid non-malignancy ratings within the exact StudyInstanceUID/SeriesInstanceUID; malignancy is never used for matching.",
        "geometry_units": "Contour extents are in XML image pixel coordinates; no physical millimeter scale is available in these XML records.",
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    if columns is None:
        columns = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_xml_tree(xml_root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(xml_root)).replace("\\", "/"): sha256_file(path)
        for path in sorted(xml_root.rglob("*.xml"))
    }


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def rows_to_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([
        [parse_float(row.get(column)) if row.get(column) not in (None, "") else np.nan for column in FEATURE_COLUMNS]
        for row in rows
    ], dtype=np.float64)


class CumulativeOrdinalModel:
    """Four cumulative logistic classifiers with monotone class probabilities."""

    def __init__(self, model_kind: str = "ordinal", seed: int = 42):
        self.model_kind = model_kind
        self.seed = seed
        self.estimators: list[Any] = []

    def fit(self, features: np.ndarray, ratings: np.ndarray) -> "CumulativeOrdinalModel":
        self.estimators = []
        for threshold in range(1, MALIGNANCY_MAX):
            binary = (ratings > threshold).astype(np.int8)
            if len(np.unique(binary)) != 2:
                raise ValueError(f"Training labels do not straddle ordinal threshold {threshold}")
            if self.model_kind == "ordinal":
                estimator = make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                    StandardScaler(),
                    LogisticRegression(C=1.0, max_iter=1000, random_state=self.seed),
                )
            elif self.model_kind == "gradient_boosted":
                estimator = make_pipeline(
                    SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                    HistGradientBoostingClassifier(
                        max_iter=80, max_leaf_nodes=7, learning_rate=0.08,
                        l2_regularization=1.0, random_state=self.seed,
                    ),
                )
            else:
                raise ValueError(f"Unknown model_kind: {self.model_kind}")
            estimator.fit(features, binary)
            self.estimators.append(estimator)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if len(self.estimators) != MALIGNANCY_MAX - 1:
            raise RuntimeError("Ordinal model has not been fitted")
        cumulative = np.column_stack([
            estimator.predict_proba(features)[:, list(estimator.classes_).index(1)]
            for estimator in self.estimators
        ])
        cumulative = np.minimum.accumulate(np.clip(cumulative, 0.0, 1.0), axis=1)
        probabilities = np.column_stack((1.0 - cumulative[:, 0], cumulative[:, :-1] - cumulative[:, 1:], cumulative[:, -1]))
        probabilities = np.clip(probabilities, 0.0, 1.0)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, features: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(features)
        return probabilities @ np.arange(1, MALIGNANCY_MAX + 1, dtype=float)


def make_evaluation_rows(nodule_rows: list[dict[str, Any]], split_name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    selected = [row for row in nodule_rows if row.get("split") == split_name]
    feature_rows: list[dict[str, Any]] = []
    truth: list[int] = []
    nodule_ids: list[str] = []
    for row in selected:
        ratings = json.loads(row["target_ratings_json"])
        for rating in ratings:
            feature_rows.append(row)
            truth.append(int(rating))
            nodule_ids.append(row["stable_nodule_id"])
    return rows_to_matrix(feature_rows), np.asarray(truth, dtype=np.int64), nodule_ids


def evaluate_predictions(truth: np.ndarray, probabilities: np.ndarray,
                         baseline_ratings: np.ndarray | None = None) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=np.int64)
    predicted_scores = probabilities @ np.arange(1, MALIGNANCY_MAX + 1, dtype=float)
    predicted = np.argmax(probabilities, axis=1) + 1
    qwk = cohen_kappa_score(truth, predicted, labels=list(range(1, MALIGNANCY_MAX + 1)), weights="quadratic")
    rho = spearmanr(truth, predicted_scores).statistic if len(np.unique(truth)) > 1 and len(np.unique(predicted_scores)) > 1 else float("nan")
    result = {
        "sample_count": int(len(truth)),
        "missing_target_count": 0,
        "protocol": "Each valid XML reader malignancy rating is scored separately; the nodule prediction is repeated for its readers, preserving disagreement without treating a consensus as ground truth.",
        "qwk": float(qwk) if np.isfinite(qwk) else None,
        "spearman": float(rho) if np.isfinite(rho) else None,
        "mae": float(np.mean(np.abs(predicted_scores - truth))) if len(truth) else None,
        "rmse": float(np.sqrt(np.mean((predicted_scores - truth) ** 2))) if len(truth) else None,
        "exact_rating_accuracy": float(np.mean(predicted == truth)) if len(truth) else None,
        "confusion_matrix": confusion_matrix(truth, predicted, labels=list(range(1, MALIGNANCY_MAX + 1))).tolist(),
        "true_rating_distribution": {str(rating): int(np.sum(truth == rating)) for rating in range(1, MALIGNANCY_MAX + 1)},
        "predicted_rating_distribution": {str(rating): int(np.sum(predicted == rating)) for rating in range(1, MALIGNANCY_MAX + 1)},
        "target_rating_distribution": {str(rating): float(np.mean(truth == rating)) for rating in range(1, MALIGNANCY_MAX + 1)} if len(truth) else {},
        "predicted_rating_distribution_fraction": {str(rating): float(np.mean(predicted == rating)) for rating in range(1, MALIGNANCY_MAX + 1)} if len(truth) else {},
    }
    if baseline_ratings is not None and len(truth):
        baseline = np.asarray(baseline_ratings, dtype=np.int64)
        baseline_qwk = cohen_kappa_score(truth, baseline, labels=list(range(1, MALIGNANCY_MAX + 1)), weights="quadratic")
        result["baseline"] = {
            "qwk": float(baseline_qwk) if np.isfinite(baseline_qwk) else None,
            "mae": float(np.mean(np.abs(baseline - truth))),
            "rmse": float(np.sqrt(np.mean((baseline - truth) ** 2))),
            "exact_rating_accuracy": float(np.mean(baseline == truth)),
            "predicted_rating_distribution": {str(rating): int(np.sum(baseline == rating)) for rating in range(1, MALIGNANCY_MAX + 1)},
            "protocol": "Training-only global reader-rating majority; median baseline rounds the training-only median to the nearest ordinal category.",
        }
    return result


def majority_and_median(training_ratings: np.ndarray) -> tuple[int, int]:
    counts = Counter(int(value) for value in training_ratings)
    majority = min(counts, key=lambda value: (-counts[value], value))
    median = int(np.clip(np.floor(float(np.median(training_ratings)) + 0.5), 1, MALIGNANCY_MAX))
    return majority, median


def elapsed_seconds(start: float) -> float:
    return round(time.perf_counter() - start, 3)