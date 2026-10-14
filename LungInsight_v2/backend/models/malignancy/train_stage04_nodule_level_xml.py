#!/usr/bin/env python3
"""Train Stage 04 malignancy rating from one XML-derived example per physical nodule.

Stage 03 remains frozen. Features are nodule-level medians of six XML reader
attribute ratings; target is median reader malignancy rating (1..5). XML reader
annotations are spatially matched across readers using ROI contour centroids and
DICOM headers only. Pixel data, H5 patches, and Stage 03 checkpoints are never read.

The fixed patient split is used unchanged. Only train/validation patient
annotations are parsed. The optional Stage 03 evaluation reads ONLY val.npz and
compares the trained model with the nodule-level median malignancy target.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import warnings
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error, mean_squared_error

FEATURES = ["subtlety", "sphericity", "margin", "lobulation", "spiculation", "texture"]
RATING_FIELDS = FEATURES + ["malignancy"]
NS = "{http://www.nih.gov}"
MODEL_PARAMS = {
    "n_estimators": 500,
    "min_samples_leaf": 8,
    "max_features": 1.0,
    "max_depth": None,
    "random_state": 2026,
    "n_jobs": -1,
}


def local(tag: str) -> str:
    return tag[len(NS):] if tag.startswith(NS) else tag


def numeric_rating(value: Any) -> float:
    if value is None or isinstance(value, bool) or str(value).strip() == "":
        return float("nan")
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return x if math.isfinite(x) and 1 <= x <= 5 else float("nan")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def patient_ids_for_split(split_obj: Any, name: str) -> set[str]:
    if not isinstance(split_obj, dict) or name not in split_obj:
        raise ValueError(f"split.json must contain '{name}'")
    item = split_obj[name]
    patients = item.get("patients") if isinstance(item, dict) else item
    if not isinstance(patients, list):
        raise ValueError(f"split.json['{name}'] must be a list or contain a patients list")
    result = {str(v).strip() for v in patients if str(v).strip()}
    if not result:
        raise ValueError(f"No patient IDs found for split '{name}'")
    return result


def parse_xml_header(path: Path) -> tuple[str | None, str | None]:
    """Read only the early ResponseHeader; do not parse reader annotation bodies."""
    try:
        for event, elem in ET.iterparse(path, events=("end",)):
            if local(elem.tag) == "ResponseHeader":
                study = series = None
                for el in elem:
                    if local(el.tag) == "StudyInstanceUID":
                        study = (el.text or "").strip()
                    elif local(el.tag) == "SeriesInstanceUid":
                        series = (el.text or "").strip()
                return study or None, series or None
    except Exception:
        return None, None
    return None, None


def load_metadata(metadata_csv: Path, allowed_patients: set[str]):
    patients: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with metadata_csv.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            patient = str(row.get("PatientID", "")).strip()
            study = str(row.get("StudyInstanceUID", "")).strip()
            series = str(row.get("SeriesInstanceUID", "")).strip()
            if patient in allowed_patients and study and series:
                item = (study, series)
                if item not in patients[patient]:
                    patients[patient].append(item)
    absent = allowed_patients - set(patients)
    if absent:
        warnings.warn(f"{len(absent)} train/validation patients have no metadata row; example={sorted(absent)[:3]}")
    return dict(patients)


def build_xml_index(xml_root: Path, wanted_studies: set[str], wanted_series: set[str]):
    """Read XML response headers to index files; parse annotation bodies only for train/val."""
    by_study: dict[str, Path] = {}
    by_series: dict[str, Path] = {}
    scanned = 0
    for path in xml_root.rglob("*.xml"):
        scanned += 1
        study, series = parse_xml_header(path)
        if study in wanted_studies:
            by_study.setdefault(study, path)
        if series in wanted_series:
            by_series.setdefault(series, path)
    return by_study, by_series, scanned


def read_dicom_header(path: Path):
    """Return geometry metadata only. stop_before_pixels=True prevents pixel reads."""
    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError("pydicom is required for XML spatial grouping. Activate l-env and run: pip install pydicom") from exc
    try:
        ds = pydicom.dcmread(
            str(path), stop_before_pixels=True, force=True,
            specific_tags=[
                "SOPInstanceUID", "StudyInstanceUID", "SeriesInstanceUID", "Modality",
                "Rows", "Columns", "ImagePositionPatient", "ImageOrientationPatient",
                "PixelSpacing", "InstanceNumber", "SliceThickness", "SpacingBetweenSlices",
            ],
        )
        if getattr(ds, "Modality", "") != "CT":
            return None
        required = ("SOPInstanceUID", "SeriesInstanceUID", "ImagePositionPatient", "ImageOrientationPatient", "PixelSpacing")
        if any(not getattr(ds, key, None) for key in required):
            return None
        return ds
    except Exception:
        return None


def build_dicom_index(dicom_root: Path, allowed_patients: set[str], wanted_series: set[str], wanted_studies: set[str]):
    """Index train/validation CT headers only; never read PixelData or enter test-patient folders."""
    index: dict[str, list[tuple[Path, Any]]] = defaultdict(list)
    scanned = accepted = 0
    if not dicom_root.is_dir():
        raise FileNotFoundError(f"DICOM root not found: {dicom_root}")
    wanted_patient_prefix = "LIDC-IDRI-"
    wanted_uids = wanted_studies | wanted_series
    def uid_looking(name: str) -> bool:
        return name.startswith("1.") and name.count(".") >= 4
    for current, dirnames, filenames in os.walk(dicom_root):
        current_path = Path(current)
        # Prune held-out patient folders and non-target Study/Series UID folders.
        dirnames[:] = [
            d for d in dirnames
            if not (d.startswith(wanted_patient_prefix) and d not in allowed_patients)
            and (not uid_looking(d) or d in wanted_uids)
        ]
        parts = set(current_path.parts)
        patient_context = bool(parts & allowed_patients)
        if not patient_context:
            # Continue through outer wrapper directories but do not read files yet.
            continue
        for filename in filenames:
            path = current_path / filename
            scanned += 1
            ds = read_dicom_header(path)
            if ds is None:
                continue
            series = str(getattr(ds, "SeriesInstanceUID", ""))
            if series not in wanted_series:
                continue
            index[series].append((path, ds))
            accepted += 1
        if scanned and scanned % 25000 == 0:
            print(f"  scanned {scanned:,} files under train/validation patient folders; indexed {accepted:,} CT headers")
    missing = wanted_series - set(index)
    if missing:
        warnings.warn(f"No CT header index for {len(missing)} train/validation series; example={sorted(missing)[:3]}")
    return dict(index), {"dicom_files_examined_under_train_val": scanned, "ct_headers_indexed_for_requested_series": accepted, "series_indexed": len(index), "series_without_headers": len(missing)}


def make_geometry(headers: list[tuple[Path, Any]]):
    first = headers[0][1]
    iop = np.asarray(first.ImageOrientationPatient, dtype=np.float64)
    row_dir = iop[:3]
    col_dir = iop[3:]
    normal = np.cross(row_dir, col_dir)
    normal /= np.linalg.norm(normal)

    def projected_position(item):
        return float(np.dot(np.asarray(item[1].ImagePositionPatient, dtype=np.float64), normal))

    headers = sorted(headers, key=projected_position)
    positions = np.asarray([projected_position(item) for item in headers], dtype=np.float64)
    ps = np.asarray(first.PixelSpacing, dtype=np.float64)
    sop_to = {
        str(ds.SOPInstanceUID): {"index": i, "ipp": np.asarray(ds.ImagePositionPatient, dtype=np.float64)}
        for i, (_, ds) in enumerate(headers)
    }
    return {
        "headers": headers, "row_dir": row_dir, "col_dir": col_dir,
        "row_spacing": float(ps[0]), "col_spacing": float(ps[1]),
        "positions": positions, "sop_to": sop_to,
    }


def parse_xml_sessions(path: Path):
    root = ET.parse(path).getroot()
    sessions = []
    for session_el in root:
        if local(session_el.tag) != "readingSession":
            continue
        reader_id = None
        nodules = []
        for child in session_el:
            tag = local(child.tag)
            if tag == "servicingRadiologistID":
                reader_id = (child.text or "").strip()
            elif tag == "unblindedReadNodule":
                nodule_id = None
                ratings: dict[str, Any] = {}
                rois = []
                for nchild in child:
                    ntag = local(nchild.tag)
                    if ntag == "noduleID":
                        nodule_id = (nchild.text or "").strip()
                    elif ntag == "characteristics":
                        for cchild in nchild:
                            field = local(cchild.tag)
                            if field == "internalStructure":
                                continue
                            if field in RATING_FIELDS:
                                ratings[field] = (cchild.text or "").strip()
                    elif ntag == "roi":
                        sop = None
                        zpos = None
                        pts = []
                        for rchild in nchild:
                            rtag = local(rchild.tag)
                            if rtag == "imageSOP_UID":
                                sop = (rchild.text or "").strip()
                            elif rtag == "imageZposition":
                                try:
                                    zpos = float(rchild.text)
                                except (TypeError, ValueError):
                                    zpos = None
                            elif rtag == "edgeMap":
                                x = y = None
                                for echild in rchild:
                                    etag = local(echild.tag)
                                    if etag == "xCoord":
                                        try:
                                            x = int(float(echild.text))
                                        except (TypeError, ValueError):
                                            x = None
                                    elif etag == "yCoord":
                                        try:
                                            y = int(float(echild.text))
                                        except (TypeError, ValueError):
                                            y = None
                                if x is not None and y is not None:
                                    pts.append((x, y))
                        if pts:
                            rois.append({"sop": sop, "z": zpos, "points": pts})
                if nodule_id:
                    nodules.append({"nodule_id": nodule_id, "reader_id": reader_id, "ratings": ratings, "rois": rois})
        sessions.append({"reader_id": reader_id, "nodules": nodules})
    return sessions


def roi_center_patient(roi: dict[str, Any], geom: dict[str, Any]):
    pts = roi.get("points") or []
    if not pts:
        return None
    x = float(np.mean([p[0] for p in pts])) - 1.0
    y = float(np.mean([p[1] for p in pts])) - 1.0
    sop = roi.get("sop")
    info = geom["sop_to"].get(sop) if sop else None
    if info is not None:
        ipp = info["ipp"]
        return ipp + geom["row_dir"] * x * geom["col_spacing"] + geom["col_dir"] * y * geom["row_spacing"]
    zpos = roi.get("z")
    if zpos is None or not len(geom["positions"]):
        return None
    idx = int(np.argmin(np.abs(geom["positions"] - float(zpos))))
    ipp = np.asarray(geom["headers"][idx][1].ImagePositionPatient, dtype=np.float64)
    return ipp + geom["row_dir"] * x * geom["col_spacing"] + geom["col_dir"] * y * geom["row_spacing"]


def annotation_center(nodule: dict[str, Any], geom: dict[str, Any]):
    centers = [roi_center_patient(roi, geom) for roi in nodule.get("rois", [])]
    centers = [c for c in centers if c is not None and np.all(np.isfinite(c))]
    if not centers:
        return None
    return np.mean(np.asarray(centers), axis=0)


def collect_reader_annotations(metadata_rows, xml_by_study, xml_by_series, dicom_index, allowed_patients, stats: Counter):
    annotations = []
    for patient_id in sorted(metadata_rows):
        for study_uid, series_uid in metadata_rows[patient_id]:
            xml_path = xml_by_study.get(study_uid) or xml_by_series.get(series_uid)
            if xml_path is None:
                stats["train_val_metadata_without_xml"] += 1
                continue
            headers = dicom_index.get(series_uid)
            if not headers:
                stats["train_val_series_without_dicom_headers"] += 1
                continue
            try:
                geometry = make_geometry(headers)
                sessions = parse_xml_sessions(xml_path)
            except Exception as exc:
                stats["xml_geometry_parse_failures"] += 1
                warnings.warn(f"Skipping {patient_id}/{series_uid}: {exc}")
                continue
            for session in sessions:
                reader_id = str(session.get("reader_id") or "unknown")
                for nodule in session.get("nodules", []):
                    stats["xml_reader_nodules_seen"] += 1
                    center = annotation_center(nodule, geometry)
                    if center is None:
                        stats["reader_nodules_without_roi_centroid"] += 1
                        continue
                    annotations.append({
                        "patient_id": patient_id,
                        "study_uid": study_uid,
                        "series_uid": series_uid,
                        "reader_id": reader_id,
                        "nodule_id": str(nodule.get("nodule_id") or ""),
                        "ratings": {key: numeric_rating(nodule.get("ratings", {}).get(key)) for key in RATING_FIELDS},
                        "center_patient": center,
                    })
                    stats["reader_nodules_with_roi_centroid"] += 1
    return annotations


def spatial_groups(annotations: list[dict[str, Any]], radius_mm: float):
    """Deterministic greedy cross-reader centroid grouping; reader-local IDs are never grouping keys."""
    by_series: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for a in annotations:
        by_series[(a["patient_id"], a["study_uid"], a["series_uid"])].append(a)
    groups = []
    for series_key, values in sorted(by_series.items()):
        values = sorted(values, key=lambda a: (
            float(a["center_patient"][2]), float(a["center_patient"][1]), float(a["center_patient"][0]),
            str(a["reader_id"]), str(a["nodule_id"])))
        local_groups = []
        for ann in values:
            best_i = None
            best_d = float("inf")
            for i, group in enumerate(local_groups):
                if ann["reader_id"] in group["reader_ids"]:
                    continue
                d = float(np.linalg.norm(ann["center_patient"] - group["center_patient"]))
                if d <= radius_mm and d < best_d:
                    best_i, best_d = i, d
            if best_i is None:
                local_groups.append({
                    "patient_id": ann["patient_id"], "study_uid": ann["study_uid"], "series_uid": ann["series_uid"],
                    "center_patient": ann["center_patient"].copy(), "records": [ann], "reader_ids": {ann["reader_id"]},
                })
            else:
                group = local_groups[best_i]
                group["records"].append(ann)
                group["reader_ids"].add(ann["reader_id"])
                group["center_patient"] = np.mean([r["center_patient"] for r in group["records"]], axis=0)
        groups.extend(local_groups)
    return groups


def aggregate_nodule_groups(groups: list[dict[str, Any]], split_by_patient: dict[str, str], stats: Counter):
    rows = []
    for i, group in enumerate(groups):
        split_name = split_by_patient[group["patient_id"]]
        records = group["records"]
        medians = {}
        counts = {}
        for key in RATING_FIELDS:
            values = [r["ratings"].get(key, float("nan")) for r in records]
            values = [v for v in values if math.isfinite(v) and 1 <= v <= 5]
            counts[key] = len(values)
            medians[key] = float(np.median(values)) if values else float("nan")
        if not math.isfinite(medians["malignancy"]):
            stats[f"{split_name}_dropped_no_valid_malignancy"] += 1
            continue
        missing_features = [key for key in FEATURES if not math.isfinite(medians[key])]
        if missing_features:
            stats[f"{split_name}_dropped_missing_six_feature_medians"] += 1
            continue
        rows.append({
            "split": split_name,
            "patient_id": group["patient_id"],
            "study_uid": group["study_uid"],
            "series_uid": group["series_uid"],
            "nodule_group_index": int(i),
            "center_patient_x": float(group["center_patient"][0]),
            "center_patient_y": float(group["center_patient"][1]),
            "center_patient_z": float(group["center_patient"][2]),
            "reader_count": len({str(r["reader_id"]) for r in records}),
            "reader_annotation_count": len(records),
            "feature_rating_counts": {key: counts[key] for key in FEATURES},
            "target_reader_count": counts["malignancy"],
            "X": [medians[key] for key in FEATURES],
            "y": medians["malignancy"],
        })
        stats[f"{split_name}_usable_physical_nodules"] += 1
    return rows


def rounded_half_up(values):
    return np.clip(np.floor(np.asarray(values, dtype=float) + 0.5), 1, 5).astype(int)


def score(y, pred, baseline):
    y = np.asarray(y, dtype=float)
    pred = np.clip(np.asarray(pred, dtype=float), 1.0, 5.0)
    y_int = rounded_half_up(y)
    p_int = rounded_half_up(pred)
    bpred = np.full(len(y), float(baseline), dtype=float)
    b_int = rounded_half_up(bpred)
    qwk = cohen_kappa_score(y_int, p_int, weights="quadratic")
    bqwk = cohen_kappa_score(y_int, b_int, weights="quadratic")
    mae = float(mean_absolute_error(y, pred))
    bmae = float(mean_absolute_error(y, bpred))
    return {
        "n_nodules": int(len(y)),
        "target_mean": float(np.mean(y)), "target_std": float(np.std(y)),
        "prediction_mean": float(np.mean(pred)), "prediction_std": float(np.std(pred)),
        "mae_continuous": mae, "rmse_continuous": float(math.sqrt(mean_squared_error(y, pred))),
        "constant_train_median": float(baseline), "constant_baseline_mae": bmae,
        "mae_gain_over_baseline": bmae - mae,
        "accuracy_rounded_half_up": float(accuracy_score(y_int, p_int)),
        "qwk_rounded_half_up": float(qwk) if math.isfinite(float(qwk)) else None,
        "baseline_accuracy_rounded_half_up": float(accuracy_score(y_int, b_int)),
        "baseline_qwk_rounded_half_up": float(bqwk) if math.isfinite(float(bqwk)) else None,
        "target_class_counts": {str(c): int((y_int == c).sum()) for c in range(1, 6)},
        "prediction_class_counts": {str(c): int((p_int == c).sum()) for c in range(1, 6)},
    }


def save_table(path: Path, rows: list[dict[str, Any]]):
    fields = ["split", "patient_id", "study_uid", "series_uid", "nodule_group_index", "center_patient_x", "center_patient_y", "center_patient_z", "reader_count", "reader_annotation_count", "target_reader_count"] + FEATURES + ["malignancy_median"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({
                "split": r["split"], "patient_id": r["patient_id"], "study_uid": r["study_uid"], "series_uid": r["series_uid"],
                "nodule_group_index": r["nodule_group_index"], "center_patient_x": f"{r['center_patient_x']:.4f}",
                "center_patient_y": f"{r['center_patient_y']:.4f}", "center_patient_z": f"{r['center_patient_z']:.4f}",
                "reader_count": r["reader_count"], "reader_annotation_count": r["reader_annotation_count"],
                "target_reader_count": r["target_reader_count"],
                **{key: f"{value:.6f}" for key, value in zip(FEATURES, r["X"])},
                "malignancy_median": f"{r['y']:.6f}",
            })


def load_ranker_val(path: Path, train_patients: set[str], val_patients: set[str]):
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        required = {"X", "feature_names", "patient_id", "malignancy_reader_ratings", "malignancy_reader_mask"}
        missing = required - set(archive.files)
        if missing:
            raise KeyError(f"Ranker val archive missing keys: {sorted(missing)}")
        X = np.asarray(archive["X"], dtype=np.float64)
        names = [v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v) for v in archive["feature_names"].tolist()]
        patients = np.asarray([v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v) for v in archive["patient_id"].tolist()])
        ratings = np.asarray(archive["malignancy_reader_ratings"], dtype=np.float64)
        masks = np.asarray(archive["malignancy_reader_mask"], dtype=bool)
    if names != FEATURES:
        raise ValueError(f"Validation archive feature order mismatch: {names}")
    if X.ndim != 2 or X.shape[1] != len(FEATURES) or ratings.shape != masks.shape or ratings.shape[0] != len(X):
        raise ValueError("Validation archive shapes are inconsistent")
    observed_patients = set(patients)
    overlap = observed_patients & train_patients
    if overlap:
        raise ValueError(f"Stage 03 validation archive overlaps train patients: {sorted(overlap)[:4]}")
    outside_val = observed_patients - val_patients
    if outside_val:
        raise ValueError(f"Stage 03 validation archive contains patients outside the canonical validation split: {sorted(outside_val)[:4]}")
    valid = masks & np.isfinite(ratings) & (ratings >= 1) & (ratings <= 5)
    safe = np.where(valid, ratings, np.nan)
    keep = (valid.sum(axis=1) > 0) & np.isfinite(X).all(axis=1)
    med = np.nanmedian(safe[keep], axis=1)
    mean = np.nanmean(safe[keep], axis=1)
    return {"X": X[keep], "patients": patients[keep], "y_median": med, "y_mean": mean,
            "raw_rows": int(len(X)), "rows_used": int(keep.sum()),
            "reader_observations": int(valid[keep].sum()), "archive_sha256": sha256_file(path)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lidc-root", type=Path, default=Path("datasets/LIDC"))
    ap.add_argument("--metadata", type=Path, default=None)
    ap.add_argument("--xml-root", type=Path, default=None)
    ap.add_argument("--dicom-root", type=Path, default=None)
    ap.add_argument("--split-json", type=Path, required=True)
    ap.add_argument("--ranker-val-archive", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--group-distance-mm", type=float, default=10.0)
    ap.add_argument("--skip-ranker-validation", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = args.lidc_root.resolve()
    metadata_csv = (args.metadata or root / "metadata" / "metadata.csv").resolve()
    xml_root = (args.xml_root or root / "LIDC-XML-only" / "tcia-lidc-xml").resolve()
    dicom_root = (args.dicom_root or root / "lidc_idri").resolve()
    split_path = args.split_json.resolve()
    val_archive_path = args.ranker_val_archive.resolve() if args.ranker_val_archive else None
    out = args.out_dir.resolve()

    for path, description in ((metadata_csv, "metadata.csv"), (xml_root, "XML directory"), (dicom_root, "DICOM root"), (split_path, "canonical split.json")):
        if not path.exists():
            raise FileNotFoundError(f"Missing {description}: {path}")
    if args.group_distance_mm <= 0:
        raise ValueError("--group-distance-mm must be positive")
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {out}. Choose a new directory or pass --overwrite")

    split_obj = read_json(split_path)
    train_patients = patient_ids_for_split(split_obj, "train")
    val_patients = patient_ids_for_split(split_obj, "val")
    overlap = train_patients & val_patients
    if overlap:
        raise ValueError(f"Canonical train/validation split overlaps ({len(overlap)} patients)")
    split_by_patient = {p: "train" for p in train_patients}
    split_by_patient.update({p: "val" for p in val_patients})
    train_val_patients = train_patients | val_patients

    stats: Counter = Counter()
    metadata_rows = load_metadata(metadata_csv, train_val_patients)
    wanted_studies = {study for rows in metadata_rows.values() for study, _ in rows}
    wanted_series = {series for rows in metadata_rows.values() for _, series in rows}
    print(f"Train patients: {len(train_patients):,}; validation patients: {len(val_patients):,}; overlap: 0")
    print(f"Indexing XML headers for {len(wanted_studies):,} train/validation studies...")
    by_study, by_series, xml_header_scanned = build_xml_index(xml_root, wanted_studies, wanted_series)
    print(f"XML files checked for response headers: {xml_header_scanned:,}; relevant studies indexed: {len(by_study):,}; series indexed: {len(by_series):,}")
    print("Indexing DICOM headers for train/validation patients only (PixelData is not read)...")
    dicom_index, dicom_stats = build_dicom_index(dicom_root, train_val_patients, wanted_series, wanted_studies)
    annotations = collect_reader_annotations(metadata_rows, by_study, by_series, dicom_index, train_val_patients, stats)
    print(f"Reader annotations with contour centroids: {len(annotations):,}")
    groups = spatial_groups(annotations, args.group_distance_mm)
    stats["physical_nodule_groups_before_label_filter"] = len(groups)
    rows = aggregate_nodule_groups(groups, split_by_patient, stats)
    train_rows = [r for r in rows if r["split"] == "train"]
    val_rows = [r for r in rows if r["split"] == "val"]
    if not train_rows or not val_rows:
        raise RuntimeError(f"No usable physical nodules after aggregation: train={len(train_rows)}, val={len(val_rows)}")
    X_train = np.asarray([r["X"] for r in train_rows], dtype=np.float64)
    y_train = np.asarray([r["y"] for r in train_rows], dtype=np.float64)
    X_val_xml = np.asarray([r["X"] for r in val_rows], dtype=np.float64)
    y_val_xml = np.asarray([r["y"] for r in val_rows], dtype=np.float64)
    if not np.isfinite(X_train).all() or not np.isfinite(X_val_xml).all():
        raise ValueError("Aggregated XML feature table contains non-finite values")

    model = ExtraTreesRegressor(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    train_median = float(np.median(y_train))
    train_mean = float(np.mean(y_train))
    xml_pred = np.clip(model.predict(X_val_xml), 1.0, 5.0)
    xml_val_metrics = score(y_val_xml, xml_pred, train_median)
    xml_val_metrics["constant_train_mean_mae"] = float(mean_absolute_error(y_val_xml, np.full(len(y_val_xml), train_mean)))

    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "malignancy_from_nodule_level_xml.joblib"
    bundle = {
        "model": model,
        "feature_names": FEATURES,
        "target": "median LIDC reader-assigned malignancy/suspicion rating per spatially grouped physical nodule",
        "target_scale": [1, 5],
        "target_aggregation": "median of valid reader malignancy ratings per spatially grouped physical nodule",
        "input_semantics": "six scalar ordinal attribute ratings; compatible with frozen Stage 03 scalar outputs",
        "model_type": "ExtraTreesRegressor",
        "hyperparameters": MODEL_PARAMS,
        "feature_training_source": "median XML reader attribute ratings per physical nodule",
        "feature_names_in_order": FEATURES,
        "grouping_method": "greedy cross-reader 3D physical ROI-centroid grouping",
        "group_distance_mm": float(args.group_distance_mm),
        "patient_split": {"train": "existing canonical train patients", "validation": "existing canonical validation patients"},
        "stage03_rankers_modified": False,
        "h5_or_ct_pixels_used": False,
    }
    joblib.dump(bundle, model_path)
    save_table(out / "nodule_level_xml_table.csv", rows)
    val_pred_path = out / "xml_nodule_validation_predictions.csv"
    with val_pred_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "study_uid", "series_uid", "nodule_group_index", "reader_count", "target_malignancy_median", "predicted_malignancy_continuous", "predicted_malignancy_round_half_up"])
        for row, pred in zip(val_rows, xml_pred):
            w.writerow([row["patient_id"], row["study_uid"], row["series_uid"], row["nodule_group_index"], row["reader_count"], f"{row['y']:.6f}", f"{float(pred):.6f}", int(rounded_half_up([pred])[0])])

    ranker_val_metrics = None
    ranker_val_info = {"skipped": True, "reason": "not requested or archive not found"}
    if not args.skip_ranker_validation and val_archive_path is not None and val_archive_path.is_file():
        ranker = load_ranker_val(val_archive_path, train_patients, val_patients)
        pred = np.clip(model.predict(ranker["X"]), 1.0, 5.0)
        ranker_val_metrics = {
            "against_nodule_median_target": score(ranker["y_median"], pred, train_median),
            "against_nodule_mean_target": score(ranker["y_mean"], pred, train_mean),
            "feature_std": {name: float(np.std(ranker["X"][:, i])) for i, name in enumerate(FEATURES)},
            "prediction_distribution": {"mean": float(np.mean(pred)), "std": float(np.std(pred)), "min": float(np.min(pred)), "max": float(np.max(pred))},
        }
        with (out / "stage03_validation_predictions.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["patient_id", "target_malignancy_median", "target_malignancy_mean", "prediction_continuous", "prediction_round_half_up", *[f"input_{x}" for x in FEATURES]])
            for i, p in enumerate(pred):
                w.writerow([ranker["patients"][i], f"{ranker['y_median'][i]:.6f}", f"{ranker['y_mean'][i]:.6f}", f"{p:.6f}", int(rounded_half_up([p])[0]), *[f"{v:.6f}" for v in ranker["X"][i]]])
        ranker_val_info = {
            "path": str(val_archive_path), "sha256": ranker["archive_sha256"], "rows_used": ranker["rows_used"],
            "raw_rows": ranker["raw_rows"], "reader_observations": ranker["reader_observations"],
            "patient_count": int(len(set(ranker["patients"]))), "patient_overlap_with_xml_train": 0,
        }
    elif val_archive_path is not None and val_archive_path.exists() and not args.skip_ranker_validation:
        raise RuntimeError(f"Ranker validation archive exists but could not be evaluated: {val_archive_path}")

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "script": Path(__file__).name,
        "python_version": platform.python_version(), "scikit_learn_version": sklearn.__version__,
        "annotation_source": {"metadata_csv": str(metadata_csv), "xml_root": str(xml_root), "dicom_root_for_headers_only": str(dicom_root)},
        "split_json": str(split_path), "split_json_sha256": sha256_file(split_path),
        "feature_names_in_order": FEATURES, "target": "median malignancy rating from the same spatially grouped reader annotations",
        "grouping_distance_mm": float(args.group_distance_mm), "model_hyperparameters": MODEL_PARAMS,
        "patient_counts": {"train": len(train_patients), "validation": len(val_patients), "overlap": 0},
        "nodule_counts": {"train": len(train_rows), "validation": len(val_rows), "physical_groups_before_label_filter": len(groups)},
        "dicom_geometry_index_stats": dicom_stats, "xml_header_files_checked": xml_header_scanned,
        "parse_and_aggregation_stats": dict(stats),
        "validation_on_held_out_xml_nodule_medians": xml_val_metrics,
        "validation_on_frozen_stage03_outputs": ranker_val_metrics,
        "stage03_validation_archive": ranker_val_info,
        "test_archive_opened": False,
        "test_patient_annotations_parsed": False,
        "h5_read": False,
        "dicom_pixel_data_read": False,
        "stage03_rankers_loaded_or_modified": False,
        "patient_split_changed": False,
        "inference_contract": {"six_scalar_features_only": FEATURES, "ct_or_h5_required": False, "reader_rating_target_not_pathology_truth": True},
        "outputs": {"model": str(model_path), "table": str(out / "nodule_level_xml_table.csv"), "xml_val_predictions": str(val_pred_path), "stage03_val_predictions": str(out / "stage03_validation_predictions.csv") if ranker_val_metrics is not None else None},
        "limitations": [
            "Spatial groups use a deterministic greedy 10 mm ROI-centroid heuristic; compare group counts to the fixed dataset before interpreting metrics.",
            "Only nodules with a median for all six attributes and malignancy are used.",
            "Human XML feature medians and model-predicted Stage 03 outputs may remain distributionally different.",
            "Ratings estimate LIDC reader-assigned malignancy/suspicion, not pathology-confirmed cancer probability.",
        ],
    }
    report_path = out / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "trained",
        "model_path": str(model_path), "report_path": str(report_path),
        "nodule_table_path": str(out / "nodule_level_xml_table.csv"),
        "xml_validation_predictions_path": str(val_pred_path),
        "stage03_validation_predictions_path": str(out / "stage03_validation_predictions.csv") if ranker_val_metrics is not None else None,
        "train_physical_nodules": len(train_rows), "validation_physical_nodules": len(val_rows),
        "xml_nodule_validation_metrics": xml_val_metrics, "stage03_output_validation_metrics": ranker_val_metrics,
        "train_val_patient_overlap": 0, "test_archive_opened": False, "test_patient_annotations_parsed": False,
        "h5_read": False, "dicom_pixel_data_read": False, "stage03_rankers_modified": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
