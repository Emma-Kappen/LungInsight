#!/usr/bin/env python3
"""
Build a fast LIDC-IDRI nodule manifest.

This script is designed for the LungInsight project and the existing
lidc_all_annotations_match.json format:

{
  "patient_count": ...,
  "patients": [
    {
      "patient_id": "...",
      "studies": [
        {
          "study_instance_uid": "...",
          "series_instance_uid": "...",
          "annotation_file": "...",
          "matched": true,
          "reading_sessions": [
            {
              "servicing_radiologist_id": "...",
              "nodules": [
                {
                  "nodule_id": "Nodule 001",
                  "subtlety": "5",
                  ...
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}

Important project convention:
- The neural network trains ONLY the 8 LIDC nodule characteristics.
- Malignancy is retained as metadata but is NOT a target for train.py.
- The output is a plain JSON list compatible with the current manifest-based
  training/inference workflow.

The script is intentionally optimized to avoid the slow pattern:
    rglob(patient_id)
for every nodule.

Instead:
1. Parse every XML file once.
2. Load the match JSON once.
3. Build a single match index.
4. Walk the DICOM tree once to locate referenced series UIDs.
5. Merge everything and write the final manifest.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


# The 8 neural-network targets.
ATTRIBUTE_NAMES = [
    "subtlety",
    "internalStructure",
    "calcification",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
]

# Raw LIDC rating scales documented by the project's metadata schema.
# These are stored in the manifest as metadata only; no normalization is
# performed by this script.
ATTRIBUTE_MAX_RATING = {
    "subtlety": 5,
    "internalStructure": 4,
    "calcification": 6,
    "sphericity": 5,
    "margin": 5,
    "lobulation": 5,
    "spiculation": 5,
    "texture": 5,
}

# Optional metadata label; retained for downstream XGBoost work only.
MALIGNANCY_MAX_RATING = 5


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def normalize_path_text(value: Any) -> str | None:
    """
    Normalize path separators/case for comparison only.
    The original XML path is preserved separately.
    """
    text = normalize_text(value)
    if text is None:
        return None

    text = text.replace("/", "\\")
    text = re.sub(r"\\+", r"\\", text)
    return text.lower()


def normalize_uid(value: Any) -> str | None:
    value = normalize_text(value)
    return value


def parse_int(value: Any) -> int | None:
    text = normalize_text(value)
    if text is None:
        return None

    try:
        number = float(text)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return int(number)


def parse_float(value: Any) -> float | None:
    text = normalize_text(value)
    if text is None:
        return None

    try:
        number = float(text)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return float(number)


def local_name(tag: str) -> str:
    """Remove an XML namespace, e.g. '{namespace}roi' -> 'roi'."""
    return tag.rsplit("}", 1)[-1]


def descendants(element: ET.Element, tag_name: str) -> Iterable[ET.Element]:
    wanted = tag_name.lower()
    for node in element.iter():
        if local_name(node.tag).lower() == wanted:
            yield node


def first_text(element: ET.Element, tag_name: str) -> str | None:
    for node in descendants(element, tag_name):
        if node.text and node.text.strip():
            return node.text.strip()
    return None


def attribute_score(nodule_element: ET.Element, tag_name: str) -> int | None:
    """
    Read one LIDC characteristic from XML.

    This is only a fallback/diagnostic value.  For the final manifest,
    the normalized values from lidc_all_annotations_match.json are preferred.
    """
    wanted = re.sub(r"[^a-z0-9]", "", tag_name.lower())

    for node in nodule_element.iter():
        node_name = re.sub(r"[^a-z0-9]", "", local_name(node.tag).lower())
        if node_name != wanted:
            continue

        score = parse_int(node.text)
        if score is not None:
            return score

    return None


def extract_roi_geometry(roi_element: ET.Element) -> dict[str, Any]:
    """
    Extract geometry from one ROI.

    Returns:
      imageSOP_UID
      imageZposition
      center_x
      center_y
      edge_count

    center_x/center_y are the mean contour coordinates for this ROI.
    """
    image_sop_uid = normalize_uid(first_text(roi_element, "imageSOP_UID"))
    image_z = parse_float(first_text(roi_element, "imageZposition"))

    xs: list[float] = []
    ys: list[float] = []

    for edge_map in descendants(roi_element, "edgeMap"):
        x = parse_float(first_text(edge_map, "xCoord"))
        y = parse_float(first_text(edge_map, "yCoord"))

        if x is not None:
            xs.append(x)
        if y is not None:
            ys.append(y)

    return {
        "imageSOP_UID": image_sop_uid,
        "imageZposition": image_z,
        "center_x": (sum(xs) / len(xs)) if xs else None,
        "center_y": (sum(ys) / len(ys)) if ys else None,
        "edge_count": min(len(xs), len(ys)),
    }


def parse_xml_file(xml_path: Path) -> list[dict[str, Any]]:
    """
    Parse one XML annotation file.

    One output item is created for each unblindedReadNodule.
    """
    try:
        root = ET.parse(xml_path).getroot()
    except Exception as exc:
        warnings.warn(f"Could not parse XML: {xml_path} ({exc})")
        return []

    patient_id = normalize_text(first_text(root, "patientId"))
    output_rows: list[dict[str, Any]] = []

    sessions = list(descendants(root, "readingSession"))

    for session_index, session in enumerate(sessions):
        radiologist_id = normalize_text(
            first_text(session, "servicingRadiologistID")
        )

        nodule_elements = list(descendants(session, "unblindedReadNodule"))

        for nodule_element in nodule_elements:
            nodule_id = normalize_text(first_text(nodule_element, "noduleID"))
            if not nodule_id:
                continue

            roi_elements = list(descendants(nodule_element, "roi"))
            roi_records = [
                extract_roi_geometry(roi)
                for roi in roi_elements
            ]

            # Prefer the ROI with the most contour points.  This gives us a
            # stable representative slice for downstream DICOM cropping.
            representative = None
            if roi_records:
                representative = max(
                    roi_records,
                    key=lambda r: (
                        int(r.get("edge_count") or 0),
                        1 if r.get("imageSOP_UID") else 0,
                    ),
                )

            row = {
                "patient_id_xml": patient_id,
                "annotation_file_xml": str(xml_path),
                "nodule_id": nodule_id,
                "xml_session_index": session_index,
                "xml_radiologist_id": radiologist_id,
                "roi_count_xml": len(roi_records),
                "roi_z_positions_xml": [
                    r["imageZposition"]
                    for r in roi_records
                    if r.get("imageZposition") is not None
                ],
                "imageSOP_UID": (
                    representative.get("imageSOP_UID")
                    if representative
                    else None
                ),
                "imageZposition": (
                    representative.get("imageZposition")
                    if representative
                    else None
                ),
                "center_x": (
                    representative.get("center_x")
                    if representative
                    else None
                ),
                "center_y": (
                    representative.get("center_y")
                    if representative
                    else None
                ),
            }

            # Keep XML values for diagnostics/fallback.
            for attr in ATTRIBUTE_NAMES:
                xml_tag = attr
                row[f"{attr}_xml"] = attribute_score(
                    nodule_element, xml_tag
                )

            row["malignancy_xml"] = attribute_score(
                nodule_element, "malignancy"
            )

            output_rows.append(row)

    return output_rows


def load_match_json(match_path: Path) -> dict[str, Any]:
    with match_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Match JSON root must be an object.")

    patients = data.get("patients")
    if not isinstance(patients, list):
        raise ValueError(
            "Match JSON does not contain the expected top-level "
            "'patients' list."
        )

    return data


def annotation_path_variants(
    annotation_file: str | None,
    xml_path: Path,
    xml_root: Path,
) -> list[str]:
    """
    Return many normalized path forms so that an annotation_file such as:

        Imaging\\LIDC\\LIDC-XML-only\\tcia-lidc-xml\\185\\069.xml

    can be matched to the user's extracted XML root even if the wrapper
    directories differ.
    """
    variants: list[str] = []

    if annotation_file:
        normalized = normalize_path_text(annotation_file)
        if normalized:
            variants.append(normalized)

            parts = normalized.split("\\")
            for i in range(len(parts)):
                suffix = "\\".join(parts[i:])
                if suffix:
                    variants.append(suffix)

    try:
        relative = str(xml_path.relative_to(xml_root))
    except ValueError:
        relative = xml_path.name

    normalized_relative = normalize_path_text(relative)
    if normalized_relative:
        variants.append(normalized_relative)

        parts = normalized_relative.split("\\")
        for i in range(len(parts)):
            suffix = "\\".join(parts[i:])
            if suffix:
                variants.append(suffix)

    variants.append(xml_path.name.lower())

    # Keep order, remove duplicates.
    return list(dict.fromkeys(variants))


def build_match_index(
    match_data: dict[str, Any],
) -> tuple[
    dict[tuple[str, str], list[dict[str, Any]]],
    set[str],
]:
    """
    Build:
        (annotation-path-form, nodule_id) -> matching records

    and a set of all referenced SeriesInstanceUID values.
    """
    index: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    referenced_series_uids: set[str] = set()

    patients = match_data.get("patients") or []

    for patient in patients:
        patient_id = normalize_text(patient.get("patient_id"))
        if not patient_id:
            continue

        for study in patient.get("studies") or []:
            series_uid = normalize_uid(study.get("series_instance_uid"))
            study_uid = normalize_uid(study.get("study_instance_uid"))
            annotation_file = normalize_path_text(
                study.get("annotation_file")
            )

            if not series_uid or not annotation_file:
                continue

            referenced_series_uids.add(series_uid)

            for session_index, session in enumerate(
                study.get("reading_sessions") or []
            ):
                radiologist_id = normalize_text(
                    session.get("servicing_radiologist_id")
                )

                for nodule in session.get("nodules") or []:
                    nodule_id = normalize_text(nodule.get("nodule_id"))
                    if not nodule_id:
                        continue

                    row = {
                        "patient_id": patient_id,
                        "study_instance_uid": study_uid,
                        "series_instance_uid": series_uid,
                        "annotation_file_match": annotation_file,
                        "servicing_radiologist_id": radiologist_id,
                        "match_session_index": session_index,
                        "nodule_id_match": nodule_id,
                    }

                    # Raw LIDC characteristics.
                    for attr in ATTRIBUTE_NAMES:
                        row[attr] = parse_int(nodule.get({
                            "internalStructure": "internal_structure"
                        }.get(attr, attr)))

                    row["malignancy"] = parse_int(
                        nodule.get("malignancy")
                    )

                    row["roi_count_match"] = parse_int(
                        nodule.get("roi_count")
                    )

                    roi_z = nodule.get("roi_z_positions") or []
                    if isinstance(roi_z, list):
                        row["roi_z_positions_match"] = [
                            parse_float(v)
                            for v in roi_z
                            if parse_float(v) is not None
                        ]
                    else:
                        row["roi_z_positions_match"] = []

                    # Index every useful path form from the match JSON.
                    parts = annotation_file.split("\\")
                    variants = [annotation_file]
                    for i in range(len(parts)):
                        variants.append("\\".join(parts[i:]))
                    variants.append(parts[-1])

                    for annotation_key in dict.fromkeys(variants):
                        index[(annotation_key, nodule_id)].append(row)

    return index, referenced_series_uids


def merge_unique_records(
    existing: dict[tuple[Any, ...], dict[str, Any]],
    record: dict[str, Any],
) -> None:
    """
    Add a record using a stable reader-aware key.

    Multiple XML occurrences of the same reader/nodule are collapsed.
    """
    key = (
        record.get("patient_id"),
        record.get("series_instance_uid"),
        record.get("nodule_id"),
        record.get("servicing_radiologist_id"),
    )

    if key not in existing:
        existing[key] = record
        return

    current = existing[key]

    # Prefer the geometry record with more contour information.
    old_roi = int(current.get("roi_count_xml") or 0)
    new_roi = int(record.get("roi_count_xml") or 0)

    if new_roi > old_roi:
        for field in (
            "annotation_file",
            "imageSOP_UID",
            "imageZposition",
            "center_x",
            "center_y",
            "roi_count_xml",
            "roi_z_positions_xml",
        ):
            current[field] = record.get(field)

    # Preserve any non-null geometry.
    for field in (
        "imageSOP_UID",
        "imageZposition",
        "center_x",
        "center_y",
    ):
        if current.get(field) is None and record.get(field) is not None:
            current[field] = record.get(field)


def locate_series_directories(
    lidc_root: Path,
    target_series_uids: set[str],
) -> dict[str, str]:
    """
    Find referenced DICOM series directories in one filesystem traversal.

    Fast path:
        if a directory name itself is a target SeriesInstanceUID, use it.

    Fallback:
        for candidate directories containing DICOM files, read only one
        header and inspect SeriesInstanceUID.

    There is no per-nodule recursive search.
    """
    if not target_series_uids:
        return {}

    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError(
            "pydicom is required. Install it with: pip install pydicom"
        ) from exc

    series_dirs: dict[str, str] = {}
    checked_headers = 0
    directories_seen = 0
    started = time.time()

    # Avoid scanning obvious non-data folders when present.
    skip_names = {
        "__pycache__",
        ".git",
        ".github",
        ".ipynb_checkpoints",
    }

    for root, dirs, files in os.walk(lidc_root):
        dirs[:] = [d for d in dirs if d not in skip_names]

        directories_seen += 1
        root_path = Path(root)
        base = root_path.name.strip()

        # Fast path: series UID encoded directly in directory name.
        if base in target_series_uids:
            series_dirs.setdefault(base, str(root_path))
            if len(series_dirs) == len(target_series_uids):
                print(
                    "All referenced series directories located by directory "
                    "name; stopping DICOM traversal early.",
                    flush=True,
                )
                break

        # Only inspect one likely DICOM file in this directory.
        candidate_file: Path | None = None

        for file_name in files:
            lower = file_name.lower()
            if lower.endswith(".dcm"):
                candidate_file = root_path / file_name
                break

        # Some LIDC extractions use numeric DICOM filenames with no .dcm.
        if candidate_file is None:
            for file_name in files[:20]:
                if file_name.startswith("."):
                    continue
                candidate_file = root_path / file_name
                break

        if candidate_file is None:
            continue

        try:
            ds = pydicom.dcmread(
                str(candidate_file),
                stop_before_pixels=True,
                force=True,
                specific_tags=[
                    "SeriesInstanceUID",
                    "Modality",
                ],
            )
        except Exception:
            continue

        checked_headers += 1
        series_uid = normalize_uid(
            getattr(ds, "SeriesInstanceUID", None)
        )

        if series_uid in target_series_uids:
            series_dirs.setdefault(series_uid, str(root_path))

            if len(series_dirs) == len(target_series_uids):
                print(
                    "All referenced DICOM series located; stopping traversal "
                    "early.",
                    flush=True,
                )
                break

        if directories_seen % 5000 == 0:
            elapsed = time.time() - started
            print(
                f"Scanned {directories_seen:,} DICOM directories; "
                f"located {len(series_dirs):,}/{len(target_series_uids):,} "
                f"referenced series; elapsed={elapsed:.1f}s",
                flush=True,
            )

    print(
        f"DICOM lookup complete: found {len(series_dirs):,}/"
        f"{len(target_series_uids):,} referenced series; "
        f"header reads={checked_headers:,}; "
        f"directories visited={directories_seen:,}.",
        flush=True,
    )

    return series_dirs


def merge_xml_with_match(
    xml_rows: list[dict[str, Any]],
    xml_root: Path,
    match_index: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    Match XML nodule geometry to the authoritative match JSON record.
    """
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}

    unmatched = 0

    for xml_row in xml_rows:
        xml_path = Path(xml_row["annotation_file_xml"])
        nodule_id = normalize_text(xml_row.get("nodule_id"))

        if not nodule_id:
            continue

        variants = annotation_path_variants(
            None,
            xml_path,
            xml_root,
        )

        matches: list[dict[str, Any]] = []
        seen_match_keys: set[tuple[Any, ...]] = set()

        for variant in variants:
            for candidate in match_index.get(
                (variant, nodule_id), []
            ):
                candidate_key = (
                    candidate.get("series_instance_uid"),
                    candidate.get("servicing_radiologist_id"),
                    candidate.get("nodule_id_match"),
                )
                if candidate_key in seen_match_keys:
                    continue
                seen_match_keys.add(candidate_key)
                matches.append(candidate)

        # Conservative filename fallback.
        if not matches:
            basename = xml_path.name.lower()
            for (ann_key, ann_nodule_id), candidates in match_index.items():
                if (
                    ann_nodule_id == nodule_id
                    and ann_key == basename
                ):
                    for candidate in candidates:
                        candidate_key = (
                            candidate.get("series_instance_uid"),
                            candidate.get("servicing_radiologist_id"),
                            candidate.get("nodule_id_match"),
                        )
                        if candidate_key not in seen_match_keys:
                            seen_match_keys.add(candidate_key)
                            matches.append(candidate)

        if not matches:
            unmatched += 1
            continue

        for match in matches:
            record = {
                "patient_id": (
                    match.get("patient_id")
                    or xml_row.get("patient_id_xml")
                ),
                "study_instance_uid": match.get(
                    "study_instance_uid"
                ),
                "series_instance_uid": match.get(
                    "series_instance_uid"
                ),
                "annotation_file": xml_row.get(
                    "annotation_file_xml"
                ),
                "annotation_file_match": match.get(
                    "annotation_file_match"
                ),
                "servicing_radiologist_id": match.get(
                    "servicing_radiologist_id"
                ),
                "nodule_id": match.get(
                    "nodule_id_match"
                ) or nodule_id,
                "imageSOP_UID": xml_row.get(
                    "imageSOP_UID"
                ),
                "imageZposition": xml_row.get(
                    "imageZposition"
                ),
                "center_x": xml_row.get("center_x"),
                "center_y": xml_row.get("center_y"),
                "roi_count_xml": xml_row.get(
                    "roi_count_xml"
                ),
                "roi_z_positions_xml": xml_row.get(
                    "roi_z_positions_xml"
                ) or [],
                "roi_count_match": match.get(
                    "roi_count_match"
                ),
                "roi_z_positions_match": match.get(
                    "roi_z_positions_match"
                ) or [],
                "xml_radiologist_id": xml_row.get(
                    "xml_radiologist_id"
                ),
            }

            # Authoritative characteristics from match JSON.
            for attr in ATTRIBUTE_NAMES:
                record[attr] = match.get(attr)

            # Retained for downstream XGBoost work only.
            record["malignancy"] = match.get("malignancy")

            # Source marker for reproducibility.
            record["label_source"] = "lidc_all_annotations_match.json"
            record["geometry_source"] = "LIDC-XML"

            merge_unique_records(merged, record)

    print(
        f"XML/match merge complete: {len(merged):,} reader-level records "
        f"matched; {unmatched:,} XML nodules had no match-json entry.",
        flush=True,
    )

    return list(merged.values())


def finalize_manifest(
    rows: list[dict[str, Any]],
    series_dirs: dict[str, str],
) -> list[dict[str, Any]]:
    """
    Attach DICOM directories and collapse duplicate reader-level records
    into one record per patient/series/nodule.

    Attribute values are averaged across available readers.

    NOTE:
    This is intended for the current train.py workflow.  It does NOT perform
    label scaling, discretization, rounding, or binary malignancy conversion.
    """
    groups: dict[
        tuple[Any, ...],
        list[dict[str, Any]]
    ] = defaultdict(list)

    missing_series = 0

    for row in rows:
        series_uid = row.get("series_instance_uid")
        dicom_dir = series_dirs.get(series_uid) # type: ignore

        if not dicom_dir:
            missing_series += 1
            continue

        row = dict(row)
        row["dicom_dir"] = dicom_dir

        key = (
            row.get("patient_id"),
            row.get("series_instance_uid"),
            row.get("nodule_id"),
        )
        groups[key].append(row)

    final_rows: list[dict[str, Any]] = []

    for group_key, records in groups.items():
        base = dict(records[0])

        for attr in ATTRIBUTE_NAMES + ["malignancy"]:
            values = [
                float(r[attr])
                for r in records
                if r.get(attr) is not None
            ]

            if values:
                base[attr] = sum(values) / len(values)
            else:
                base[attr] = None

        base["reader_count"] = len(records)

        # Number of readers providing each characteristic.
        base["attribute_observation_count"] = {
            attr: sum(
                r.get(attr) is not None
                for r in records
            )
            for attr in ATTRIBUTE_NAMES + ["malignancy"]
        }

        # Prefer the geometry record with the most contour information.
        representative = max(
            records,
            key=lambda r: (
                int(r.get("roi_count_xml") or 0),
                int(r.get("roi_count_match") or 0),
            ),
        )

        for field in (
            "imageSOP_UID",
            "imageZposition",
            "center_x",
            "center_y",
            "roi_count_xml",
            "roi_count_match",
            "roi_z_positions_xml",
            "roi_z_positions_match",
            "dicom_dir",
            "annotation_file",
            "annotation_file_match",
        ):
            base[field] = representative.get(field)

        # Explicitly preserve only the 8 neural-network training targets in
        # the "training_attributes" field.  Malignancy remains metadata.
        base["training_attributes"] = list(ATTRIBUTE_NAMES)

        base["attribute_rating_scales"] = {
            attr: {
                "min": 1,
                "max": ATTRIBUTE_MAX_RATING[attr],
            }
            for attr in ATTRIBUTE_NAMES
        }

        base["malignancy_rating_scale"] = {
            "min": 1,
            "max": MALIGNANCY_MAX_RATING,
        }

        final_rows.append(base)

    final_rows.sort(
        key=lambda r: (
            str(r.get("patient_id") or ""),
            str(r.get("series_instance_uid") or ""),
            str(r.get("nodule_id") or ""),
        )
    )

    print(
        f"Final manifest records: {len(final_rows):,}; "
        f"records without a located DICOM series: {missing_series:,}.",
        flush=True,
    )

    return final_rows


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(
        path.name + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            data,
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    temporary.replace(path)


def print_summary(
    match_data: dict[str, Any],
    manifest: list[dict[str, Any]],
) -> None:
    patients = sorted(
        {
            row.get("patient_id")
            for row in manifest
            if row.get("patient_id")
        } # type: ignore
    )

    series = sorted(
        {
            row.get("series_instance_uid")
            for row in manifest
            if row.get("series_instance_uid")
        } # type: ignore
    )

    print()
    print("=" * 72)
    print("LIDC MANIFEST SUMMARY")
    print("=" * 72)
    print(
        f"Match JSON patient_count : "
        f"{match_data.get('patient_count', 'unknown')}"
    )
    print(f"Manifest records         : {len(manifest):,}")
    print(f"Unique patients          : {len(patients):,}")
    print(f"Unique series             : {len(series):,}")
    print()

    print("Training attributes (8):")
    for attr in ATTRIBUTE_NAMES:
        present = sum(
            row.get(attr) is not None
            for row in manifest
        )
        max_rating = ATTRIBUTE_MAX_RATING[attr]
        print(
            f"  {attr:18s}: {present:7,}/{len(manifest):7,} "
            f"(rating 1-{max_rating})"
        )

    malignancy_present = sum(
        row.get("malignancy") is not None
        for row in manifest
    )

    print()
    print(
        f"Malignancy metadata   : "
        f"{malignancy_present:,}/{len(manifest):,} "
        f"(rating 1-{MALIGNANCY_MAX_RATING}; "
        f"NOT a neural-network target)"
    )
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fast LIDC-IDRI nodule manifest using "
            "lidc_all_annotations_match.json."
        )
    )

    parser.add_argument(
        "--lidc-root",
        required=True,
        help=(
            "Root directory of the extracted LIDC-IDRI DICOM dataset."
        ),
    )

    parser.add_argument(
        "--xml-root",
        required=True,
        help=(
            "Root directory containing the extracted LIDC XML files."
        ),
    )

    parser.add_argument(
        "--match-file",
        default=None,
        help=(
            "Path to lidc_all_annotations_match.json. "
            "Default: <LIDC root parent>/lidc_all_annotations_match.json"
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path for the final JSON manifest.",
    )

    args = parser.parse_args()

    started = time.time()

    lidc_root = Path(args.lidc_root).expanduser().resolve()
    xml_root = Path(args.xml_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if args.match_file:
        match_path = Path(args.match_file).expanduser().resolve()
    else:
        match_path = (
            lidc_root.parent
            / "lidc_all_annotations_match.json"
        )

    if not lidc_root.exists():
        raise FileNotFoundError(
            f"LIDC root does not exist: {lidc_root}"
        )

    if not xml_root.exists():
        raise FileNotFoundError(
            f"XML root does not exist: {xml_root}"
        )

    if not match_path.exists():
        raise FileNotFoundError(
            f"Match JSON does not exist: {match_path}"
        )

    print("LungInsight LIDC manifest builder")
    print()
    print(f"LIDC root : {lidc_root}")
    print(f"XML root  : {xml_root}")
    print(f"Match JSON: {match_path}")
    print(f"Output    : {output_path}")
    print()

    # ------------------------------------------------------------------
    # 1. Load match JSON.
    # ------------------------------------------------------------------
    print("[1/5] Loading match JSON...", flush=True)

    match_data = load_match_json(match_path)
    match_index, referenced_series_uids = build_match_index(
        match_data
    )

    print(
        f"Match JSON loaded: "
        f"{len(match_data.get('patients', [])):,} patients; "
        f"{len(referenced_series_uids):,} referenced series; "
        f"{len(match_index):,} path/nodule index entries.",
        flush=True,
    )

    # ------------------------------------------------------------------
    # 2. Parse XML files once.
    # ------------------------------------------------------------------
    print()
    print("[2/5] Parsing XML files...", flush=True)

    xml_files = sorted(xml_root.rglob("*.xml"))
    print(
        f"Found {len(xml_files):,} XML files.",
        flush=True,
    )

    xml_rows: list[dict[str, Any]] = []

    for index, xml_file in enumerate(xml_files, start=1):
        xml_rows.extend(
            parse_xml_file(xml_file)
        )

        if index % 100 == 0 or index == len(xml_files):
            print(
                f"Parsed {index:,}/{len(xml_files):,} XML files; "
                f"nodule annotations={len(xml_rows):,}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # 3. Merge XML geometry with match JSON.
    # ------------------------------------------------------------------
    print()
    print("[3/5] Matching XML nodules to match JSON...", flush=True)

    matched_rows = merge_xml_with_match(
        xml_rows,
        xml_root,
        match_index,
    )

    # ------------------------------------------------------------------
    # 4. Locate DICOM series once.
    # ------------------------------------------------------------------
    print()
    print("[4/5] Locating referenced DICOM series...", flush=True)

    series_dirs = locate_series_directories(
        lidc_root,
        referenced_series_uids,
    )

    # ------------------------------------------------------------------
    # 5. Finalize and save.
    # ------------------------------------------------------------------
    print()
    print("[5/5] Finalizing manifest...", flush=True)

    manifest = finalize_manifest(
        matched_rows,
        series_dirs,
    )

    if not manifest:
        raise RuntimeError(
            "The resulting manifest is empty. "
            "Check the LIDC root, XML root, match JSON, and DICOM layout."
        )

    save_json(
        output_path,
        manifest,
    )

    print_summary(
        match_data,
        manifest,
    )

    elapsed = time.time() - started
    size_mb = output_path.stat().st_size / (1024 * 1024)

    print()
    print(
        f"Saved manifest: {output_path}"
    )
    print(
        f"Manifest size : {size_mb:.2f} MB"
    )
    print(
        f"Total time    : {elapsed:.1f} seconds"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"\nERROR: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
