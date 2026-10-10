#!/usr/bin/env python3
"""Build a reader-aware XML-only malignancy suspicion dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from xml_malignancy_common import (
    FEATURE_COLUMNS,
    H5_DEFAULT,
    OUTPUT_DEFAULT,
    REPO_ROOT,
    SPLIT_DEFAULT,
    XML_ROOT_DEFAULT,
    aggregate_nodules,
    feature_schema,
    match_reader_records,
    parse_xml_file,
    save_json,
    sha256_file,
    sha256_xml_tree,
    validate_canonical_split,
    write_csv,
)


OUTPUT_FILES = (
    "xml_features.csv", "xml_reader_records.csv", "unmatched_records.csv",
    "feature_schema.json", "split_manifest.json", "run_manifest.json",
)


def _hash_json(values: Any) -> str:
    return hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _reader_csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        flattened = {
            key: row.get(key) for key in (
                "patient_id", "study_instance_uid", "series_instance_uid", "annotation_file",
                "session_index", "reader_id", "xml_nodule_id", "match_status", "match_reason",
                "h5_sample_index", "h5_reader_slot", "h5_nodule_id", "malignancy_rating",
                "target_h5_rating", "target_crosscheck",
            )
        }
        flattened.update(row["feature_values"])
        flattened.update(row["geometry"])
        output.append(flattened)
    return output


def _coverage(
    split_indices: dict[str, list[int]],
    split_meta: dict[str, Any],
    index_to_split: dict[int, str],
    nodule_rows: list[dict[str, Any]],
    reader_rows: list[dict[str, Any]],
    sample_metadata: dict[int, dict[str, str]],
) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in nodule_rows:
        by_split[row["split"]].append(row)
    series_splits: dict[str, set[str]] = defaultdict(set)
    for index, split_name in index_to_split.items():
        series_splits[sample_metadata[index]["series_instance_uid"]].add(split_name)
    unmatched_by_split: Counter[str] = Counter()
    unmatched_unallocated = 0
    for row in reader_rows:
        if row["match_status"] == "matched":
            continue
        candidate_splits = series_splits.get(row["series_instance_uid"], set())
        if len(candidate_splits) == 1:
            unmatched_by_split[next(iter(candidate_splits))] += 1
        else:
            unmatched_unallocated += 1
    result: dict[str, Any] = {}
    for split_name, sample_indices in split_indices.items():
        nodules = by_split[split_name]
        targeted = [row for row in nodules if int(row["target_reader_count"]) > 0]
        result[split_name] = {
            "canonical_sample_count": len(sample_indices),
            "canonical_patient_count": split_meta["split_patient_counts"][split_name],
            "matched_physical_nodule_count": len(nodules),
            "matched_target_nodule_count": len(targeted),
            "matched_reader_annotation_count": sum(int(row["matched_reader_count"]) for row in nodules),
            "valid_xml_target_rating_count": sum(int(row["target_reader_count"]) for row in nodules),
            "nodule_coverage_fraction": len(nodules) / len(sample_indices) if sample_indices else None,
            "target_nodule_coverage_fraction": len(targeted) / len(sample_indices) if sample_indices else None,
            "unmatched_xml_annotation_count_for_series_in_split": unmatched_by_split[split_name],
        }
    result["unmatched_xml_annotation_count_unallocated_to_split"] = unmatched_unallocated
    return result


def build_dataset(xml_root: Path, h5_path: Path, split_path: Path, out_dir: Path) -> None:
    start = time.perf_counter()
    for filename in OUTPUT_FILES:
        if (out_dir / filename).exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {out_dir / filename}")
    if not xml_root.is_dir():
        raise FileNotFoundError(f"XML root does not exist: {xml_root}")
    if not h5_path.is_file() or not split_path.is_file():
        raise FileNotFoundError("The requested H5 or canonical split file does not exist")

    split_indices, index_to_split, split_meta = validate_canonical_split(
        h5_path, split_path, enforce_project_canonical=True
    )
    print(
        f"Validated canonical split: samples={split_meta['sample_count']}; "
        f"train/val/test={split_meta['split_sample_counts']}; "
        f"patient_disjoint={split_meta['patient_disjoint']}", flush=True,
    )

    xml_paths = sorted(xml_root.rglob("*.xml"))
    reader_rows: list[dict[str, Any]] = []
    parse_issues: list[dict[str, Any]] = []
    source_tag_counts: Counter[str] = Counter()
    for file_index, xml_path in enumerate(xml_paths, start=1):
        records, status = parse_xml_file(xml_path)
        reader_rows.extend(records)
        source_tag_counts.update(status.get("tag_counts", {}))
        if status["status"] != "parsed":
            parse_issues.append(status)
        if file_index % 100 == 0 or file_index == len(xml_paths):
            print(
                f"Parsed XML files={file_index}/{len(xml_paths)}; "
                f"reader_records={len(reader_rows)}; issues={len(parse_issues)}; "
                f"elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )

    match_report = match_reader_records(reader_rows, h5_path)
    if match_report["target_crosscheck_mismatch_count"]:
        raise ValueError(
            "XML malignancy ratings disagree with H5 reader slots for "
            f"{match_report['target_crosscheck_mismatch_count']} unique feature matches"
        )
    nodules = aggregate_nodules(
        reader_rows, index_to_split, match_report["sample_metadata"]
    )
    if len({row["stable_nodule_id"] for row in nodules}) != len(nodules):
        raise ValueError("Duplicate stable physical-nodule identifiers after aggregation")

    schema = feature_schema(nodules, reader_rows, parse_issues, dict(source_tag_counts))
    schema_hash = _hash_json(schema)
    xml_hashes = sha256_xml_tree(xml_root)
    xml_tree_hash = _hash_json(xml_hashes)
    coverage = _coverage(
        split_indices, split_meta, index_to_split, nodules,
        reader_rows, match_report["sample_metadata"],
    )
    split_manifest = {
        "canonical_split_path": str(split_path),
        "canonical_split_sha256": sha256_file(split_path),
        "h5_sha256": sha256_file(h5_path),
        "split_integrity": split_meta,
        "canonical_indices": split_indices,
        "coverage": coverage,
        "matched_reader_record_count": match_report["matched_record_count"],
        "unmatched_reader_record_count": len(reader_rows) - match_report["matched_record_count"],
        "unmatched_reason_counts": match_report["unmatched_reason_counts"],
        "target_crosscheck_mismatch_count": match_report["target_crosscheck_mismatch_count"],
        "xml_parse_issue_count": len(parse_issues),
        "xml_parse_issues": parse_issues,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "xml_features.csv", nodules)
    write_csv(out_dir / "xml_reader_records.csv", _reader_csv_rows(reader_rows))
    unmatched = [row for row in _reader_csv_rows(reader_rows) if row["match_status"] != "matched"]
    write_csv(out_dir / "unmatched_records.csv", unmatched)
    save_json(out_dir / "feature_schema.json", schema)
    save_json(out_dir / "split_manifest.json", split_manifest)

    run_manifest = {
        "workflow": "xml_only_radiologist_malignancy_suspicion",
        "target_definition": schema["target_definition"],
        "feature_schema_sha256": schema_hash,
        "sources": {
            "xml_root": str(xml_root),
            "xml_file_count": len(xml_hashes),
            "xml_file_sha256": xml_hashes,
            "xml_tree_index_sha256": xml_tree_hash,
            "h5_path": str(h5_path),
            "h5_sha256": sha256_file(h5_path),
            "canonical_split_path": str(split_path),
            "canonical_split_sha256": sha256_file(split_path),
        },
        "artifacts": {
            "xml_features.csv": sha256_file(out_dir / "xml_features.csv"),
            "xml_reader_records.csv": sha256_file(out_dir / "xml_reader_records.csv"),
            "unmatched_records.csv": sha256_file(out_dir / "unmatched_records.csv"),
            "feature_schema.json": sha256_file(out_dir / "feature_schema.json"),
            "split_manifest.json": sha256_file(out_dir / "split_manifest.json"),
        },
        "matched_reader_record_count": match_report["matched_record_count"],
        "unmatched_reader_record_count": len(reader_rows) - match_report["matched_record_count"],
        "valid_target_rating_count": sum(int(row["target_reader_count"]) for row in nodules),
        "target_labeled_nodule_count": sum(int(row["target_reader_count"]) > 0 for row in nodules),
        "feature_columns": list(FEATURE_COLUMNS),
    }
    save_json(out_dir / "run_manifest.json", run_manifest)

    print(
        f"Dataset complete: physical_nodules={len(nodules)}; "
        f"matched_reader_records={match_report['matched_record_count']}; "
        f"unmatched_reader_records={len(reader_rows) - match_report['matched_record_count']}; "
        f"target_labeled_nodules={run_manifest['target_labeled_nodule_count']}; "
        f"valid_target_ratings={run_manifest['valid_target_rating_count']}",
        flush=True,
    )
    for split_name in ("train", "val", "test"):
        print(f"coverage[{split_name}]={coverage[split_name]}", flush=True)
    print(f"feature_missingness={{{', '.join(f'{key}: {value['missing_count']}' for key, value in schema['features'].items())}}}", flush=True)
    print(f"elapsed={time.perf_counter() - start:.1f}s output={out_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--xml-root", type=Path, default=XML_ROOT_DEFAULT)
    parser.add_argument("--h5", type=Path, default=H5_DEFAULT)
    parser.add_argument("--split", type=Path, default=SPLIT_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DEFAULT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    build_dataset(args.xml_root.resolve(), args.h5.resolve(), args.split.resolve(), args.out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())