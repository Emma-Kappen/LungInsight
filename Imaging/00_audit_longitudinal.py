"""
00_audit_longitudinal.py

AUDIT script -- run this BEFORE investing in longitudinal (multi-
time-point) nodule tracking, to find out how many patients in your
local LIDC-IDRI copy actually HAVE more than one scan to compare.

    01_dicom_to_hu.py       -> DICOM -> HU volume
    02_mask_and_crop.py     -> lung segmentation + non-lung blanking + Z-crop
    03_visualize.py         -> viewing
    00_audit_longitudinal.py <- this file: census of multi-study patients

Why this matters: LIDC-IDRI is organized PatientID/StudyUID/SeriesUID,
but it was built primarily as a SINGLE-time-point annotation dataset
(four radiologists marking nodules on one scan per patient). Some
patients do have more than one StudyInstanceUID (a re-scan on a later
date), but that's the minority. Building nodule-growth / volume-
doubling-time logic is wasted effort if only a handful of patients in
your local copy actually have two dated scans to compare.

This script does NOT read pixel data or do any segmentation -- it only
reads DICOM headers (fast) to answer:

  1. How many PatientIDs are under the given root?
  2. Of those, how many have MORE THAN ONE StudyInstanceUID?
  3. For multi-study patients, what's the calendar gap between their
     earliest and latest StudyDate (a proxy for "was this actually a
     follow-up scan, or two studies done the same day for some other
     reason, e.g. contrast + non-contrast")?
  4. Within each study, how many CT series are present (helps spot
     scout/localizer/duplicate-kernel series, same issue
     01_dicom_to_hu.py already handles when loading a single study).

Output:
  - A per-patient CSV report (--out-csv, default
    'longitudinal_audit.csv') with one row per PatientID.
  - A printed summary: patient count, multi-study patient count, and
    the distribution of day-gaps for multi-study patients.

Usage:
    python 00_audit_longitudinal.py "Imaging/LIDC/lidc_idri"
    python 00_audit_longitudinal.py "Imaging/LIDC/lidc_idri" --out-csv audit.csv

Note on speed: reads DICOM headers only (stop_before_pixels=True), and
only needs to open ONE file per series to get StudyInstanceUID /
StudyDate / SeriesInstanceUID / Modality -- it does not open every
slice. Still, LIDC-IDRI has 1000+ patients, so this scans directory-
by-directory rather than opening every single file blindly: it grabs
just enough files per series folder to identify the series, assuming
LIDC-IDRI's standard PatientID/StudyUID/SeriesUID nesting.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

try:
    import pydicom
    from pydicom.errors import InvalidDicomError
except ImportError:
    print(
        "pydicom is required. Install it with:\n"
        "    pip install pydicom --break-system-packages",
        file=sys.stderr,
    )
    sys.exit(1)


@dataclass
class SeriesInfo:
    series_uid: str
    modality: Optional[str] = None
    num_files: int = 0


@dataclass
class StudyInfo:
    study_uid: str
    study_date: Optional[str] = None  # DICOM DA format: YYYYMMDD
    series: Dict[str, SeriesInfo] = field(default_factory=dict)


def read_header_only(path: str):
    """Read just enough of a DICOM file to identify it, without
    decoding pixel data (much faster over thousands of files)."""
    try:
        return pydicom.dcmread(path, stop_before_pixels=True, force=False)
    except (InvalidDicomError, Exception):
        return None


def parse_study_date(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y%m%d")
    except ValueError:
        return None


def audit_patient(patient_dir: str) -> Dict[str, StudyInfo]:
    """
    Walk one PatientID folder and group everything found by
    StudyInstanceUID -> SeriesInstanceUID, reading headers only.

    Assumes the standard LIDC-IDRI nesting (PatientID/StudyUID/
    SeriesUID/*.dcm) for efficiency: we sample files per leaf
    directory rather than opening every single slice, since all
    slices in one series folder share the same Study/Series UIDs and
    Modality.
    """
    studies: Dict[str, StudyInfo] = {}

    for root, _dirs, files in os.walk(patient_dir):
        if not files:
            continue

        # Sample a handful of files in this leaf directory -- enough
        # to reliably catch the real image series even if the first
        # file or two happens to be a non-CT / no-pixel-data object
        # (e.g. a stray SR report dropped in the same folder).
        sample_files = [f for f in files if not f.startswith(".")][:5]

        for fname in sample_files:
            ds = read_header_only(os.path.join(root, fname))
            if ds is None:
                continue

            study_uid = getattr(ds, "StudyInstanceUID", None)
            series_uid = getattr(ds, "SeriesInstanceUID", None)
            if study_uid is None or series_uid is None:
                continue

            study = studies.setdefault(
                study_uid,
                StudyInfo(study_uid=study_uid, study_date=getattr(ds, "StudyDate", None)),
            )
            if study.study_date is None:
                study.study_date = getattr(ds, "StudyDate", None)

            series = study.series.setdefault(series_uid, SeriesInfo(series_uid=series_uid))
            series.modality = getattr(ds, "Modality", series.modality)

            # num_files is approximate (counts files actually seen in
            # this leaf dir, capped by the sample) -- good enough to
            # flag "this looks like a scout, not a real CT volume."
            series.num_files = len(files)

            # One representative file per series folder is enough;
            # move to the next leaf directory.
            break

    return studies


def find_patient_dirs(root_dir: str) -> List[str]:
    """
    LIDC-IDRI patient folders are typically named 'LIDC-IDRI-####' at
    the top level of the dataset root. Fall back to treating every
    immediate subdirectory as a patient folder if that naming isn't
    found, so this still works against a differently-named local copy.
    """
    entries = sorted(
        e for e in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, e))
    )
    if not entries:
        raise FileNotFoundError(f"No subdirectories found under '{root_dir}'.")

    lidc_named = [e for e in entries if e.upper().startswith("LIDC-IDRI")]
    chosen = lidc_named if lidc_named else entries

    if lidc_named and len(lidc_named) != len(entries):
        print(
            f"[info] Found {len(entries)} subdirectories, "
            f"{len(lidc_named)} matching 'LIDC-IDRI-*' naming; using those."
        )
    return [os.path.join(root_dir, e) for e in chosen]


def run_audit(root_dir: str, out_csv: str):
    patient_dirs = find_patient_dirs(root_dir)
    print(f"[info] Found {len(patient_dirs)} patient folders under '{root_dir}'.")

    rows = []
    multi_study_gaps_days = []
    num_multi_study = 0
    num_ct_series_total = 0

    for i, patient_dir in enumerate(patient_dirs, 1):
        patient_id = os.path.basename(patient_dir.rstrip("/\\"))
        if i % 100 == 0 or i == len(patient_dirs):
            print(f"[info] Scanned {i}/{len(patient_dirs)} patients...")

        studies = audit_patient(patient_dir)

        ct_study_dates = []
        for study in studies.values():
            ct_series = [s for s in study.series.values() if s.modality == "CT"]
            num_ct_series_total += len(ct_series)
            if ct_series:
                ct_study_dates.append(parse_study_date(study.study_date))

        num_studies = len(studies)
        num_ct_bearing_studies = len(ct_study_dates)
        is_multi_study = num_ct_bearing_studies > 1

        gap_days = None
        if is_multi_study:
            valid_dates = [d for d in ct_study_dates if d is not None]
            if len(valid_dates) >= 2:
                gap_days = (max(valid_dates) - min(valid_dates)).days
            num_multi_study += 1
            if gap_days is not None:
                multi_study_gaps_days.append(gap_days)

        rows.append({
            "patient_id": patient_id,
            "num_studies_total": num_studies,
            "num_ct_bearing_studies": num_ct_bearing_studies,
            "is_multi_study": is_multi_study,
            "earliest_study_date": min(
                (d for d in ct_study_dates if d is not None), default=None
            ),
            "latest_study_date": max(
                (d for d in ct_study_dates if d is not None), default=None
            ),
            "gap_days": gap_days,
        })

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "patient_id", "num_studies_total", "num_ct_bearing_studies",
            "is_multi_study", "earliest_study_date", "latest_study_date", "gap_days",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("\n=== Longitudinal audit summary ===")
    print(f"Total patients scanned:               {len(patient_dirs)}")
    print(f"Total CT series found (all studies):  {num_ct_series_total}")
    print(f"Patients with >1 CT-bearing study:     {num_multi_study} "
          f"({100 * num_multi_study / max(1, len(patient_dirs)):.1f}%)")

    if multi_study_gaps_days:
        gaps = sorted(multi_study_gaps_days)
        n = len(gaps)
        print(f"  Of those with a known date gap (n={n}):")
        print(f"    min gap:    {gaps[0]} days")
        print(f"    median gap: {gaps[n // 2]} days")
        print(f"    max gap:    {gaps[-1]} days")
        same_day = sum(1 for g in gaps if g == 0)
        if same_day:
            print(
                f"    [note] {same_day} of these have a 0-day gap -- likely "
                f"contrast/non-contrast pairs or repeat reconstructions on "
                f"the same visit, NOT a follow-up scan. Worth excluding "
                f"those from any 'growth over time' analysis."
            )
    else:
        print("  No date information available to compute gaps.")

    print(f"\n[done] Wrote per-patient CSV report to '{out_csv}'")
    print(
        "[info] Next: filter the CSV to rows where is_multi_study=True and "
        "gap_days > 0 to get your real candidate list for longitudinal "
        "nodule tracking."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit a local LIDC-IDRI copy for patients with more "
        "than one scan (StudyInstanceUID) -- i.e. real candidates for "
        "longitudinal nodule-growth tracking."
    )
    parser.add_argument(
        "root_dir",
        help="Path to the LIDC-IDRI dataset root containing patient "
        "folders, e.g. 'Imaging/LIDC/lidc_idri'.",
    )
    parser.add_argument(
        "--out-csv",
        default="longitudinal_audit.csv",
        help="Path to write the per-patient CSV report "
        "(default: 'longitudinal_audit.csv').",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_audit(args.root_dir, args.out_csv)


if __name__ == "__main__":
    main()