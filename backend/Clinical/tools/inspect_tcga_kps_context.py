import csv
import json
from pathlib import Path
from collections import Counter

artifacts = Path(r"C:\Users\emkap\Documents\Projects\LungInsight\backend\Clinical\artifacts")
source = artifacts / "tcga_kps_xml_structure_20261029T165441Z.csv"

with source.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    columns = reader.fieldnames or []

required = {
    "cohort", "barcode_from_path", "filename", "full_path",
    "kps_value_raw", "kps_is_nonempty", "kps_element_path",
    "ancestor_context", "same_parent_nonempty_children",
    "timing_evidence_json",
}
missing = required - set(columns)
if missing:
    raise SystemExit(f"Missing expected CSV columns: {sorted(missing)}")

def nonempty(value):
    return str(value or "").strip().lower() not in {
        "", "na", "n/a", "null", "none", "not available", "unknown"
    }

valid = [r for r in rows if nonempty(r.get("kps_value_raw"))]
print("INPUT:", source)
print("TOTAL ROWS:", len(rows))
print("NONEMPTY KPS ROWS:", len(valid))
print("UNIQUE PATIENT BARCODES:", len({
    (r.get("cohort", ""), r.get("barcode_from_path", ""))
    for r in valid
}))

for field in ("cohort", "kps_value_raw", "kps_element_path",
              "ancestor_context", "same_parent_nonempty_children"):
    counts = Counter(
        str(r.get(field, "")).strip()
        for r in valid if nonempty(r.get(field))
    )
    print(f"\n=== {field} ===")
    for value, count in counts.most_common(15):
        print(f"[{count}] {value[:700]}")

print("\n=== TIMING EVIDENCE EXAMPLES ===")
shown = 0
for r in valid:
    raw = (r.get("timing_evidence_json") or "").strip()
    if raw and raw not in ("[]", "{}", "null"):
        print(json.dumps({
            "cohort": r.get("cohort"),
            "barcode": r.get("barcode_from_path"),
            "filename": r.get("filename"),
            "kps_value_raw": r.get("kps_value_raw"),
            "kps_element_path": r.get("kps_element_path"),
            "ancestor_context": r.get("ancestor_context"),
            "same_parent_nonempty_children": r.get("same_parent_nonempty_children"),
            "timing_evidence_json": raw,
        }, ensure_ascii=False))
        shown += 1
        if shown >= 12:
            break
print("Timing examples printed:", shown)

# Write up to 60 nonempty KPS rows for focused inspection.
sample = valid[:60]
out_csv = artifacts / "tcga_kps_context_sample_20261029T165441Z.csv"
with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(sample)

print("\nSAMPLE CSV:", out_csv)
print("Read-only inspection complete.")
