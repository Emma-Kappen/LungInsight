from pathlib import Path
import os
import re
import shutil

root = Path(os.environ["LI_BUILD_DIR"])
backup = Path(os.environ["LI_BACKUP_DIR"])
path = root / "build_compiled_dataset.py"
source = path.read_text(encoding="utf-8")

def replace_once(old, new, label):
    global source
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one occurrence for {label}; found {count}. "
            "No builder changes were written."
        )
    source = source.replace(old, new, 1)

# Keep existing canonical_smoking() unchanged. Add a separate detailed decoder.
replace_once(
'''def canonical_label(v):
''',
'''def detailed_smoking(v):
    """Preserve detailed source categories without splitting combined MSK labels."""
    if not v:
        return "Unknown"

    raw = v.strip()
    x = re.sub(r"[_-]+", " ", raw.lower()).strip()

    # TCGA CDE 2181650: retain the source-specific quit-duration distinctions.
    tcga_codes = {
        "1": "Never smoker (lifelong)",
        "2": "Current smoker",
        "3": "Former smoker (quit >15 years)",
        "4": "Former smoker (quit <=15 years)",
        "5": "Former smoker (duration unspecified)",
    }
    if raw in tcga_codes:
        return tcga_codes[raw]

    if "former" in x and "current" in x:
        return "Former/Current smoker (combined source category)"
    if "never" in x or x in {"non smoker", "nonsmoker", "lifelong non smoker"}:
        return "Never smoker (lifelong)"
    if "current" in x or "active smoker" in x:
        return "Current smoker"
    if "former" in x or "ex smoker" in x or "previous smoker" in x:
        return "Former smoker (duration unspecified)"
    if x in {"[not available]", "[unknown]", "[discrepancy]", "unknown"}:
        return "Unknown"
    return "Unmapped source category: " + raw

def smoking_source_code(v):
    """Return a TCGA numeric code as text; never coerce non-code source labels."""
    if v is None:
        return ""
    raw = str(v).strip()
    return raw if raw in {"1", "2", "3", "4", "5"} else ""

def canonical_label(v):
''',
"add detailed smoking decoder"
)

replace_once(
'''        "smoking_status": canonical_smoking(raw_smoking),
        "smoking_status_source_raw": raw_smoking,
''',
'''        "smoking_status": canonical_smoking(raw_smoking),
        "smoking_status_source_raw": raw_smoking,
        "smoking_status_source_code": smoking_source_code(raw_smoking),
        "smoking_status_detailed": detailed_smoking(raw_smoking),
''',
"add smoking fields to compiled rows"
)

replace_once(
'''        "smoking_status",
        "smoking_duration_years",
''',
'''        "smoking_status_detailed",
        "smoking_duration_years",
''',
"use detailed smoking field in model matrix"
)

replace_once(
'''        {"field": "smoking_status", "type": "categorical", "allowed": ["Never", "Former", "Current", "Unknown"], "role": "predictor",
         "notes": "Combined Former/Current source category maps to Unknown; raw category preserved."},
''',
'''        {"field": "smoking_status", "type": "categorical", "allowed": ["Never", "Former", "Current", "Unknown"], "role": "audit/legacy comparison",
         "notes": "Legacy coarse mapping retained for comparison; do not use in place of smoking_status_detailed."},
        {"field": "smoking_status_source_raw", "type": "string", "role": "source provenance",
         "notes": "Original smoking history value preserved as read from source."},
        {"field": "smoking_status_source_code", "type": "string", "role": "source provenance",
         "notes": "TCGA numeric source code 1-5 preserved as text; blank for non-code source values."},
        {"field": "smoking_status_detailed", "type": "categorical", "role": "predictor",
         "allowed": ["Never smoker (lifelong)", "Current smoker", "Former smoker (quit >15 years)", "Former smoker (quit <=15 years)", "Former smoker (duration unspecified)", "Former/Current smoker (combined source category)", "Unknown"],
         "notes": "Detailed category derived from TCGA CDE 2181650 codes or explicit source text. Combined MSK former/current category remains combined. Unmapped values are labelled and must be reviewed."},
''',
"update smoking feature dictionary"
)

# Write atomically only after all expected replacements succeeded.
tmp = path.with_suffix(".py.tmp")
tmp.write_text(source, encoding="utf-8")
tmp.replace(path)
print("Patched builder:", path)
print("Backup directory:", backup)
