#!/usr/bin/env python3
"""Compare candidate metadata fields across dataset profiles from explore_dataset_metadata.py."""
from __future__ import annotations
import argparse, re
from pathlib import Path
import pandas as pd

ALIASES = {
    'histology_label': re.compile(r'histolog|cancer.?type|tumou?r.?type|diagnos|subtype|classification|disease.?type|primary.?diagnosis|morpholog', re.I),
    'cohort_or_project': re.compile(r'cohort|project|study|source|dataset|program|collection|repository', re.I),
    'patient_or_case_id': re.compile(r'patient.?id|case.?id|submitter.?id|participant|subject|barcode|case_uuid', re.I),
    'sample_or_specimen_type': re.compile(r'sample.?type|specimen.?type|sample.?class|tissue.?type|analyte', re.I),
    'age': re.compile(r'(^|_)age(_|$)|age_at|age_at_diagnosis', re.I),
    'sex_or_gender': re.compile(r'(^|_)(sex|gender)(_|$)', re.I),
    'stage': re.compile(r'tnm|clinical.?stage|pathologic.?stage|tumou?r.?stage', re.I),
    'smoking': re.compile(r'smok|tobacco|pack.?year', re.I),
    'molecular_or_biomarker': re.compile(r'mutation|variant|gene|expression|biomarker|egfr|kras|tp53|alk|ros1|braf|erbb2|met', re.I),
}

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profiles', type=Path, required=True, help='column_profiles.csv from explorer')
    ap.add_argument('--output-dir', type=Path, default=Path('metadata_exploration'))
    args = ap.parse_args()
    df = pd.read_csv(args.profiles)
    if df.empty:
        raise SystemExit('No profiled columns found. Check the explorer output and dataset paths.')
    df['column_name'] = df['column'].astype(str)
    df['normalized_column'] = df['column_name'].str.lower().str.replace(r'[^a-z0-9]+', '_', regex=True).str.strip('_')
    for category, pattern in ALIASES.items():
        df[category] = df['column_name'].map(lambda x: bool(pattern.search(x)))
    categorized = []
    for _, r in df.iterrows():
        cats = [c for c in ALIASES if bool(r[c])]
        if cats:
            categorized.append({'dataset': r.get('dataset'), 'relative_path': r.get('relative_path'), 'column': r['column_name'], 'categories': ';'.join(cats), 'dtype_in_sample': r.get('dtype_in_sample'), 'missing_pct_in_sample': r.get('missing_pct_in_sample'), 'unique_non_null_in_sample': r.get('unique_non_null_in_sample'), 'sample_values': r.get('sample_values'), 'top_values_if_low_cardinality': r.get('top_values_if_low_cardinality')})
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(categorized).to_csv(out / 'categorized_metadata_candidates.csv', index=False)
    # Similar column names present in multiple datasets are useful, but do not imply equivalent semantics.
    presence = df.groupby(['normalized_column','column_name','dataset'], dropna=False).size().reset_index(name='files_with_column')
    presence.to_csv(out / 'column_presence_by_dataset.csv', index=False)
    if categorized:
        print(f'Wrote {out / "categorized_metadata_candidates.csv"}')
    print(f'Wrote {out / "column_presence_by_dataset.csv"}')
    print('Interpret column matches manually: same-looking field names may encode different definitions across cohorts.')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
