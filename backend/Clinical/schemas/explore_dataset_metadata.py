#!/usr/bin/env python3
"""Inventory and profile metadata files in TCGA-LUAD, TCGA-LUSC, and MSK-CHORD.

This is an exploratory tool, not a training script. It does not modify source data.
It samples large tables and writes schema/missingness/value summaries to an output folder.
"""
from __future__ import annotations
import argparse, csv, json, re
from collections import Counter
from pathlib import Path
from typing import Any
import pandas as pd

TABLE_EXTS = {'.csv', '.tsv', '.txt', '.tab', '.psv', '.xlsx', '.xls', '.parquet', '.jsonl', '.ndjson'}
TEXT_META_EXTS = {'.json', '.yaml', '.yml', '.xml', '.md', '.txt', '.tsv', '.csv'}
ID_RE = re.compile(r'(^|_)(id|uuid|barcode|submitter_id|case_id|patient_id|sample_id|aliquot_id)(_|$)', re.I)
LABEL_RE = re.compile(r'(histolog|cancer.?type|tumou?r.?type|diagnos|primary.?site|disease.?type|project|cohort|subtype|classification|specimen.?type|sample.?type|patholog)', re.I)


def read_table(path: Path, nrows: int) -> tuple[pd.DataFrame, int | None, str]:
    ext = path.suffix.lower()
    if ext in {'.csv', '.tsv', '.tab', '.txt', '.psv'}:
        sep = {' .tsv': '\t'}.get(' '+ext, None)
        if ext in {'.tsv', '.tab'}: sep = '\t'
        elif ext == '.psv': sep = '|'
        elif ext == '.csv': sep = ','
        else:
            # Let pandas infer common text delimiters from a small sample.
            sep = None
        df = pd.read_csv(path, sep=sep, engine='python', nrows=nrows, on_bad_lines='skip')
        return df, None, f'text table (sample up to {nrows} rows; row count not scanned)'
    if ext in {'.xlsx', '.xls'}:
        df = pd.read_excel(path, nrows=nrows)
        return df, None, f'Excel (sample up to {nrows} rows; first sheet only)'
    if ext == '.parquet':
        df = pd.read_parquet(path)
        return df.head(nrows), len(df), 'Parquet (row count exact; profile sampled)' 
    if ext in {'.jsonl', '.ndjson'}:
        df = pd.read_json(path, lines=True, nrows=nrows)
        return df, None, f'JSON Lines (sample up to {nrows} rows; row count not scanned)'
    raise ValueError(f'Unsupported table type: {ext}')


def jsonable(v: Any) -> Any:
    if pd.isna(v): return None
    if hasattr(v, 'item'):
        try: return v.item()
        except Exception: pass
    return str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v


def profile_column(series: pd.Series, col: str) -> dict[str, Any]:
    nonnull = series.dropna()
    nunique = int(nonnull.nunique(dropna=True))
    sample_values = [str(x)[:160] for x in nonnull.astype(str).drop_duplicates().head(8).tolist()]
    top_values = {str(k)[:160]: int(v) for k, v in nonnull.astype(str).value_counts().head(10).items()} if nunique <= 100 else {}
    return {
        'column': str(col), 'dtype_in_sample': str(series.dtype),
        'sample_rows': int(len(series)), 'non_null_sample': int(series.notna().sum()),
        'missing_pct_in_sample': round(float(series.isna().mean() * 100), 2) if len(series) else None,
        'unique_non_null_in_sample': nunique,
        'looks_like_identifier': bool(ID_RE.search(str(col))),
        'possible_label_or_cohort_field': bool(LABEL_RE.search(str(col))),
        'sample_values': sample_values, 'top_values_if_low_cardinality': top_values,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets-root', type=Path, required=True, help='Folder containing TCGA-LUAD, TCGA-LUSC, and msk_chord_2024')
    p.add_argument('--output-dir', type=Path, default=Path('metadata_exploration'))
    p.add_argument('--sample-rows', type=int, default=5000, help='Maximum rows sampled per table (Parquet is counted exactly)')
    p.add_argument('--max-file-mb', type=float, default=2048, help='Skip profiling files larger than this; still inventory them')
    args = p.parse_args()
    root = args.datasets_root.resolve(); out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    dataset_dirs = [
        ('TCGA-LUAD', root / 'TCGA-LUAD'),
        ('TCGA-LUSC', root / 'TCGA-LUSC'),
        ('MSK-CHORD-2024', root / 'msk_chord_2024'),
    ]
    inventory, table_profiles, column_profiles, label_candidates = [], [], [], []
    report: dict[str, Any] = {'datasets_root': str(root), 'sampling': {'max_rows_per_table': args.sample_rows, 'max_file_mb': args.max_file_mb}, 'datasets': {}, 'warnings': []}
    for dataset_name, folder in dataset_dirs:
        ds_summary = {'path': str(folder), 'exists': folder.exists(), 'file_count': 0, 'extension_counts': {}, 'tables_profiled': 0, 'tables_skipped_or_failed': []}
        if not folder.exists():
            report['warnings'].append(f'{dataset_name}: directory not found: {folder}')
            report['datasets'][dataset_name] = ds_summary
            continue
        files = sorted(x for x in folder.rglob('*') if x.is_file())
        ds_summary['file_count'] = len(files)
        ds_summary['extension_counts'] = dict(Counter((x.suffix.lower() or '[no extension]') for x in files))
        for f in files:
            rel = str(f.relative_to(root))
            size = f.stat().st_size
            row = {'dataset': dataset_name, 'relative_path': rel, 'extension': f.suffix.lower(), 'size_bytes': size, 'size_mb': round(size / 1024**2, 3), 'profile_status': 'not_tabular'}
            if f.suffix.lower() in TABLE_EXTS:
                if size > args.max_file_mb * 1024**2:
                    row['profile_status'] = 'skipped_too_large'
                    ds_summary['tables_skipped_or_failed'].append({'path': rel, 'reason': f'larger than --max-file-mb ({args.max_file_mb})'})
                else:
                    try:
                        df, exact_rows, method = read_table(f, args.sample_rows)
                        row['profile_status'] = 'profiled'; row['columns'] = len(df.columns); row['sample_rows_read'] = len(df); row['row_count_exact'] = exact_rows; row['profile_method'] = method
                        table_profiles.append({'dataset': dataset_name, 'relative_path': rel, 'format': f.suffix.lower(), 'size_bytes': size, 'sample_rows_read': len(df), 'row_count_exact': exact_rows, 'column_count': len(df.columns), 'columns': ' | '.join(map(str, df.columns)), 'profile_method': method})
                        for col in df.columns:
                            cp = profile_column(df[col], str(col)); cp.update({'dataset': dataset_name, 'relative_path': rel})
                            column_profiles.append(cp)
                            if cp['possible_label_or_cohort_field']:
                                label_candidates.append({k: cp.get(k) for k in ['dataset','relative_path','column','dtype_in_sample','missing_pct_in_sample','unique_non_null_in_sample','sample_values','top_values_if_low_cardinality']})
                        ds_summary['tables_profiled'] += 1
                    except Exception as e:
                        row['profile_status'] = 'failed'; row['error'] = f'{type(e).__name__}: {e}'[:500]
                        ds_summary['tables_skipped_or_failed'].append({'path': rel, 'reason': row['error']})
            inventory.append(row)
        report['datasets'][dataset_name] = ds_summary
    # CSVs are easy to inspect in Excel; JSON preserves nested values and detailed diagnostics.
    pd.DataFrame(inventory).to_csv(out / 'file_inventory.csv', index=False)
    pd.DataFrame(table_profiles).to_csv(out / 'table_profiles.csv', index=False)
    pd.DataFrame(column_profiles).to_csv(out / 'column_profiles.csv', index=False)
    pd.DataFrame(label_candidates).to_csv(out / 'label_and_cohort_candidates.csv', index=False)
    (out / 'metadata_report.json').write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
    print(json.dumps({'output_dir': str(out), 'dataset_summary': report['datasets'], 'outputs': ['file_inventory.csv','table_profiles.csv','column_profiles.csv','label_and_cohort_candidates.csv','metadata_report.json']}, indent=2))
    print('\nIMPORTANT: value counts and missingness are sample-based for most formats. Review label provenance and patient identifiers before training.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
