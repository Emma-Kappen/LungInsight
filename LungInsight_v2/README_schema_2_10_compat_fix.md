# LungInsight v2 — schema 2.10 loader compatibility repair

## Why this patch is needed

The Stage 01 traceback says `backend/Imaging/pipeline_schema.py` permits only schema versions `2.6`, `2.7`, and `2.8`, while the installed mask-tracking bundle writes `meta_schema.json` as version `2.10`. The loader fails before Stage 01 starts.

This repair keeps the existing local `pipeline_schema.py` implementation and applies a targeted edit to its literal `SUPPORTED_SCHEMA_VERSIONS` assignment. It also adds `2.10` to `meta_schema.json` → `metadata_policy.supported_schema_versions`. It does **not** replace `pipeline_schema.py` wholesale or modify checkpoints/models/CT files.

## Install/run

1. Extract this archive at the `LungInsight_v2` project root. It adds `tools/fix_pipeline_schema_2_10_compat.py`, its 5 synthetic unit tests under `tools/tests/`, and this README. It does not include or blindly overwrite `pipeline_schema.py`; the script patches your existing copy after validating its supported-version declaration.
2. From PowerShell:

```powershell
Set-Location 'C:\Users\emkap\Documents\Projects\LungInsight\LungInsight_v2'
. .\l-env\Scripts\Activate.ps1
python tools\fix_pipeline_schema_2_10_compat.py --root .
if ($LASTEXITCODE -ne 0) { throw 'Schema loader compatibility patch failed.' }
```

The tool makes timestamped backups of the two files under `backups\schema_2_10_compat_YYYYMMDD_HHMMSS\` before writing. It is idempotent. It fails without modifying files if the loader assignment or existing version contract differs from expected.

## Verify

```powershell
python -m unittest discover -s tools\tests -p 'test_fix_pipeline_schema_2_10_compat.py' -v
python -m py_compile backend\Imaging\pipeline_schema.py backend\Imaging\01_hu_and_crop.py backend\Imaging\heart_rod_tracking.py
python tools\audit_lunginsight_compatibility.py --root .
```

Then rerun Stage 01:

```powershell
.\tools\run_stage01_from_lidc.ps1 -PatientId 'LIDC-IDRI-0001'
if ($LASTEXITCODE -ne 0) { throw 'Stage 01 failed.' }
```

## Safety/limitations

The compatibility patch only resolves the explicit loader-version mismatch. It does not establish that the rod-tracking algorithm segments the heart accurately, and it does not authorize downstream inference. Re-run mask QC and validate `heart_mask.npy` / `volume_hu_heart_excluded.npy` before proceeding. Do not remove or weaken the fail-closed mask gate.
