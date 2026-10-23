# Imaging pipeline configuration

The canonical schema remains `LungInsight_v2/meta_schema.json` for backward
compatibility with `run_pipeline.ps1`, existing command lines, and installed
project bundles. `backend/Imaging/pipeline_schema.py` resolves that root schema.

Do not add a competing schema copy here. This directory is reserved for future
configuration assets that do not replace the canonical file.
