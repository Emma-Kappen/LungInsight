# LungInsight Multimodal Clinical + CT Backend

## Start
From the LungInsight project root:
`pip install -r Clinical/backend_multimodal/requirements.txt`
`uvicorn Clinical.backend_multimodal.app:app --host 127.0.0.1 --port 8000`

Set `LUNGINSIGHT_MODEL_DIR` if the models are outside `Clinical/models`.

## Required
- clinical_preprocessor.joblib
- stage_model_bundle.joblib OR stage_rf_model.joblib

## Optional for multimodal inference
- histology_model_bundle.joblib / histology_rf_model.joblib
- survival_coxph_model.joblib
- imaging_feature_extractor.pt OR imaging_model.pt
- multimodal_fusion_model.joblib

The imaging extractor must be a callable/evaluable PyTorch model that accepts `(N,1,Z,Y,X)` and returns an embedding tensor. A raw state_dict alone is intentionally rejected because the architecture cannot be inferred safely.

## CT
Accepts DICOM `.dcm`, ZIP/TAR.GZ archives, or NIfTI. DICOM slices are sorted by axial position, RescaleSlope/Intercept are applied, and volumes are resampled to 1 mm isotropic spacing.

The current API returns a representative preview. The UI contract leaves room for a `/api/ct/{id}/slice/{index}` endpoint for true server-side interactive slice retrieval; this is preferable to returning hundreds of base64 images in one inference response.

## Safety
This is a research/decision-support application, not a diagnostic device. Outputs require qualified clinical review.
