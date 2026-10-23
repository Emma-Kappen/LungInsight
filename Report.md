### 3.2 Stage 02 — ViTDet3D nodule detection and candidate processing

**Relevant implementation:** `backend/Imaging/stages/02_detect_and_patch.py`, `backend/Imaging/lunginsight_candidate_fusion.py`, and associated validation and audit utilities.

Stage 02 uses **ViTDet3D as the active nodule detector**. The LoG proposal-generation path is not part of the intended active workflow because its runtime was too high.

#### ViTDet3D detection

ViTDet3D generates learned nodule-detection evidence. Its detections are processed through the configured post-processing and non-maximum-suppression steps before accepted candidates are passed downstream.

An earlier LIDC-IDRI-0001 run reported two raw ViTDet3D detections and two candidates after NMS. This is a single-run observation, not a detector sensitivity, precision, or FROC evaluation.

#### Candidate processing and acceptance

Candidate processing must:

- Validate detection geometry against the CT volume, voxel spacing, and crop metadata.
- Deduplicate overlapping detections using the configured spatial and identity rules.
- Preserve detector provenance and stable candidate identifiers.
- Apply valid anatomical and quality filters.
- Record candidate counts and rejection reasons at each processing gate.
- Pass only accepted candidates to subsequent stages.
- Preserve audit artifacts needed to reproduce candidate selection.

The configured downstream limit is the top 50 accepted candidates, where applicable. This limit must be applied after candidate processing and acceptance rules; it does not imply that every candidate is a true nodule.

The candidate-fusion infrastructure may remain in the codebase for deduplication, auditing, and future detector integration, but the active run must not invoke LoG or expect LoG evidence. Any legacy fields that describe LoG detections must be clearly marked as unused or absent rather than populated with fabricated evidence.

#### Performance and acceptance requirements

- Measure Stage 02 runtime on representative CT studies.
- Verify candidate coordinates, bounding boxes, and spatial consistency.
- Record the number of raw detections, post-NMS candidates, rejected candidates, and accepted candidates.
- Verify that missing anatomical prerequisites are handled according to the configured acceptance policy.
- Evaluate detector sensitivity and false-positive performance against a defined reference dataset before making accuracy claims.

Removing LoG reduces the cost of candidate generation but does not, by itself, establish that detection accuracy is adequate. Detector performance must be evaluated independently.

