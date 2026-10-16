# Stable Nodule Identity Specification

## Canonical identity

`stable_nodule_id` is generated as `nodule_` followed by the first 24 lowercase
hex characters of SHA-256 over canonical JSON containing:

- `patient_id`
- DICOM `study_uid`
- DICOM `series_uid`
- `centroid_lps_mm`, rounded to the nearest micrometre and encoded as integer
  micrometres in LPS axis order
- fused-record `detector_source`: `ViTDet3D`, `LoG`, or `Merged`

The JSON uses sorted keys and compact separators. Empty identifiers, malformed
centroids, and non-finite physical coordinates are errors; they are never
replaced with fabricated values. The 96-bit displayed digest is an identifier,
not a security credential. Persist and collision-check it against the complete
identity payload when writing a study; a conflicting payload for one ID is a
hard error.

## Fusion, lineage, and persistence

Stage 02 chooses representative physical geometry during deduplication and
generates the stable ID once for the resulting accepted record. It retains all
constituent candidate IDs and detector records in `source_members`. The ID is
copied unchanged into the patch manifest and every later stage, explanation,
report, and API record. Downstream stages must join by this ID, never by array
order, patch index, or a detector's transient candidate ID.

Changing the representative centroid, patient/study/series identity, or fused
detector source changes the computed identity. A rerun over the same canonical
inputs yields the same ID. Distinct records that hash to one ID must be
collision-checked and rejected rather than silently joined.

## Contract naming

`stable_nodule_id` is the canonical serialized field. Where an older consumer
requires `nodule_id`, it is an exact alias and both fields must compare equal.
Detector `candidate_id` and each member's original detector ID remain separate
lineage values and are not substitutes for the stable ID.
