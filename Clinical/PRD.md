# LungInsight — Product Requirements Document

**Feature area:** Clinical inference workflow (intake → AI-assisted read → report)
**Status:** Draft, prototype scope
**Audience:** Engineering, clinical informatics, compliance review

## 1. Problem statement

Clinicians reviewing chest imaging want a fast, structured second read that surfaces likely findings and a visual explanation (heatmap) alongside the patient's clinical context, and want that read captured in a signable report for the chart — without adding meaningful workflow latency or introducing unreviewed "black box" outputs into care decisions.

## 2. Goals / non-goals

**Goals**
- Let a clinician submit patient metadata + one image and get back structured, explainable findings in the browser.
- Keep human review mandatory: every output is framed as decision *support*, requires a physician signature before it means anything clinically.
- Produce a print/PDF report suitable for the physical or scanned chart.

**Non-goals (explicitly out of scope for this spec)**
- Autonomous diagnosis or triage without physician sign-off.
- Multi-image / prior-comparison (longitudinal) reads — v2 candidate.
- Mobile native app — this is a responsive web workflow only.
- Replacing PACS/RIS — this is a standalone decision-support layer, not an imaging archive.

## 3. Users & context

- **Primary user:** Attending or resident physician / radiologist reviewing a single study during or after a clinical encounter.
- **Secondary user:** Clinical staff entering intake metadata ahead of physician review.
- **Environment:** Desktop browser in a clinical setting; occasionally tablet.

## 4. Functional requirements

### 4.1 Web intake (browser, HTML/CSS/JS)
| ID | Requirement |
|---|---|
| F1 | Form captures age, sex, smoking history (status + pack-years), free-text clinical notes. |
| F2 | File upload accepts DICOM (`.dcm`), PNG, JPEG; client-side shows a preview thumbnail for PNG/JPEG and a metadata summary (rows/cols, modality) for DICOM. |
| F3 | Inline validation before submit: required fields, numeric ranges, file type/size, with field-level error text (not just a banner). |
| F4 | Submit shows a loading state; disables double-submit. |

### 4.2 Inference
| ID | Requirement |
|---|---|
| F5 | Server returns findings as a list of `{label, probability, flag}` covering at minimum: normal, nodule/mass, effusion, consolidation/pneumonia pattern, suspicious-for-malignancy. |
| F6 | Server returns a Grad-CAM (or equivalent) heatmap overlay aligned to the source image. |
| F7 | Server returns an explicit uncertainty indicator; low-quality input triggers a distinct "insufficient quality" flag instead of a numeric score. |
| **NF-latency** | P50 end-to-end (upload → rendered result) under 3s for a single 512×512 image on CPU inference; P95 under 8s. |

### 4.3 Results UI
| ID | Requirement |
|---|---|
| F8 | Findings render as labeled confidence bars, sorted descending by probability. |
| F9 | Heatmap overlay renders adjustable opacity against the source image. |
| F10 | Every result view carries a persistent, non-dismissable disclaimer: "Investigational output — requires physician review." |

### 4.4 Report generation
| ID | Requirement |
|---|---|
| F11 | One-click "Generate report" produces a print-ready layout: patient identifiers (as entered), findings table, heatmap image, model version + timestamp, and two signature lines (reviewing physician, co-signer/attending). |
| F12 | Report is available both as an in-browser `@media print` view and a downloadable PDF. |
| F13 | Report explicitly states the tool's investigational status and that findings do not constitute a diagnosis absent physician sign-off. |

### 4.5 Validation & security
| ID | Requirement |
|---|---|
| F14 | All inputs re-validated server-side regardless of client-side checks. |
| F15 | Uploaded files checked by content (magic bytes), not just extension, before processing. |
| F16 | File size capped (e.g., 25 MB) and rejected files return actionable error messages. |
| F17 | Auth required on every API route; no anonymous access to `/analyze` or `/report`. |

## 5. Success metrics (prototype phase)

- Time-to-first-result per case (target: <3s median, per NF-latency).
- % of sessions that reach report generation (proxy for perceived usefulness).
- Zero P0 incidents involving PHI exposure in logs/metrics during pilot.
- Clinician-reported trust/usability score (qualitative, pilot survey) — not a substitute for a real clinical validation study before any diagnostic claim is made.

## 6. Explicitly required disclaimers (product copy, not just legal boilerplate)

Every surface — intake form, results panel, and report — must carry visible language equivalent to: *"LungInsight is an investigational decision-support tool. It has not been validated as a diagnostic device. All outputs require independent review and sign-off by a licensed physician before informing patient care."* This is a product requirement, not an optional footer.

## 7. HIPAA / PHI considerations (see also ARCHITECTURE.md §7)

- Data minimization: don't collect or display PHI fields beyond what's needed for the specific report a clinician requests.
- Encryption in transit (TLS) and at rest for anything persisted.
- Audit logging of every access to a case record, separate from PHI storage.
- Business Associate Agreement required with any cloud/infra vendor before real PHI touches the system.
- De-identification of DICOM headers before any data reaches logs, metrics, or third-party model APIs.
- Role-based access control; a "clinical staff" role should not necessarily see the same fields as a "physician" role in the UI.
- This PRD does not itself make the system HIPAA-compliant — compliance is an infrastructure + process + legal review outcome, not a checkbox in a feature spec.

## 8. Open questions for clinical/compliance stakeholders

- Does this tool's output classification (decision-support vs. diagnostic) trigger SaMD (Software as a Medical Device) regulatory review in the target jurisdiction?
- What is the required retention period for generated reports, and who owns deletion requests?
- Who is accountable if a physician signs a report without materially reviewing the AI output ("automation bias") — is a review-time minimum or attestation checkbox needed in the UI?

## 9. Milestones (prototype)

1. **M0** — Static UI + mocked JSON responses (no real model), used to validate UX with clinicians.
2. **M1** — Backend wired to placeholder imaging model + Grad-CAM, end-to-end happy path working.
3. **M2** — Report generation (print + PDF) with full disclaimer language and signature block.
4. **M3** — Security review: validation, auth, PHI-handling gate.
5. **M4 (post-PRD)** — Real trained/validated imaging model swapped in; clinical validation study scoped separately; regulatory pathway determined before any pilot with real patients.
