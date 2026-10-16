"use strict";

const form = document.querySelector("#clinical-form");
const folderInput = document.querySelector("#dicom-folder");
const filesInput = document.querySelector("#dicom-files");
const selection = document.querySelector("#scan-selection");
const removeScanButton = document.querySelector("#remove-scan");
const uploadStatus = document.querySelector("#upload-status");
const analyseButton = document.querySelector("#analyse-button");
const formError = document.querySelector("#form-error");
const reviewSummary = document.querySelector("#review-summary");
const results = document.querySelector("#results");
const resultSummary = document.querySelector("#result-summary");
const resultSections = document.querySelector("#result-sections");
const technicalResult = document.querySelector("#technical-result");

const imagingApiBase = window.LUNGINSIGHT_IMAGING_API_BASE || "/api/v1/imaging";
let selectedFiles = [];
let uploadResult = null;
let requestInProgress = false;
let uploadedPatientId = null;

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[character]));
}

function optionalNumber(value) {
  return value === "" ? null : Number(value);
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function setStatus(message, kind = "") {
  uploadStatus.className = `status ${kind}`.trim();
  uploadStatus.textContent = message;
}

function updateReview() {
  const data = new FormData(form);
  const patient = String(data.get("patient_id") || "").trim();
  const kps = data.get("kps") || "not supplied";
  reviewSummary.innerHTML = `
    <strong>Review before submission</strong>
    <dl>
      <dt>Patient</dt><dd>${escapeHtml(patient || "not supplied")}</dd>
      <dt>Prediction timestamp</dt><dd>Generated automatically when the run starts (UTC)</dd>
      <dt>KPS</dt><dd>${escapeHtml(kps)}</dd>
      <dt>CT series</dt><dd>${escapeHtml(uploadResult?.series ? `${uploadResult.series.slice_count} slices uploaded and validated` : selectedFiles.length ? "selected; not uploaded" : "not selected")}</dd>
    </dl>`;
  const alreadyUploaded = uploadResult && uploadedPatientId === patient;
  analyseButton.disabled = requestInProgress || !selectedFiles.length || !patient || alreadyUploaded;
  analyseButton.textContent = alreadyUploaded
    ? "Imaging review required before analysis"
    : "Upload CT for validation";
}

function showSelection(files) {
  selectedFiles = Array.from(files || []);
  uploadResult = null;
  uploadedPatientId = null;
  if (!selectedFiles.length) {
    selection.className = "selection empty";
    selection.textContent = "No CT scan selected.";
    removeScanButton.hidden = true;
    setStatus("");
    updateReview();
    return;
  }
  const total = selectedFiles.reduce((sum, file) => sum + file.size, 0);
  const firstPath = selectedFiles[0].webkitRelativePath || selectedFiles[0].name;
  const folder = firstPath.includes("/") ? firstPath.split("/")[0] : "Selected DICOM files";
  selection.className = "selection";
  selection.innerHTML = `<strong>${escapeHtml(folder)}</strong><span>${selectedFiles.length} file(s), ${formatBytes(total)}</span>`;
  removeScanButton.hidden = false;
  setStatus("Scan selected. Enter the patient identifier, then upload it in step 4 for DICOM validation.", "info");
  updateReview();
}

function uploadFiles(files, patientId) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", `${imagingApiBase}/dicom/upload`);
    request.responseType = "json";
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) setStatus(`Uploading CT: ${Math.round(event.loaded / event.total * 100)}%`, "info");
    };
    request.onerror = () => reject(new Error("The CT upload could not reach the Imaging service."));
    request.ontimeout = () => reject(new Error("The CT upload timed out. Try the scan again."));
    request.timeout = 120000;
    request.onload = () => {
      const body = request.response || (() => {
        try { return JSON.parse(request.responseText); } catch { return {}; }
      })();
      if (request.status < 200 || request.status >= 300) {
        reject(new Error(body.detail || `Imaging upload failed (HTTP ${request.status}).`));
        return;
      }
      resolve(body);
    };
    const payload = new FormData();
    payload.append("patient_id", patientId);
    files.forEach((file) => payload.append("files", file, file.webkitRelativePath || file.name));
    request.send(payload);
  });
}

function renderResult(body) {
  results.hidden = false;
  resultSummary.className = body.analysis_status === "blocked" ? "summary warning" : "summary success";
  resultSummary.textContent = body.message || "Processing completed.";
  const series = body.series || {};
  resultSections.innerHTML = `
    <article class="result-card">
      <h3>CT scan and processing</h3>
      <p><strong>Upload:</strong> ${escapeHtml(body.status || "unknown")}</p>
      <p><strong>Analysis:</strong> ${escapeHtml(body.analysis_status || "unknown")}</p>
      <p><strong>Series:</strong> ${escapeHtml(series.slice_count || 0)} slices, volume ${escapeHtml((series.volume_shape_zyx || []).join(" × "))} voxels</p>
      ${body.next_step ? `<p class="warning"><strong>Next step:</strong> ${escapeHtml(body.next_step)}</p>` : ""}
      ${body.runner_command ? `<p><strong>Operator command:</strong></p><code class="operator-command">${escapeHtml(body.runner_command)}</code>` : ""}
      <p class="warning">No nodule or Clinical prediction is displayed unless the Imaging pipeline completes its required validation gates.</p>
    </article>`;
  technicalResult.textContent = JSON.stringify(body, null, 2);
}

function clinicalPayload(data, imagingEvidence) {
  return {
    schema_version: "1.0.0",
    patient_id: String(data.get("patient_id")).trim(),
    baseline_definition: "other_or_unverified",
    requested_malignancy_method: "brock",
    clinician_inputs: {
      age_at_baseline_years: optionalNumber(data.get("age")),
      sex_at_baseline: data.get("sex") || null,
      kps: optionalNumber(data.get("kps")),
    },
    imaging_evidence: imagingEvidence,
  };
}

async function requestClinicalPrediction(data, imagingEvidence) {
  const response = await fetch("/api/v1/clinical/predict", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(clinicalPayload(data, imagingEvidence)),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Clinical API returned HTTP ${response.status}.`);
  return body;
}

function validateForm(data) {
  const kps = data.get("kps");
  if (kps !== "" && (Number(kps) < 0 || Number(kps) > 100 || Number(kps) % 10 !== 0)) {
    throw new Error("KPS must be a whole number from 0 to 100 in 10-point increments.");
  }
  if (!data.get("patient_id")) {
    throw new Error("Enter the patient identifier.");
  }
  if (!selectedFiles.length) {
    throw new Error("Select a CT series before uploading.");
  }
}

folderInput.addEventListener("change", () => showSelection(folderInput.files));
filesInput.addEventListener("change", () => showSelection(filesInput.files));
removeScanButton.addEventListener("click", () => {
  folderInput.value = "";
  filesInput.value = "";
  showSelection([]);
});
form.addEventListener("input", () => {
  const patientId = String(new FormData(form).get("patient_id") || "").trim();
  if (uploadResult && patientId !== uploadedPatientId) {
    uploadResult = null;
    uploadedPatientId = null;
    setStatus("Patient identifier changed. Upload the selected CT again for this patient.", "warning");
  }
  updateReview();
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (requestInProgress) return;
  formError.textContent = "";
  try {
    const data = new FormData(form);
    validateForm(data);
    requestInProgress = true;
    analyseButton.disabled = true;
    setStatus("Validating and reconstructing the CT series…", "info");
    uploadResult = await uploadFiles(selectedFiles, String(data.get("patient_id")).trim());
    uploadedPatientId = String(data.get("patient_id")).trim();
    renderResult(uploadResult);
    if (uploadResult.analysis_status === "ready" && uploadResult.imaging_evidence) {
      renderResult(await requestClinicalPrediction(data, uploadResult.imaging_evidence));
    }
    setStatus(uploadResult.analysis_status === "blocked"
      ? "CT uploaded and DICOM validation completed. Imaging analysis requires the authorized Stage 01-07 workflow and its review attestation; no Clinical prediction has been run."
      : "CT processing completed.", uploadResult.analysis_status === "blocked" ? "warning" : "success");
  } catch (error) {
    uploadResult = null;
    uploadedPatientId = null;
    setStatus("CT processing failed.", "error");
    formError.textContent = error.message;
  } finally {
    requestInProgress = false;
    updateReview();
  }
});

updateReview();
