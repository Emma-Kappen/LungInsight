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
const printReportButton = document.querySelector("#print-report");
const packYearsOutput = document.querySelector("#pack-years");
const saveClinicalButton = document.querySelector("#save-clinical-record");
const loadClinicalButton = document.querySelector("#load-clinical-record");
const loadStudiesButton = document.querySelector("#load-studies");
const clinicalRecordStatus = document.querySelector("#clinical-record-status");
const savedStudies = document.querySelector("#saved-studies");

const imagingApiBase = window.LUNGINSIGHT_IMAGING_API_BASE || "/api/v1/imaging";
let selectedFiles = [];
let uploadResult = null;
let analysisResult = null;
let requestInProgress = false;
let uploadedPatientId = null;

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[character]));
}

function clinicalInputs(data) {
  return {
    age_at_baseline_years: optionalNumber(data.get("age")),
    age_at_baseline_years_status: data.get("age_status"),
    sex_at_baseline: data.get("sex"),
    race_at_baseline: data.get("race"),
    ethnicity_at_baseline: data.get("ethnicity"),
    smoking_status: data.get("smoking_status"),
    years_smoked: optionalNumber(data.get("years_smoked")),
    years_smoked_status: data.get("years_smoked_status"),
    packs_per_day: optionalNumber(data.get("packs_per_day")),
    packs_per_day_status: data.get("packs_per_day_status"),
    pack_years: calculatedPackYears(data),
    years_since_cessation: optionalNumber(data.get("years_since_cessation")),
    family_history_lung_cancer: data.get("family_history_lung_cancer"),
    medical_history_status: data.get("medical_history_status"),
    medical_history_conditions: data.getAll("medical_history"),
    symptoms_status: data.get("symptoms_status"),
    symptoms: data.getAll("symptoms"),
    kps: optionalNumber(data.get("kps")),
  };
}

function optionalNumber(value) {
  return value === "" ? null : Number(value);
}
if (metadata.nodules?.length) showNoduleDetails(metadata.nodules[0]);

function calculatedPackYears(data) {
  if (data.get("smoking_status") === "never") return 0;
  const years = optionalNumber(data.get("years_smoked"));
  const packs = optionalNumber(data.get("packs_per_day"));
  return years === null || packs === null ? null : Math.round(years * packs * 100) / 100;
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
  const packYears = calculatedPackYears(data);
  if (packYearsOutput) packYearsOutput.textContent = packYears === null ? "Unavailable" : `${packYears} pack-years`;
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
  analyseButton.textContent = alreadyUploaded ? "Analysis submitted" : "Upload CT and start analysis";
}

function clinicalSummary(prediction) {
  if (!prediction) return "";
  const models = [
    ["Cancer-type model", prediction.cancer_type],
    ["Survival model", prediction.survival],
  ];
  return `<section class="clinical-results"><h4>Clinical results</h4>
    ${models.map(([label, model]) => `<article class="model-result">
      <strong>${label}</strong>
      <span>Status: ${escapeHtml(model?.status || "unavailable")}</span>
      <span>${escapeHtml(model?.reason || (model?.prediction == null ? "No numeric result returned." : JSON.stringify(model.prediction)))}</span>
    </article>`).join("")}
    <p class="hint">Clinical outputs are research-only and are not a replacement for pathology or clinician judgment.</p>
  </section>`;
}

function showSelection(files) {
  selectedFiles = Array.from(files || []);
  uploadResult = null;
  analysisResult = null;
  if (!selectedFiles.length) {
    selection.className = "selection empty";
    selection.textContent = "No CT scan selected.";
    removeScanButton.hidden = true;
    setStatus("");
    updateReview();
    printReportButton.addEventListener("click", () => window.print());
    return;
  }
  const total = selectedFiles.reduce((sum, file) => sum + file.size, 0);
  const firstPath = selectedFiles[0].webkitRelativePath || selectedFiles[0].name;
  const folder = firstPath.includes("/") ? firstPath.split("/")[0] : "Selected DICOM files";
  selection.className = "selection";
  selection.innerHTML = `<strong>${escapeHtml(folder)}</strong><span>${selectedFiles.length} file(s), ${formatBytes(total)}</span>`;
  removeScanButton.hidden = false;
  setStatus("Scan selected. Enter the patient identifier, then submit for DICOM validation.", "info");
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
  const status = body.status || body.analysis_status || "unknown";
  const blocked = ["blocked", "awaiting_imaging_review", "failed", "interrupted"].includes(status);
  resultSummary.className = blocked ? "summary warning" : status === "completed" ? "summary success" : "summary";
  resultSummary.textContent = body.message || `Current status: ${status}.`;
  const series = body.series || {};
  const study = body.study || body.study_summary || {};
  const counts = study.brock_prediction_counts || {};
  const viewer = body.run_id ? `
    <article class="result-card viewer-card">
      <h3>CT viewer</h3>
      <p class="hint">This viewer uses the current run's reconstructed CT slices. Overlays are shown only when the corresponding validated artifacts exist.</p>
      <div id="viewer-status" class="status info">Loading viewer metadata...</div>
      <div class="viewer-controls" hidden>
        <label>Slice <input id="viewer-slice" type="range" min="0" max="0" value="0"></label>
        <span id="viewer-slice-label"></span>
        <button id="viewer-previous" type="button">Previous slice</button>
        <button id="viewer-next" type="button">Next slice</button>
        <label><input id="viewer-lung" type="checkbox"> Lung mask</label>
        <label><input id="viewer-heart" type="checkbox"> Heart/exclusion mask</label>
        <label>Grad-CAM <select id="viewer-gradcam"><option value="">Off</option></select></label>
      </div>
      <canvas id="viewer-canvas" class="ct-viewer-image" aria-label="Current CT axial slice" hidden></canvas>
      <div id="viewer-nodules" class="nodule-list"></div>
      <div id="viewer-nodule-detail" class="nodule-detail"></div>
    </article>` : "";
  resultSections.innerHTML = `
    <article class="result-card">
      <h3>CT scan and processing</h3>
      <p><strong>Upload:</strong> ${escapeHtml(body.status || "unknown")}</p>
      <p><strong>Analysis:</strong> ${escapeHtml(body.analysis_status || body.status || "unknown")}</p>
      ${body.run_id ? `<p><strong>Run:</strong> ${escapeHtml(body.run_id)}</p>` : ""}
      ${body.current_stage ? `<p><strong>Current stage:</strong> ${escapeHtml(body.current_stage)}</p>` : ""}
      ${body.completed_stages?.length ? `<p><strong>Completed stages:</strong> ${escapeHtml(body.completed_stages.join(", "))}</p>` : ""}
      ${series.slice_count ? `<p><strong>Series:</strong> ${escapeHtml(series.slice_count)} slices, volume ${escapeHtml((series.volume_shape_zyx || []).join(" x "))} voxels</p>` : ""}
      ${body.next_step ? `<p class="warning"><strong>Next step:</strong> ${escapeHtml(body.next_step)}</p>` : ""}
      ${body.blocking_requirements?.length ? `<p class="warning"><strong>Blocking requirements:</strong> ${escapeHtml(body.blocking_requirements.join("; "))}</p>` : ""}
      ${study.processing_status ? `<p><strong>Study processing:</strong> ${escapeHtml(study.processing_status)} · preprocessing ${escapeHtml(study.preprocessing_status || "unknown")} · masks ${escapeHtml(study.mask_validation_status || "unknown")}</p>` : ""}
      ${study.eligible_nodule_count != null ? `<p><strong>Nodules:</strong> ${escapeHtml(study.eligible_nodule_count)} eligible · ${escapeHtml(study.excluded_candidate_count ?? "unknown")} excluded · Brock available ${escapeHtml(counts.completed ?? 0)}, incomplete ${escapeHtml(counts.incomplete ?? 0)}, failed ${escapeHtml(counts.failed ?? 0)}, not run ${escapeHtml(counts.not_run ?? 0)}</p>` : ""}
      ${study.warnings?.length ? `<p class="warning"><strong>Warnings:</strong> ${escapeHtml(study.warnings.join("; "))}</p>` : ""}
      ${study.errors?.length ? `<p class="error"><strong>Errors:</strong> ${escapeHtml(study.errors.join("; "))}</p>` : ""}
      ${body.error ? `<p class="error"><strong>Failure:</strong> ${escapeHtml(body.error)}</p>` : ""}
      ${body.run_id ? `<p class="report-links"><a href="/api/v1/analysis/runs/${encodeURIComponent(body.run_id)}/report/html" target="_blank" rel="noopener">Open readable study report</a> · <a href="/api/v1/analysis/runs/${encodeURIComponent(body.run_id)}/report" target="_blank" rel="noopener">Open structured JSON result</a></p>` : ""}
      ${clinicalSummary(body.clinical_prediction)}
      <p class="warning">Only artifacts and predictions produced by this run are displayed. Blocked or unavailable model components are not treated as negative results.</p>
    </article>`;
  resultSections.insertAdjacentHTML("beforeend", viewer);
  technicalResult.textContent = JSON.stringify(body, null, 2);
  if (body.run_id) loadViewer(body.run_id);
}

async function loadViewer(runId) {
  const viewerStatus = document.querySelector("#viewer-status");
  const controls = document.querySelector(".viewer-controls");
  const canvas = document.querySelector("#viewer-canvas");
  if (!viewerStatus || !controls || !canvas) return;
  try {
    const response = await fetch(`/api/v1/analysis/runs/${encodeURIComponent(runId)}/viewer`);
    const metadata = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(metadata.detail || "Viewer artifacts are unavailable.");
    const slice = document.querySelector("#viewer-slice");
    const label = document.querySelector("#viewer-slice-label");
    const lung = document.querySelector("#viewer-lung");
    const heart = document.querySelector("#viewer-heart");
    const gradcam = document.querySelector("#viewer-gradcam");
    const previous = document.querySelector("#viewer-previous");
    const next = document.querySelector("#viewer-next");
    const maxIndex = Math.max(0, Number(metadata.shape_zyx?.[0] || 1) - 1);
    slice.max = String(maxIndex);
    controls.hidden = false;
    lung.disabled = !metadata.overlays?.lung_mask;
    heart.disabled = !metadata.overlays?.heart_mask;
    let requestNumber = 0;
    for (const option of metadata.gradcam_options || []) {
      const item = document.createElement("option");
      item.value = `${option.stable_nodule_id}|${option.attribute}`;
      item.textContent = `${option.stable_nodule_id} / ${option.attribute}`;
      gradcam.appendChild(item);
    }
    const update = async () => {
      const index = Math.min(maxIndex, Math.max(0, Number(slice.value) || 0));
      const params = new URLSearchParams({
        lung_overlay: String(lung.checked && !lung.disabled),
        heart_overlay: String(heart.checked && !heart.disabled),
      });
      const [gradcamNodule, gradcamAttribute] = gradcam.value.split("|");
      if (gradcamNodule && gradcamAttribute) {
        params.set("gradcam_nodule_id", gradcamNodule);
        params.set("gradcam_attribute", gradcamAttribute);
      }
      const currentRequest = ++requestNumber;
      const response = await fetch(`/api/v1/analysis/runs/${encodeURIComponent(runId)}/viewer/slices/${index}?${params}`);
      if (!response.ok || currentRequest !== requestNumber) return;
      const blob = await response.blob();
      const image = new Image();
      image.onload = () => {
        if (currentRequest !== requestNumber) return;
        canvas.width = image.naturalWidth;
        canvas.height = image.naturalHeight;
        const context = canvas.getContext("2d");
        context.drawImage(image, 0, 0);
        context.strokeStyle = "#ffcf33";
        context.lineWidth = 3;
        for (const nodule of metadata.nodules || []) {
          const geometry = nodule.geometry || {};
          const start = geometry.bbox_start_zyx;
          const end = geometry.bbox_end_zyx_exclusive;
          if (!Array.isArray(start) || !Array.isArray(end) || index < start[0] || index >= end[0]) continue;
          context.strokeRect(start[2], start[1], end[2] - start[2], end[1] - start[1]);
          context.fillStyle = "#ffcf33";
          context.font = "12px sans-serif";
          context.fillText(nodule.stable_nodule_id || "eligible nodule", start[2], Math.max(12, start[1] - 3));
        }
        canvas.hidden = false;
        URL.revokeObjectURL(image.src);
      };
      image.src = URL.createObjectURL(blob);
      label.textContent = `Slice ${index + 1} of ${maxIndex + 1}`;
      previous.disabled = index === 0;
      next.disabled = index === maxIndex;
    };
    slice.addEventListener("input", update);
    lung.addEventListener("change", update);
    heart.addEventListener("change", update);
    gradcam.addEventListener("change", update);
    previous.addEventListener("click", () => {
      slice.value = String(Math.max(0, Number(slice.value) - 1));
      update().catch(() => {});
    });
    next.addEventListener("click", () => {
      slice.value = String(Math.min(maxIndex, Number(slice.value) + 1));
      update().catch(() => {});
    });
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      const direction = event.deltaY > 0 ? 1 : -1;
      const target = Math.min(maxIndex, Math.max(0, Number(slice.value) + direction));
      if (target !== Number(slice.value)) {
        slice.value = String(target);
        update().catch(() => {});
      }
    }, {passive: false});
    update().catch(() => {
      viewerStatus.className = "status warning";
      viewerStatus.textContent = "The requested CT slice could not be loaded.";
    });
    viewerStatus.className = "status success";
    viewerStatus.textContent = `CT volume available: ${metadata.shape_zyx.join(" x ")} voxels.`;
    const noduleList = document.querySelector("#viewer-nodules");
    if (metadata.nodules?.length) {
      noduleList.innerHTML = `<h4>Detected nodules</h4>${metadata.nodules.map((nodule, index) => `
        <button type="button" class="nodule-item" data-nodule-index="${index}">
          <strong>${escapeHtml(nodule.stable_nodule_id)}</strong>
          <span>${escapeHtml(nodule.geometry?.diameter_mm ?? "unknown")} mm</span>
          <span>${escapeHtml(nodule.anatomical_location?.region || nodule.anatomical_location?.primary || "Location unavailable")}</span>
          <span>Sources: ${escapeHtml((nodule.detection_sources || []).join(", ") || "unavailable")}</span>
          <span>Classification: ${escapeHtml(Object.entries(nodule.classification?.attributes || {}).map(([name, value]) => `${name}: ${value?.expected_rating ?? value?.value ?? "unavailable"}`).join("; ") || "unavailable")}</span>
          <span>Brock: ${escapeHtml(nodule.malignancy?.selected_prediction?.status || nodule.malignancy?.brock?.status || "unavailable")}</span>
          <span>Quality: ${escapeHtml(nodule.quality_status || "unknown")}</span>
        </button>`).join("")}`;
      noduleList.querySelectorAll("[data-nodule-index]").forEach((button) => {
        button.addEventListener("click", () => {
          const nodule = metadata.nodules[Number(button.dataset.noduleIndex)];
          showNoduleDetails(nodule);
          const center = nodule?.geometry?.center_zyx?.[0];
          if (Number.isFinite(center)) {
            slice.value = String(Math.min(maxIndex, Math.max(0, Math.round(center))));
            update().catch(() => {});
          }
        });
      });
    } else {
      noduleList.textContent = metadata.status === "completed"
        ? "No nodule records were included in the validated contract."
        : "Nodule list unavailable until candidate detection completes.";
    }
  } catch (error) {
    viewerStatus.className = "status warning";
    viewerStatus.textContent = error.message;
  }
}

function showNoduleDetails(nodule) {
  const target = document.querySelector("#viewer-nodule-detail");
  if (!target || !nodule) return;
  const classification = nodule.classification?.attributes || {};
  const brock = nodule.malignancy?.brock || {};
  const explanations = nodule.gradcam_refs || {};
  const classificationRows = Object.entries(classification).map(([name, item]) =>
    `<li>${escapeHtml(name)}: ${escapeHtml(item?.expected_rating ?? item?.value ?? "unavailable")} (${escapeHtml(item?.status || "prediction")})</li>`,
  ).join("");
  const explanationRows = Object.entries(explanations).map(([name, refs]) => {
    const path = refs?.ct_cam_path || refs?.patch_cam_path;
    if (!path) return `<li>${escapeHtml(name)}: unavailable</li>`;
    const href = `/api/v1/analysis/runs/${encodeURIComponent(analysisResult?.run_id || "")}/artifacts/${path.split("/").map(encodeURIComponent).join("/")}`;
    return `<li>${escapeHtml(name)}: <a href="${href}" target="_blank" rel="noopener">Grad-CAM artifact</a></li>`;
  }).join("");
  target.innerHTML = `
    <h4>${escapeHtml(nodule.stable_nodule_id)}</h4>
    <p>Diameter: ${escapeHtml(nodule.geometry?.diameter_mm ?? "unavailable")} mm · Sources: ${escapeHtml((nodule.detection_sources || []).join(", "))}</p>
    <p>Brock malignancy-risk estimate: ${escapeHtml(brock.probability ?? "unavailable")} (${escapeHtml(brock.status || "unavailable")}); missing inputs: ${escapeHtml((brock.features_missing || brock.missing_inputs || []).join(", ") || "none reported")}.</p>
    <p>Classification model outputs (reader-attribute predictions, not diagnosis):</p><ul>${classificationRows || "<li>Unavailable</li>"}</ul>
    <p>Explanation artifacts:</p><ul>${explanationRows || "<li>Unavailable</li>"}</ul>
    <p class="hint">Brock is a risk estimate, not a confirmed diagnosis or pathology result.</p>`;
}

async function startAnalysis(data, upload) {
  const response = await fetch("/api/v1/analysis/runs", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      patient_id: String(data.get("patient_id")).trim(),
      upload_job_id: upload.job_id,
      requested_malignancy_method: "brock",
      data_provenance: "operator_submitted",
      clinician_inputs: clinicalInputs(data),
    }),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Analysis could not be started (HTTP ${response.status}).`);
  return body;
}

async function pollAnalysis(runId) {
  for (;;) {
    const response = await fetch(`/api/v1/analysis/runs/${encodeURIComponent(runId)}`);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || `Analysis status could not be read (HTTP ${response.status}).`);
    analysisResult = body;
    renderResult(body);
    if (["completed", "failed", "blocked", "awaiting_imaging_review"].includes(body.status)) return body;
    setStatus(body.message || `Analysis status: ${body.status}`, "info");
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
}

async function persistClinicalRecord(data, {quiet = false} = {}) {
  const patientId = String(data.get("patient_id") || "").trim();
  const response = await fetch(`/api/v1/clinical/records/${encodeURIComponent(patientId)}`, {
    method: "PUT",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({clinician_inputs: clinicalInputs(data)}),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Clinical information could not be saved (HTTP ${response.status}).`);
  if (!quiet) {
    clinicalRecordStatus.className = "status success";
    clinicalRecordStatus.textContent = `Clinical information saved for ${patientId}.`;
  }
  return body;
}

function applyClinicalRecord(inputs) {
  const fields = {
    age_at_baseline_years: "age",
    age_at_baseline_years_status: "age_status",
    sex_at_baseline: "sex",
    race_at_baseline: "race",
    ethnicity_at_baseline: "ethnicity",
    smoking_status: "smoking_status",
    years_smoked: "years_smoked",
    years_smoked_status: "years_smoked_status",
    packs_per_day: "packs_per_day",
    packs_per_day_status: "packs_per_day_status",
    pack_years: null,
    years_since_cessation: "years_since_cessation",
    family_history_lung_cancer: "family_history_lung_cancer",
    medical_history_status: "medical_history_status",
    symptoms_status: "symptoms_status",
    kps: "kps",
  };
  for (const [key, name] of Object.entries(fields)) {
    if (!name) continue;
    const control = form.elements.namedItem(name);
    if (control) control.value = inputs[key] == null ? "" : String(inputs[key]);
  }
  for (const [name, values] of [
    ["medical_history", inputs.medical_history_conditions || []],
    ["symptoms", inputs.symptoms || []],
  ]) {
    const selected = new Set(values);
    form.querySelectorAll(`[name="${name}"]`).forEach((checkbox) => {
      checkbox.checked = selected.has(checkbox.value);
    });
  }
  updateReview();
}

async function restoreClinicalRecord() {
  const patientId = String(new FormData(form).get("patient_id") || "").trim();
  if (!patientId) throw new Error("Enter a patient identifier to load its saved clinical information.");
  const response = await fetch(`/api/v1/clinical/records/${encodeURIComponent(patientId)}`);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Clinical information could not be loaded (HTTP ${response.status}).`);
  applyClinicalRecord(body.clinician_inputs || {});
  clinicalRecordStatus.className = "status success";
  clinicalRecordStatus.textContent = `Saved clinical information loaded (saved ${body.saved_at}).`;
}

async function restoreStudies() {
  const patientId = String(new FormData(form).get("patient_id") || "").trim();
  if (!patientId) throw new Error("Enter a patient identifier to find saved studies.");
  const response = await fetch(`/api/v1/analysis/patients/${encodeURIComponent(patientId)}/studies`);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Saved studies could not be loaded (HTTP ${response.status}).`);
  const studies = body.studies || [];
  savedStudies.innerHTML = studies.length
    ? `<strong>Persisted studies for ${escapeHtml(patientId)}</strong>${studies.map((study) => `
        <button class="saved-study" type="button" data-run-id="${escapeHtml(study.run_id)}">
          ${escapeHtml(study.run_id)} · ${escapeHtml(study.status)} · ${escapeHtml(study.eligible_nodule_count ?? "unknown")} eligible nodule(s)
        </button>`).join("")}`
    : "No persisted analysis runs found for this patient.";
  savedStudies.querySelectorAll("[data-run-id]").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        const result = await fetch(`/api/v1/analysis/runs/${encodeURIComponent(button.dataset.runId)}`);
        const saved = await result.json().catch(() => ({}));
        if (!result.ok) throw new Error(saved.detail || "Saved study could not be reopened.");
        analysisResult = saved;
        renderResult(saved);
        results.scrollIntoView({behavior: "smooth", block: "start"});
      } catch (error) {
        clinicalRecordStatus.className = "status error";
        clinicalRecordStatus.textContent = error.message;
      }
    });
  });
}

function validateForm(data, {requireScan = true} = {}) {
  const kps = data.get("kps");
  if (kps !== "" && (Number(kps) < 0 || Number(kps) > 100 || Number(kps) % 10 !== 0)) {
    throw new Error("KPS must be a whole number from 0 to 100 in 10-point increments.");
  }
  if (!data.get("patient_id")) throw new Error("Enter the patient identifier.");
  const age = optionalNumber(data.get("age"));
  if (data.get("age_status") === "reported" && (age === null || age < 18 || age > 120)) {
    throw new Error("Enter an age from 18 to 120 or mark age as unknown/not reported.");
  }
  if (data.get("age_status") !== "reported" && age !== null) {
    throw new Error("Clear the age value when it is marked unknown or not reported.");
  }
  for (const [name, statusName, max] of [
    ["years_smoked", "years_smoked_status", 100],
    ["packs_per_day", "packs_per_day_status", 20],
  ]) {
    const value = optionalNumber(data.get(name));
    const status = data.get(statusName);
    if (status === "reported" && (value === null || value < 0 || value > max)) {
      throw new Error(`Enter a valid ${name.replaceAll("_", " ")} value or mark it unknown/not reported.`);
    }
    if (status !== "reported" && value !== null) {
      throw new Error(`Clear ${name.replaceAll("_", " ")} when it is marked unknown or not reported.`);
    }
  }
  if (data.get("smoking_status") !== "former" && optionalNumber(data.get("years_since_cessation")) !== null) {
    throw new Error("Years since quitting can only be entered for former smokers.");
  }
  for (const [statusName, listName] of [["medical_history_status", "medical_history"], ["symptoms_status", "symptoms"]]) {
    if (data.get(statusName) !== "reported" && data.getAll(listName).length) {
      throw new Error(`Clear selected ${listName.replaceAll("_", " ")} when the information is unknown or not reported.`);
    }
  }
  if (requireScan && !selectedFiles.length) throw new Error("Select a CT series before uploading.");
}

folderInput.addEventListener("change", () => showSelection(folderInput.files));
filesInput.addEventListener("change", () => showSelection(filesInput.files));
saveClinicalButton.addEventListener("click", async () => {
  try {
    validateForm(new FormData(form), {requireScan: false});
    await persistClinicalRecord(new FormData(form));
  } catch (error) {
    clinicalRecordStatus.className = "status error";
    clinicalRecordStatus.textContent = error.message;
  }
});
loadClinicalButton.addEventListener("click", () => {
  restoreClinicalRecord().catch((error) => {
    clinicalRecordStatus.className = "status error";
    clinicalRecordStatus.textContent = error.message;
  });
});
loadStudiesButton.addEventListener("click", () => {
  restoreStudies().catch((error) => {
    savedStudies.textContent = error.message;
  });
});
removeScanButton.addEventListener("click", () => {
  folderInput.value = "";
  filesInput.value = "";
  showSelection([]);
});
form.addEventListener("input", () => {
  const target = document.activeElement;
  if (target?.name === "years_smoked" && target.value !== "") form.elements.years_smoked_status.value = "reported";
  if (target?.name === "packs_per_day" && target.value !== "") form.elements.packs_per_day_status.value = "reported";
  if (target?.name === "age" && target.value !== "") form.elements.age_status.value = "reported";
  const patientId = String(new FormData(form).get("patient_id") || "").trim();
  if (uploadResult && patientId !== uploadedPatientId) {
    uploadResult = null;
    analysisResult = null;
    uploadedPatientId = null;
    setStatus("Patient identifier changed. Upload the selected CT again for this patient.", "warning");
  }
  updateReview();
});
form.addEventListener("change", (event) => {
  const target = event.target;
  if (target?.name === "age_status" && target.value !== "reported") form.elements.age.value = "";
  if (target?.name === "years_smoked_status" && target.value !== "reported") form.elements.years_smoked.value = "";
  if (target?.name === "packs_per_day_status" && target.value !== "reported") form.elements.packs_per_day.value = "";
  if (target?.name === "medical_history_status" && target.value !== "reported") {
    form.querySelectorAll('[name="medical_history"]').forEach((input) => { input.checked = false; });
  }
  if (target?.name === "symptoms_status" && target.value !== "reported") {
    form.querySelectorAll('[name="symptoms"]').forEach((input) => { input.checked = false; });
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
    await persistClinicalRecord(data, {quiet: true});
    requestInProgress = true;
    analyseButton.disabled = true;
    setStatus("Validating and reconstructing the CT series...", "info");
    uploadResult = await uploadFiles(selectedFiles, String(data.get("patient_id")).trim());
    uploadedPatientId = String(data.get("patient_id")).trim();
    renderResult(uploadResult);
    setStatus("CT uploaded and DICOM validation completed. Starting the operator-controlled analysis run...", "info");
    analysisResult = await startAnalysis(data, uploadResult);
    const finalResult = await pollAnalysis(analysisResult.run_id);
    setStatus(
      finalResult.status === "completed" ? "Analysis completed." : "Analysis did not complete.",
      finalResult.status === "completed" ? "success" : "warning",
    );
  } catch (error) {
    setStatus(analysisResult ? `Analysis did not complete: ${error.message}` : `CT upload did not complete: ${error.message}`, "error");
    formError.textContent = error.message;
    if (analysisResult) renderResult({...analysisResult, status: analysisResult.status || "failed", message: error.message, error: error.message});
  } finally {
    requestInProgress = false;
    updateReview();
  }
});

updateReview();
