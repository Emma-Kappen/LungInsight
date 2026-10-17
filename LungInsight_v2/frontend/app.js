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
const runMessage = document.querySelector("#run-message");
const stagesList = document.querySelector("#stages");
const viewerSection = document.querySelector("#viewer-section");
const detailSection = document.querySelector("#detail");
const technicalResult = document.querySelector("#technical-result");
const printReportButton = document.querySelector("#print-report");
const packYearsOutput = document.querySelector("#pack-years");
const saveClinicalButton = document.querySelector("#save-clinical-record");
const loadClinicalButton = document.querySelector("#load-clinical-record");
const loadStudiesButton = document.querySelector("#load-studies");
const clinicalRecordStatus = document.querySelector("#clinical-record-status");
const savedStudies = document.querySelector("#saved-studies");
const confirmUploadButton = document.querySelector("#confirm-upload");
const uploadProgress = document.querySelector("#upload-progress");
const activityLog = document.querySelector("#activity-log");

const imagingApiBase = window.LUNGINSIGHT_IMAGING_API_BASE || "/api/v1/imaging";
let selectedFiles = [];
let uploadResult = null;
let analysisResult = null;
let requestInProgress = false;
let uploadedPatientId = null;
let requestSequence = 0;

function logEvent(event, fields = {}) {
  console.info("[LungInsight]", event, fields);
  if (!activityLog) return;
  const item = document.createElement("li");
  const time = new Date().toLocaleTimeString();
  const detail = Object.entries(fields).map(([key, value]) => `${key}=${value}`).join(" ");
  item.textContent = `${time} ${event}${detail ? ` (${detail})` : ""}`;
  activityLog.appendChild(item);
  while (activityLog.children.length > 200) activityLog.firstElementChild.remove();
}

function logError(event, error, fields = {}) {
  console.error("[LungInsight]", event, error, fields);
  logEvent(event, {...fields, message: error?.message || String(error)});
}

async function apiFetch(url, options = {}) {
  const method = options.method || "GET";
  const id = ++requestSequence;
  const started = performance.now();
  logEvent("api_request", {id, method, url});
  let response;
  try {
    response = await fetch(url, options);
  } catch (error) {
    logError("api_network_error", error, {id, method, url});
    throw new Error(`Could not reach the server (${method} ${url}). Check that the LungInsight server is still running.`);
  }
  logEvent("api_response", {id, method, url, status: response.status, duration_ms: Math.round(performance.now() - started)});
  return response;
}

function errorMessage(body, fallback) {
  const detail = body?.detail;
  if (typeof detail === "string" && detail) return detail;
  if (Array.isArray(detail) && detail.length) {
    return detail.map((item) => `${(item.loc || []).slice(1).join(".") || "field"}: ${item.msg}`).join("; ");
  }
  return fallback;
}

window.addEventListener("error", (event) => {
  logError("window_error", event.error || new Error(event.message), {source: event.filename, line: event.lineno});
});
window.addEventListener("unhandledrejection", (event) => {
  logError("unhandled_rejection", event.reason instanceof Error ? event.reason : new Error(String(event.reason)));
});

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
    symptoms_status: symptomsStatus(data),
    symptoms: symptomsStatus(data) === "reported" ? data.getAll("symptoms") : [],
    kps: data.get("kps_unknown") ? null : optionalNumber(data.get("kps")),
  };
}

function optionalNumber(value) {
  return value === "" ? null : Number(value);
}

// "no_symptoms" is a form-only choice; it is sent as a reported status with an empty symptom list.
function symptomsStatus(data) {
  const value = data.get("symptoms_status");
  return value === "no_symptoms" ? "reported" : value;
}

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

function setRunMessage(message) {
  runMessage.textContent = message;
}

const STAGE_LABELS = [
  ["stage01", "Stage 01 · CT conversion, lung masks and crop"],
  ["stage02", "Stage 02 · Nodule detection and patch extraction"],
  ["stage03", "Stage 03 · Nodule attribute classification"],
  ["stage04", "Stage 04 · Brock malignancy-risk estimate"],
  ["stage05", "Stage 05 · Grad-CAM explanations"],
  ["stage06", "Stage 06 · Full-CT Grad-CAM projection"],
  ["stage07", "Stage 07 · Report and viewer artifacts"],
  ["clinical", "Clinical models"],
];

function badge(label, kind) {
  return `<span class="badge b-${kind}">${escapeHtml(String(label).replaceAll("_", " "))}</span>`;
}

function renderStages(body) {
  const completed = new Set(body.completed_stages || []);
  const failed = ["failed", "blocked", "interrupted"].includes(body.status);
  const finished = body.status === "completed";
  stagesList.innerHTML = STAGE_LABELS.map(([key, label]) => {
    let state = "queued";
    if (completed.has(key) || (finished && key === "clinical")) state = "completed";
    else if (body.current_stage === key) state = failed ? body.status : "running";
    const kind = {completed: "ok", running: "warn", queued: "mute"}[state] || "bad";
    return `<li><span>${escapeHtml(label)}</span>${badge(state, kind)}</li>`;
  }).join("");
}

function updateReview() {
  const data = new FormData(form);
  const patient = String(data.get("patient_id") || "").trim();
  const kps = data.get("kps_unknown") ? "unknown" : data.get("kps") || "not supplied";
  const packYears = calculatedPackYears(data);
  if (packYearsOutput) packYearsOutput.textContent = packYears === null ? "–" : `${packYears}`;
  reviewSummary.innerHTML = `
    <dl>
      <dt>Patient</dt><dd>${escapeHtml(patient || "not supplied")}</dd>
      <dt>Prediction timestamp</dt><dd>Generated automatically when the run starts (UTC)</dd>
      <dt>KPS</dt><dd>${escapeHtml(kps)}</dd>
      <dt>CT series</dt><dd>${escapeHtml(uploadResult?.series ? `${uploadResult.series.slice_count} slices uploaded and validated` : selectedFiles.length ? "selected; not uploaded" : "not selected")}</dd>
    </dl>`;
  const alreadyUploaded = Boolean(uploadResult && uploadedPatientId === patient);
  confirmUploadButton.disabled = requestInProgress || !selectedFiles.length || !patient || alreadyUploaded;
  confirmUploadButton.textContent = alreadyUploaded ? "CT uploaded" : "Confirm and upload CT";
  analyseButton.disabled = requestInProgress || !alreadyUploaded;
  analyseButton.textContent = "Submit case and run analysis";
}


const MODEL_ISSUES = {
  prediction_baseline_does_not_match_training_baseline: "The CT was not confirmed as acquired at initial pathologic diagnosis. Tick the confirmation in section 1 and re-run.",
  required_model_inputs_missing: "Required inputs are missing.",
  age_at_baseline_years_outside_training_range: "Age is outside the range the model was trained on (33 to 90 years).",
  sex_at_baseline_category_not_seen_in_training: "Sex category was not seen in training.",
  no_validated_horizon_within_model_support: "No validated time horizon is supported for this input.",
};

function modelIssues(model) {
  const issues = [...(model?.warnings || [])].map((code) => MODEL_ISSUES[code] || String(code).replaceAll("_", " "));
  if (model?.missing_features?.length) issues.push(`Missing: ${model.missing_features.join(", ").replaceAll("_", " ")}.`);
  return issues;
}

function modelOutcome(model) {
  if (model?.status === "research_prediction") return "ok";
  return model?.status === "failed" ? "failed" : "unavailable";
}

function modelResultHtml(model) {
  if (!model) return "Unavailable";
  if (model.status === "research_prediction" && model.probabilities) {
    const rows = Object.entries(model.probabilities).map(([label, p]) => `<div style="max-width:360px"><small>${escapeHtml(label)} ${(p * 100).toFixed(1)}%</small><div class="bar"><i style="width:${Math.min(100, Math.max(0, p * 100))}%"></i></div></div>`).join("");
    return `${badge("research prediction", "ok")} Predicted: <b>${escapeHtml(model.predicted_class)}</b> (histology, conditional on LUAD or LUSC)${rows}`;
  }
  if (model.status === "research_prediction" && model.prediction) {
    const horizons = Object.entries(model.prediction.probability_at_horizons_days || {})
      .map(([days, p]) => `<dt>${Math.round(days / 365.25 * 10) / 10} yr survival</dt><dd>${(p * 100).toFixed(1)}%</dd>`).join("");
    return `${badge("research prediction", "ok")} Overall survival from diagnosis<dl>${horizons}</dl>`;
  }
  const issues = modelIssues(model);
  return `${badge(model.status || "unavailable", model.status === "failed" ? "bad" : "warn")} ${escapeHtml(model.reason || issues.join(" ") || "No result returned.")}`;
}

function clinicalSummary(prediction) {
  if (!prediction) return "";
  const models = [
    ["Cancer-type model", prediction.cancer_type],
    ["Survival model", prediction.survival],
  ];
  return `<h3 style="margin-top:1rem">Clinical models</h3>
    <dl>${models.map(([label, model]) => `<dt>${label}</dt><dd>${modelResultHtml(model)}</dd>`).join("")}</dl>
    <p class="hint">Research-only. These models use only age, sex and recorded history; they are not a replacement for pathology or clinician judgment.</p>`;
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
    return;
  }
  const total = selectedFiles.reduce((sum, file) => sum + file.size, 0);
  const firstPath = selectedFiles[0].webkitRelativePath || selectedFiles[0].name;
  const folder = firstPath.includes("/") ? firstPath.split("/")[0] : "Selected DICOM files";
  selection.className = "selection";
  selection.innerHTML = `<strong>${escapeHtml(folder)}</strong><span>${selectedFiles.length} file(s), ${formatBytes(total)}</span>`;
  removeScanButton.hidden = false;
  logEvent("ct_selection_changed", {files: selectedFiles.length, folder, bytes: total});
  setStatus("Review the selected CT folder, then confirm the upload.", "info");
  updateReview();
}

function uploadFiles(files, patientId) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", `${imagingApiBase}/dicom/upload`);
    logEvent("api_request", {method: "POST", url: `${imagingApiBase}/dicom/upload`, files: files.length});
    request.responseType = "json";
    request.upload.onprogress = (event) => {
      if (!event.lengthComputable) return;
      const percent = Math.round(event.loaded / event.total * 100);
      uploadProgress.value = percent;
      setStatus(`Uploading CT: ${percent}% (${formatBytes(event.loaded)} of ${formatBytes(event.total)})`, "info");
    };
    request.upload.onload = () => {
      uploadProgress.value = 100;
      logEvent("upload_body_sent");
      setStatus("Upload received. The server is validating and reconstructing the CT series; large scans can take a minute.", "info");
    };
    request.onerror = () => {
      logError("upload_network_error", new Error("network error"));
      reject(new Error("The CT upload could not reach the Imaging service. Check that the LungInsight server is still running."));
    };
    request.ontimeout = () => reject(new Error("The CT upload timed out. Try the scan again."));
    request.timeout = 120000;
    request.onload = () => {
      const body = request.response || (() => {
        try { return JSON.parse(request.responseText); } catch { return {}; }
      })();
      logEvent("api_response", {method: "POST", url: `${imagingApiBase}/dicom/upload`, status: request.status});
      if (request.status < 200 || request.status >= 300) {
        reject(new Error(errorMessage(body, `Imaging upload failed (HTTP ${request.status}).`)));
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

const TERMINAL = ["completed", "failed", "blocked", "awaiting_imaging_review", "interrupted"];
// Notes the interface never shows (stage progress text and the unreviewed-mask notice), including in older saved runs.
const HIDDEN_NOTES = [/ is running\.?$/i, /not visually reviewed/i];
// Attributes the interface does not present; the backend does not support them yet.
const HIDDEN_ATTRIBUTES = new Set(["internal_structure", "calcification"]);

function methodRows(body, metadata) {
  const study = body.study || body.study_summary || {};
  const counts = study.brock_prediction_counts || {};
  const eligible = study.eligible_nodule_count;
  const rows = [];
  if (eligible != null) {
    rows.push(["Detection", eligible, study.errors?.length ? 1 : 0, 0, `${study.excluded_candidate_count ?? "unknown"} candidate(s) excluded by lung-mask gate`]);
  }
  if (metadata?.nodules?.length) {
    const attributed = metadata.nodules.filter((n) => Object.keys(n.classification?.attributes || {}).some((name) => !HIDDEN_ATTRIBUTES.has(name))).length;
    rows.push(["Attribute classification", attributed, 0, metadata.nodules.length - attributed, metadata.nodules.length - attributed ? "No attribute output for some nodules" : "–"]);
  }
  if (eligible != null) {
    const unavailable = (counts.incomplete ?? 0) + (counts.not_run ?? 0);
    rows.push(["Brock risk", counts.completed ?? 0, counts.failed ?? 0, unavailable, unavailable ? "Missing clinical inputs or not run" : "–"]);
  }
  const prediction = body.clinical_prediction;
  if (prediction) {
    for (const [label, model] of [["Cancer-type model", prediction.cancer_type], ["Survival model", prediction.survival]]) {
      const outcome = modelOutcome(model);
      rows.push([label, outcome === "ok" ? 1 : 0, outcome === "failed" ? 1 : 0, outcome === "unavailable" ? 1 : 0, outcome === "ok" ? "–" : model?.reason || modelIssues(model).join(" ") || model?.status || "unavailable"]);
    }
  }
  return rows;
}

function renderMethods(body, metadata) {
  const rows = methodRows(body, metadata);
  document.querySelector("#methods").innerHTML = rows.length
    ? rows.map((row) => `<tr>${row.map((cell) => `<td>${escapeHtml(cell)}</td>`).join("")}</tr>`).join("")
    : '<tr><td colspan="5">No method outcomes are available for this run.</td></tr>';
}

function renderResult(body) {
  if (!body.run_id) return;
  const status = body.status || "unknown";
  renderStages(body);
  if (!TERMINAL.includes(status)) return;
  results.hidden = false;
  const study = body.study || body.study_summary || {};
  const series = body.series || {};
  const stats = [
    ["Status", String(status).replaceAll("_", " ")],
    ["Eligible nodules", study.eligible_nodule_count ?? "unknown"],
    ["Excluded candidates", study.excluded_candidate_count ?? "unknown"],
    ["Run ID", body.run_id],
  ];
  if (series.slice_count) stats.splice(1, 0, ["Slices", series.slice_count]);
  document.querySelector("#stats").innerHTML = stats
    .map(([key, value]) => `<div class="stat"><b>${escapeHtml(value)}</b>${escapeHtml(key)}</div>`).join("");
  const gateOk = status === "completed" && !["blocked", "failed"].includes(study.mask_validation_status);
  document.querySelector("#gate").innerHTML = `<div class="alert ${gateOk ? "ok" : "bad"}">Mask validation: <b>${escapeHtml(study.mask_validation_status || "unknown")}</b>. Preprocessing: <b>${escapeHtml(study.preprocessing_status || "unknown")}</b>. Downstream analysis: <b>${escapeHtml(gateOk ? "eligible" : status.replaceAll("_", " "))}</b>.</div>`;
  const notes = [];
  if (body.message && status !== "completed" && !HIDDEN_NOTES.some((pattern) => pattern.test(body.message))) notes.push(["warn", body.message]);
  if (body.next_step) notes.push(["warn", `Next step: ${body.next_step}`]);
  if (body.blocking_requirements?.length) notes.push(["bad", `Blocking requirements: ${body.blocking_requirements.join("; ")}`]);
  const warnings = (study.warnings || []).filter((text) => !HIDDEN_NOTES.some((pattern) => pattern.test(text)));
  if (warnings.length) notes.push(["warn", `Warnings: ${warnings.join("; ")}`]);
  const errors = [...new Set(study.errors || [])];
  if (errors.length) notes.push(["bad", `Errors: ${errors.join("; ")}`]);
  if (body.error && !errors.includes(body.error)) notes.push(["bad", `Failure: ${body.error}`]);
  document.querySelector("#result-notes").innerHTML = notes.map(([kind, text]) => `<div class="alert ${kind}">${escapeHtml(text)}</div>`).join("");
  renderMethods(body, null);
  const base = `/api/v1/analysis/runs/${encodeURIComponent(body.run_id)}`;
  document.querySelector("#report-html").href = `${base}/report/html`;
  document.querySelector("#report-json").href = `${base}/report`;
  document.querySelector("#clinical-results").innerHTML = clinicalSummary(body.clinical_prediction);
  technicalResult.textContent = JSON.stringify(body, null, 2);
  loadViewer(body.run_id);
}

const viewer = {runId: null, metadata: null, selected: null, requestNumber: 0, wired: false};

function viewerElements() {
  return {
    status: document.querySelector("#viewer-status"),
    controls: document.querySelector(".viewer-controls"),
    canvas: document.querySelector("#viewer-canvas"),
    slice: document.querySelector("#viewer-slice"),
    label: document.querySelector("#viewer-slice-label"),
    lung: document.querySelector("#viewer-lung"),
    heart: document.querySelector("#viewer-heart"),
    gradcamAttribute: document.querySelector("#viewer-gradcam-attribute"),
    gradcamNodules: document.querySelector("#viewer-gradcam-nodules"),
    gradcamAlpha: document.querySelector("#viewer-gradcam-alpha"),
    gradcamAlphaLabel: document.querySelector("#viewer-gradcam-alpha-label"),
    previous: document.querySelector("#viewer-previous"),
    next: document.querySelector("#viewer-next"),
    nodules: document.querySelector("#viewer-nodules"),
  };
}

function setViewerStatus(message, kind = "") {
  const {status} = viewerElements();
  status.className = `sub ${kind}`.trim();
  status.textContent = message;
}

function stepSlice(delta) {
  const {slice} = viewerElements();
  const target = Math.min(Number(slice.max), Math.max(0, Number(slice.value) + delta));
  if (target === Number(slice.value)) return;
  slice.value = String(target);
  updateViewer().catch(() => setViewerStatus("The requested CT slice could not be loaded.", "warning"));
}

async function updateViewer() {
  const el = viewerElements();
  const metadata = viewer.metadata;
  if (!metadata) return;
  const maxIndex = Math.max(0, Number(metadata.shape_zyx?.[0] || 1) - 1);
  const index = Math.min(maxIndex, Math.max(0, Number(el.slice.value) || 0));
  const params = new URLSearchParams({
    lung_overlay: String(el.lung.checked && !el.lung.disabled),
    heart_overlay: String(el.heart.checked && !el.heart.disabled),
  });
  const attribute = el.gradcamAttribute.value;
  const camNodules = [...el.gradcamNodules.querySelectorAll("input:checked")].map((box) => box.value);
  if (attribute && camNodules.length) {
    params.set("gradcam_attribute", attribute);
    for (const id of camNodules) params.append("gradcam_nodule_id", id);
    params.set("gradcam_alpha", String(Number(el.gradcamAlpha.value) / 100));
  }
  const currentRequest = ++viewer.requestNumber;
  const response = await apiFetch(`/api/v1/analysis/runs/${encodeURIComponent(viewer.runId)}/viewer/slices/${index}?${params}`);
  if (!response.ok) throw new Error("slice unavailable");
  if (currentRequest !== viewer.requestNumber) return;
  const blob = await response.blob();
  const image = new Image();
  image.onload = () => {
    if (currentRequest !== viewer.requestNumber) return;
    el.canvas.width = image.naturalWidth;
    el.canvas.height = image.naturalHeight;
    const context = el.canvas.getContext("2d");
    context.drawImage(image, 0, 0);
    context.lineWidth = 3;
    context.font = "12px sans-serif";
    for (const nodule of metadata.nodules || []) {
      const start = nodule.geometry?.bbox_start_zyx;
      const end = nodule.geometry?.bbox_end_zyx_exclusive;
      if (!Array.isArray(start) || !Array.isArray(end) || index < start[0] || index >= end[0]) continue;
      const selected = nodule.stable_nodule_id === viewer.selected;
      context.strokeStyle = selected ? "#ffd24a" : "rgba(255, 207, 51, .6)";
      context.strokeRect(start[2], start[1], end[2] - start[2], end[1] - start[1]);
      context.fillStyle = "#ffd24a";
      context.fillText(nodule.stable_nodule_id || "eligible nodule", start[2], Math.max(12, start[1] - 3));
    }
    URL.revokeObjectURL(image.src);
  };
  image.src = URL.createObjectURL(blob);
  el.label.textContent = `${index + 1} of ${maxIndex + 1}`;
  el.previous.disabled = index === 0;
  el.next.disabled = index === maxIndex;
}

function wireViewer() {
  if (viewer.wired) return;
  viewer.wired = true;
  const el = viewerElements();
  const refresh = () => updateViewer().catch(() => setViewerStatus("The requested CT slice could not be loaded.", "warning"));
  el.slice.addEventListener("input", refresh);
  for (const control of [el.lung, el.heart, el.gradcamNodules]) control.addEventListener("change", refresh);
  el.gradcamAttribute.addEventListener("change", () => {
    renderGradcamNodules();
    refresh();
  });
  let alphaTimer = null;
  el.gradcamAlpha.addEventListener("input", () => {
    el.gradcamAlphaLabel.textContent = `${el.gradcamAlpha.value}%`;
    clearTimeout(alphaTimer);
    alphaTimer = setTimeout(refresh, 120);
  });
  el.previous.addEventListener("click", () => stepSlice(-1));
  el.next.addEventListener("click", () => stepSlice(1));
  el.canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    stepSlice(event.deltaY > 0 ? 1 : -1);
  }, {passive: false});
  el.canvas.addEventListener("keydown", (event) => {
    const delta = ["ArrowUp", "ArrowRight"].includes(event.key) ? 1 : ["ArrowDown", "ArrowLeft"].includes(event.key) ? -1 : 0;
    if (!delta) return;
    event.preventDefault();
    stepSlice(delta);
  });
  el.nodules.addEventListener("click", (event) => {
    const button = event.target.closest("[data-nodule-index]");
    if (button) selectNodule(Number(button.dataset.noduleIndex), true);
  });
}

function selectNodule(index, focus = false) {
  const nodule = viewer.metadata?.nodules?.[index];
  if (!nodule) return;
  const {slice, nodules} = viewerElements();
  viewer.selected = nodule.stable_nodule_id;
  nodules.querySelectorAll("button").forEach((button, i) => button.setAttribute("aria-pressed", String(i === index)));
  showNoduleDetails(nodule, focus);
  const center = nodule.geometry?.center_zyx?.[0];
  if (Number.isFinite(center)) slice.value = String(Math.min(Number(slice.max), Math.max(0, Math.round(center))));
  updateViewer().catch(() => setViewerStatus("The requested CT slice could not be loaded.", "warning"));
}

function formatMm(value) {
  return Number.isFinite(value) ? `${value.toFixed(1)} mm` : "unknown";
}

function renderGradcamNodules() {
  const el = viewerElements();
  const attribute = el.gradcamAttribute.value;
  const ids = (viewer.metadata?.gradcam_options || [])
    .filter((option) => option.attribute === attribute)
    .map((option) => option.stable_nodule_id);
  el.gradcamNodules.innerHTML = attribute
    ? ids.map((id) => `<label class="check"><input type="checkbox" value="${escapeHtml(id)}" checked> ${escapeHtml(id)}</label>`).join("")
    : "";
}

function brockLabel(nodule) {
  const brock = nodule.malignancy?.selected_prediction || nodule.malignancy?.brock || {};
  const status = brock.status || "unavailable";
  const kind = ["available", "completed"].includes(status) ? "ok" : status === "failed" ? "bad" : "warn";
  const probability = Number.isFinite(brock.probability) ? ` ${(brock.probability * 100).toFixed(0)}%` : "";
  return badge(`Brock ${status}${probability}`, kind);
}

function brockShapHtml(brock) {
  const items = (brock.feature_contributions || []).filter((item) => Number.isFinite(item.contribution_logit));
  if (!items.length) return '<p class="hint">Brock explanation unavailable for this nodule.</p>';
  const base = items.find((item) => item.feature === "intercept");
  const drivers = items.filter((item) => item.feature !== "intercept" && item.contribution_logit !== 0)
    .sort((a, b) => Math.abs(b.contribution_logit) - Math.abs(a.contribution_logit));
  const zeroCount = items.length - drivers.length - (base ? 1 : 0);
  const scale = Math.max(...drivers.map((item) => Math.abs(item.contribution_logit)), 1e-9);
  const label = (item) => {
    const name = String(item.feature).replaceAll("_", " ");
    return Number.isFinite(item.input_value) ? `${name} = ${item.input_value}${item.unit ? ` ${item.unit}` : ""}` : name;
  };
  const rows = drivers.map((item) => {
    const value = item.contribution_logit;
    const width = Math.abs(value) / scale * 50;
    return `<div class="shap-row"><span>${escapeHtml(label(item))}</span><div class="shap-track"><i class="shap-bar ${value > 0 ? "pos" : "neg"}" style="width:${width}%"></i></div><span class="shap-val">${value > 0 ? "+" : ""}${value.toFixed(2)}</span></div>`;
  }).join("");
  const probability = Number.isFinite(brock.probability) ? `${(brock.probability * 100).toFixed(1)}%` : "unavailable";
  return `<div class="shap" role="img" aria-label="Brock feature contributions to the malignancy log-odds">${rows || "<p class='hint'>All inputs contribute zero.</p>"}</div>
    <div class="shap-key"><span><b style="background:var(--bad)"></b>raises risk</span><span><b style="background:var(--accent)"></b>lowers risk</span></div>
    <p class="hint">Contributions are in log-odds from the published Brock (PanCan) formula, which is additive, so they are exact. Baseline (intercept) ${base ? base.contribution_logit.toFixed(2) : "n/a"}; total log-odds ${Number.isFinite(brock.logit) ? brock.logit.toFixed(2) : "n/a"}; probability ${probability}.${zeroCount > 0 ? ` ${zeroCount} input(s) contribute zero and are not shown.` : ""}</p>`;
}

async function loadViewer(runId) {
  if (viewer.runId === runId && viewer.metadata) return;
  const el = viewerElements();
  wireViewer();
  viewerSection.hidden = false;
  setViewerStatus("Loading viewer metadata...");
  try {
    const response = await apiFetch(`/api/v1/analysis/runs/${encodeURIComponent(runId)}/viewer`);
    const metadata = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(errorMessage(metadata, "Viewer artifacts are unavailable."));
    viewer.runId = runId;
    viewer.metadata = metadata;
    viewer.selected = null;
    const maxIndex = Math.max(0, Number(metadata.shape_zyx?.[0] || 1) - 1);
    el.slice.max = String(maxIndex);
    el.slice.value = String(Math.round(maxIndex / 2));
    el.controls.hidden = false;
    el.lung.disabled = !metadata.overlays?.lung_mask;
    el.heart.disabled = !metadata.overlays?.heart_mask;
    el.lung.checked = false;
    el.heart.checked = false;
    el.gradcamAttribute.innerHTML = '<option value="">Off</option>';
    const attributes = [...new Set((metadata.gradcam_options || []).map((option) => option.attribute))]
      .filter((name) => !HIDDEN_ATTRIBUTES.has(name));
    for (const attribute of attributes) {
      const item = document.createElement("option");
      item.value = attribute;
      item.textContent = attribute.replaceAll("_", " ");
      el.gradcamAttribute.appendChild(item);
    }
    renderGradcamNodules();
    setViewerStatus(`CT volume available: ${metadata.shape_zyx.join(" x ")} voxels. Overlays are shown only when validated artifacts exist.`);
    renderMethods(analysisResult || {}, metadata);
    if (metadata.nodules?.length) {
      el.nodules.innerHTML = metadata.nodules.map((nodule, index) => `<li>
        <button type="button" data-nodule-index="${index}" aria-pressed="false">
          <span><b>${escapeHtml(nodule.stable_nodule_id)}</b> · ${escapeHtml(formatMm(nodule.geometry?.diameter_mm))}</span>
          ${brockLabel(nodule)}
          <span class="sub" style="margin:0;flex-basis:100%">${escapeHtml(nodule.anatomical_location?.region || nodule.anatomical_location?.primary || "Location unavailable")} · quality ${escapeHtml(nodule.quality_status || "unknown")}</span>
        </button></li>`).join("");
      selectNodule(0);
    } else {
      el.nodules.innerHTML = `<li>${metadata.status === "completed"
        ? "No nodule records were included in the validated contract."
        : "Nodule list unavailable until candidate detection completes."}</li>`;
      detailSection.hidden = true;
      await updateViewer();
    }
  } catch (error) {
    viewer.metadata = null;
    el.controls.hidden = true;
    el.nodules.innerHTML = "";
    setViewerStatus(error.message, "warning");
  }
}

function showNoduleDetails(nodule, focus = false) {
  const target = document.querySelector("#viewer-nodule-detail");
  if (!target || !nodule) return;
  const classification = Object.entries(nodule.classification?.attributes || {}).filter(([name]) => !HIDDEN_ATTRIBUTES.has(name));
  const brock = nodule.malignancy?.brock || {};
  const explanations = Object.entries(nodule.gradcam_refs || {}).filter(([name]) => !HIDDEN_ATTRIBUTES.has(name));
  const classificationRows = classification.map(([name, item]) =>
    `<dt>${escapeHtml(name)}</dt><dd>${escapeHtml(item?.expected_rating ?? item?.value ?? "unavailable")} (${escapeHtml(item?.status || "prediction")})</dd>`,
  ).join("");
  const explanationRows = explanations.map(([name, refs]) => {
    const path = refs?.ct_cam_path || refs?.patch_cam_path;
    if (!path) return `<dt>${escapeHtml(name)}</dt><dd>unavailable</dd>`;
    const href = `/api/v1/analysis/runs/${encodeURIComponent(viewer.runId || "")}/artifacts/${path.split("/").map(encodeURIComponent).join("/")}`;
    return `<dt>${escapeHtml(name)}</dt><dd><a href="${href}" target="_blank" rel="noopener">Grad-CAM artifact</a></dd>`;
  }).join("");
  const missing = (brock.features_missing || brock.missing_inputs || []);
  const brockStatus = brock.status || "unavailable";
  const probability = Number.isFinite(brock.probability) ? Math.min(100, Math.max(0, brock.probability * 100)) : null;
  const geometry = nodule.geometry || {};
  const box = Array.isArray(geometry.bbox_start_zyx) && Array.isArray(geometry.bbox_end_zyx_exclusive)
    ? `${geometry.bbox_start_zyx.join(",")} to ${geometry.bbox_end_zyx_exclusive.join(",")} (end exclusive)`
    : "unavailable";
  const mm = Array.isArray(geometry.center_lps_mm) ? geometry.center_lps_mm.map((v) => Number(v).toFixed(1)).join(", ") : "unavailable";
  target.innerHTML = `
    <h3>${escapeHtml(nodule.stable_nodule_id)}</h3>
    <h4>Brock malignancy-risk estimate</h4>
    ${probability === null
      ? `<div class="alert warn">Brock ${escapeHtml(brockStatus)}. Missing inputs: ${escapeHtml(missing.join(", ") || "none reported")}.</div>`
      : `<p><b>${probability.toFixed(0)}%</b> (${escapeHtml(brockStatus)})</p><div class="bar" style="max-width:360px"><i style="width:${probability}%"></i></div>`}
    <h4>Brock SHAP-style explanation</h4>
    ${brockShapHtml(brock)}
    <h4>Attributes (reader-attribute predictions, not diagnosis)</h4>
    <dl>${classificationRows || "<dt>Classification</dt><dd>Unavailable</dd>"}</dl>
    <h4>Explanation artifacts</h4>
    <dl>${explanationRows || "<dt>Grad-CAM</dt><dd>Unavailable</dd>"}</dl>
    <h4>Location</h4>
    <dl>
      <dt>Diameter</dt><dd>${escapeHtml(formatMm(geometry.diameter_mm))}</dd>
      <dt>Detection sources</dt><dd>${escapeHtml((nodule.detection_sources || []).join(", ") || "unavailable")}</dd>
      <dt>Physical LPS (mm)</dt><dd>${escapeHtml(mm)}</dd>
      <dt>Voxel box (z,y,x)</dt><dd>${escapeHtml(box)}</dd>
    </dl>
    <p class="hint">Brock is a risk estimate, not a confirmed diagnosis or pathology result.</p>`;
  detailSection.hidden = false;
  if (focus) detailSection.focus();
}

async function startAnalysis(data, upload) {
  const response = await apiFetch("/api/v1/analysis/runs", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      patient_id: String(data.get("patient_id")).trim(),
      upload_job_id: upload.job_id,
      requested_malignancy_method: "brock",
      data_provenance: "operator_submitted",
      baseline_definition: data.get("baseline_confirmed") ? "initial_pathologic_diagnosis" : "other_or_unverified",
      clinician_inputs: clinicalInputs(data),
    }),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(errorMessage(body, `Analysis could not be started (HTTP ${response.status}).`));
  return body;
}

async function pollAnalysis(runId) {
  for (;;) {
    const response = await apiFetch(`/api/v1/analysis/runs/${encodeURIComponent(runId)}`);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(errorMessage(body, `Analysis status could not be read (HTTP ${response.status}).`));
    analysisResult = body;
    renderResult(body);
    if (TERMINAL.includes(body.status)) return body;
    setRunMessage("Analysis running…");
    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
}

async function persistClinicalRecord(data, {quiet = false} = {}) {
  const patientId = String(data.get("patient_id") || "").trim();
  const response = await apiFetch(`/api/v1/clinical/records/${encodeURIComponent(patientId)}`, {
    method: "PUT",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({clinician_inputs: clinicalInputs(data)}),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(errorMessage(body, `Clinical information could not be saved (HTTP ${response.status}).`));
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
  form.elements.symptoms_status.value = inputs.symptoms_status === "reported" && !(inputs.symptoms || []).length
    ? "no_symptoms"
    : (inputs.symptoms_status ?? "");
  form.elements.kps_unknown.checked = inputs.kps == null;
  form.elements.kps.disabled = inputs.kps == null;
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
  const response = await apiFetch(`/api/v1/clinical/records/${encodeURIComponent(patientId)}`);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(errorMessage(body, `Clinical information could not be loaded (HTTP ${response.status}).`));
  applyClinicalRecord(body.clinician_inputs || {});
  clinicalRecordStatus.className = "status success";
  clinicalRecordStatus.textContent = `Saved clinical information loaded (saved ${body.saved_at}).`;
}

async function restoreStudies() {
  const patientId = String(new FormData(form).get("patient_id") || "").trim();
  if (!patientId) throw new Error("Enter a patient identifier to find saved studies.");
  const response = await apiFetch(`/api/v1/analysis/patients/${encodeURIComponent(patientId)}/studies`);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(errorMessage(body, `Saved studies could not be loaded (HTTP ${response.status}).`));
  const studies = body.studies || [];
  savedStudies.innerHTML = studies.length
    ? studies.map((study) => `<li>
        <button type="button" data-run-id="${escapeHtml(study.run_id)}">
          <span>${escapeHtml(study.run_id)} · ${escapeHtml(study.created_at || "date unknown")}</span>
          <span>${badge(study.status || "unknown", study.status === "completed" ? "ok" : "warn")} ${escapeHtml(study.eligible_nodule_count ?? "unknown")} eligible nodule(s)</span>
        </button></li>`).join("")
    : "<li>No persisted analysis runs found for this patient.</li>";
  savedStudies.querySelectorAll("[data-run-id]").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        const result = await apiFetch(`/api/v1/analysis/runs/${encodeURIComponent(button.dataset.runId)}`);
        const saved = await result.json().catch(() => ({}));
        if (!result.ok) throw new Error(errorMessage(saved, "Saved study could not be reopened."));
        analysisResult = saved;
        detailSection.hidden = true;
        renderResult(saved);
        setRunMessage(`Reopened ${button.dataset.runId} (${saved.status}).`);
        results.focus();
      } catch (error) {
        savedStudies.insertAdjacentHTML("afterbegin", `<li class="err" role="alert">${escapeHtml(error.message)}</li>`);
      }
    });
  });
}

const requiredFields = [
  ["patient_id", "Patient ID"],
  ["age_status", "Age"],
  ["sex", "Sex"],
  ["race", "Race"],
  ["ethnicity", "Ethnicity"],
  ["smoking_status", "Smoking information"],
  ["family_history_lung_cancer", "Family history"],
  ["symptoms_status", "Symptoms"],
];

function setFieldError(name, message) {
  const control = form.elements.namedItem(name);
  if (!control) return;
  const host = control.closest("label") || control.parentElement;
  let note = host.querySelector(`.field-error[data-error-for="${name}"]`);
  if (!note) {
    note = document.createElement("span");
    note.className = "field-error";
    note.dataset.errorFor = name;
    note.id = `error-${name}`;
    host.append(note);
  }
  note.textContent = message;
  control.setAttribute("aria-invalid", message ? "true" : "false");
  if (message) control.setAttribute("aria-describedby", note.id);
  else control.removeAttribute("aria-describedby");
}

// Returns true only when every required field has an explicit answer; each missing field gets an inline message.
function checkRequiredFields(data) {
  let valid = true;
  for (const [name, label] of requiredFields) {
    const hint = name === "patient_id" ? "" : " Choose an option.";
    const message = String(data.get(name) || "").trim() ? "" : `${label} is required.${hint}`;
    setFieldError(name, message);
    valid = valid && !message;
  }
  const ageMessage = data.get("age_status") === "reported" && optionalNumber(data.get("age")) === null
    ? "Enter age in years (18 to 120), or choose Unknown or Not reported."
    : "";
  setFieldError("age", ageMessage);
  const kpsMessage = data.get("kps_unknown") || data.get("kps") !== ""
    ? ""
    : "KPS is required. Enter a value from 0 to 100, or tick KPS unknown.";
  setFieldError("kps", kpsMessage);
  return valid && !ageMessage && !kpsMessage;
}

async function confirmAndUploadCt() {
  const patientId = String(new FormData(form).get("patient_id") || "").trim();
  logEvent("confirm_upload_clicked", {files: selectedFiles.length});
  if (!selectedFiles.length) {
    setStatus("Select a CT folder or DICOM files before uploading.", "error");
    return;
  }
  if (!patientId) {
    setFieldError("patient_id", "Patient ID is required before uploading the CT.");
    form.elements.patient_id.focus();
    setStatus("Enter the Patient ID before uploading the CT.", "error");
    return;
  }
  setFieldError("patient_id", "");
  requestInProgress = true;
  uploadResult = null;
  uploadedPatientId = null;
  uploadProgress.hidden = false;
  uploadProgress.value = 0;
  updateReview();
  try {
    uploadResult = await uploadFiles(selectedFiles, patientId);
    uploadedPatientId = patientId;
    setStatus(`CT uploaded and validated: ${uploadResult.series?.slice_count ?? "unknown"} slices. Review the case, then submit it.`, "success");
    logEvent("upload_completed", {job_id: uploadResult.job_id, status: uploadResult.status});
  } catch (error) {
    logError("upload_failed", error);
    setStatus(`CT upload failed: ${error.message}`, "error");
  } finally {
    requestInProgress = false;
    uploadProgress.hidden = true;
    updateReview();
  }
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

document.querySelector("#select-folder").addEventListener("click", () => {
  logEvent("select_folder_clicked");
  folderInput.click();
});
document.querySelector("#select-files").addEventListener("click", () => {
  logEvent("select_files_clicked");
  filesInput.click();
});
folderInput.addEventListener("change", () => showSelection(folderInput.files));
filesInput.addEventListener("change", () => showSelection(filesInput.files));
confirmUploadButton.addEventListener("click", () => {
  confirmAndUploadCt().catch((error) => logError("confirm_upload_crashed", error));
});
printReportButton.addEventListener("click", () => {
  logEvent("print_report_clicked");
  window.print();
});
form.addEventListener("change", (event) => {
  if (event.target?.name !== "kps_unknown") return;
  form.elements.kps.disabled = event.target.checked;
  if (event.target.checked) form.elements.kps.value = "";
  updateReview();
});
saveClinicalButton.addEventListener("click", async () => {
  try {
    const data = new FormData(form);
    logEvent("save_clinical_clicked");
    if (!checkRequiredFields(data)) {
      clinicalRecordStatus.className = "status error";
      clinicalRecordStatus.textContent = "Complete the highlighted required fields before saving.";
      form.querySelector('[aria-invalid="true"]')?.focus();
      logEvent("save_blocked_missing_fields");
      return;
    }
    validateForm(data, {requireScan: false});
    await persistClinicalRecord(data);
  } catch (error) {
    logError("save_clinical_failed", error);
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
    savedStudies.innerHTML = `<li class="err" role="alert">${escapeHtml(error.message)}</li>`;
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
    logEvent("submit_case_clicked");
    if (!checkRequiredFields(data)) {
      formError.textContent = "Complete the highlighted required fields before submitting.";
      form.querySelector('[aria-invalid="true"]')?.focus();
      logEvent("submit_blocked_missing_fields");
      return;
    }
    validateForm(data);
    if (!uploadResult || uploadedPatientId !== String(data.get("patient_id")).trim()) {
      throw new Error("Confirm and upload the CT scan before submitting the case.");
    }
    requestInProgress = true;
    updateReview();
    results.hidden = true;
    viewerSection.hidden = true;
    detailSection.hidden = true;
    viewer.metadata = null;
    stagesList.innerHTML = "";
    await persistClinicalRecord(data, {quiet: true});
    setRunMessage("Clinical information saved. Starting the analysis run...");
    analysisResult = await startAnalysis(data, uploadResult);
    const finalResult = await pollAnalysis(analysisResult.run_id);
    setRunMessage(finalResult.status === "completed"
      ? "Analysis completed."
      : `Analysis did not complete (${String(finalResult.status).replaceAll("_", " ")}). ${finalResult.message || ""}`.trim());
  } catch (error) {
    logError("case_submission_failed", error);
    setRunMessage(analysisResult ? `Analysis did not complete: ${error.message}` : `Case submission did not complete: ${error.message}`);
    formError.textContent = error.message;
    if (analysisResult?.run_id) renderResult({...analysisResult, status: "failed", message: error.message, error: error.message});
  } finally {
    requestInProgress = false;
    updateReview();
  }
});

updateReview();
