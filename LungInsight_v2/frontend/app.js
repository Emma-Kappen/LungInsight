"use strict";

const $ = (id) => document.getElementById(id);
const form = $("clinical-form");
const imagingApiBase = window.LUNGINSIGHT_IMAGING_API_BASE || "/api/v1/imaging";
const analysisBase = "/api/v1/analysis";

const TERMINAL = ["completed", "failed", "blocked", "awaiting_imaging_review", "interrupted"];
// The full pipeline runs several minutes on CPU; stop waiting after 30 minutes.
const POLL_LIMIT_MS = 30 * 60 * 1000;
const KNOWN_STATUS = /^(analysis_queued|stage\d\d_running|clinical_running|imaging_complete|completed|failed|blocked|awaiting_imaging_review|interrupted)$/;
const PATIENT_ID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;
// Attributes the interface does not present; the backend does not support them yet.
const HIDDEN_ATTRIBUTES = new Set(["internal_structure", "calcification"]);
// Notes the interface never shows (stage progress text and the unreviewed-mask notice), including in older saved runs.
const HIDDEN_NOTES = [/ is running\.?$/i, /not visually reviewed/i];

const S = {
  files: [], upload: null, uploadPid: null, busy: false, xhr: null,
  runId: null, poll: {stop: false}, run: null,
  touched: new Set(), attempted: false,
};
const V = {runId: null, meta: null, nodules: [], selected: -1, request: 0, wired: false, camChecked: new Set()};

/* ---------- utilities ---------- */
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const fin = (v) => typeof v === "number" && Number.isFinite(v);
const label = (text) => String(text).replaceAll("_", " ");
const val = (name) => (form.elements.namedItem(name)?.value ?? "");
const checkedValues = (name) => [...form.querySelectorAll(`[name="${name}"]:checked`)].map((box) => box.value);

function logEvent(event, fields = {}) {
  console.info("[LungInsight]", event, fields);
}

async function apiFetch(url, options = {}) {
  const method = options.method || "GET";
  const started = performance.now();
  let response;
  try {
    response = await fetch(url, options);
  } catch (error) {
    console.error("[LungInsight] network error", method, url, error);
    throw new Error(`Could not reach the server (${method} ${url}). Check that the LungInsight server is still running.`);
  }
  logEvent("api", {method, url, status: response.status, ms: Math.round(performance.now() - started)});
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

const badge = (text, kind) => `<span class="badge b-${kind}">${esc(label(text))}</span>`;

/* ---------- views and windows ---------- */
function setView(name) {
  document.body.dataset.view = name;
  for (const view of ["entry", "report"]) {
    $(`tab-${view}`).setAttribute("aria-selected", String(view === name));
    $(`view-${view}`).hidden = view !== name;
  }
  if (name === "report") resizeCanvas();
}
$("tab-entry").addEventListener("click", () => setView("entry"));
$("tab-report").addEventListener("click", () => setView("report"));

document.addEventListener("click", (event) => {
  const toggle = event.target.closest(".win-t");
  if (!toggle) return;
  const win = toggle.closest(".win");
  const collapsed = win.classList.toggle("collapsed");
  toggle.setAttribute("aria-expanded", String(!collapsed));
  toggle.textContent = collapsed ? "Expand" : "Collapse";
  resizeCanvas();
});

/* ---------- case entry: derived values ---------- */
function ageFromDob(text, now = new Date()) {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(text || "");
  if (!match) return {error: "Enter a valid date of birth."};
  const [year, month, day] = [Number(match[1]), Number(match[2]), Number(match[3])];
  const birth = new Date(Date.UTC(year, month - 1, day));
  if (birth.getUTCFullYear() !== year || birth.getUTCMonth() !== month - 1 || birth.getUTCDate() !== day) {
    return {error: "Enter a valid date of birth."};
  }
  const ty = now.getUTCFullYear(), tm = now.getUTCMonth() + 1, td = now.getUTCDate();
  if (birth.getTime() > Date.UTC(ty, tm - 1, td)) return {error: "Date of birth cannot be in the future."};
  let age = ty - year;
  if (tm < month || (tm === month && td < day)) age -= 1;
  if (age < 18 || age > 110) return {error: `Age ${age} is outside the supported range of 18 to 110 years.`};
  return {age};
}

/* Age for display: stored records carry the date of birth, so derive it when no age is present. */
function ageOfInputs(inputs) {
  if (fin(inputs?.age_at_baseline_years)) return inputs.age_at_baseline_years;
  const derived = inputs?.date_of_birth ? ageFromDob(inputs.date_of_birth) : null;
  return derived && fin(derived.age) ? derived.age : null;
}

function readForm() {
  const number = (name) => {
    const text = val(name).trim();
    return text === "" ? null : Number(text);
  };
  const smoking = val("smoking_status");
  return {
    patientId: val("patient_id").trim(),
    dob: val("dob"),
    sex: val("sex"), race: val("race"), ethnicity: val("ethnicity"),
    stage: val("stage_at_baseline"),
    smoking,
    hasDetail: smoking === "former" || smoking === "current",
    years: number("years_smoked"), packs: number("packs_per_day"), cessation: number("years_since_cessation"),
    family: val("family_history_lung_cancer"),
    history: checkedValues("medical_history"),
    symptoms: checkedValues("symptoms"),
    kps: number("kps"), kpsUnknown: Boolean(form.elements.kps_unknown.checked),
    baselineConfirmed: Boolean(form.elements.baseline_confirmed.checked),
  };
}

function packYears(f) {
  if (f.smoking === "never") return 0;
  if (!f.hasDetail || !fin(f.years) || !fin(f.packs)) return null;
  return Math.round(f.years * f.packs * 100) / 100;
}

function evaluate() {
  const f = readForm();
  const issues = {};
  const required = {};
  const range = (name, value, low, high, text) => {
    if (value !== null && !(fin(value) && value >= low && value <= high)) issues[name] = `${text} must be between ${low} and ${high}.`;
  };
  if (!f.patientId) required.patient_id = "Enter a patient ID.";
  else if (!PATIENT_ID.test(f.patientId)) issues.patient_id = "Use letters, digits, dot, underscore or hyphen (must start with a letter or digit).";
  const age = f.dob ? ageFromDob(f.dob) : null;
  if (!f.dob) required.dob = "Enter the date of birth.";
  else if (age.error) issues.dob = age.error;
  for (const [name, text] of [["sex", "Sex"], ["race", "Race"], ["ethnicity", "Ethnicity"], ["smoking_status", "Smoking status"], ["family_history_lung_cancer", "Family history"]]) {
    const key = {sex: f.sex, race: f.race, ethnicity: f.ethnicity, smoking_status: f.smoking, family_history_lung_cancer: f.family}[name];
    if (!key) required[name] = `${text} is required.`;
  }
  range("years_smoked", f.years, 0, 90, "Years smoked");
  range("packs_per_day", f.packs, 0, 10, "Packs per day");
  range("years_since_cessation", f.cessation, 0, 80, "Years since quitting");
  if (!f.symptoms.length) required.symptoms = "Tick at least one symptom, or “No symptoms”.";
  if (!f.kpsUnknown) {
    if (f.kps === null) required.kps = "Enter KPS (0 to 100), or tick KPS unknown.";
    else if (!(fin(f.kps) && f.kps >= 0 && f.kps <= 100 && f.kps % 10 === 0)) issues.kps = "KPS must be 0 to 100 in steps of 10.";
  }
  const ctOk = Boolean(S.upload && S.uploadPid === f.patientId && f.patientId);
  const items = [
    ["Patient ID", Boolean(f.patientId) && !issues.patient_id],
    ["CT uploaded", ctOk],
    ["Date of birth", Boolean(f.dob) && !issues.dob],
    ["Sex", Boolean(f.sex)],
    ["Race", Boolean(f.race)],
    ["Ethnicity", Boolean(f.ethnicity)],
    ["Smoking status", Boolean(f.smoking)],
    ["Family history", Boolean(f.family)],
    ["Symptoms", f.symptoms.length > 0],
    ["KPS or KPS unknown", f.kpsUnknown || (f.kps !== null && !issues.kps)],
  ];
  const clinicalReady = items.filter(([name]) => name !== "CT uploaded").every(([, ok]) => ok) && Object.keys(issues).length === 0;
  return {f, issues, required, items, age, clinicalReady, ready: clinicalReady && ctOk};
}

/* ---------- case entry: rendering ---------- */
function setBadge(id, state, text) {
  const el = $(`sb-${id}`);
  el.className = `sb ${state}`;
  el.textContent = text;
}

function showFieldError(name, message, control) {
  const el = $(`${name}-e`);
  if (el) el.textContent = message;
  if (control) {
    control.setAttribute("aria-invalid", message ? "true" : "false");
  }
}

function syncSmokingFields(f) {
  for (const [name, enabled] of [["years_smoked", f.hasDetail], ["packs_per_day", f.hasDetail], ["years_since_cessation", f.smoking === "former"]]) {
    const input = form.elements.namedItem(name);
    input.disabled = !enabled;
    if (!enabled && input.value !== "") input.value = "";
  }
}

function refresh() {
  syncSmokingFields(readForm());
  const e = evaluate();
  const {f} = e;
  $("age-out").textContent = e.age && !e.age.error ? `${e.age.age} years` : "Unavailable";
  const py = packYears(f);
  $("pack-years").textContent = py === null ? "Unavailable" : `${py.toFixed(1)} pack-years`;

  setBadge("dob", e.age && !e.age.error ? "ok" : "warn", e.age && !e.age.error ? "Reported" : "Not reported");
  for (const [id, enabled, value, issue] of [
    ["years_smoked", f.hasDetail, f.years, e.issues.years_smoked],
    ["packs_per_day", f.hasDetail, f.packs, e.issues.packs_per_day],
    ["years_since_cessation", f.smoking === "former", f.cessation, e.issues.years_since_cessation],
  ]) {
    if (!enabled) setBadge(id, "na", "Not applicable");
    else if (value !== null && !issue) setBadge(id, "ok", "Reported");
    else setBadge(id, "warn", "Not reported");
  }
  setBadge("medical_history", f.history.length ? "ok" : "warn", f.history.length ? "Reported" : "Not reported");
  setBadge("symptoms", f.symptoms.length ? "ok" : "warn", f.symptoms.length ? "Reported" : "Not reported");
  const kpsReported = !f.kpsUnknown && f.kps !== null && !e.issues.kps;
  setBadge("kps", kpsReported ? "ok" : "warn", kpsReported ? "Reported" : "Not reported");
  form.elements.kps.disabled = f.kpsUnknown;
  if (f.kpsUnknown && form.elements.kps.value !== "") form.elements.kps.value = "";

  const fieldControls = {
    patient_id: form.elements.patient_id, dob: form.elements.dob, sex: form.elements.sex, race: form.elements.race,
    ethnicity: form.elements.ethnicity, smoking_status: form.elements.smoking_status,
    years_smoked: form.elements.years_smoked, packs_per_day: form.elements.packs_per_day,
    years_since_cessation: form.elements.years_since_cessation,
    family_history_lung_cancer: form.elements.family_history_lung_cancer, kps: form.elements.kps,
    symptoms: $("symptoms-group"),
  };
  for (const [name, control] of Object.entries(fieldControls)) {
    const shown = S.attempted || S.touched.has(name);
    showFieldError(name, shown ? (e.issues[name] || e.required[name] || "") : "", control);
  }

  $("checklist").innerHTML = e.items.map(([name, ok]) => `<li><span>${esc(name)}</span><span class="${ok ? "done" : "missing"}">${ok ? "Done" : "Missing"}</span></li>`).join("");
  const done = e.items.filter(([, ok]) => ok).length;
  $("checklist-count").textContent = `${done} of ${e.items.length} required items complete.`;

  const hasFiles = S.files.length > 0;
  const alreadyUploaded = Boolean(S.upload && S.uploadPid === f.patientId);
  $("confirm-upload").disabled = S.busy || !hasFiles || !f.patientId || !!e.issues.patient_id || alreadyUploaded;
  $("confirm-upload").textContent = alreadyUploaded ? "CT uploaded" : "Confirm and upload CT";
  $("analyse-button").disabled = S.busy || !e.ready;
}

/* Exclusive "none" boxes */
form.addEventListener("change", (event) => {
  const box = event.target;
  if (box.type === "checkbox" && (box.name === "medical_history" || box.name === "symptoms") && box.checked) {
    form.querySelectorAll(`[name="${box.name}"]`).forEach((other) => {
      if (other !== box && ((box.value === "none") || other.value === "none")) other.checked = false;
    });
  }
  S.touched.add(box.name || box.id);
  refresh();
});
form.addEventListener("input", (event) => {
  if (event.target.name === "patient_id") invalidateUpload();
  refresh();
});
form.addEventListener("focusout", (event) => {
  const name = event.target.name || event.target.id;
  if (name) S.touched.add(name);
  refresh();
});

function invalidateUpload() {
  const pid = val("patient_id").trim();
  if (S.upload && S.uploadPid !== pid) {
    S.upload = null;
    $("upload-status").className = "status warning";
    $("upload-status").textContent = "Patient ID changed. Upload the CT again for this patient.";
  }
}

/* ---------- CT selection and upload ---------- */
const formatBytes = (bytes) => (bytes < 1048576 ? `${(bytes / 1024).toFixed(1)} KB` : `${(bytes / 1048576).toFixed(1)} MB`);

function setUploadStatus(message, kind = "") {
  $("upload-status").className = `status ${kind}`.trim();
  $("upload-status").textContent = message;
}

function showSelection(files) {
  S.files = Array.from(files || []);
  S.upload = null;
  S.uploadPid = null;
  if (!S.files.length) {
    $("scan-selection").textContent = "No CT scan selected. Not uploaded yet.";
    $("remove-scan").hidden = true;
    setUploadStatus("");
  } else {
    const total = S.files.reduce((sum, file) => sum + file.size, 0);
    const first = S.files[0].webkitRelativePath || S.files[0].name;
    const folder = first.includes("/") ? `${first.split("/")[0]}: ` : "";
    $("scan-selection").textContent = `${folder}${S.files.length} file(s), ${formatBytes(total)}. Selected; not uploaded yet.`;
    $("remove-scan").hidden = false;
    setUploadStatus("Review the selection, then confirm the upload.");
  }
  refresh();
}

function uploadFiles(files, patientId) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    S.xhr = request;
    request.open("POST", `${imagingApiBase}/dicom/upload`);
    request.responseType = "json";
    request.timeout = 300000;
    request.upload.onprogress = (event) => {
      if (!event.lengthComputable) return;
      const percent = Math.round(event.loaded / event.total * 100);
      $("upload-progress").value = percent;
      setUploadStatus(`Uploading CT: ${percent}% (${formatBytes(event.loaded)} of ${formatBytes(event.total)})`);
    };
    request.upload.onload = () => {
      $("upload-progress").value = 100;
      setUploadStatus("Upload received. The server is validating and reconstructing the CT series; large scans can take a minute.");
    };
    request.onerror = () => reject(new Error("The CT upload could not reach the server. Check that the LungInsight server is still running."));
    request.ontimeout = () => reject(new Error("The CT upload timed out. Try the scan again."));
    request.onabort = () => reject(new DOMException("Cancelled", "AbortError"));
    request.onload = () => {
      const body = request.response || (() => { try { return JSON.parse(request.responseText); } catch { return {}; } })();
      logEvent("upload", {status: request.status});
      if (request.status < 200 || request.status >= 300) {
        reject(new Error(errorMessage(body, `Upload failed (HTTP ${request.status}).`)));
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

async function confirmAndUpload() {
  const patientId = val("patient_id").trim();
  if (!S.files.length) return setUploadStatus("Select a CT folder or DICOM files first.", "error");
  if (!PATIENT_ID.test(patientId)) {
    S.touched.add("patient_id");
    refresh();
    form.elements.patient_id.focus();
    return setUploadStatus("Enter a valid Patient ID before uploading.", "error");
  }
  S.busy = true;
  S.upload = null;
  $("upload-progress").hidden = false;
  $("upload-progress").value = 0;
  $("cancel-upload").hidden = false;
  refresh();
  try {
    const body = await uploadFiles(S.files, patientId);
    S.upload = body;
    S.uploadPid = patientId;
    setUploadStatus(`CT uploaded and validated: ${body.series?.slice_count ?? "unknown"} slices${body.series?.volume_shape_zyx ? `, volume ${body.series.volume_shape_zyx.join(" × ")}` : ""}.`, "success");
  } catch (error) {
    setUploadStatus(error.name === "AbortError" ? "Upload cancelled." : `CT upload failed: ${error.message}`, error.name === "AbortError" ? "warning" : "error");
  } finally {
    S.busy = false;
    S.xhr = null;
    $("upload-progress").hidden = true;
    $("cancel-upload").hidden = true;
    refresh();
  }
}

$("select-folder").addEventListener("click", () => $("dicom-folder").click());
$("select-files").addEventListener("click", () => $("dicom-files").click());
$("dicom-folder").addEventListener("change", (event) => showSelection(event.target.files));
$("dicom-files").addEventListener("change", (event) => showSelection(event.target.files));
$("confirm-upload").addEventListener("click", () => confirmAndUpload());
$("cancel-upload").addEventListener("click", () => S.xhr?.abort());
$("remove-scan").addEventListener("click", () => {
  $("dicom-folder").value = "";
  $("dicom-files").value = "";
  showSelection([]);
});

/* ---------- clinical payload, save/load ---------- */
function clinicalInputs(f) {
  const hasHistory = f.history.length > 0;
  return {
    date_of_birth: f.dob || null,
    sex_at_baseline: f.sex,
    stage_at_baseline: f.stage || null,
    race_at_baseline: f.race,
    ethnicity_at_baseline: f.ethnicity,
    smoking_status: f.smoking,
    years_smoked: f.hasDetail ? f.years : null,
    years_smoked_status: f.hasDetail && f.years !== null ? "reported" : "not_reported",
    packs_per_day: f.hasDetail ? f.packs : null,
    packs_per_day_status: f.hasDetail && f.packs !== null ? "reported" : "not_reported",
    pack_years: packYears(f),
    years_since_cessation: f.smoking === "former" ? f.cessation : null,
    family_history_lung_cancer: f.family,
    medical_history_status: hasHistory ? "reported" : "not_reported",
    medical_history_conditions: f.history.filter((item) => item !== "none"),
    symptoms_status: "reported",
    symptoms: f.symptoms.filter((item) => item !== "none"),
    kps: f.kpsUnknown ? null : f.kps,
  };
}

function setRecordStatus(message, kind = "") {
  $("clinical-record-status").className = `status ${kind}`.trim();
  $("clinical-record-status").textContent = message;
}

async function persistClinicalRecord(inputs, patientId) {
  const response = await apiFetch(`/api/v1/clinical/records/${encodeURIComponent(patientId)}`, {
    method: "PUT",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({clinician_inputs: inputs}),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(errorMessage(body, `Clinical information could not be saved (HTTP ${response.status}).`));
  return body;
}

$("save-clinical-record").addEventListener("click", async () => {
  S.attempted = true;
  const e = evaluate();
  refresh();
  if (!e.clinicalReady) return setRecordStatus("Complete the highlighted required fields before saving.", "error");
  try {
    await persistClinicalRecord(clinicalInputs(e.f), e.f.patientId);
    setRecordStatus(`Clinical information saved for ${e.f.patientId}. The date of birth is stored; age is calculated from it when used.`, "success");
  } catch (error) {
    setRecordStatus(error.message, "error");
  }
});

function applyClinicalRecord(inputs) {
  const set = (name, value) => { form.elements.namedItem(name).value = value == null ? "" : String(value); };
  set("dob", inputs.date_of_birth);
  set("sex", inputs.sex_at_baseline ?? (inputs.sex_at_baseline_status ? "not_reported" : ""));
  set("stage_at_baseline", inputs.stage_at_baseline ?? (inputs.stage_at_baseline_status === "unknown" ? "unknown" : ""));
  set("race", inputs.race_at_baseline);
  set("ethnicity", inputs.ethnicity_at_baseline);
  set("smoking_status", inputs.smoking_status);
  set("family_history_lung_cancer", inputs.family_history_lung_cancer);
  syncSmokingFields(readForm());
  set("years_smoked", inputs.years_smoked);
  set("packs_per_day", inputs.packs_per_day);
  set("years_since_cessation", inputs.years_since_cessation);
  set("kps", inputs.kps);
  form.elements.kps_unknown.checked = inputs.kps == null;
  const lists = [
    ["medical_history", inputs.medical_history_status === "reported" ? (inputs.medical_history_conditions?.length ? inputs.medical_history_conditions : ["none"]) : []],
    ["symptoms", inputs.symptoms_status === "reported" ? (inputs.symptoms?.length ? inputs.symptoms : ["none"]) : []],
  ];
  for (const [name, values] of lists) {
    form.querySelectorAll(`[name="${name}"]`).forEach((box) => { box.checked = values.includes(box.value); });
  }
  syncSmokingFields(readForm());
  set("years_smoked", inputs.years_smoked);
  set("packs_per_day", inputs.packs_per_day);
  set("years_since_cessation", inputs.years_since_cessation);
  refresh();
}

$("load-clinical-record").addEventListener("click", async () => {
  const patientId = val("patient_id").trim();
  if (!patientId) { S.touched.add("patient_id"); refresh(); return setRecordStatus("Enter a patient ID to load its saved clinical information.", "error"); }
  try {
    const response = await apiFetch(`/api/v1/clinical/records/${encodeURIComponent(patientId)}`);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(errorMessage(body, `Clinical information could not be loaded (HTTP ${response.status}).`));
    applyClinicalRecord(body.clinician_inputs || {});
    const dob = body.clinician_inputs?.date_of_birth;
    setRecordStatus(dob
      ? `Saved clinical information loaded (saved ${body.saved_at}), including the date of birth.`
      : `Saved clinical information loaded (saved ${body.saved_at}). This older record has no date of birth${fin(body.clinician_inputs?.age_at_baseline_years) ? ` (saved age ${body.clinician_inputs.age_at_baseline_years})` : ""}: please enter it.`, "success");
  } catch (error) {
    setRecordStatus(error.message, "error");
  }
});

/* ---------- saved studies ---------- */
$("load-studies").addEventListener("click", async () => {
  const patientId = val("patient_id").trim();
  const list = $("saved-studies");
  if (!patientId) { S.touched.add("patient_id"); refresh(); list.innerHTML = ""; return setRecordStatus("Enter a patient ID to find saved studies.", "error"); }
  try {
    const response = await apiFetch(`${analysisBase}/patients/${encodeURIComponent(patientId)}/studies`);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(errorMessage(body, `Saved studies could not be loaded (HTTP ${response.status}).`));
    const studies = body.studies || [];
    setRecordStatus(studies.length ? `${studies.length} saved study(ies) for ${patientId}. Select one to open its report.` : `No saved studies for ${patientId}.`);
    list.innerHTML = studies.map((study) => `<li><button type="button" data-run-id="${esc(study.run_id)}">
      <span>${esc(study.run_id)} · ${esc(study.created_at || "date unknown")}</span>
      <span>${badge(study.status || "unknown", study.status === "completed" ? "ok" : "warn")} ${esc(study.eligible_nodule_count ?? "unknown")} eligible nodule(s)</span></button></li>`).join("");
  } catch (error) {
    list.innerHTML = "";
    setRecordStatus(error.message, "error");
  }
});

$("saved-studies").addEventListener("click", async (event) => {
  const button = event.target.closest("[data-run-id]");
  if (!button) return;
  try {
    await openRun(button.dataset.runId);
  } catch (error) {
    setRecordStatus(error.message, "error");
  }
});

/* ---------- analysis run ---------- */
const STAGE_LABELS = [
  ["stage01", "CT conversion, lung masks, crop"],
  ["stage02", "Nodule detection and patches"],
  ["stage03", "Attribute ranking"],
  ["stage04", "Brock malignancy risk"],
  ["stage05", "Grad-CAM explanations"],
  ["stage06", "Full-CT Grad-CAM projection"],
  ["stage07", "Report and viewer artifacts"],
  ["clinical", "Clinical models"],
];

function renderStages(body) {
  const completed = new Set(body.completed_stages || []);
  const failed = ["failed", "blocked", "interrupted"].includes(body.status);
  $("stages").innerHTML = STAGE_LABELS.map(([key, text]) => {
    let state = "queued";
    if (completed.has(key) || (body.status === "completed" && key === "clinical")) state = "completed";
    else if (body.current_stage === key) state = failed ? body.status : "running";
    const kind = {completed: "ok", running: "warn", queued: "mute"}[state] || "bad";
    return `<li><span>${esc(text)}</span>${badge(state, kind)}</li>`;
  }).join("");
}

const setRunMessage = (message) => { $("run-message").textContent = message; };

async function startAnalysis(f, inputs) {
  const response = await apiFetch(analysisBase + "/runs", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      patient_id: f.patientId,
      upload_job_id: S.upload.job_id,
      requested_malignancy_method: "brock",
      data_provenance: "operator_submitted",
      baseline_definition: f.baselineConfirmed ? "initial_pathologic_diagnosis" : "other_or_unverified",
      clinician_inputs: inputs,
    }),
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(errorMessage(body, `Analysis could not be started (HTTP ${response.status}).`));
  return body;
}

async function pollRun(runId) {
  S.poll = {stop: false};
  const mine = S.poll;
  const startedAt = Date.now();
  $("stop-waiting").hidden = false;
  $("retry-run").hidden = true;
  try {
    while (!mine.stop) {
      if (Date.now() - startedAt > POLL_LIMIT_MS) throw new Error("Timed out waiting for the run to finish. It may still be running on the server; use Retry to check again.");
      const response = await apiFetch(`${analysisBase}/runs/${encodeURIComponent(runId)}`);
      const body = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(errorMessage(body, `Analysis status could not be read (HTTP ${response.status}).`));
      renderStages(body);
      if (!KNOWN_STATUS.test(String(body.status))) throw new Error(`Unknown status “${body.status}” from the server.`);
      if (TERMINAL.includes(body.status)) return body;
      setRunMessage("Analysis running. This page checks every 1.5 seconds.");
      await sleep(1500);
    }
    setRunMessage("Stopped waiting. The run may still be going on the server; use Retry to check again.");
    $("retry-run").hidden = false;
    return null;
  } finally {
    $("stop-waiting").hidden = true;
  }
}

async function finishRun(runId) {
  const body = await pollRun(runId);
  if (!body) return;
  if (body.status === "awaiting_imaging_review") {
    setRunMessage("Awaiting imaging review. Results appear once the review is done.");
    return;
  }
  setRunMessage(body.status === "completed" ? "Analysis completed." : `Analysis ${label(body.status)}. ${body.message || body.error || ""}`.trim());
  await openRun(runId);
}

async function submitCase() {
  if (S.busy) return;
  S.attempted = true;
  const e = evaluate();
  refresh();
  $("form-error").textContent = "";
  if (!e.ready) {
    $("form-error").textContent = "Complete the required items marked Missing before submitting.";
    form.querySelector('[aria-invalid="true"]')?.focus();
    return;
  }
  S.busy = true;
  S.runId = null;
  $("retry-run").hidden = true;
  $("stages").innerHTML = "";
  refresh();
  try {
    setRunMessage("Saving clinical information…");
    const inputs = clinicalInputs(e.f);
    await persistClinicalRecord(inputs, e.f.patientId);
    setRunMessage("Starting the analysis run…");
    const run = await startAnalysis(e.f, inputs);
    S.runId = run.run_id;
    await finishRun(run.run_id);
  } catch (error) {
    $("form-error").textContent = error.message;
    setRunMessage(`Case submission did not complete: ${error.message}`);
    $("retry-run").hidden = false;
  } finally {
    S.busy = false;
    refresh();
  }
}

async function retryRun() {
  if (S.busy) return;
  $("form-error").textContent = "";
  if (S.runId) {
    S.busy = true;
    refresh();
    try {
      await finishRun(S.runId);
    } catch (error) {
      $("form-error").textContent = error.message;
      $("retry-run").hidden = false;
    } finally {
      S.busy = false;
      refresh();
    }
  } else {
    await submitCase();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitCase().catch((error) => { $("form-error").textContent = error.message; });
});
$("stop-waiting").addEventListener("click", () => { S.poll.stop = true; });
$("retry-run").addEventListener("click", () => retryRun());

/* ---------- report: data loading ---------- */
async function openRun(runId) {
  const response = await apiFetch(`${analysisBase}/runs/${encodeURIComponent(runId)}`);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(errorMessage(body, "Saved study could not be reopened."));
  let meta = null;
  let metaError = null;
  try {
    const viewerResponse = await apiFetch(`${analysisBase}/runs/${encodeURIComponent(runId)}/viewer`);
    const viewerBody = await viewerResponse.json().catch(() => ({}));
    if (viewerResponse.ok) meta = viewerBody;
    else metaError = errorMessage(viewerBody, "Viewer artifacts are unavailable for this run.");
  } catch (error) {
    metaError = error.message;
  }
  let inputs = null;
  try {
    const reportResponse = await apiFetch(`${analysisBase}/runs/${encodeURIComponent(runId)}/report`);
    if (reportResponse.ok) inputs = (await reportResponse.json()).clinical_inputs || null;
    if (!inputs && body.patient_id) {
      const recordResponse = await apiFetch(`/api/v1/clinical/records/${encodeURIComponent(body.patient_id)}`);
      if (recordResponse.ok) inputs = (await recordResponse.json()).clinician_inputs || null;
    }
  } catch {
    inputs = null;
  }
  S.run = {body, meta, metaError, inputs};
  renderReport();
  setView("report");
}

/* ---------- report: helpers ---------- */
function validateBox(nodule, shape) {
  if (!nodule || typeof nodule.stable_nodule_id !== "string" || !nodule.stable_nodule_id) return "Missing stable nodule ID";
  const start = nodule.geometry?.bbox_start_zyx;
  const end = nodule.geometry?.bbox_end_zyx_exclusive;
  if (!Array.isArray(start) || !Array.isArray(end) || start.length !== 3 || end.length !== 3) return "Box coordinates missing";
  if (![...start, ...end].every(fin)) return "Box has non-finite values";
  if (!start.every((v, i) => end[i] > v)) return "Box has zero or negative size";
  if (start.some((v) => v < 0) || !Array.isArray(shape) || shape.length !== 3 || end.some((v, i) => v > shape[i])) return "Box outside image bounds";
  return null;
}

const MALIGNANCY_THRESHOLD = 0.10; // BTS-style 10% Brock threshold: a display setting, not a diagnosis
const manualBrockOf = (nodule) => (nodule.manual_brock?.brock?.status === "available" ? nodule.manual_brock.brock : null);
const contractBrockOf = (nodule) => nodule.malignancy?.brock || {};
const brockIsManual = (nodule) => contractBrockOf(nodule).status !== "available" && Boolean(manualBrockOf(nodule));
const brockOf = (nodule) => (brockIsManual(nodule) ? manualBrockOf(nodule) : contractBrockOf(nodule));
const brockUsable = (b) => b.status === "available" && fin(b.probability);

function assessment(nodule) {
  const b = brockOf(nodule);
  if (!brockUsable(b)) return {state: "unavailable", label: "Not assessed"};
  return b.probability >= MALIGNANCY_THRESHOLD ? {state: "elevated", label: "Elevated", p: b.probability} : {state: "unlikely", label: "Unlikely", p: b.probability};
}

/* Detector box grown to the nodule's CT extent by the backend; falls back to the detector box. */
function boxOf(nodule) {
  const refined = nodule.geometry?.bbox_refined;
  if (refined && Array.isArray(refined.start) && Array.isArray(refined.end)) {
    return {start: refined.start, end: refined.end, slices: refined.slice_boxes || {}, method: refined.method};
  }
  return {start: nodule.geometry?.bbox_start_zyx, end: nodule.geometry?.bbox_end_zyx_exclusive, slices: {}, method: "detector_box"};
}
const contributions = (b) => Object.fromEntries((b.feature_contributions || []).map((item) => [item.feature, item]));
const pct = (p) => `${(p * 100).toFixed(p < 0.1 ? 1 : 0)}%`;
const formatMm = (v) => (fin(v) ? `${v.toFixed(1)} mm` : "Unavailable");
const flagText = (item, yes, no) => (item && fin(item.encoded_value) ? (item.encoded_value === 1 ? yes : no) : null);

const BROCK_PARAMS = [
  {key: "age", name: "Age", source: "Case entry (date of birth)", missing: ["age_years"], enters: "0.0287 × (age − 62)",
    value: (c) => (fin(c.age_years?.input_value) ? `${c.age_years.input_value} years` : null),
    fallback: (x) => (fin(ageOfInputs(x.inputs)) ? `${ageOfInputs(x.inputs)} years` : null),
    contrib: (c) => c.age_years?.contribution_logit},
  {key: "sex", name: "Sex", source: "Case entry", missing: ["sex"], enters: "0.6011 × sex (female = 1, male = 0)",
    value: (c) => flagText(c.female, "Female", "Male"),
    fallback: (x) => ({female: "Female", male: "Male"}[x.inputs?.sex_at_baseline] ?? null),
    contrib: (c) => c.female?.contribution_logit},
  {key: "fh", name: "Family history", source: "Case entry", missing: ["family_history_lung_cancer"], enters: "0.2961 × family history (yes = 1)",
    value: (c) => flagText(c.family_history_lung_cancer, "Yes", "No"),
    fallback: (x) => ({yes: "Yes", no: "No"}[x.inputs?.family_history_lung_cancer] ?? null),
    contrib: (c) => c.family_history_lung_cancer?.contribution_logit},
  {key: "emph", name: "Emphysema", source: "Case entry (medical history)", missing: ["emphysema"], enters: "0.2953 × emphysema (yes = 1)",
    value: (c) => flagText(c.emphysema, "Yes", "No"),
    fallback: (x) => (x.inputs?.medical_history_status === "reported" ? ((x.inputs.medical_history_conditions || []).includes("emphysema") ? "Yes" : "No") : null),
    contrib: (c) => c.emphysema?.contribution_logit},
  {key: "size", name: "Nodule size", source: "Imaging (detector)", missing: ["nodule_size_mm"], enters: "−5.3854 × ((diameter/10)^−0.5 − 1.58113883)",
    value: (c) => (fin(c.nodule_size_mm?.input_value) ? `${c.nodule_size_mm.input_value.toFixed(1)} mm` : null),
    fallback: (x) => (fin(x.nodule.geometry?.diameter_mm) ? `${x.nodule.geometry.diameter_mm.toFixed(1)} mm` : null),
    contrib: (c) => c.nodule_size_mm?.contribution_logit},
  {key: "type", name: "Nodule type", source: "Attribute models (texture)", missing: ["nodule_type"], enters: "Solid 0 (reference), part-solid +0.3770, non-solid −0.1276",
    value: (c) => (c.pure_ground_glass && c.part_solid ? (c.part_solid.encoded_value === 1 ? "Part-solid" : c.pure_ground_glass.encoded_value === 1 ? "Nonsolid (GGO)" : "Solid") : null),
    contrib: (c) => (c.pure_ground_glass || c.part_solid ? (c.pure_ground_glass?.contribution_logit ?? 0) + (c.part_solid?.contribution_logit ?? 0) : undefined)},
  {key: "lobe", name: "Upper-lobe location", source: "Imaging (location)", missing: ["upper_lobe"], enters: "0.6581 × upper lobe (yes = 1)",
    value: (c) => flagText(c.upper_lobe, "Upper lobe", "Not upper lobe"), contrib: (c) => c.upper_lobe?.contribution_logit},
  {key: "count", name: "Nodule count", source: "Imaging (valid boxes)", missing: ["nodule_count"], enters: "−0.0824 × (nodule count − 4)",
    value: (c) => (fin(c.nodule_count?.input_value) ? String(c.nodule_count.input_value) : null),
    fallback: (x) => String(x.nodules.filter((n) => !n._boxErr).length),
    contrib: (c) => c.nodule_count?.contribution_logit},
  {key: "spic", name: "Spiculation", source: "Attribute models", missing: ["spiculation"], enters: "0.7729 × spiculation (yes = 1)",
    value: (c) => flagText(c.spiculation, "Present", "Absent"), contrib: (c) => c.spiculation?.contribution_logit},
];

const MISSING_LABELS = {
  age_years: "Age (date of birth)", sex: "Sex", family_history_lung_cancer: "Family history", emphysema: "Emphysema (medical history)",
  nodule_size_mm: "Nodule size", upper_lobe: "Upper-lobe location", spiculation: "Spiculation", nodule_type: "Nodule type", nodule_count: "Nodule count",
};

function missingBrockInputs(nodule) {
  if (brockIsManual(nodule)) return [];
  const b = contractBrockOf(nodule);
  const names = new Set([...(b.features_missing || []), ...(b.missing_inputs || []), ...(nodule.input_completeness?.fields_missing || [])]);
  return [...names];
}

/* Controls for supplying a missing Brock input by hand (posted to the backend, which refuses to overwrite known values). */
const MANUAL_CONTROLS = {
  upper_lobe: {label: "Upper-lobe location", options: [["", "Select"], ["true", "Upper lobe"], ["false", "Not upper lobe"]], parse: (v) => v === "true"},
  emphysema: {label: "Emphysema", options: [["", "Select"], ["true", "Yes"], ["false", "No"]], parse: (v) => v === "true"},
  family_history_lung_cancer: {label: "Family history", options: [["", "Select"], ["true", "Yes"], ["false", "No"]], parse: (v) => v === "true"},
  spiculation: {label: "Spiculation", options: [["", "Select"], ["1", "Present"], ["0", "Absent"]], parse: (v) => Number(v)},
  nodule_type: {label: "Nodule type", options: [["", "Select"], ["pure_ground_glass", "Nonsolid (GGO)"], ["part_solid", "Part-solid"], ["solid", "Solid"]], parse: (v) => v},
  sex: {label: "Sex", options: [["", "Select"], ["female", "Female"], ["male", "Male"]], parse: (v) => v},
  nodule_size_mm: {label: "Nodule size (mm)", number: {min: 1, max: 100, step: 0.1}, parse: (v) => Number(v)},
  age_years: {label: "Age (years)", number: {min: 18, max: 120, step: 1}, parse: (v) => Number(v)},
  nodule_count: {label: "Nodule count", number: {min: 1, max: 20, step: 1}, parse: (v) => Number(v)},
};

function manualFormHtml(nodule, missingNames) {
  const fields = missingNames.filter((name) => MANUAL_CONTROLS[name]);
  if (!fields.length) return "";
  const controls = fields.map((name) => {
    const def = MANUAL_CONTROLS[name];
    const control = def.number
      ? `<input type="number" data-manual="${name}" min="${def.number.min}" max="${def.number.max}" step="${def.number.step}">`
      : `<select data-manual="${name}">${def.options.map(([value, text]) => `<option value="${value}">${esc(text)}</option>`).join("")}</select>`;
    return `<label class="manual-row"><span>${esc(def.label)}</span>${control}</label>`;
  }).join("");
  return `<form class="manual" id="manual-brock-form" data-nodule="${esc(nodule.stable_nodule_id)}" novalidate>
    <b>Supply the missing input${fields.length > 1 ? "s" : ""}</b>
    <p class="hint" style="margin:.2rem 0 .4rem">Entered by you, not derived by the pipeline. Brock is then calculated with the same published formula and labelled as clinician-supplied. Inputs that already have a known value cannot be changed here.</p>
    ${controls}
    <div class="btns" style="margin-top:.4rem"><button type="submit">Calculate Brock with these values</button></div>
    <p class="err" id="manual-brock-error" role="alert"></p></form>`;
}

async function submitManualBrock(formEl) {
  const nodule = V.nodules.find((n) => n.stable_nodule_id === formEl.dataset.nodule);
  const error = formEl.querySelector("#manual-brock-error");
  error.textContent = "";
  const values = {};
  for (const control of formEl.querySelectorAll("[data-manual]")) {
    if (control.value === "") { error.textContent = "Fill in every listed input."; control.focus(); return; }
    values[control.dataset.manual] = MANUAL_CONTROLS[control.dataset.manual].parse(control.value);
  }
  const button = formEl.querySelector("button[type=submit]");
  button.disabled = true;
  try {
    const response = await apiFetch(`${analysisBase}/runs/${encodeURIComponent(V.runId)}/nodules/${encodeURIComponent(nodule.stable_nodule_id)}/brock`, {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({values}),
    });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(errorMessage(body, `Brock could not be calculated (HTTP ${response.status}).`));
    nodule.manual_brock = body;
    renderNoduleTable();
    renderBrock();
    renderShap();
    renderMethods();
    renderSummary();
  } catch (failure) {
    error.textContent = failure.message;
    button.disabled = false;
  }
}

/* Survival curve as an SVG line graph (years on x, survival probability on y). */
function survivalChartHtml(model) {
  const curve = model?.prediction?.survival_function;
  const days = curve?.time_days, prob = curve?.probability;
  if (!Array.isArray(days) || !Array.isArray(prob) || days.length < 2 || days.length !== prob.length) return "";
  const W = 480, H = 230, L = 42, R = 14, T = 12, B = 38;
  const horizons = Object.entries(model.prediction.probability_at_horizons_days || {}).map(([d, p]) => [Number(d) / 365.25, p]);
  const maxYears = Math.max(...days) / 365.25;
  const xMax = Math.max(1, Math.ceil(maxYears));
  const x = (years) => L + (years / xMax) * (W - L - R);
  const y = (p) => T + (1 - p) * (H - T - B);
  const points = days.map((d, i) => `${x(d / 365.25).toFixed(1)},${y(prob[i]).toFixed(1)}`).join(" ");
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((p) => `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(p)}" y2="${y(p)}"/><text x="${L - 6}" y="${y(p) + 4}" text-anchor="end">${Math.round(p * 100)}%</text>`).join("");
  const xTicks = Array.from({length: xMax + 1}, (_, i) => `<line class="grid" x1="${x(i)}" x2="${x(i)}" y1="${T}" y2="${H - B}"/><text x="${x(i)}" y="${H - B + 16}" text-anchor="middle">${i}</text>`).join("");
  const marks = horizons.map(([years, p]) => `<circle class="pt" cx="${x(years)}" cy="${y(p)}" r="4"/><text class="lbl" x="${x(years)}" y="${y(p) - 9}" text-anchor="middle">${Math.round(p * 100)}%</text>`).join("");
  const summary = horizons.map(([years, p]) => `${Math.round(years * 10) / 10} years ${Math.round(p * 100)}%`).join(", ");
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Predicted overall survival curve: ${esc(summary)}">
    <title>Predicted overall survival from diagnosis</title>${yTicks}${xTicks}
    <line class="axis" x1="${L}" x2="${L}" y1="${T}" y2="${H - B}"/><line class="axis" x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}"/>
    <polyline class="curve" points="${points}"/>${marks}
    <text x="${(L + W - R) / 2}" y="${H - 6}" text-anchor="middle">Years since diagnosis</text>
    <text transform="translate(11 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">Survival probability</text></svg>`;
}

const CANCER_STAGE_LABELS = Object.assign(Object.create(null), {stage_1_3: "Stage I–III", stage_4: "Stage IV"});
const SMOKING_HISTORY_LABELS = Object.assign(Object.create(null), {never: "never smoked", former_or_current: "former or current smoker"});

/* Exact histology SHAP values: contribution to the predicted class probability. */
function histologyShapHtml(explanation, predictedClass) {
  if (!explanation || explanation.status !== "available" || !Array.isArray(explanation.feature_contributions)) {
    const reason = explanation?.status === "unavailable"
      ? "SHAP is unavailable for this run; the histology prediction is unchanged."
      : "No histology SHAP explanation was returned.";
    return `<div class="hint" style="margin-top:.6rem"><b>Histology SHAP:</b> ${esc(reason)}</div>`;
  }

  const base = explanation.base_values?.[predictedClass];
  const finalValue = explanation.prediction_values?.[predictedClass];
  if (!fin(base) || !fin(finalValue)) {
    return '<div class="hint" style="margin-top:.6rem">Histology SHAP values are incomplete for the predicted class.</div>';
  }

  const entries = explanation.feature_contributions.map((item) => ({
    feature: item.feature,
    value: item.feature_value,
    contribution: item.values?.[predictedClass],
  })).filter((item) => fin(item.contribution));

  if (!entries.length) return '<div class="hint" style="margin-top:.6rem">No usable histology SHAP values.</div>';

  const peak = Math.max(...entries.map((item) => Math.abs(item.contribution)), 1e-9);
  const rows = entries.map((item) => {
    const pctPoints = item.contribution * 100;
    const width = Math.abs(item.contribution) / peak * 50;
    const positive = item.contribution >= 0;
    const direction = positive ? "supports" : "opposes";
    const featureName = label(item.feature);
    const barClass = positive ? "pos" : "neg";

    return `<div class="shap-row">
      <span class="shap-name"><b>${esc(featureName)}</b><small>${esc(item.value ?? "missing")}</small></span>
      <span class="shap-track" role="img" aria-label="${esc(item.feature)} ${direction} ${esc(predictedClass)} by ${Math.abs(pctPoints).toFixed(2)} percentage points">
        <i aria-hidden="true" class="shap-bar ${barClass}" style="width:${width}%"></i>
      </span>
      <b class="shap-val" style="color:${positive ? "var(--raise)" : "var(--lower)"}">${positive ? "+" : "−"}${Math.abs(pctPoints).toFixed(2)} pp</b>
    </div>`;
  }).join("");

  const sum = entries.reduce((total, item) => total + item.contribution, 0);
  const localError = Math.abs((base + sum) - finalValue);
  const ref = explanation.reference_population || {};

  return `<section aria-label="Histology SHAP explanation" style="margin-top:.7rem;padding-top:.6rem;border-top:1px solid var(--border,#ddd)">
    <h4 style="margin:.1rem 0 .35rem">Exact SHAP explanation for ${esc(predictedClass)}</h4>
    <p class="hint" style="margin:.2rem 0">Reference probability: ${(base * 100).toFixed(2)}% → model probability: ${(finalValue * 100).toFixed(2)}%.</p>
    <div class="shap-legend"><span class="lo">← Lowers ${esc(predictedClass)} probability</span><span class="hi">Raises ${esc(predictedClass)} probability →</span></div>
    ${rows}
    <p class="hint" style="margin:.35rem 0">Base value + feature SHAP values reconstruct the model probability (absolute check error: ${localError.toExponential(1)}). Contributions are probability percentage points, not causal effects.</p>
    <p class="hint" style="margin:.2rem 0">Reference: ${esc(ref.cohort || "documented training cohort")}, ${esc(ref.rows ?? "unknown")} training rows. ${esc((explanation.limitations || []).find((x) => /biological causation/i.test(x)) || "Research-only model explanation; not a diagnosis.")}</p>
  </section>`;
}

/* Patient-level histology panel: independent of any nodule's Brock risk and of the survival model. */
function cancerTypePanelHtml(model) {
  const head = `<h3 class="sub-h">Histology: LUAD vs LUSC (patient-level research model)</h3>`;
  const limits = (Array.isArray(model?.limitations) && model.limitations.length ? model.limitations : [
    "Research-only; not clinically validated.",
    "Compares LUAD with LUSC only; it does not detect cancer, separate cancer from benign disease, or establish a diagnosis.",
  ]).map((text) => `<li>${esc(text)}</li>`).join("");
  const limitsHtml = `<ul class="hint" style="margin:.3rem 0 0;padding-left:1.1rem">${limits}</ul>`;
  if (model?.status === "research_prediction" && model.probabilities) {
    const rows = Object.entries(model.probabilities).sort((a, b) => b[1] - a[1]).map(([name, p]) => {
      const percent = Number.isFinite(p) ? Math.min(100, Math.max(0, p * 100)) : 0;
      return `<div class="ct-row"><span><b>${esc(name)}</b></span><div class="bar" role="img" aria-label="${esc(name)} probability ${percent.toFixed(1)} percent"><i style="width:${percent}%"></i></div><span class="ct-val">${percent.toFixed(1)}%</span></div>`;
    }).join("");
    const values = model.feature_values || {};
    const used = [
      values.sex_at_baseline && `sex: ${values.sex_at_baseline}`,
      values.smoking_history && `smoking: ${SMOKING_HISTORY_LABELS[values.smoking_history] || label(values.smoking_history)}`,
      values.stage && `stage: ${CANCER_STAGE_LABELS[values.stage] || label(values.stage)}`,
    ].filter(Boolean).join("; ");
    return `<div class="ct">${head}
      <p style="margin:.2rem 0">${badge("research model output", "ok")} Highest-probability class: <b>${esc(model.predicted_class)}</b>. Not a diagnosis.</p>${rows}
      <p class="hint" style="margin:.3rem 0 0">Applies to patients already known to have LUAD or LUSC. Inputs used: ${esc(used || "not reported")}.</p>
      ${histologyShapHtml(model.explanation, model.predicted_class)}${limitsHtml}</div>`;
  }
  return `<div class="ct">${head}${modelResultHtml(model)}${limitsHtml}</div>`;
}

function assessmentHtml(nodule) {
  const a = assessment(nodule);
  const manual = brockIsManual(nodule) ? " Based on clinician-supplied inputs." : "";
  const note = `<p class="hint" style="margin:.2rem 0 0">Threshold ${Math.round(MALIGNANCY_THRESHOLD * 100)}% Brock risk; a research display setting, not a diagnosis.${esc(manual)}</p>`;
  if (a.state === "unavailable") {
    return `<div class="alert warn"><b>Not assessed.</b> Brock is unavailable for this nodule. This is not a negative result.</div>`;
  }
  if (a.state === "unlikely") {
    return `<div class="alert ok"><b>Chances of cancer are unlikely.</b> Brock risk ${esc(pct(a.p))} is below ${Math.round(MALIGNANCY_THRESHOLD * 100)}%.</div>${note}`;
  }
  return `<div class="alert bad"><b>Elevated malignancy risk.</b> Brock risk ${esc(pct(a.p))} is at or above ${Math.round(MALIGNANCY_THRESHOLD * 100)}%.</div>${note}`;
}

function modelResultHtml(model) {
  if (!model) return "Unavailable";
  if (model.status === "research_prediction" && model.probabilities) {
    const rows = Object.entries(model.probabilities).map(([name, p]) => `<small>${esc(name)} ${(p * 100).toFixed(1)}%</small>`).join(" · ");
    return `${badge("research prediction", "ok")} <b>${esc(model.predicted_class)}</b> (histology, conditional on LUAD or LUSC)<br>${rows}`;
  }
  if (model.status === "research_prediction" && model.prediction) {
    const horizons = Object.entries(model.prediction.probability_at_horizons_days || {})
      .map(([days, p]) => `${Math.round(days / 365.25 * 10) / 10} y: ${(p * 100).toFixed(0)}%`).join(" · ");
    return `${badge("research prediction", "ok")} Overall survival from diagnosis<br><small>${esc(horizons)}</small>`;
  }
  const codes = {
    prediction_baseline_does_not_match_training_baseline: "The CT was not confirmed as acquired at initial pathologic diagnosis (tick the confirmation in Case entry and re-run).",
    required_model_inputs_missing: "Required inputs are missing.",
    age_at_baseline_years_outside_training_range: "Age is outside the training range (33 to 90 years).",
    sex_at_baseline_category_not_seen_in_training: "Sex category was not seen in training.",
    no_validated_horizon_within_model_support: "No validated time horizon is supported for this input.",
  };
  const issues = (model.warnings || []).map((code) => codes[code] || label(code));
  if (model.missing_features?.length) issues.push(`Missing: ${model.missing_features.map(label).join(", ")}.`);
  return `${badge(model.status || "unavailable", model.status === "failed" ? "bad" : "warn")} ${esc(model.reason || issues.join(" ") || "No result returned.")}`;
}

/* ---------- report: rendering ---------- */
function gateText(body) {
  const study = body.study || body.study_summary || {};
  const maskOk = !["blocked", "failed"].includes(study.mask_validation_status);
  if (body.status === "completed" && study.preprocessing_status === "completed" && maskOk) return {ok: true, text: "Passed, eligible for nodule analysis"};
  if (body.status === "completed") return {ok: false, text: `Not passed (${label(study.mask_validation_status || "unknown")})`};
  return {ok: false, text: `Run ${label(body.status || "unknown")}`};
}

function renderSummary() {
  const {body} = S.run;
  const gate = gateText(body);
  const nodules = V.nodules;
  const withBrock = nodules.filter((n) => brockUsable(brockOf(n))).length;
  const when = body.clinical_prediction?.prediction_baseline_datetime || body.started_at || body.created_at || "unknown";
  const study = body.study || body.study_summary || {};
  const notes = [];
  if (body.status !== "completed" && body.message && !HIDDEN_NOTES.some((p) => p.test(body.message))) notes.push(["bad", body.message]);
  const errors = [...new Set(study.errors || [])];
  if (errors.length) notes.push(["bad", `Errors: ${errors.join("; ")}`]);
  if (body.error && !errors.includes(body.error) && body.error !== body.message) notes.push(["bad", `Failure: ${body.error}`]);
  const warnings = (study.warnings || []).filter((text) => !HIDDEN_NOTES.some((p) => p.test(text)));
  if (warnings.length) notes.push(["warn", `Warnings: ${warnings.join("; ")}`]);
  if (body.next_step) notes.push(["warn", `Next step: ${body.next_step}`]);
  if (body.blocking_requirements?.length) notes.push(["bad", `Blocking requirements: ${body.blocking_requirements.join("; ")}`]);
  const base = `${analysisBase}/runs/${encodeURIComponent(body.run_id)}`;
  $("summary").innerHTML = `
    <span>Patient <b>${esc(body.patient_id)}</b></span>
    <span>Run <b>${esc(body.run_id)}</b></span>
    <span>Predicted (UTC) <b>${esc(when)}</b></span>
    <span>Gate <b>${esc(gate.text)}</b></span>
    <span>Nodules <b>${nodules.length}</b> (${withBrock} with a Brock result)</span>
    <span class="links"><a href="${base}/report/html" target="_blank" rel="noopener">HTML report</a><a href="${base}/report" target="_blank" rel="noopener">JSON report</a>
      <button type="button" id="print-report" class="ghost">Print</button><button type="button" id="back-to-entry" class="ghost">Back to case entry</button></span>
    ${notes.map(([kind, text]) => `<div class="alert ${kind}" style="flex-basis:100%;margin:.1rem 0">${esc(text)}</div>`).join("")}`;
}

function renderNoduleTable() {
  const body = $("nodule-table").querySelector("tbody");
  if (!V.nodules.length) {
    body.innerHTML = `<tr><td colspan="5">${V.meta ? "No nodule records in the validated contract." : esc(S.run.metaError || "Nodule records unavailable.")}</td></tr>`;
    return;
  }
  body.innerHTML = V.nodules.map((n, index) => {
    const b = brockOf(n);
    const a = assessment(n);
    const kind = {unlikely: "ok", elevated: "bad", unavailable: "mute"}[a.state];
    return `<tr data-row="${index}" class="${index === V.selected ? "sel" : ""}">
      <td><button type="button" class="link" data-nodule-index="${index}" aria-pressed="${index === V.selected}" title="${esc(n.stable_nodule_id)}">${esc(n.stable_nodule_id)}</button></td>
      <td>${esc(formatMm(n.geometry?.diameter_mm))}</td>
      <td>${brockUsable(b) ? esc(pct(b.probability)) + (brockIsManual(n) ? " (supplied)" : "") : "Unavailable"}</td>
      <td>${badge(a.label, kind)}</td>
      <td title="${esc(n._boxErr || "Box passes validation")}">${n._boxErr ? badge("Invalid", "bad") : badge("Valid", "ok")}</td></tr>`;
  }).join("");
}

const ATTRIBUTES = [
  ["subtlety", "Subtlety", "extremely subtle", "obvious"],
  ["sphericity", "Sphericity", "linear", "round"],
  ["margin", "Margin", "poorly defined", "sharp"],
  ["lobulation", "Lobulation", "none", "marked"],
  ["spiculation", "Spiculation", "none", "marked"],
  ["texture", "Texture", "ground-glass", "solid"],
];

function renderCharacteristics() {
  const target = $("b-chars");
  const n = V.nodules[V.selected];
  if (!n) { target.textContent = "Select a nodule."; return; }
  if (n._boxErr) {
    target.innerHTML = `<div class="alert warn">Attribute ranking models unavailable for ${esc(n.stable_nodule_id)}: ${esc(n._boxErr)}. This is not a negative result.</div>`;
    return;
  }
  const attrs = n.classification?.attributes || {};
  const rows = ATTRIBUTES.filter(([key]) => !HIDDEN_ATTRIBUTES.has(key)).map(([key, name, low, high]) => {
    const item = attrs[key];
    const rating = item?.expected_rating;
    if (!fin(rating)) return `<div class="scale-row"><span>${esc(name)}</span><span>unavailable</span><span></span></div>`;
    const [lo, hi] = Array.isArray(item.rating_scale) && item.rating_scale.length === 2 ? item.rating_scale : [1, 5];
    const position = Math.min(100, Math.max(0, (rating - lo) / (hi - lo) * 100));
    return `<div class="scale-row"><span>${esc(name)}</span><div><div class="scale" role="img" aria-label="${esc(name)} rank ${rating.toFixed(1)} on a ${lo} to ${hi} scale"><i style="left:${position}%"></i></div>
      <div class="scale-ends" style="display:flex;justify-content:space-between;font-size:.7rem;color:var(--mute)"><span>${lo} ${esc(low)}</span><span>${esc(high)} ${hi}</span></div></div><b>${rating.toFixed(1)}</b></div>`;
  }).join("");
  target.innerHTML = `<p class="hint" style="margin:0 0 .2rem">${esc(n.stable_nodule_id)}: predicted LIDC-style reader ranks, not measurements or a diagnosis.</p>${rows}`;
}

function renderBrock() {
  const target = $("b-brock");
  const n = V.nodules[V.selected];
  if (!n) { target.textContent = "Select a nodule."; return; }
  const b = brockOf(n);
  const c = contributions(b);
  const missingNames = missingBrockInputs(n);
  const usable = brockUsable(b);
  const manualNote = brockIsManual(n) ? `<div class="hint" style="margin:0 0 .3rem">Calculated from clinician-supplied inputs (${esc(Object.keys(n.manual_brock.supplied || {}).map(label).join(", "))}).</div>` : "";
  const head = usable
    ? `<div class="big">${esc(pct(b.probability))}</div><div class="hint" style="margin:0 0 .4rem">Probability of malignancy (Brock)</div>${manualNote}`
    : `<div class="alert bad"><b>Brock unavailable.</b> ${missingNames.length ? `Missing: ${esc(missingNames.map((m) => MISSING_LABELS[m] || label(m)).join(", "))}.` : esc(b.status ? `Status: ${label(b.status)}.` : "No Brock result for this nodule.")}</div>`;
  const ctx = {inputs: S.run?.inputs, nodule: n, nodules: V.nodules};
  const supplied = brockIsManual(n) ? (n.manual_brock.supplied || {}) : {};
  const rows = BROCK_PARAMS.map((p) => {
    const reported = p.value(c);
    const value = reported ?? p.fallback?.(ctx) ?? null;
    const missing = p.missing.some((name) => missingNames.includes(name)) || (value == null && !usable && !p.fallback);
    let cell;
    if (missing) cell = '<span class="tag-miss">Missing</span>';
    else if (value == null) cell = "Derived by the model";
    else cell = esc(value) + (p.missing.some((name) => name in supplied) ? " (supplied)" : "");
    return `<tr><td>${esc(p.source)}</td><td>${esc(p.name)}</td><td>${cell}</td><td>${esc(p.enters)}</td></tr>`;
  }).join("");
  const vertical = n.anatomical_location?.vertical_region;
  const why = missingNames.includes("upper_lobe") && vertical
    ? `<p class="hint">Upper-lobe location is withheld, not guessed: a “${esc(vertical)}” vertical position has no unambiguous upper-lobe mapping.</p>` : "";
  target.innerHTML = `${head}<h3 class="sub-h">Assessment</h3>${assessmentHtml(n)}
    <div class="scroll"><table class="dt"><caption class="sr">Brock inputs for ${esc(n.stable_nodule_id)}</caption><thead><tr><th>Source</th><th>Brock input</th><th>Value</th><th>Enters model as</th></tr></thead><tbody>${rows}</tbody></table></div>${why}${usable ? "" : manualFormHtml(n, missingNames)}
    <p class="hint">Brock (PanCan) derives malignancy risk from the nodule characteristics and clinical data; there is no separate malignancy classifier. ${esc(b.reference || "")}</p>`;
}

/* The nine Brock terms with the patient's values substituted, exactly as the published formula computes them. */
const sgn = (v, d = 2) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(d)}`;
const BROCK_TERMS = {
  age: (c) => `0.0287 × (${c.age_years.input_value} − 62)`,
  sex: (c) => `0.6011 × ${c.female.encoded_value}`,
  fh: (c) => `0.2961 × ${c.family_history_lung_cancer.encoded_value}`,
  emph: (c) => `0.2953 × ${c.emphysema.encoded_value}`,
  size: (c) => `−5.3854 × ((${c.nodule_size_mm.input_value.toFixed(1)}/10)^−0.5 − 1.5811)`,
  type: (c) => (c.part_solid.encoded_value ? "part-solid: +0.3770" : c.pure_ground_glass.encoded_value ? "non-solid: −0.1276" : "solid: 0 (reference)"),
  lobe: (c) => `0.6581 × ${c.upper_lobe.encoded_value}`,
  count: (c) => `−0.0824 × (${c.nodule_count.input_value} − 4)`,
  spic: (c) => `0.7729 × ${c.spiculation.encoded_value}`,
};

function renderShap() {
  const target = $("b-shap");
  const n = V.nodules[V.selected];
  if (!n) { target.textContent = "Select a nodule."; return; }
  const b = brockOf(n);
  if (!brockUsable(b)) {
    target.innerHTML = '<div class="alert warn">Blocked: Brock result unavailable, so there is nothing to explain.</div>';
    return;
  }
  const c = contributions(b);
  const items = BROCK_PARAMS.map((p) => {
    let term = "";
    try { term = BROCK_TERMS[p.key](c); } catch { term = ""; }
    return {p, value: p.value(c), contribution: p.contrib(c), term};
  });
  const intercept = c.intercept?.contribution_logit;
  const peak = Math.max(...items.map((item) => Math.abs(item.contribution ?? 0)), 1e-9);
  const rows = items.map(({p, value, contribution, term}) => {
    const known = fin(contribution);
    const width = known ? Math.abs(contribution) / peak * 48 : 0;
    const bar = known && contribution !== 0 ? `<i class="shap-bar ${contribution > 0 ? "pos" : "neg"}" style="width:${width}%"></i>` : "";
    return `<div class="shap-row"><span class="shap-name" title="${esc(`${p.name}: ${p.enters}`)}">${esc(p.name)}: <b>${esc(value ?? "Missing")}</b><small>${esc(term)}</small></span><div class="shap-track">${bar}</div><span class="shap-val">${known ? sgn(contribution) : "n/a"}</span></div>`;
  }).join("");
  const net = items.reduce((sum, item) => sum + (fin(item.contribution) ? item.contribution : 0), 0);
  const lp = fin(b.logit) ? b.logit : (fin(intercept) ? intercept + net : null);
  const baseline = fin(intercept) ? 1 / (1 + Math.exp(-intercept)) : null;
  target.innerHTML = `<p class="hint formula" style="margin:0 0 .3rem">P = 1 / (1 + e<sup>−LP</sup>), with LP = −6.7892 + Σ the nine terms below. Each bar is that term's push on the log-odds relative to the reference nodule
      (62-year-old male, no family history or emphysema, 4 mm solid nodule, lower/middle lobe, 4 nodules, not spiculated), whose LP is the intercept alone${baseline === null ? "" : ` (risk ${esc(pct(baseline))})`}.</p>
    <div class="shap-legend"><span class="lo">← lowers risk</span><span class="hi">raises risk →</span></div>${rows}
    <div class="shap-sum" role="group" aria-label="Sum of terms">
      <span>Intercept <b>${fin(intercept) ? sgn(intercept) : "n/a"}</b></span><span>+ terms <b>${sgn(net)}</b></span>
      <span>= LP <b>${lp === null ? "n/a" : sgn(lp)}</b></span><span>→ P <b>${esc(pct(b.probability))}</b></span></div>
    <p class="hint">Exact for this additive model (not a sampled approximation). Model: ${esc(b.model_name || "Brock PanCan")}, ${esc(b.formula_version || "")}.</p>`;
}
function renderClinicalInputs() {
  const inputs = S.run.inputs;
  const target = $("b-inputs");
  if (!inputs) { target.innerHTML = "<p class='hint' style='margin:0'>Clinical inputs were not stored for this run. Not reported.</p>"; return; }
  const text = (value, fallback = "Not reported") => (value == null || value === "" ? fallback : label(value));
  const sex = inputs.sex_at_baseline ?? null;
  const emphysema = inputs.medical_history_status === "reported" ? (inputs.medical_history_conditions || []).includes("emphysema") ? "Yes" : "No" : "Not reported";
  const packs = fin(inputs.pack_years) ? ` (${inputs.pack_years.toFixed(1)} pack-years)` : "";
  target.innerHTML = `<dl class="kv">
    <dt>Age</dt><dd>${fin(ageOfInputs(inputs)) ? `${ageOfInputs(inputs)} years` : "Not reported"}</dd>
    <dt>Sex</dt><dd>${esc(text(sex))}</dd>
    <dt>Smoking</dt><dd>${esc(text(inputs.smoking_status))}${esc(packs)}</dd>
    <dt>Stage</dt><dd>${esc(CANCER_STAGE_LABELS[inputs.stage_at_baseline] || "Not reported")}</dd>
    <dt>Family history</dt><dd>${esc(text(inputs.family_history_lung_cancer))}</dd>
    <dt>Emphysema</dt><dd>${esc(emphysema)}</dd>
    <dt>KPS</dt><dd>${fin(inputs.kps) ? esc(inputs.kps) : "Not reported"}</dd></dl>`;
}

function renderClinicalModels() {
  const prediction = S.run.body.clinical_prediction;
  const survival = prediction?.survival;
  const chart = survivalChartHtml(survival);
  const survivalHtml = !prediction
    ? "<p class='hint' style='margin:0'>Survival model not available for this run.</p>"
    : chart
      ? `${chart}<p class="hint" style="margin:.2rem 0 0">Predicted overall survival from diagnosis (random survival forest on age and sex). Research-only, not calibrated for absolute survival.</p>`
      : `<div>${modelResultHtml(survival)}</div>`;

  // Histology UI gate: require at least one nodule with an available Brock risk >= the existing threshold.
  // This is a display/eligibility gate; backend inference must also be gated to prevent model execution.
  const eligibleNodules = (Array.isArray(V.nodules) ? V.nodules : [])
    .map((nodule) => ({nodule, result: assessment(nodule)}))
    .filter(({result}) => result.state === "elevated");
  const cancerHtml = !prediction
    ? "<p class='hint' style='margin:0'>Histology model not available for this run.</p>"
    : eligibleNodules.length === 0
      ? `<div class="ct"><h3 class="sub-h">Histology: LUAD vs LUSC</h3><p class="hint">Locked: histology is shown only when at least one nodule has an available Brock malignancy risk at or above ${Math.round(MALIGNANCY_THRESHOLD * 100)}%. Missing Brock results do not qualify.</p></div>`
      : `<p class="hint" style="margin:.2rem 0">Histology display enabled by ${eligibleNodules.length} nodule(s) at or above the ${Math.round(MALIGNANCY_THRESHOLD * 100)}% Brock threshold. The histology model remains patient-level and does not diagnose the eligible nodule.</p>${cancerTypePanelHtml(prediction.cancer_type)}`;
  $("b-models").innerHTML = `<section aria-label="Overall survival model"><h3 class="sub-h">Overall survival</h3>${survivalHtml}</section>
    <section aria-label="Histology model" style="margin-top:.8rem">${cancerHtml}</section>`;
}

function renderMethods() {
  const body = S.run.body;
  const study = body.study || body.study_summary || {};
  const counts = study.brock_prediction_counts || {};
  const eligible = study.eligible_nodule_count;
  const nodules = V.nodules;
  const rows = [];
  if (eligible != null) {
    const preprocessing = study.preprocessing_status && study.preprocessing_status !== "completed" ? `Preprocessing ${label(study.preprocessing_status)}. ` : "";
    rows.push(["Detection and segmentation", eligible, study.errors?.length ? 1 : 0, 0, `${preprocessing}${study.excluded_candidate_count ?? "unknown"} candidate(s) excluded by the lung-mask gate`]);
  }
  if (nodules.length) {
    const ranked = nodules.filter((n) => Object.keys(n.classification?.attributes || {}).some((k) => !HIDDEN_ATTRIBUTES.has(k))).length;
    rows.push(["Attribute ranking", ranked, 0, nodules.length - ranked, nodules.length - ranked ? "No attribute output for some nodules" : "–"]);
  }
  if (eligible != null) {
    const manualOk = nodules.filter((n) => brockIsManual(n)).length;
    const unavailable = Math.max(0, (counts.incomplete ?? 0) + (counts.not_run ?? 0) - manualOk);
    rows.push(["Brock", (counts.completed ?? 0) + manualOk, counts.failed ?? 0, unavailable, unavailable ? "Missing inputs or not run" : manualOk ? `${manualOk} calculated from clinician-supplied inputs` : "–"]);
  }
  if (nodules.length) {
    const explained = nodules.filter((n) => brockUsable(brockOf(n)) && (brockOf(n).feature_contributions || []).length).length;
    rows.push(["Brock formula contributions", explained, 0, nodules.length - explained, nodules.length - explained ? "Analytical Brock formula terms; not SHAP" : "Analytical Brock formula terms; not SHAP"]);
  }
  $("methods").innerHTML = rows.length
    ? rows.map((row) => `<tr>${row.map((cell) => `<td>${esc(cell)}</td>`).join("")}</tr>`).join("")
    : '<tr><td colspan="5">No method outcomes are available for this run.</td></tr>';
}

function renderReport() {
  const {body, meta} = S.run;
  V.meta = meta;
  V.runId = body.run_id;
  V.nodules = (meta?.nodules || []).map((n) => ({...n, _boxErr: validateBox(n, meta.shape_zyx)}));
  V.selected = -1;
  V.camChecked = new Set();
  $("report-empty").hidden = true;
  $("report-grid").hidden = false;
  $("summary").hidden = false;
  renderSummary();
  renderMethods();
  renderClinicalInputs();
  renderClinicalModels();
  setupViewer();
  if (V.nodules.length) selectNodule(0);
  else {
    renderNoduleTable();
    renderCharacteristics();
    renderBrock();
    renderShap();
    updateViewer().catch(() => {});
  }
  resizeCanvas();
}

$("summary").addEventListener("click", (event) => {
  if (event.target.closest("#print-report")) window.print();
  if (event.target.closest("#back-to-entry")) setView("entry");
});
$("nodule-table").addEventListener("click", (event) => {
  const button = event.target.closest("[data-nodule-index]");
  if (button) selectNodule(Number(button.dataset.noduleIndex), true);
});
$("b-brock").addEventListener("submit", (event) => {
  if (event.target.id !== "manual-brock-form") return;
  event.preventDefault();
  submitManualBrock(event.target);
});

/* ---------- report: viewer ---------- */
function viewerEls() {
  return {
    canvas: $("viewer-canvas"), wrap: $("canvas-wrap"), slice: $("viewer-slice"), label: $("viewer-slice-label"),
    lung: $("viewer-lung"), heart: $("viewer-heart"), cam: $("viewer-gradcam"), camControls: $("cam-controls"),
    attribute: $("viewer-gradcam-attribute"), camNodules: $("viewer-gradcam-nodules"),
    alpha: $("viewer-gradcam-alpha"), alphaLabel: $("viewer-gradcam-alpha-label"),
    previous: $("viewer-previous"), next: $("viewer-next"), status: $("viewer-status"),
  };
}

function resizeCanvas() {
  const {canvas, wrap} = viewerEls();
  const size = Math.floor(Math.min(wrap.clientWidth, wrap.clientHeight));
  if (size > 0) {
    canvas.style.width = `${size}px`;
    canvas.style.height = `${size}px`;
  }
}
new ResizeObserver(resizeCanvas).observe($("canvas-wrap"));

function setViewerStatus(message, kind = "") {
  const {status} = viewerEls();
  status.className = `status ${kind}`.trim();
  status.textContent = message;
}

function camAttributes() {
  return [...new Set((V.meta?.gradcam_options || []).map((o) => o.attribute))].filter((name) => !HIDDEN_ATTRIBUTES.has(name));
}

function renderCamNodules() {
  const el = viewerEls();
  const attribute = el.attribute.value;
  const ids = (V.meta?.gradcam_options || []).filter((o) => o.attribute === attribute).map((o) => o.stable_nodule_id);
  V.camChecked = new Set([...V.camChecked].filter((id) => ids.includes(id)));
  const selectedId = V.nodules[V.selected]?.stable_nodule_id;
  if (!V.camChecked.size && ids.includes(selectedId)) V.camChecked.add(selectedId);
  el.camNodules.innerHTML = ids.map((id) => `<label class="check"><input type="checkbox" value="${esc(id)}" ${V.camChecked.has(id) ? "checked" : ""}> ${esc(id.slice(-6))}</label>`).join("");
}

function setupViewer() {
  const el = viewerEls();
  const meta = V.meta;
  if (!V.wired) {
    V.wired = true;
    const refresh = () => updateViewer().catch(() => setViewerStatus("The requested CT slice could not be loaded.", "warning"));
    el.slice.addEventListener("input", refresh);
    for (const control of [el.lung, el.heart]) control.addEventListener("change", refresh);
    el.cam.addEventListener("change", () => {
      el.camControls.hidden = !el.cam.checked;
      if (el.cam.checked) {
        if (!el.attribute.value) el.attribute.value = el.attribute.options[0]?.value ?? "";
        renderCamNodules();
      }
      resizeCanvas();
      refresh();
    });
    el.attribute.addEventListener("change", () => { renderCamNodules(); refresh(); });
    el.camNodules.addEventListener("change", (event) => {
      if (event.target.checked) V.camChecked.add(event.target.value);
      else V.camChecked.delete(event.target.value);
      refresh();
    });
    let timer = null;
    el.alpha.addEventListener("input", () => {
      el.alphaLabel.textContent = `${el.alpha.value}%`;
      clearTimeout(timer);
      timer = setTimeout(refresh, 120);
    });
    el.previous.addEventListener("click", () => stepSlice(-1));
    el.next.addEventListener("click", () => stepSlice(1));
    el.canvas.addEventListener("wheel", (event) => { event.preventDefault(); stepSlice(event.deltaY > 0 ? 1 : -1); }, {passive: false});
    el.canvas.addEventListener("keydown", (event) => {
      const delta = ["ArrowUp", "ArrowRight"].includes(event.key) ? 1 : ["ArrowDown", "ArrowLeft"].includes(event.key) ? -1 : 0;
      if (!delta) return;
      event.preventDefault();
      stepSlice(delta);
    });
  }
  const available = Boolean(meta);
  for (const control of [el.slice, el.lung, el.heart, el.cam, el.previous, el.next]) control.disabled = !available;
  el.lung.checked = false;
  el.heart.checked = false;
  el.cam.checked = false;
  el.camControls.hidden = true;
  if (!available) {
    el.slice.max = "0";
    el.label.textContent = "–";
    el.canvas.getContext("2d").clearRect(0, 0, el.canvas.width, el.canvas.height);
    setViewerStatus(S.run.metaError || "CT viewer unavailable for this run.", "warning");
    return;
  }
  const maxIndex = Math.max(0, Number(meta.shape_zyx?.[0] || 1) - 1);
  el.slice.max = String(maxIndex);
  el.slice.value = String(Math.round(maxIndex / 2));
  el.lung.disabled = !meta.overlays?.lung_mask;
  el.heart.disabled = !meta.overlays?.heart_mask;
  const attributes = camAttributes();
  el.cam.disabled = !attributes.length;
  el.cam.title = attributes.length ? "" : "No Grad-CAM artifacts for this run";
  el.attribute.innerHTML = attributes.map((name) => `<option value="${esc(name)}">${esc(label(name))}</option>`).join("");
  setViewerStatus(`CT volume ${meta.shape_zyx.join(" × ")} voxels.`);
}

function stepSlice(delta) {
  const {slice} = viewerEls();
  const target = Math.min(Number(slice.max), Math.max(0, Number(slice.value) + delta));
  if (target === Number(slice.value)) return;
  slice.value = String(target);
  updateViewer().catch(() => setViewerStatus("The requested CT slice could not be loaded.", "warning"));
}

function boxStatus(nodule, index) {
  if (!nodule) return "";
  if (nodule._boxErr) return `${nodule.stable_nodule_id}: box not drawn. ${nodule._boxErr}.`;
  const box = boxOf(nodule);
  const z0 = box.start[0], z1 = box.end[0];
  const kind = box.method === "detector_box" ? "detector box (no distinct tissue found to refine it)" : "box fitted to the nodule in the CT";
  return index >= z0 && index < z1
    ? `${nodule.stable_nodule_id}: ${kind} shown on this slice (z ${z0 + 1}–${z1}).`
    : `${nodule.stable_nodule_id}: box spans slices ${z0 + 1}–${z1}; this slice is outside it.`;
}

/* Box for one slice: the nodule's own outline on that slice when known, otherwise the overall extent (dashed). */
function sliceRect(nodule, index) {
  const box = boxOf(nodule);
  if (index < box.start[0] || index >= box.end[0]) return null;
  const own = box.slices[String(index)];
  if (own) return {y0: own[0], x0: own[1], y1: own[2], x1: own[3], dashed: false};
  return {y0: box.start[1], x0: box.start[2], y1: box.end[1], x1: box.end[2], dashed: true};
}

async function updateViewer() {
  const el = viewerEls();
  const meta = V.meta;
  if (!meta) return;
  const maxIndex = Math.max(0, Number(meta.shape_zyx?.[0] || 1) - 1);
  const index = Math.min(maxIndex, Math.max(0, Number(el.slice.value) || 0));
  const params = new URLSearchParams({
    lung_overlay: String(el.lung.checked && !el.lung.disabled),
    heart_overlay: String(el.heart.checked && !el.heart.disabled),
  });
  const camOn = el.cam.checked && el.attribute.value && V.camChecked.size > 0;
  if (camOn) {
    params.set("gradcam_attribute", el.attribute.value);
    for (const id of V.camChecked) params.append("gradcam_nodule_id", id);
    params.set("gradcam_alpha", String(Number(el.alpha.value) / 100));
  }
  const request = ++V.request;
  const response = await apiFetch(`${analysisBase}/runs/${encodeURIComponent(V.runId)}/viewer/slices/${index}?${params}`);
  if (!response.ok) throw new Error("slice unavailable");
  if (request !== V.request) return;
  const blob = await response.blob();
  const image = new Image();
  image.onload = () => {
    if (request !== V.request) return;
    el.canvas.width = image.naturalWidth;
    el.canvas.height = image.naturalHeight;
    const context = el.canvas.getContext("2d");
    context.drawImage(image, 0, 0);
    context.font = "13px sans-serif";
    V.nodules.forEach((nodule, i) => {
      if (nodule._boxErr) return;
      const rect = sliceRect(nodule, index);
      if (!rect) return;
      const selected = i === V.selected;
      context.lineWidth = selected ? 3 : 1.5;
      context.strokeStyle = selected ? "#ffd24a" : "rgba(255, 210, 74, .5)";
      context.setLineDash(rect.dashed ? [6, 4] : []);
      context.strokeRect(rect.x0, rect.y0, rect.x1 - rect.x0, rect.y1 - rect.y0);
      context.setLineDash([]);
      if (selected) {
        context.fillStyle = "#ffd24a";
        context.fillText(nodule.stable_nodule_id.slice(-6), rect.x0, Math.max(13, rect.y0 - 4));
      }
    });
    URL.revokeObjectURL(image.src);
  };
  image.src = URL.createObjectURL(blob);
  el.label.textContent = `${index + 1} of ${maxIndex + 1}`;
  el.previous.disabled = index === 0;
  el.next.disabled = index === maxIndex;
  const inverted = camOn && (meta.gradcam_inverted_attributes || []).includes(el.attribute.value);
  document.querySelector(".cam-legend").classList.toggle("inv", inverted);
  const note = camOn ? ` Grad-CAM is drawn over the nodule-centred patch (the square). Colours are relative within that patch (${inverted ? "inverted for this attribute: red = lowest, blue = highest" : "red = highest"}); a model-sensitivity map, not an attribute probability.` : "";
  setViewerStatus(`${boxStatus(V.nodules[V.selected], index)}${note}`.trim() || `CT volume ${meta.shape_zyx.join(" × ")} voxels.`);
}

function selectNodule(index, focus = false) {
  const nodule = V.nodules[index];
  if (!nodule) return;
  V.selected = index;
  const el = viewerEls();
  if (!nodule._boxErr) {
    const box = boxOf(nodule);
    const own = Object.entries(box.slices).map(([z, [y0, x0, y1, x1]]) => [Number(z), (y1 - y0) * (x1 - x0)]);
    const z = own.length ? own.sort((a, b) => b[1] - a[1])[0][0] : Math.floor((box.start[0] + box.end[0] - 1) / 2);
    el.slice.value = String(Math.min(Number(el.slice.max), Math.max(0, z)));
  }
  if (el.cam.checked) {
    V.camChecked = new Set([nodule.stable_nodule_id]);
    renderCamNodules();
  }
  renderNoduleTable();
  renderCharacteristics();
  renderBrock();
  renderShap();
  updateViewer().catch(() => setViewerStatus("The requested CT slice could not be loaded.", "warning"));
  if (focus) el.canvas.focus({preventScroll: true});
}

/* ---------- start ---------- */
refresh();
setView("entry");
