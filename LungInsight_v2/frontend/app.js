"use strict";

const form = document.querySelector("#clinical-form");
const result = document.querySelector("#result");

function optionalNumber(value) {
  return value === "" ? null : Number(value);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  result.textContent = "Validating inputs…";
  try {
    const data = new FormData(form);
    const selectedFile = document.querySelector("#imaging-file").files[0];
    if (!selectedFile) {
      throw new Error("A validated Imaging evidence JSON file is required.");
    }
    const imagingEvidence = JSON.parse(await selectedFile.text());
    const localBaseline = new Date(data.get("baseline_at"));
    if (Number.isNaN(localBaseline.getTime())) {
      throw new Error("The prediction baseline is invalid.");
    }
    const payload = {
      schema_version: "1.0.0",
      patient_id: String(data.get("patient_id")).trim(),
      baseline_at: localBaseline.toISOString(),
      baseline_definition: data.get("baseline_definition"),
      clinician_inputs: {
        age_at_baseline_years: optionalNumber(data.get("age")),
        sex_at_baseline: data.get("sex") || null,
        stage_at_baseline: String(data.get("stage")).trim() || null,
        kps: optionalNumber(data.get("kps")),
        ecog: optionalNumber(data.get("ecog")),
      },
      imaging_evidence: imagingEvidence,
    };
    const response = await fetch("/api/v1/clinical/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.detail || `Clinical API returned HTTP ${response.status}.`);
    }
    result.textContent = JSON.stringify(body, null, 2);
  } catch (error) {
    result.textContent = `No prediction returned: ${error.message}`;
  }
});
