let ctFiles = [];
let runId = null, result = null, ctWindow = 'lung', chart = null;

const $ = id => document.getElementById(id);

const fields = [
  'PATIENT_ID','AGE','SEX','RACE','ETHNICITY','SMOKING_STATUS',
  'ever_pdl1_positive','num_treatment_events',
  'eastern_cancer_oncology_group','karnofsky_performance_score'
];

function patient() {
  const x = {};
  fields.forEach(k => {
    const el = $(k);
    if (el) x[k] = el.value;
  });
  ['AGE','num_treatment_events',
   'eastern_cancer_oncology_group','karnofsky_performance_score']
    .forEach(k => x[k] = Number(x[k]));
  return x;
}

function setStatus(msg, isError = false) {
  if ($('status')) $('status').textContent = msg || '';
  if ($('error')) {
    $('error').textContent = isError ? (msg || 'Request failed.') : '';
    $('error').classList.toggle('d-none', !isError);
  }
}

async function readJsonResponse(r) {
  const text = await r.text();
  let data = {};
  try { data = text ? JSON.parse(text) : {}; } catch (_) {
    data = { detail: text || `HTTP ${r.status}` };
  }
  if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
  return data;
}

$('sample').addEventListener('click', async () => {
  try {
    const d = await readJsonResponse(await fetch('/api/sample'));
    fields.forEach(k => {
      if ($(k) && d[k] !== undefined && d[k] !== null) $(k).value = d[k];
    });
    setStatus('Sample patient loaded.');
  } catch (e) {
    setStatus(e.message, true);
  }
});

function updateFileDisplay() {
  if (!$('fileName')) return;
  if (!ctFiles.length) {
    $('fileName').textContent = '';
    return;
  }
  const dcmCount = ctFiles.filter(f => /\.dcm$/i.test(f.name)).length;
  $('fileName').textContent =
    ctFiles.length === 1
      ? ctFiles[0].name
      : `${ctFiles.length} files selected (${dcmCount} DICOM)`;
}

$('drop').addEventListener('click', () => $('ct').click());

$('ct').addEventListener('change', e => {
  ctFiles = Array.from(e.target.files || []);
  updateFileDisplay();
});

['dragover', 'dragenter'].forEach(ev => {
  $('drop').addEventListener(ev, e => {
    e.preventDefault();
    $('drop').style.background = '#eef6ff';
  });
});

['dragleave', 'drop'].forEach(ev => {
  $('drop').addEventListener(ev, e => {
    e.preventDefault();
    $('drop').style.background = '#fbfdff';
  });
});

$('drop').addEventListener('drop', e => {
  const files = Array.from(e.dataTransfer.files || []);
  ctFiles = files;
  updateFileDisplay();
});

$('run').addEventListener('click', async e => {
  e.preventDefault();

  try {
    $('run').disabled = true;
    setStatus('Running clinical inference…');

    const fd = new FormData();
    fd.append('patient_json', JSON.stringify(patient()));

    // Single archive/file OR a whole DICOM folder selected with webkitdirectory.
    for (const f of ctFiles) {
      // Preserve relative folder structure for folder uploads.
      fd.append('ct_files', f, f.webkitRelativePath || f.name);
    }

    const r = await fetch('/api/predict', { method: 'POST', body: fd });
    const d = await readJsonResponse(r);

    result = d;
    runId = d.run_id;

    if (d.imaging?.stage04_nodule_count != null) {
      $('noduleCount').textContent = d.imaging.stage04_nodule_count;
    } else {
      $('noduleCount').textContent = '—';
    }

    const stage09Frame = $('stage09Frame');
    const ctStage09 = $('ctStage09');
    if (d.stage09_viewer_url) {
      stage09Frame.src = d.stage09_viewer_url + '?t=' + Date.now();
      ctStage09.classList.add('loaded');
    } else {
      stage09Frame.removeAttribute('src');
      ctStage09.classList.remove('loaded');
    }

    $('stage').textContent = d.stage?.label ?? 'Unavailable';
    $('stageConf').textContent =
      d.stage?.confidence != null
        ? `Confidence ${(d.stage.confidence * 100).toFixed(1)}% · threshold ${d.stage.threshold}`
        : '—';

    $('hist').textContent = d.histology?.label ?? 'Unavailable';
    $('histConf').textContent =
      d.histology?.confidence != null
        ? `Confidence ${(d.histology.confidence * 100).toFixed(1)}%`
        : '—';

    $('riskScore').textContent =
      d.survival?.risk_score != null
        ? Number(d.survival.risk_score).toFixed(3)
        : '—';

    if (chart) chart.destroy();
    const times = d.survival?.times_months || [];
    const surv = d.survival?.survival_probability || [];

    if (times.length && surv.length) {
      chart = new Chart($('survival'), {
        type: 'line',
        data: {
          labels: times,
          datasets: [{
            label: 'Survival probability',
            data: surv,
            borderWidth: 2,
            tension: .25,
            pointRadius: 0
          }]
        },
        options: {
          responsive: true,
          scales: {
            x: { title: { display: true, text: 'Months' } },
            y: { min: 0, max: 1, title: { display: true, text: 'Probability' } }
          }
        }
      });
    }

    const sh = d.shap || {};
    if (sh.waterfall_png_b64) {
      $('shap').src = 'data:image/png;base64,' + sh.waterfall_png_b64;
      $('shap').style.display = 'block';
    }

    $('pos').innerHTML = (sh.positive || [])
      .map(x => `<li>${x.feature}: <b>${x.value >= 0 ? '+' : ''}${Number(x.value).toFixed(3)}</b></li>`)
      .join('');

    $('neg').innerHTML = (sh.negative || [])
      .map(x => `<li>${x.feature}: <b>${Number(x.value).toFixed(3)}</b></li>`)
      .join('');

    $('pdf').disabled = false;

    const fused = d.multimodal_fusion?.used;
    setStatus(
      fused
        ? 'Clinical + imaging late fusion used.'
        : (d.imaging?.used
            ? 'Clinical inference completed; imaging was uploaded but no usable fusion embedding was found.'
            : 'Clinical inference completed.')
    );
  } catch (e) {
    console.error(e);
    setStatus(e.message, true);
  } finally {
    $('run').disabled = false;
  }
});

$('pdf').addEventListener('click', async () => {
  if (!result) return;
  try {
    setStatus('Generating PDF…');
    const r = await fetch('/api/report', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        patient: result.patient,
        prediction_results: result
      })
    });
    if (!r.ok) throw new Error(await r.text());
    const b = await r.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(b);
    a.download = 'LungInsight_Report.pdf';
    a.click();
    URL.revokeObjectURL(a.href);
    setStatus('PDF generated.');
  } catch (e) {
    setStatus(e.message, true);
  }
});
