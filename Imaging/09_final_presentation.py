"""
09_final_presentation.py

LungInsight -- Stage 09
========================

Final Presentation: Combined Multi-Candidate, Multi-Head Sliding CT Viewer.

Stage 09 consumes the results of Stage 08 (`08_full_ct_gradcam.py`) and
produces a SINGLE self-contained HTML viewer that shows every candidate
nodule, for every classifier head, overlaid together on the same full
Stage 02 native CT volume -- unlike Stage 08's per-candidate viewers,
which only show one candidate at a time.

Stage 09 is the SOLE presentation authority in the LungInsight pipeline.
It does no new projection math of its own: it reuses Stage 08's exact
projection functions (imported directly from `08_full_ct_gradcam.py`,
byte-for-byte the same code, no re-derivation) and Stage 08's own
`08_visualization/report.json` to determine exactly which candidates and
which heads actually projected successfully. Stage 09 never re-decides
what counts as a successful projection; it only re-renders what Stage 08
already certified as `"status": "PROJECTED"`.

Why the heatmaps are recomputed rather than "just loaded"
-----------------------------------------------------------
Stage 08 does not persist full-volume (or even per-candidate) projected
heatmap arrays to disk -- only PNG slices/MIPs/GIFs and per-candidate
HTML viewers, all baked at Stage-08-decided crops/thresholds. To combine
*all* candidates and *all* heads into one interactive full-CT view (with
its own independent head toggles and a global Z-slider), Stage 09 needs
the raw per-(candidate, head) heatmap arrays back. It gets them by
calling Stage 08's own `project_cam_to_native()` again on the same
Stage 02 CT / Stage 05 geometry / Stage 07 Grad-CAMs Stage 08 used --
which is deterministic and produces bit-identical results to what Stage
08 already validated -- rather than re-deriving the projection math
in a second, potentially-diverging implementation.

What "loading Stage 08's results" means here, concretely:
    1. `08_visualization/report.json` is read first and is authoritative
       for *which* candidates/heads to include (only `"status":
       "PROJECTED"` candidates; only their listed `projected_heads`).
    2. Stage 08's module is imported and its functions are called to
       regenerate the (small, per-candidate-bounding-box) heatmap arrays
       backing those already-certified projections.
    3. Nothing is included in the Stage 09 viewer that Stage 08's report
       did not already mark as successfully projected.

Output
------
output/<patient>/09_presentation/
    viewer.html      -- the combined interactive viewer (open in any browser)
    manifest.json     -- what went into the viewer, for provenance

Viewer controls
----------------
- A single Z-slider scrubs through the FULL native Stage 02 CT volume
  (every slice, not just candidate-adjacent ones).
- A DROPDOWN selects exactly ONE classifier head to display at a time
  (each head keeps its own fixed, distinguishable color).
- A checkbox PER CANDIDATE toggles whether that candidate contributes to
  the displayed overlay. Any subset of candidates can be selected; the
  selected candidates' heatmaps are combined per-pixel-max for whichever
  head is currently active, entirely in the browser (no server round
  trip), so the combination updates instantly as heads/candidates change.
- Hovering the mouse over the CT reports a live readout of the cursor's
  displayed-pixel and native-voxel coordinates plus the combined
  Grad-CAM heatmap value at that location for the active head.
- Small crosshair markers (togglable) show every candidate's location
  on whichever slice its center falls on, with a legend that can jump
  the slider straight to any candidate.
- A global alpha slider controls overlay opacity.
"""

from __future__ import annotations

import argparse
import base64
import colorsys
import hashlib
import importlib.util
import io
import json
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_THRESHOLD = 0.4
DEFAULT_ALPHA = 0.45
DEFAULT_MAX_DIM = 360          # downsample cap (px) for embedded viewer frames
DEFAULT_STAGE08_SCRIPT_NAME = "08_full_ct_gradcam.py"

# Fixed, hand-picked, mutually-distinguishable colors for the standard
# LungInsight classifier heads (matches Stage 07/08's EXPECTED_HEADS).
# Any head not in this table gets a deterministic hash-based color instead
# (see `color_for_head`), so behavior is stable even for unexpected heads.
HEAD_COLOR_TABLE: Dict[str, str] = {
    "calcification": "#e6194b",  # red
    "lobulation": "#3cb44b",     # green
    "malignancy": "#4363d8",     # blue
    "margin": "#f58231",         # orange
    "sphericity": "#911eb4",     # purple
    "spiculation": "#42d4f4",    # cyan
    "subtlety": "#f032e6",       # magenta
    "texture": "#bfef45",        # lime
}


# ============================================================================
# STAGE 08 MODULE LOADING (reuse, never re-derive, the projection math)
# ============================================================================


def load_stage08_module(script_path: Path) -> types.ModuleType:
    """
    Dynamically import `08_full_ct_gradcam.py` (its filename starts with a
    digit, so it cannot be imported with a normal `import` statement).

    Stage 09 relies on this module for every piece of projection and
    rendering logic it reuses: `load_stage02`, `load_stage05_manifest`,
    `discover_candidate_directories`, `extract_candidate_id`,
    `discover_head_files`, `resolve_candidate_geometry`, `normalize_heatmap`,
    `project_cam_to_native`, `window_ct_uint8_rgb`, and `_encode_png_b64`.
    """

    if not script_path.is_file():
        raise FileNotFoundError(
            f"Could not find Stage 08 script at {script_path}. "
            "Pass --stage08-script to point Stage 09 at "
            f"'{DEFAULT_STAGE08_SCRIPT_NAME}' explicitly."
        )

    spec = importlib.util.spec_from_file_location(
        "lunginsight_stage08_full_ct_gradcam", script_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an import spec for {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    required = (
        "load_stage02",
        "load_stage05_manifest",
        "discover_candidate_directories",
        "extract_candidate_id",
        "discover_head_files",
        "resolve_candidate_geometry",
        "normalize_heatmap",
        "project_cam_to_native",
        "window_ct_uint8_rgb",
        "_encode_png_b64",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise ImportError(
            f"Stage 08 module at {script_path} is missing expected "
            f"attributes: {missing}. Is this really 08_full_ct_gradcam.py?"
        )

    return module


# ============================================================================
# STAGE 08 REPORT (authoritative list of what actually projected)
# ============================================================================


def load_stage08_report(report_path: Path) -> Dict[str, Any]:
    if not report_path.is_file():
        raise FileNotFoundError(
            f"Stage 08 report not found: {report_path}\n"
            "Stage 09 presents Stage 08's results and refuses to guess at "
            "them -- run 08_full_ct_gradcam.py for this patient first."
        )
    return json.loads(report_path.read_text(encoding="utf-8"))


def projected_candidates_from_report(
    report: Dict[str, Any],
    candidates_filter: Optional[List[int]],
) -> Dict[int, Dict[str, Any]]:
    """
    Return {candidate_id: candidate_report_entry} for every candidate Stage
    08 marked `"status": "PROJECTED"`, optionally narrowed by
    `candidates_filter`.
    """

    out: Dict[int, Dict[str, Any]] = {}
    for entry in report.get("candidates", []):
        if entry.get("status") != "PROJECTED":
            continue
        candidate_id = entry.get("candidate_id")
        if candidate_id is None:
            continue
        candidate_id = int(candidate_id)
        if candidates_filter is not None and candidate_id not in candidates_filter:
            continue
        out[candidate_id] = entry
    return out


# ============================================================================
# COLOR HELPERS
# ============================================================================


def color_for_head(head: str) -> str:
    """Fixed color for known heads; stable hash-derived color otherwise."""

    if head in HEAD_COLOR_TABLE:
        return HEAD_COLOR_TABLE[head]

    digest = hashlib.sha256(head.encode("utf-8")).hexdigest()
    hue = (int(digest[:8], 16) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return (
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    )


# ============================================================================
# RE-PROJECT EACH (CANDIDATE, HEAD) USING STAGE 08'S OWN FUNCTIONS
# ============================================================================


def compute_candidate_head_projections(
    mod: types.ModuleType,
    candidate_dir: Path,
    stage02: Dict[str, Any],
    stage05_manifest: Dict[str, Dict[str, Any]],
    allowed_heads: List[str],
    heads_filter: Optional[List[str]],
) -> Optional[Dict[str, Any]]:
    """
    Recompute the (small, candidate-bounding-box) projected heatmap for
    every head Stage 08's report already certified as `PROJECTED` for this
    candidate, using Stage 08's own `resolve_candidate_geometry` /
    `normalize_heatmap` / `project_cam_to_native`. Returns None if nothing
    usable remains after filtering.
    """

    candidate_id = mod.extract_candidate_id(candidate_dir)
    metadata_path = candidate_dir / "metadata.json"
    if not metadata_path.is_file():
        return None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    stage05_record = stage05_manifest.get(str(candidate_id))

    geometry = mod.resolve_candidate_geometry(
        metadata, stage05_record, stage02["spacing_zyx_mm"]
    )

    head_files = mod.discover_head_files(candidate_dir)

    wanted_heads = set(allowed_heads)
    if heads_filter:
        wanted_heads &= set(heads_filter)

    head_files = {h: p for h, p in head_files.items() if h in wanted_heads}
    if not head_files:
        return None

    heads_data: Dict[str, Dict[str, Any]] = {}
    for head, path in sorted(head_files.items()):
        raw_cam = np.load(path, allow_pickle=False)
        cam = mod.normalize_heatmap(raw_cam)

        result = mod.project_cam_to_native(cam, geometry, stage02["volume_shape_zyx"])
        if result is None:
            continue

        heads_data[head] = result

    if not heads_data:
        return None

    return {
        "candidate_id": candidate_id,
        "geometry": geometry,
        "heads": heads_data,
    }


# ============================================================================
# COMBINED FULL-CT COMPOSITING (all candidates -> one overlay per head)
# ============================================================================


def build_base_frames(
    mod: types.ModuleType,
    ct_volume: np.ndarray,
    max_dim: int,
) -> Dict[str, str]:
    """Base64 PNG per native Z-slice, covering the FULL CT depth."""

    base_frames: Dict[str, str] = {}
    depth = ct_volume.shape[0]
    for z in range(depth):
        rgb = mod.window_ct_uint8_rgb(ct_volume[z, :, :])
        base_frames[str(z)] = mod._encode_png_b64(rgb, max_dim)
    return base_frames


def build_candidate_head_overlays(
    mod: types.ModuleType,
    projections: Dict[int, Dict[str, Any]],
    full_shape_yx: Tuple[int, int],
    depth: int,
    threshold: float,
    max_dim: int,
) -> Tuple[Dict[int, Dict[str, Dict[str, Any]]], Dict[str, str]]:
    """
    Build a PER-CANDIDATE, per-head, per-slice grayscale heat overlay PNG
    (intensity in [threshold, 1] mapped to [1, 255] in every RGB channel,
    fully opaque, zero elsewhere), keeping each candidate's contribution
    separate instead of pre-combining across candidates.

    This lets the browser do two things Stage 09 used to do server-side:
      1. Combine only the currently-TOGGLED-ON candidates (per-pixel max,
         via canvas 'lighten' compositing) for whichever single head is
         currently selected in the dropdown.
      2. Read the raw combined intensity value back out of that canvas
         under the mouse cursor for the hover readout.

    Returns:
        candidate_overlays: {candidate_id: {head: {"z_start": int,
            "z_end": int, "overlays": {str(z): base64_png}}}}
        head_colors: {head: "#rrggbb"} for every head seen, for the
            dropdown's option list and the active overlay's tint.
    """

    candidate_overlays: Dict[int, Dict[str, Dict[str, Any]]] = {}
    head_colors: Dict[str, str] = {}

    for candidate_id, cand in projections.items():
        head_out: Dict[str, Dict[str, Any]] = {}

        for head, res in cand["heads"].items():
            head_colors.setdefault(head, color_for_head(head))

            heat = res["heatmap"]
            lo = res["native_start_zyx"]

            overlays: Dict[str, str] = {}
            for lz in range(heat.shape[0]):
                gz = int(lo[0]) + lz
                if gz < 0 or gz >= depth:
                    continue

                slice_local = heat[lz]
                if float(slice_local.max()) < threshold:
                    continue

                y0, x0 = int(lo[1]), int(lo[2])
                y1, x1 = y0 + slice_local.shape[0], x0 + slice_local.shape[1]

                cy0, cx0 = max(y0, 0), max(x0, 0)
                cy1, cx1 = min(y1, full_shape_yx[0]), min(x1, full_shape_yx[1])
                if cy1 <= cy0 or cx1 <= cx0:
                    continue

                sub = slice_local[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]

                buf = np.zeros(full_shape_yx, dtype=np.float32)
                buf[cy0:cy1, cx0:cx1] = sub

                intensity = np.clip(buf, 0.0, 1.0)
                if not np.any(intensity > 0.0):
                    continue

                # Preserve the projected value itself. Display thresholding and
                # Jet coloring happen in JavaScript, after selected candidates
                # have been combined by per-pixel maximum.
                gray_u8 = np.rint(intensity * 255.0).astype(np.uint8)

                rgba = np.zeros((*full_shape_yx, 4), dtype=np.uint8)
                rgba[..., 0] = gray_u8
                rgba[..., 1] = gray_u8
                rgba[..., 2] = gray_u8
                rgba[..., 3] = np.where(intensity > 0.0, 255, 0).astype(np.uint8)

                overlays[str(gz)] = mod._encode_png_b64(rgba, max_dim)

            if overlays:
                z_values = [int(z) for z in overlays.keys()]
                head_out[head] = {
                    "z_start": min(z_values),
                    "z_end": max(z_values),
                    "overlays": overlays,
                }

        if head_out:
            candidate_overlays[candidate_id] = head_out

    return candidate_overlays, head_colors


# ============================================================================
# HTML VIEWER TEMPLATE
# ============================================================================

_COMBINED_VIEWER_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LungInsight -- {patient_id} -- final presentation</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ background:#111318; color:#e8e8ec; font-family:-apple-system,Segoe UI,Roboto,sans-serif; margin:0; padding:24px; }}
  h1 {{ font-size:17px; font-weight:600; margin:0 0 4px; }}
  .sub {{ color:#9aa0ab; font-size:12px; margin-bottom:16px; }}
  .badge {{ display:inline-block; background:#1f2430; border-radius:4px; padding:2px 8px; font-size:11px; color:#8fb3ff; margin-right:6px; }}
  .layout {{ display:flex; gap:24px; align-items:flex-start; flex-wrap:wrap; }}
  .frame-wrap {{ position:relative; background:#000; border-radius:8px; overflow:hidden; line-height:0; width:{frame_width}px; max-width:100%; cursor:crosshair; }}
  .frame-wrap img, .frame-wrap canvas {{ display:block; position:absolute; top:0; left:0; width:100%; height:auto; image-rendering:pixelated; }}
  .frame-wrap img.base {{ position:static; }}
  #overlayCanvas {{ pointer-events:none; }}
  .marker {{ position:absolute; transform:translate(-50%, -50%); pointer-events:none; }}
  .marker .dot {{ width:10px; height:10px; border-radius:50%; border:2px solid #fff; box-shadow:0 0 4px rgba(0,0,0,0.8); }}
  .marker .label {{ position:absolute; top:8px; left:8px; font-size:10px; background:rgba(0,0,0,0.65); padding:1px 4px; border-radius:3px; white-space:nowrap; color:#fff; }}
  .controls {{ min-width:280px; display:flex; flex-direction:column; gap:16px; }}
  .control label {{ display:block; font-size:12px; color:#9aa0ab; margin-bottom:4px; }}
  input[type=range] {{ width:100%; }}
  select {{ width:100%; background:#1a1e27; color:#e8e8ec; border:1px solid #2a2f3a; border-radius:6px; padding:6px 8px; font-size:13px; }}
  .readout {{ font-variant-numeric: tabular-nums; font-size:13px; color:#e8e8ec; }}
  .candidate-list {{ display:flex; flex-direction:column; gap:6px; max-height:260px; overflow-y:auto; padding-right:4px; }}
  .head-row {{ display:flex; align-items:center; gap:8px; font-size:13px; }}
  .swatch {{ width:12px; height:12px; border-radius:3px; flex:none; }}
  .head-row label {{ margin:0; color:#e8e8ec; cursor:pointer; }}
  .candidate-row {{ display:flex; align-items:center; justify-content:space-between; gap:8px; font-size:12px; background:#1a1e27; border-radius:6px; padding:6px 10px; }}
  .candidate-row label {{ display:flex; align-items:center; gap:8px; cursor:pointer; }}
  .candidate-row button {{ background:#283042; color:#8fb3ff; border:none; border-radius:4px; padding:4px 8px; font-size:11px; cursor:pointer; flex:none; }}
  .candidate-row button:hover {{ background:#334063; }}
  fieldset {{ border:1px solid #2a2f3a; border-radius:8px; padding:10px 12px; }}
  legend {{ font-size:12px; color:#9aa0ab; padding:0 6px; }}
  .hover-readout {{ font-size:12px; color:#e8e8ec; line-height:1.7; font-variant-numeric: tabular-nums; }}
  .hover-readout strong {{ color:#8fb3ff; }}
</style>
</head>
<body>
<h1>{patient_id} &mdash; final presentation: toggle candidates, one head at a time, full CT</h1>
<div class="sub">
  <span class="badge">coordinate_order: ZYX</span>
  <span class="badge">source_space: stage02_native_ct</span>
  <span class="badge">projection_authority: stage08 (re-rendered by stage09)</span>
  <span class="badge">candidates: {num_candidates}</span>
  <span class="badge">heads: {num_heads}</span>
</div>
<div class="layout">
  <div class="frame-wrap" id="frameWrap">
    <img class="base" id="baseImg" alt="CT slice">
    <canvas id="overlayCanvas"></canvas>
    <div id="markerLayer"></div>
  </div>
  <div class="controls">
    <div class="control">
      <label for="zRange">Z-slice (full native CT)</label>
      <input type="range" id="zRange" min="0" max="0" step="1" value="0">
      <div class="readout" id="zReadout">Slice: -- / --</div>
    </div>
    <div class="control">
      <label for="alphaRange">Overlay transparency (alpha)</label>
      <input type="range" id="alphaRange" min="0" max="1" step="0.01" value="{default_alpha}">
    </div>
    <div class="control">
      <label for="headSelect">Classifier head (one at a time)</label>
      <select id="headSelect"></select>
    </div>
    <div class="control">
      <label class="head-row"><input type="checkbox" id="markerToggle" checked> Show candidate markers</label>
    </div>
    <fieldset>
      <legend>Candidates -- toggle which are combined into the overlay ({num_candidates})</legend>
      <div class="candidate-list" id="candidateList"></div>
    </fieldset>
    <fieldset>
      <legend>Cursor / Grad-CAM readout</legend>
      <div class="hover-readout" id="hoverReadout">Hover over the CT to inspect coordinates and heatmap values.</div>
    </fieldset>
  </div>
</div>
<script>
const VIEWER_DATA = {viewer_data_json};

const zRange = document.getElementById('zRange');
const zReadout = document.getElementById('zReadout');
const alphaRange = document.getElementById('alphaRange');
const baseImg = document.getElementById('baseImg');
const overlayCanvas = document.getElementById('overlayCanvas');
const overlayCtx = overlayCanvas.getContext('2d');
const markerLayer = document.getElementById('markerLayer');
const headSelect = document.getElementById('headSelect');
const candidateList = document.getElementById('candidateList');
const markerToggle = document.getElementById('markerToggle');
const hoverReadout = document.getElementById('hoverReadout');
const frameWrap = document.getElementById('frameWrap');

const combineCanvas = document.createElement('canvas');
const combineCtx = combineCanvas.getContext('2d', {{ willReadFrequently: true }});

const headNames = Object.keys(VIEWER_DATA.heads);
const candidateSelected = {{}};
let currentGrid = null;   // {{ width, height, data }} -- combined intensity grid for the active head/z
let lastHoverEvent = null;
let renderToken = 0;

function hexToRgb(hex) {{
  hex = hex.replace('#', '');
  return [
    parseInt(hex.substring(0, 2), 16),
    parseInt(hex.substring(2, 4), 16),
    parseInt(hex.substring(4, 6), 16),
  ];
}}

function jetColor(v) {{
  // Matplotlib-style Jet approximation for normalized Grad-CAM intensity.
  v = Math.max(0, Math.min(1, v));
  const r = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * v - 3)));
  const g = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * v - 2)));
  const b = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * v - 1)));
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}}

function loadImage(b64) {{
  return new Promise((resolve, reject) => {{
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = 'data:image/png;base64,' + b64;
  }});
}}

const heatImagePromiseCache = {{}};

function getHeatImagePromise(candidateId, head, z) {{
  const key = candidateId + '|' + head + '|' + z;
  if (key in heatImagePromiseCache) return heatImagePromiseCache[key];

  const candData = VIEWER_DATA.candidate_overlays[String(candidateId)];
  const headData = candData && candData[head];
  const b64 = headData && headData.overlays[String(z)];

  const promise = b64 ? loadImage(b64) : Promise.resolve(null);
  heatImagePromiseCache[key] = promise;
  return promise;
}}

for (const head of headNames) {{
  const opt = document.createElement('option');
  opt.value = head;
  opt.textContent = head;
  headSelect.appendChild(opt);
}}
if (headNames.length) headSelect.value = headNames[0];

for (const cand of VIEWER_DATA.candidates_all) {{
  candidateSelected[cand.id] = true;

  const row = document.createElement('div');
  row.className = 'candidate-row';

  const left = document.createElement('label');

  const checkbox = document.createElement('input');
  checkbox.type = 'checkbox';
  checkbox.checked = true;
  checkbox.addEventListener('change', () => {{
    candidateSelected[cand.id] = checkbox.checked;
    updateOverlay();
  }});

  const info = document.createElement('span');
  info.textContent = 'Candidate ' + cand.id + ' (' + cand.heads.join(', ') + ')';

  left.appendChild(checkbox);
  left.appendChild(info);

  const jump = document.createElement('button');
  jump.textContent = 'Jump';
  jump.addEventListener('click', () => {{
    zRange.value = cand.z;
    render();
  }});

  row.appendChild(left);
  row.appendChild(jump);
  candidateList.appendChild(row);
}}

function renderMarkers() {{
  markerLayer.innerHTML = '';
  if (!markerToggle.checked) return;

  const z = parseInt(zRange.value, 10);
  for (const cand of VIEWER_DATA.candidates) {{
    if (cand.z !== z) continue;

    const marker = document.createElement('div');
    marker.className = 'marker';
    marker.style.left = (cand.x_frac * 100) + '%';
    marker.style.top = (cand.y_frac * 100) + '%';

    const dot = document.createElement('div');
    dot.className = 'dot';
    dot.style.background = 'transparent';
    dot.style.borderColor = '#ffffff';
    dot.style.outline = '1px solid #000';

    const labelEl = document.createElement('div');
    labelEl.className = 'label';
    labelEl.textContent = '#' + cand.id;

    marker.appendChild(dot);
    marker.appendChild(labelEl);
    markerLayer.appendChild(marker);
  }}
}}

async function updateOverlay() {{
  const myToken = ++renderToken;
  const head = headSelect.value;
  const z = parseInt(zRange.value, 10);
  const alpha = parseFloat(alphaRange.value);

  currentGrid = null;
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  if (!head) {{
    updateHoverFromLastPosition();
    return;
  }}

  const selectedIds = VIEWER_DATA.candidates_all
    .map((c) => c.id)
    .filter((id) => candidateSelected[id]);

  const images = await Promise.all(
    selectedIds.map((id) => getHeatImagePromise(id, head, z))
  );

  if (myToken !== renderToken) return; // a newer render superseded this one

  const validImages = images.filter(Boolean);
  if (!validImages.length) {{
    updateHoverFromLastPosition();
    return;
  }}

  const w = validImages[0].naturalWidth;
  const h = validImages[0].naturalHeight;

  combineCanvas.width = w;
  combineCanvas.height = h;
  combineCtx.globalCompositeOperation = 'source-over';
  combineCtx.clearRect(0, 0, w, h);
  combineCtx.fillStyle = '#000000';
  combineCtx.fillRect(0, 0, w, h);
  combineCtx.globalCompositeOperation = 'lighten'; // per-channel max == per-pixel max intensity
  for (const img of validImages) {{
    combineCtx.drawImage(img, 0, 0, w, h);
  }}
  combineCtx.globalCompositeOperation = 'source-over';

  const grid = combineCtx.getImageData(0, 0, w, h);
  currentGrid = {{ width: w, height: h, data: grid.data }};

  overlayCanvas.width = w;
  overlayCanvas.height = h;

  const colored = overlayCtx.createImageData(w, h);
  for (let i = 0; i < grid.data.length; i += 4) {{
    const intensity = grid.data[i] / 255;
    const [cr, cg, cb] = jetColor(intensity);
    colored.data[i] = cr;
    colored.data[i + 1] = cg;
    colored.data[i + 2] = cb;
    const visible = intensity >= VIEWER_DATA.threshold;
    colored.data[i + 3] = visible ? Math.round(255 * alpha) : 0;
  }}
  overlayCtx.putImageData(colored, 0, 0);

  updateHoverFromLastPosition();
}}

function updateHoverFromLastPosition() {{
  if (lastHoverEvent) handleHover(lastHoverEvent);
}}

function handleHover(evt) {{
  lastHoverEvent = evt;
  const rect = baseImg.getBoundingClientRect();
  if (!rect.width || !rect.height) return;

  const fracX = Math.min(1, Math.max(0, (evt.clientX - rect.left) / rect.width));
  const fracY = Math.min(1, Math.max(0, (evt.clientY - rect.top) / rect.height));

  const z = parseInt(zRange.value, 10);
  const nativeX = Math.round(fracX * (VIEWER_DATA.native_width - 1));
  const nativeY = Math.round(fracY * (VIEWER_DATA.native_height - 1));

  let valueText = 'no data at this head/slice';
  if (currentGrid) {{
    const px = Math.min(currentGrid.width - 1, Math.floor(fracX * currentGrid.width));
    const py = Math.min(currentGrid.height - 1, Math.floor(fracY * currentGrid.height));
    const idx = (py * currentGrid.width + px) * 4;
    const intensity = currentGrid.data[idx] / 255;
    valueText = intensity.toFixed(3) + (intensity > 0 ? '' : ' (below threshold)');
  }}

  hoverReadout.innerHTML =
    'Head: <strong>' + (headSelect.value || '--') + '</strong><br>' +
    'Slice: <strong>z=' + z + '</strong><br>' +
    'Voxel (x=' + nativeX + ', y=' + nativeY + ', z=' + z + ')<br>' +
    'Heatmap value: <strong>' + valueText + '</strong>';
}}

frameWrap.addEventListener('mousemove', handleHover);
frameWrap.addEventListener('mouseleave', () => {{
  lastHoverEvent = null;
  hoverReadout.textContent = 'Hover over the CT to inspect coordinates and heatmap values.';
}});

function render() {{
  const z = parseInt(zRange.value, 10);
  const zKey = String(z);

  baseImg.src = 'data:image/png;base64,' + (VIEWER_DATA.base_frames[zKey] || '');
  zReadout.textContent = 'Slice: ' + z + ' / ' + VIEWER_DATA.z_max;

  renderMarkers();
  updateOverlay();
}}

zRange.min = VIEWER_DATA.z_min;
zRange.max = VIEWER_DATA.z_max;
zRange.value = VIEWER_DATA.z_min;
zRange.addEventListener('input', render);
alphaRange.addEventListener('input', updateOverlay);
markerToggle.addEventListener('change', renderMarkers);
headSelect.addEventListener('change', updateOverlay);

render();
</script>
</body>
</html>
"""


def build_combined_viewer_html(
    patient_id: str,
    stage02: Dict[str, Any],
    projections: Dict[int, Dict[str, Any]],
    threshold: float,
    default_alpha: float,
    base_frames: Dict[str, str],
    heads_json: Dict[str, Dict[str, Any]],
    show_markers: bool,
) -> str:
    depth, height, width = stage02["volume_shape_zyx"]

    candidates_json: List[Dict[str, Any]] = []
    for candidate_id, cand in sorted(projections.items()):
        center = cand["geometry"]["candidate_center_zyx"]
        z_center = int(round(float(center[0])))
        candidates_json.append(
            {
                "id": candidate_id,
                "z": max(0, min(depth - 1, z_center)),
                "y_frac": float(center[1]) / max(1, height - 1),
                "x_frac": float(center[2]) / max(1, width - 1),
                "heads": sorted(cand["heads"].keys()),
            }
        )

    visible_heads = {
        key: value for key, value in heads_json.items()
        if key != "_candidate_overlays"
    }

    viewer_data = {
        "patient_id": patient_id,
        "z_min": 0,
        "z_max": depth - 1,
        "threshold": threshold,
        "default_alpha": default_alpha,
        "base_frames": base_frames,
        "heads": visible_heads,
        "candidate_overlays": heads_json.get("_candidate_overlays", {}),
        "candidates_all": candidates_json,
        "candidates": candidates_json if show_markers else [],
        "native_width": width,
        "native_height": height,
    }

    return _COMBINED_VIEWER_HTML_TEMPLATE.format(
        patient_id=patient_id,
        num_candidates=len(projections),
        num_heads=len(visible_heads),
        default_alpha=default_alpha,
        frame_width=max(320, min(720, width)),
        viewer_data_json=json.dumps(viewer_data),
    )


# ============================================================================
# CLI / MAIN
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 09 -- combine Stage 08's per-candidate Grad-CAM "
            "projections into one full-CT viewer with all candidates and "
            "a toggle per classifier head."
        )
    )
    parser.add_argument("patient_id", help="Patient/output identifier, e.g. LIDC-IDRI-0141")
    parser.add_argument("--output-root", default="output", help="Root output directory. Default: output")
    parser.add_argument("--stage02-dir", default=None, help="Override Stage 02 directory.")
    parser.add_argument("--stage05-dir", default=None, help="Override Stage 05 directory.")
    parser.add_argument("--stage07-dir", default=None, help="Override Stage 07 directory.")
    parser.add_argument("--stage08-dir", default=None, help="Override Stage 08 visualization directory.")
    parser.add_argument(
        "--stage08-script",
        default=None,
        help=(
            "Path to 08_full_ct_gradcam.py, whose projection functions are "
            f"reused. Default: '{DEFAULT_STAGE08_SCRIPT_NAME}' next to this script."
        ),
    )
    parser.add_argument(
        "--heads",
        nargs="*",
        default=None,
        help="Restrict the viewer to these classifier heads. Default: all Stage 08 projected.",
    )
    parser.add_argument(
        "--candidates",
        nargs="*",
        type=int,
        default=None,
        help="Restrict the viewer to these candidate ids. Default: all Stage 08 PROJECTED candidates.",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Heatmap display threshold in [0,1].")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="Initial overlay blend alpha.")
    parser.add_argument("--max-dim", type=int, default=DEFAULT_MAX_DIM, help="Downsample cap (px) for embedded viewer frames.")
    parser.add_argument("--no-markers", action="store_true", help="Do not draw candidate location markers.")
    parser.add_argument("--out", default=None, help="Override output HTML path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    patient_dir = Path(args.output_root) / args.patient_id
    stage02_dir = Path(args.stage02_dir) if args.stage02_dir else patient_dir / "02"
    stage05_dir = Path(args.stage05_dir) if args.stage05_dir else patient_dir / "05_classifier_patches"
    stage07_dir = Path(args.stage07_dir) if args.stage07_dir else patient_dir / "07_gradcam"
    stage08_dir = Path(args.stage08_dir) if args.stage08_dir else patient_dir / "08_visualization"

    stage08_script = (
        Path(args.stage08_script)
        if args.stage08_script
        else Path(__file__).resolve().parent / DEFAULT_STAGE08_SCRIPT_NAME
    )

    output_dir = patient_dir / "09_presentation"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_html_path = Path(args.out) if args.out else output_dir / "viewer.html"
    manifest_path = output_dir / "manifest.json"

    print(f"Stage 08 script   : {stage08_script}")
    print(f"Stage 08 report   : {stage08_dir / 'report.json'}")
    print(f"Stage 02 (CT)     : {stage02_dir}")
    print(f"Stage 05 (geom)   : {stage05_dir}")
    print(f"Stage 07 (CAM)    : {stage07_dir}")
    print(f"Output            : {out_html_path}")

    mod = load_stage08_module(stage08_script)

    report = load_stage08_report(stage08_dir / "report.json")
    report_candidates = projected_candidates_from_report(report, args.candidates)

    if not report_candidates:
        raise RuntimeError(
            "Stage 08's report.json contains no candidates with "
            "\"status\": \"PROJECTED\" (after applying --candidates, if "
            "given). Nothing for Stage 09 to present."
        )

    print(f"Candidates (from Stage 08 report, status=PROJECTED): {sorted(report_candidates)}")

    stage02 = mod.load_stage02(stage02_dir)
    print(f"CT volume shape   : {stage02['volume_shape_zyx']}")

    stage05_manifest = mod.load_stage05_manifest(stage05_dir)

    candidate_dirs = {
        mod.extract_candidate_id(d): d
        for d in mod.discover_candidate_directories(stage07_dir)
    }

    projections: Dict[int, Dict[str, Any]] = {}
    skipped: List[Dict[str, Any]] = []

    for candidate_id, report_entry in sorted(report_candidates.items()):
        candidate_dir = candidate_dirs.get(candidate_id)
        if candidate_dir is None:
            skipped.append({"candidate_id": candidate_id, "reason": "STAGE07_DIR_NOT_FOUND"})
            print(f"  [SKIP] candidate {candidate_id}: Stage 07 directory not found")
            continue

        allowed_heads = report_entry.get("projected_heads") or []
        result = compute_candidate_head_projections(
            mod=mod,
            candidate_dir=candidate_dir,
            stage02=stage02,
            stage05_manifest=stage05_manifest,
            allowed_heads=allowed_heads,
            heads_filter=args.heads,
        )

        if result is None:
            skipped.append({"candidate_id": candidate_id, "reason": "NO_HEADS_AFTER_FILTERING"})
            print(f"  [SKIP] candidate {candidate_id}: no heads left after filtering/re-projection")
            continue

        projections[candidate_id] = result
        print(f"  [OK]   candidate {candidate_id}: heads = {sorted(result['heads'].keys())}")

    if not projections:
        raise RuntimeError("No candidates survived re-projection; nothing to render.")

    full_shape_yx = tuple(stage02["volume_shape_zyx"][1:])
    depth = stage02["volume_shape_zyx"][0]

    print("Rendering base CT frames (full volume)...")
    base_frames = build_base_frames(mod, stage02["ct"], args.max_dim)

    print("Encoding per-candidate projected Grad-CAM overlays...")
    candidate_overlays, head_colors = build_candidate_head_overlays(
        mod=mod,
        projections=projections,
        full_shape_yx=full_shape_yx,
        depth=depth,
        threshold=args.threshold,
        max_dim=args.max_dim,
    )
    heads_json = {
        head: {"color": color}
        for head, color in sorted(head_colors.items())
    }
    # Internal transport for build_combined_viewer_html; removed from the
    # dropdown model before the browser consumes the head list.
    heads_json["_candidate_overlays"] = candidate_overlays

    if len(heads_json) <= 1:
        print(
            "WARNING: no head has any voxel at/above threshold "
            f"{args.threshold} after combining candidates -- the viewer "
            "will show the CT with no overlays and no head toggles."
        )

    html = build_combined_viewer_html(
        patient_id=args.patient_id,
        stage02=stage02,
        projections=projections,
        threshold=args.threshold,
        default_alpha=args.alpha,
        base_frames=base_frames,
        heads_json=heads_json,
        show_markers=not args.no_markers,
    )

    out_html_path.parent.mkdir(parents=True, exist_ok=True)
    out_html_path.write_text(html, encoding="utf-8")
    print(f"\nSaved combined viewer: {out_html_path}")

    manifest = {
        "stage": 9,
        "patient_id": args.patient_id,
        "source_stage08_report": str(stage08_dir / "report.json"),
        "stage08_script": str(stage08_script),
        "threshold": args.threshold,
        "default_alpha": args.alpha,
        "max_dim": args.max_dim,
        "native_volume_shape_zyx": list(stage02["volume_shape_zyx"]),
        "num_candidates_included": len(projections),
        "candidate_ids_included": sorted(projections.keys()),
        "heads_included": sorted(k for k in heads_json.keys() if k != "_candidate_overlays"),
        "skipped_candidates": skipped,
        "viewer_path": str(out_html_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest       : {manifest_path}")


if __name__ == "__main__":
    main()