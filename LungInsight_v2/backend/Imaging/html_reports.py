"""Headless, static HTML reports for accepted imaging nodules."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
from PIL import Image, ImageDraw


PAGES = ("index.html", "masks.html", "nodules.html", "gradcam.html", "brock.html", "lineage.html")


def _href(value: Any) -> str:
    return html.escape(quote(str(value).replace("\\", "/"), safe="/:._-#"), quote=True)


def _relative_asset(path: Path, patient_root: Path, report_root: Path) -> str:
    try:
        relative_to_patient = path.resolve().relative_to(patient_root.resolve())
        return Path("..", "..", *relative_to_patient.parts).as_posix()
    except ValueError:
        return path.resolve().as_uri()


def _write_slice_assets(
    report_root: Path,
    ct: np.ndarray | None,
    lung_mask: np.ndarray | None,
    heart_mask: np.ndarray | None,
) -> list[tuple[str, str]]:
    if ct is None or lung_mask is None or heart_mask is None:
        return []
    if ct.ndim != 3 or lung_mask.shape != ct.shape or heart_mask.shape != ct.shape:
        raise ValueError("Report CT and masks must share the same 3-D ZYX geometry.")
    z = ct.shape[0] // 2
    image = np.clip((np.asarray(ct[z], dtype=np.float32) + 1000.0) / 1400.0 * 255.0, 0, 255).astype(np.uint8)
    base = np.repeat(image[..., None], 3, axis=-1).astype(np.float32)
    lung = np.asarray(lung_mask[z], dtype=bool)
    heart = np.asarray(heart_mask[z], dtype=bool)
    base[lung] = 0.65 * base[lung] + 0.35 * np.asarray([40, 210, 80], dtype=np.float32)
    base[heart] = 0.45 * base[heart] + 0.55 * np.asarray([240, 55, 60], dtype=np.float32)
    assets = report_root / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    ct_path = assets / "ct_central_slice.png"
    overlay_path = assets / "mask_overlay_central_slice.png"
    Image.fromarray(np.repeat(image[..., None], 3, axis=-1)).save(ct_path)
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)).save(overlay_path)
    lung_path = assets / "lung_mask_central_slice.png"
    heart_path = assets / "heart_mask_central_slice.png"
    Image.fromarray((lung.astype(np.uint8) * 255)).save(lung_path)
    Image.fromarray((heart.astype(np.uint8) * 255)).save(heart_path)
    return [
        ("Original CT central axial slice", "assets/ct_central_slice.png"),
        ("Lung mask central axial slice", "assets/lung_mask_central_slice.png"),
        ("Heart mask central axial slice", "assets/heart_mask_central_slice.png"),
        ("Lung and heart mask overlay (lung green, heart red)", "assets/mask_overlay_central_slice.png"),
    ]


def _overlay_cam(ct_slice: np.ndarray, cam_slice: np.ndarray, output: Path) -> None:
    if ct_slice.ndim != 2 or cam_slice.ndim != 2 or ct_slice.shape != cam_slice.shape:
        raise ValueError("CAM preview CT and CAM slices must be aligned 2-D arrays.")
    hu = np.asarray(ct_slice, dtype=np.float32)
    gray = np.clip((hu + 1000.0) / 1400.0, 0.0, 1.0)
    heat = np.asarray(cam_slice, dtype=np.float32)
    finite = np.isfinite(heat)
    if not finite.any():
        raise ValueError(f"CAM preview has no covered voxels: {output}.")
    spread = float(heat[finite].max() - heat[finite].min())
    normalized = np.zeros_like(heat)
    if spread > 1e-12:
        normalized[finite] = (heat[finite] - heat[finite].min()) / spread
    red = np.clip(1.5 * normalized, 0.0, 1.0)
    green = np.clip(1.5 - 1.5 * np.abs(2.0 * normalized - 1.0), 0.0, 1.0)
    blue = np.clip(1.5 * (1.0 - normalized), 0.0, 1.0)
    rgb = np.repeat((gray * 255.0)[..., None], 3, axis=-1)
    colors = np.stack((red, green, blue), axis=-1) * 255.0
    opacity = (finite & (np.abs(heat) > 1e-12))[..., None] * 0.48
    rgb = rgb * (1.0 - opacity) + colors * opacity
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(output)


def _write_nodule_box(
    ct: np.ndarray | None,
    record: dict[str, Any],
    stable_id: str,
    report_root: Path,
) -> str | None:
    if ct is None or ct.ndim != 3:
        return None
    geometry = record.get("geometry")
    if not isinstance(geometry, dict):
        return None
    center = np.asarray(geometry.get("center_zyx"), dtype=np.float64)
    start = np.asarray(geometry.get("bbox_start_zyx"), dtype=np.int64)
    end = np.asarray(geometry.get("bbox_end_zyx_exclusive"), dtype=np.int64)
    if center.shape != (3,) or start.shape != (3,) or end.shape != (3,):
        return None
    z = int(np.rint(center[0]))
    if (
        not np.isfinite(center).all()
        or not np.isfinite(start).all()
        or not np.isfinite(end).all()
        or not 0 <= z < ct.shape[0]
        or np.any(start < 0)
        or np.any(end <= start)
        or np.any(end > np.asarray(ct.shape))
        or np.any(center < start)
        or np.any(center >= end)
    ):
        return None
    gray = np.clip((np.asarray(ct[z], dtype=np.float32) + 1000.0) / 1400.0 * 255.0, 0, 255).astype(np.uint8)
    image = Image.fromarray(np.repeat(gray[..., None], 3, axis=-1))
    draw = ImageDraw.Draw(image)
    x0 = max(0, int(start[2]))
    y0 = max(0, int(start[1]))
    x1 = min(ct.shape[2] - 1, int(end[2]) - 1)
    y1 = min(ct.shape[1] - 1, int(end[1]) - 1)
    draw.rectangle((x0, y0, x1, y1), outline=(255, 0, 255), width=2)
    draw.text((x0, max(0, y0 - 12)), stable_id, fill=(255, 0, 255))
    filename = "nodule_" + "".join(char for char in stable_id if char.isalnum() or char in "-_") + ".png"
    path = report_root / "assets" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return f"assets/{filename}"


def _brock_decomposition_html(brock: dict[str, Any]) -> str:
    """Render Brock inputs, coefficients, and logit contributions as a table."""
    contributions = [item for item in brock.get("feature_contributions", []) if isinstance(item, dict)]
    if not contributions:
        return "<p class=\"unavailable\">Brock formula decomposition unavailable.</p>"
    rows: list[str] = []
    for item in contributions:
        feature = str(item.get("feature"))
        if "input_value" in item:
            unit = f" {item['unit']}" if item.get("unit") else ""
            input_text = f"{item['input_value']}{unit}"
        elif feature == "intercept":
            input_text = "—"
        else:
            input_text = f"indicator {item.get('encoded_value')}"
        coefficient = item.get("coefficient")
        rows.append(
            "<tr>"
            f"<td>{html.escape(feature)}</td>"
            f"<td>{html.escape(input_text)}</td>"
            f"<td>{html.escape('—' if coefficient is None else str(coefficient))}</td>"
            f"<td>{html.escape(str(item.get('contribution_logit')))}</td>"
            "</tr>"
        )
    rows.append(
        "<tr><th>Total logit</th><td>—</td><td>—</td>"
        f"<td>{html.escape(str(brock.get('logit', 'unavailable')))}</td></tr>"
        "<tr><th>Probability (sigmoid of total logit)</th><td>—</td><td>—</td>"
        f"<td>{html.escape(str(brock.get('probability', 'unavailable')))}</td></tr>"
    )
    return (
        "<table><thead><tr><th>Feature</th><th>Input</th><th>Coefficient</th>"
        "<th>Logit contribution</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def _cam_sprite(cam: np.ndarray) -> np.ndarray:
    """Encode a patch CAM as a vertical RGBA sprite for canvas rendering."""
    values = np.asarray(cam, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"Interactive viewer CAM must be ZYX, got {values.shape}.")
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("Interactive viewer CAM contains no finite values.")

    normalized = np.zeros_like(values)
    low = float(values[finite].min())
    high = float(values[finite].max())
    if high - low > 1e-12:
        normalized[finite] = (values[finite] - low) / (high - low)

    red = np.clip(1.5 * normalized, 0.0, 1.0)
    green = np.clip(1.5 - 1.5 * np.abs(2.0 * normalized - 1.0), 0.0, 1.0)
    blue = np.clip(1.5 * (1.0 - normalized), 0.0, 1.0)
    rgba = np.stack((red, green, blue, finite.astype(np.float32)), axis=-1)
    zyx = np.clip(rgba * 255.0, 0, 255).astype(np.uint8)
    return zyx.reshape(values.shape[0] * values.shape[1], values.shape[2], 4)


def _write_interactive_viewer_assets(
    report_root: Path,
    patient_root: Path,
    accepted: list[dict[str, Any]],
    ct: np.ndarray | None,
    lung_mask: np.ndarray | None,
    heart_mask: np.ndarray | None,
) -> dict[str, Any] | None:
    if ct is None or ct.ndim != 3:
        return None
    if lung_mask is not None and lung_mask.shape != ct.shape:
        raise ValueError("Interactive viewer lung mask must share CT ZYX geometry.")
    if heart_mask is not None and heart_mask.shape != ct.shape:
        raise ValueError("Interactive viewer heart mask must share CT ZYX geometry.")

    assets = report_root / "assets" / "viewer"
    assets.mkdir(parents=True, exist_ok=True)
    ct_paths: list[str] = []
    lung_paths: list[str] = []
    heart_paths: list[str] = []
    for z, ct_slice in enumerate(np.asarray(ct, dtype=np.float32)):
        image = np.clip((ct_slice + 1000.0) / 1400.0 * 255.0, 0, 255).astype(np.uint8)
        path = assets / f"ct_z_{z:03d}.png"
        Image.fromarray(image).save(path)
        ct_paths.append(path.relative_to(report_root).as_posix())
        for mask, color, paths, label in (
            (lung_mask, (40, 210, 80), lung_paths, "lung"),
            (heart_mask, (240, 55, 60), heart_paths, "heart"),
        ):
            if mask is None:
                continue
            rgba = np.zeros((*ct_slice.shape, 4), dtype=np.uint8)
            selected = np.asarray(mask[z], dtype=bool)
            rgba[selected, :3] = color
            rgba[selected, 3] = 110
            overlay_path = assets / f"{label}_z_{z:03d}.png"
            Image.fromarray(rgba, mode="RGBA").save(overlay_path)
            paths.append(overlay_path.relative_to(report_root).as_posix())

    nodules: list[dict[str, Any]] = []
    for row in accepted:
        record = row.get("stage06_record")
        if not isinstance(record, dict):
            continue
        stable_id = str(row.get("stable_nodule_id", record.get("stable_nodule_id", "")))
        geometry = record.get("geometry")
        if not stable_id or not isinstance(geometry, dict):
            continue
        crop_start = geometry.get("crop_start_zyx")
        patch_shape = geometry.get("patch_size_vox_zyx")
        center = geometry.get("center_zyx")
        bbox_start = geometry.get("bbox_start_zyx")
        bbox_end = geometry.get("bbox_end_zyx_exclusive")
        if not all(isinstance(value, list) and len(value) == 3 for value in (
            crop_start, patch_shape, center, bbox_start, bbox_end,
        )):
            continue

        gradcam = record.get("explanations", {}).get("gradcam", {})
        cams: dict[str, dict[str, Any]] = {}
        if isinstance(gradcam, dict):
            for attribute, refs in sorted(gradcam.items()):
                if not isinstance(refs, dict) or not refs.get("patch_cam_path"):
                    continue
                cam_path = Path(str(refs["patch_cam_path"]))
                if not cam_path.is_absolute():
                    cam_path = patient_root / cam_path
                if not cam_path.is_file():
                    continue
                cam = np.load(cam_path, allow_pickle=False)
                if tuple(cam.shape) != tuple(int(value) for value in patch_shape):
                    raise ValueError(
                        f"{cam_path}: Grad-CAM shape {cam.shape} does not match "
                        f"Stage 06 patch geometry {patch_shape}."
                    )
                sprite_path = assets / f"{stable_id}_{attribute}_cam_sprite.png"
                Image.fromarray(_cam_sprite(cam), mode="RGBA").save(sprite_path)
                cams[str(attribute)] = {
                    "sprite": sprite_path.relative_to(report_root).as_posix(),
                    "target_scalar": refs.get("target_scalar_definition"),
                }

        brock = record.get("malignancy", {}).get("brock", {})
        nodules.append(
            {
                "id": stable_id,
                "source": record.get("detector", {}).get("source"),
                "probability": brock.get("probability") if isinstance(brock, dict) else None,
                "brock": brock,
                "diameter_mm": record.get("geometry", {}).get("diameter_mm"),
                "classification": {
                    name: value.get("expected_rating")
                    for name, value in record.get("attributes", {}).items()
                    if isinstance(value, dict) and value.get("expected_rating") is not None
                },
                "center_zyx": center,
                "bbox_start_zyx": bbox_start,
                "bbox_end_zyx_exclusive": bbox_end,
                "crop_start_zyx": crop_start,
                "patch_shape_zyx": patch_shape,
                "cams": cams,
                "formula_contributions": brock.get("feature_contributions", []) if isinstance(brock, dict) else [],
            }
        )
    nodules.sort(
        key=lambda nodule: (
            -float(nodule["probability"]) if isinstance(nodule["probability"], (int, float)) else float("inf"),
            nodule["id"],
        )
    )
    return {
        "ct_slices": ct_paths,
        "lung_mask_slices": lung_paths,
        "heart_mask_slices": heart_paths,
        "ct_shape_zyx": list(ct.shape),
        "nodules": nodules,
    }


def _interactive_viewer_body(viewer_data: dict[str, Any] | None, *, tab: str) -> str:
    if not viewer_data or not viewer_data["nodules"]:
        return '<p class="unavailable">Interactive CT viewer unavailable: no aligned CT, Grad-CAM, and nodule records were found.</p>'

    payload = json.dumps(viewer_data, ensure_ascii=False).replace("<", "\\u003c")
    return (
        "<section class='viewer'>"
        f"<h2>Interactive {html.escape(tab.title())} viewer</h2>"
        "<p>Nodules with available Brock probabilities are ordered highest first. Check one or more nodules to display their bounding boxes, Grad-CAM overlays, and Brock formula decomposition.</p>"
        "<div class='viewer-controls'>"
        "<label>Grad-CAM attribute <select id='viewer-attribute'></select></label>"
        "<label>Z slice <input id='viewer-z' type='range' step='1'><output id='viewer-z-value'></output></label>"
        "<label>Grad-CAM alpha <input id='viewer-alpha' type='range' min='0' max='1' step='0.05' value='0.48'><output id='viewer-alpha-value'>0.48</output></label>"
        "<label><input id='viewer-show-masks' type='checkbox' checked> Show lung and heart masks</label>"
        "</div>"
        "<fieldset class='nodule-checklist'><legend>Show nodule Grad-CAM</legend><div id='viewer-nodules'></div></fieldset>"
        "<p id='viewer-status' class='viewer-status'></p>"
        f"<canvas id='viewer-canvas' width='{int(viewer_data['ct_shape_zyx'][2])}' "
        f"height='{int(viewer_data['ct_shape_zyx'][1])}' "
        "aria-label='Axial CT slice with selected eligible nodule bounding boxes and Grad-CAM overlay'></canvas>"
        "<section class='shap-panel'><h3>Last checked nodule Brock explanation</h3><p id='viewer-rating'></p><div id='viewer-shap'></div></section>"
        "</section>"
        f"<script id='viewer-data' type='application/json'>{payload}</script>"
        "<script>"
        "(()=>{const data=JSON.parse(document.getElementById('viewer-data').textContent);"
        "const checklist=document.getElementById('viewer-nodules'),attributeSelect=document.getElementById('viewer-attribute'),zInput=document.getElementById('viewer-z'),"
        "alphaInput=document.getElementById('viewer-alpha'),showMasks=document.getElementById('viewer-show-masks'),canvas=document.getElementById('viewer-canvas'),ctx=canvas.getContext('2d'),"
        "status=document.getElementById('viewer-status'),rating=document.getElementById('viewer-rating'),shap=document.getElementById('viewer-shap');"
        "const ctImages=data.ct_slices.map(src=>{const image=new Image();image.src=src;return image}),lungImages=(data.lung_mask_slices||[]).map(src=>{const image=new Image();image.src=src;return image}),heartImages=(data.heart_mask_slices||[]).map(src=>{const image=new Image();image.src=src;return image}),sprites=new Map();"
        "for(const nodule of data.nodules){for(const [attribute,cam] of Object.entries(nodule.cams)){const image=new Image();image.src=cam.sprite;sprites.set(nodule.id+'|'+attribute,image)}}"
        "let activeId=data.nodules[0].id;for(const [index,nodule] of data.nodules.entries()){const label=document.createElement('label'),box=document.createElement('input');box.type='checkbox';box.value=nodule.id;box.checked=index===0;box.dataset.noduleId=nodule.id;"
        "const classification=Object.entries(nodule.classification||{}).map(([name,value])=>name+': '+value).join(', ')||'classification unavailable',explanations=Object.keys(nodule.cams||{}).join(', ')||'explanation unavailable';"
        "label.append(box,document.createTextNode(' '+(index+1)+'. '+nodule.id+' | '+(nodule.diameter_mm??'unknown')+' mm | Brock '+(nodule.brock?.probability??'unavailable')+' ('+(nodule.brock?.status||'unavailable')+') | '+(nodule.source||'unknown source')+' | '+classification+' | Grad-CAM: '+explanations));checklist.append(label)}"
        "zInput.min='0';zInput.max=String(data.ct_shape_zyx[0]-1);"
        "function selected(){return data.nodules.find(nodule=>nodule.id===activeId)||data.nodules[0]}function checked(){return [...checklist.querySelectorAll('input:checked')].map(box=>data.nodules.find(nodule=>nodule.id===box.value)).filter(Boolean)}"
        "function populateAttributes(){const nodule=selected(),current=attributeSelect.value;attributeSelect.replaceChildren();const attributes=Object.keys(nodule.cams);"
        "for(const attribute of attributes){const option=document.createElement('option');option.value=attribute;option.textContent=attribute;attributeSelect.append(option)}"
        "if(attributes.includes(current))attributeSelect.value=current;zInput.value=String(Math.round(nodule.center_zyx[0]));renderShap();}"
        "function renderShap(){const nodule=selected();rating.textContent='Brock probability: '+(nodule.brock?.probability??'unavailable')+' ('+(nodule.brock?.status||'unavailable')+').';shap.replaceChildren();"
        "if(!nodule.formula_contributions.length){shap.textContent='Brock formula explanation unavailable.';return}"
        "for(const item of nodule.formula_contributions){const row=document.createElement('div');row.className='shap-row';row.textContent=item.feature+' (input '+(item.input_value??item.encoded_value??'n/a')+'): logit contribution '+(item.contribution_logit??'unavailable');shap.append(row)}}"
        "function render(){const z=Number(zInput.value),attribute=attributeSelect.value,ctImage=ctImages[z],selectedNodules=checked();"
        "document.getElementById('viewer-z-value').textContent=String(z);document.getElementById('viewer-alpha-value').textContent=Number(alphaInput.value).toFixed(2);"
        "ctx.clearRect(0,0,canvas.width,canvas.height);if(ctImage.complete)ctx.drawImage(ctImage,0,0,canvas.width,canvas.height);else ctImage.onload=render;"
        "if(showMasks.checked){for(const image of [lungImages[z],heartImages[z]]){if(image&&image.complete)ctx.drawImage(image,0,0,canvas.width,canvas.height)}}"
        "let overlays=0;for(const nodule of selectedNodules){const start=nodule.bbox_start_zyx,end=nodule.bbox_end_zyx_exclusive;"
        "if(z>=start[0]&&z<end[0]){ctx.strokeStyle='#ff00ff';ctx.lineWidth=2;ctx.strokeRect(start[2],start[1],end[2]-start[2],end[1]-start[1]);ctx.fillStyle='#ff00ff';ctx.font='12px sans-serif';ctx.fillText(nodule.id,start[2],Math.max(12,start[1]-3))}"
        "const cam=nodule.cams[attribute];if(!cam)continue;const relativeZ=z-nodule.crop_start_zyx[0],shape=nodule.patch_shape_zyx,sprite=sprites.get(nodule.id+'|'+attribute);"
        "if(relativeZ>=0&&relativeZ<shape[0]){const sx=Math.max(0,-nodule.crop_start_zyx[2]),sy=Math.max(0,-nodule.crop_start_zyx[1]),dx=Math.max(0,nodule.crop_start_zyx[2]),dy=Math.max(0,nodule.crop_start_zyx[1]),width=Math.min(shape[2]-sx,canvas.width-dx),height=Math.min(shape[1]-sy,canvas.height-dy);"
        "if(sprite.complete&&width>0&&height>0){ctx.save();ctx.globalAlpha=Number(alphaInput.value);ctx.drawImage(sprite,sx,relativeZ*shape[1]+sy,width,height,dx,dy,width,height);ctx.restore();overlays++}else sprite.onload=render}}"
        "status.textContent='CT slice '+z+'. Showing '+overlays+' '+attribute+' Grad-CAM overlay(s) for '+selectedNodules.length+' checked nodule(s).'}"
        "checklist.addEventListener('change',event=>{const box=event.target;if(box.checked){activeId=box.value;const nodule=selected();zInput.value=String(Math.round(nodule.center_zyx[0]));populateAttributes()}render()});attributeSelect.addEventListener('change',render);zInput.addEventListener('input',render);alphaInput.addEventListener('input',render);showMasks.addEventListener('change',render);"
        "populateAttributes();render();})();</script>"
    )


def generate_patient_report(
    *,
    patient_id: str,
    patient_root: Path,
    report_root: Path,
    accepted: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    ct: np.ndarray | None = None,
    lung_mask: np.ndarray | None = None,
    heart_mask: np.ndarray | None = None,
    input_paths: dict[str, str] | None = None,
) -> list[Path]:
    """Write six navigable report pages; missing artifacts remain explicit."""
    report_root.mkdir(parents=True, exist_ok=True)
    images = _write_slice_assets(report_root, ct, lung_mask, heart_mask)
    navigation = " ".join(
        f'<a href="{_href(page)}">{html.escape(page.removesuffix(".html").title())}</a>'
        for page in PAGES
    )
    style = (
        "<style>body{font:15px sans-serif;max-width:1200px;margin:2rem auto;padding:0 1rem;color:#202124}"
        "nav{display:flex;gap:1rem;flex-wrap:wrap;margin:1.2rem 0}table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #ddd;padding:.45rem;text-align:left;vertical-align:top}"
        "th{background:#f4f4f4}img{max-width:48%;height:auto;margin:.4rem}code{overflow-wrap:anywhere}"
        ".unavailable{color:#8a4b00}.viewer{margin:1.5rem 0;padding:1rem;border:1px solid #ccd6e0;border-radius:.5rem;background:#f8fafc}"
        ".viewer-controls{display:flex;gap:1rem;flex-wrap:wrap;align-items:center;margin:1rem 0}.viewer-controls label{display:flex;gap:.45rem;align-items:center}"
        "#viewer-canvas{display:block;width:min(100%,512px);height:auto;border:1px solid #637487;background:#000;image-rendering:auto}"
        ".viewer-status{min-height:1.4rem}.shap-panel{margin-top:1rem;background:#fff;padding:1rem;border-radius:.35rem}.shap-row{display:flex;align-items:center;gap:.5rem;margin:.35rem 0}"
        ".shap-row>span:first-child{width:15rem;max-width:55%;overflow-wrap:anywhere}.shap-bar{height:1rem;min-width:2px}.shap-bar.positive{background:#b42318}.shap-bar.negative{background:#175cd3}"
        ".nodule-checklist{max-height:15rem;overflow:auto;background:#fff;border:1px solid #ccd6e0}.nodule-checklist label{display:block;padding:.25rem .5rem;cursor:pointer}.nodule-checklist label:nth-child(even){background:#f4f7fa}</style>"
    )

    def page(title: str, body: str) -> str:
        return (
            "<!doctype html><html lang='en'><meta charset='utf-8'>"
            f"<title>{html.escape(title)}</title>{style}<body><h1>{html.escape(title)}</h1>"
            f"<nav>{navigation}</nav>{body}</body></html>"
        )

    def artifact_link(raw: Any) -> str:
        if not raw:
            return '<span class="unavailable">Unavailable</span>'
        target = Path(str(raw))
        if not target.is_absolute():
            target = patient_root / target
        href = _relative_asset(target, patient_root, report_root)
        return f'<a href="{_href(href)}">{html.escape(str(raw))}</a>'

    masks_body = []
    for label, path in images:
        masks_body.append(f"<figure><img src='{_href(path)}' alt='{html.escape(label)}'><figcaption>{html.escape(label)}</figcaption></figure>")
    review_paths = sorted(patient_root.glob("heart_mask_review_montage_*.png"))
    for path in review_paths:
        href = _relative_asset(path, patient_root, report_root)
        masks_body.append(
            f"<p>Review montage: <a href='{_href(href)}'>{html.escape(path.name)}</a></p>"
        )
    if not masks_body:
        masks_body.append('<p class="unavailable">CT or aligned masks are unavailable.</p>')

    nodule_rows = []
    gradcam_sections = []
    shap_sections = []
    lineage_sections = []
    for row in accepted:
        record = row.get("stage06_record", {})
        stable_id = str(row.get("stable_nodule_id", record.get("stable_nodule_id", record.get("nodule_id", ""))))
        if not stable_id:
            raise ValueError("Accepted report row lacks stable_nodule_id.")
        detector = record.get("detector", {})
        malignancy = record.get("malignancy", {})
        brock = malignancy.get("brock", {}) if isinstance(malignancy, dict) else {}
        selected = malignancy.get("selected_prediction", {}) if isinstance(malignancy, dict) else {}
        geometry = record.get("geometry", {})
        attributes = record.get("attributes")
        attributes = attributes if isinstance(attributes, dict) else {}
        classification = ", ".join(
            f"{name}: {value.get('expected_rating')}"
            for name, value in sorted(attributes.items())
            if isinstance(value, dict) and value.get("expected_rating") is not None
        ) or "Unavailable"
        bbox_preview = _write_nodule_box(ct, record, stable_id, report_root)
        preview_cell = (
            f"<img src='{_href(bbox_preview)}' alt='CT bounding box for {html.escape(stable_id)}'>"
            if bbox_preview else '<span class="unavailable">Bounding-box preview unavailable</span>'
        )
        nodule_rows.append(
            "<tr>"
            f"<td><code>{html.escape(stable_id)}</code></td>"
            f"<td>{html.escape(str(detector.get('source', row.get('source', 'Unavailable'))))}</td>"
            f"<td>ViTDet3D: {html.escape(str(detector.get('vitdet_score')))}</td>"
            f"<td>Diameter: {html.escape(str(geometry.get('diameter_mm', 'unavailable')))} mm<br>"
            f"Volume: {html.escape(str(geometry.get('volume_mm3', 'unavailable')))} mm³</td>"
            f"<td>{html.escape(classification)}</td>"
            f"<td>Selected: {html.escape(str(selected.get('method', 'not specified')))} "
            f"({html.escape(str(selected.get('status', 'unavailable')))})"
            f"<br>Selected value: {html.escape(str(selected.get('value', 'unavailable')))}"
            f"<br>Brock risk estimate: {html.escape(str(brock.get('probability', 'unavailable')))} "
            f"({html.escape(str(brock.get('status', 'unavailable')))})</td>"
            f"<td>{html.escape(str(row.get('center_lps_x_mm')))}, {html.escape(str(row.get('center_lps_y_mm')))}, {html.escape(str(row.get('center_lps_z_mm')))}</td>"
            f"<td>{preview_cell}</td>"
            "</tr>"
        )
        explanations = record.get("explanations", {})
        gradcam = explanations.get("gradcam", {}) if isinstance(explanations, dict) else {}
        cam_lines = [f"<h3>{html.escape(name)}</h3>" for name in sorted(gradcam)]
        for name, refs in sorted(gradcam.items()):
            refs = refs if isinstance(refs, dict) else {}
            patch_preview = None
            ct_preview = None
            if ct is not None:
                patch_path = refs.get("patch_cam_path")
                ct_cam_path = refs.get("ct_cam_path")
                crop = record.get("geometry", {}).get("crop_start_zyx")
                center = record.get("geometry", {}).get("center_zyx")
                if patch_path and crop is not None and center is not None:
                    patch_cam_file = Path(str(patch_path))
                    if not patch_cam_file.is_absolute():
                        patch_cam_file = patient_root / patch_cam_file
                    if patch_cam_file.is_file():
                        patch_cam = np.load(patch_cam_file, allow_pickle=False)
                        patch_z = int(np.rint(float(center[0]))) - int(crop[0])
                        ct_z = int(np.rint(float(center[0])))
                        y0, x0 = int(crop[1]), int(crop[2])
                        if 0 <= patch_z < patch_cam.shape[0] and 0 <= ct_z < ct.shape[0]:
                            y1 = min(ct.shape[1], y0 + patch_cam.shape[1])
                            x1 = min(ct.shape[2], x0 + patch_cam.shape[2])
                            py1, px1 = y1 - y0, x1 - x0
                            if y0 >= 0 and x0 >= 0 and py1 > 0 and px1 > 0:
                                patch_preview = report_root / "assets" / f"{stable_id}_{name}_patch.png"
                                _overlay_cam(
                                    ct[ct_z, y0:y1, x0:x1],
                                    patch_cam[patch_z, :py1, :px1],
                                    patch_preview,
                                )
                if ct_cam_path:
                    ct_cam_file = Path(str(ct_cam_path))
                    if not ct_cam_file.is_absolute():
                        ct_cam_file = patient_root / ct_cam_file
                    if ct_cam_file.is_file():
                        ct_cam = np.load(ct_cam_file, allow_pickle=False)
                        ct_z = int(np.rint(float(center[0]))) if center is not None else ct.shape[0] // 2
                        if ct_cam.shape == ct.shape and 0 <= ct_z < ct.shape[0]:
                            ct_preview = report_root / "assets" / f"{stable_id}_{name}_ct.png"
                            _overlay_cam(ct[ct_z], ct_cam[ct_z], ct_preview)
            preview_html = ""
            if patch_preview:
                preview_html += f"<figure><img src='{_href(patch_preview.relative_to(report_root).as_posix())}' alt='Patch-space {html.escape(name)} Grad-CAM'><figcaption>Patch-space view</figcaption></figure>"
            if ct_preview:
                preview_html += f"<figure><img src='{_href(ct_preview.relative_to(report_root).as_posix())}' alt='CT-space {html.escape(name)} Grad-CAM'><figcaption>Projected CT-space view</figcaption></figure>"
            cam_lines.append(
                "<p>"
                f"Patch CAM: {artifact_link(refs.get('patch_cam_path'))}<br>"
                f"CT-space CAM: {artifact_link(refs.get('ct_cam_path'))}<br>"
                f"Target scalar: {html.escape(str(refs.get('target_scalar_definition')))}<br>"
                f"Target index: {html.escape(str(refs.get('target_output_index')))}<br>"
                f"Checkpoint: {html.escape(str(refs.get('checkpoint')))}<br>"
                f"Target layer: {html.escape(str(refs.get('target_layer')))}"
                f"</p>{preview_html}"
            )
        gradcam_sections.append(
            f"<section><h2>{html.escape(stable_id)}</h2>{''.join(cam_lines) or '<p class=\"unavailable\">No CAM references.</p>'}</section>"
        )
        contributions = brock.get("feature_contributions", [])
        limitations = record.get("uncertainty_and_limitations", {})
        rating_content = (
            f"<p>Selected malignancy method: {html.escape(str(malignancy.get('selected_prediction', {}).get('method', 'not specified')))}; "
            f"status: {html.escape(str(malignancy.get('selected_prediction', {}).get('status', 'unavailable')))}; "
            f"value: {html.escape(str(malignancy.get('selected_prediction', {}).get('value', 'unavailable')))}.</p>"
            f"<p>Brock/PanCan probability: {html.escape(str(brock.get('probability', 'unavailable')))}; "
            f"status: {html.escape(str(brock.get('status', 'unavailable')))}. "
            "This is a research/POC risk estimate, not a pathology diagnosis.</p>"
            f"<p>Uncertainty: {html.escape(json.dumps(record.get('uncertainty', {'status': 'not_estimated'}), ensure_ascii=False))}</p>"
            f"<p>Limitations: {html.escape(json.dumps(limitations, ensure_ascii=False))}</p>"
        )
        shap_content = rating_content + "<h3>Brock formula decomposition</h3>" + _brock_decomposition_html(brock)
        shap_sections.append(f"<section><h2>{html.escape(stable_id)}</h2>{shap_content}</section>")
        lineage_sections.append(
            f"<section><h2>{html.escape(stable_id)}</h2><p>Detection → Stage 03 → Stage 04 → Stage 05 → Stage 06 → Stage 07</p>"
            f"<p>Detector members: <code>{html.escape(json.dumps(detector.get('source_members', []), ensure_ascii=False))}</code></p>"
            f"<p>Stage 06 record: {artifact_link(row.get('stage06_record_path'))}</p></section>"
        )

    nodules_body = (
        "<table><thead><tr><th>Stable ID</th><th>Detector source</th><th>Separate detector scores</th>"
        "<th>Measurements</th><th>Predicted classification attributes</th><th>Model output</th>"
        "<th>Centroid LPS mm</th><th>Bounding box</th></tr></thead><tbody>"
        + "".join(nodule_rows) + "</tbody></table>"
        + f"<p>Accepted: {len(accepted)}; excluded from display: {len(excluded)} (audit only).</p>"
        + "<p>Bounding boxes use magenta. Detector scores are shown separately; no cross-detector confidence color scale is applied.</p>"
    )
    gradcam_body = "".join(gradcam_sections) or "<p>No accepted nodules.</p>"
    shap_body = "".join(shap_sections) or "<p>No accepted nodules.</p>"
    lineage_body = "".join(lineage_sections) or "<p>No accepted nodules.</p>"
    summary = {
        "patient_id": patient_id,
        "accepted_nodules": len(accepted),
        "excluded_nodules_audit_only": len(excluded),
        "inputs": input_paths or {},
        "stable_nodule_ids": [
            row.get("stable_nodule_id", row.get("stage06_record", {}).get("nodule_id"))
            for row in accepted
        ],
    }
    viewer_data = _write_interactive_viewer_assets(
        report_root, patient_root, accepted, ct, lung_mask, heart_mask
    )
    index_body = _interactive_viewer_body(viewer_data, tab="CT, Grad-CAM, and Brock") + f"<h2>Report summary</h2><pre>{html.escape(json.dumps(summary, indent=2))}</pre>"
    pages = {
        "index.html": page(f"LungInsight imaging report — {patient_id}", index_body),
        "masks.html": page(f"Masks — {patient_id}", _interactive_viewer_body(viewer_data, tab="masks") + "".join(masks_body)),
        "nodules.html": page(f"Accepted nodules — {patient_id}", _interactive_viewer_body(viewer_data, tab="nodules") + nodules_body),
        "gradcam.html": page(f"Grad-CAM — {patient_id}", _interactive_viewer_body(viewer_data, tab="Grad-CAM") + gradcam_body),
        "brock.html": page(f"Brock formula explanation — {patient_id}", _interactive_viewer_body(viewer_data, tab="Brock") + shap_body),
        "lineage.html": page(f"Nodule lineage — {patient_id}", lineage_body),
    }
    written = []
    for filename, content in pages.items():
        path = report_root / filename
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written
