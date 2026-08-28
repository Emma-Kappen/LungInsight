"""
pdf_generator.py
=================

Generates a formal, 1-2 page clinical inference report as PDF bytes from the
output of ``ClinicalPipeline.predict()``.

Uses ReportLab (pure Python, no external binary dependency — unlike
WeasyPrint, which needs system libraries) so the app stays portable across
Windows/Colab/Linux deployment targets.
"""

from __future__ import annotations

import base64
import io
from datetime import datetime
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

INK = colors.HexColor("#12181F")
MUTED = colors.HexColor("#5B6672")
ACCENT = colors.HexColor("#1F6F78")
WARN = colors.HexColor("#B23A48")
RULE = colors.HexColor("#D6DBDF")
PANEL_BG = colors.HexColor("#F3F5F6")

FIELD_LABELS = {
    "AGE": "Age",
    "SEX": "Sex",
    "RACE": "Race",
    "ETHNICITY": "Ethnicity",
    "SMOKING_STATUS": "Smoking status",
    "num_treatment_events": "Prior treatment events",
    "num_distinct_tumor_sites": "Distinct tumor sites",
    "ever_pdl1_positive": "Ever PD-L1 positive",
    "eastern_cancer_oncology_group": "ECOG performance status",
    "karnofsky_performance_score": "Karnofsky performance score",
    "SOURCE_DATASET": "Source dataset",
}


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=16, textColor=INK, alignment=TA_LEFT, spaceAfter=2,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle", parent=base["Normal"], fontName="Helvetica",
            fontSize=9, textColor=MUTED, spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=11, textColor=INK, spaceBefore=12, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["Normal"], fontName="Helvetica",
            fontSize=9.5, textColor=INK, leading=13,
        ),
        "small": ParagraphStyle(
            "Small", parent=base["Normal"], fontName="Helvetica",
            fontSize=8, textColor=MUTED, leading=11,
        ),
        "bullet": ParagraphStyle(
            "Bullet", parent=base["Normal"], fontName="Helvetica",
            fontSize=9.5, textColor=INK, leading=13, leftIndent=10,
        ),
        "badge_big": ParagraphStyle(
            "BadgeBig", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=18, textColor=colors.white, alignment=TA_CENTER,
        ),
        "badge_small": ParagraphStyle(
            "BadgeSmall", parent=base["Normal"], fontName="Helvetica",
            fontSize=8, textColor=colors.white, alignment=TA_CENTER,
        ),
    }


def _fmt_pct(x: float | None) -> str:
    return "N/A" if x is None else f"{x * 100:.1f}%"


def _fmt_num(x: float | None, decimals: int = 1, suffix: str = "") -> str:
    return "N/A" if x is None else f"{x:.{decimals}f}{suffix}"


def _stage_color(label: str) -> colors.Color:
    return WARN if "4" in str(label) else ACCENT


def _demographics_table(patient_data: dict[str, Any], styles: dict) -> Table:
    rows = [[Paragraph("<b>Field</b>", styles["small"]), Paragraph("<b>Value</b>", styles["small"])]]
    for key, label in FIELD_LABELS.items():
        raw = patient_data.get(key)
        if raw is None or (isinstance(raw, float) and raw != raw):
            display = "Unknown / not provided"
        else:
            display = str(raw)
        rows.append([Paragraph(label, styles["body"]), Paragraph(display, styles["body"])])

    table = Table(rows, colWidths=[2.3 * inch, 3.4 * inch], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), INK),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PANEL_BG]),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, INK),
                ("GRID", (0, 0), (-1, -1), 0.4, RULE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _diagnostic_summary_box(prediction_results: dict[str, Any], styles: dict) -> Table:
    stage = prediction_results["stage"]
    hist = prediction_results["histology"]

    stage_label = stage["predicted_label"]
    stage_conf = stage["probabilities"].get(stage_label)
    hist_label = hist["predicted_label"]
    hist_conf = hist["probabilities"].get(hist_label)

    stage_cell = Table(
        [
            [Paragraph("PREDICTED STAGE", styles["badge_small"])],
            [Paragraph(str(stage_label), styles["badge_big"])],
            [Paragraph(f"Confidence: {_fmt_pct(stage_conf)}", styles["badge_small"])],
        ],
        colWidths=[2.85 * inch],
    )
    stage_cell.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), _stage_color(stage_label)),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (0, 0), 8),
                ("BOTTOMPADDING", (0, -1), (0, -1), 8),
                ("TOPPADDING", (0, 1), (0, 1), 2),
                ("BOTTOMPADDING", (0, 1), (0, 1), 4),
            ]
        )
    )

    hist_cell = Table(
        [
            [Paragraph("PREDICTED HISTOLOGY", styles["badge_small"])],
            [Paragraph(str(hist_label), ParagraphStyle("hb", parent=styles["badge_big"], fontSize=13))],
            [Paragraph(f"Confidence: {_fmt_pct(hist_conf)}", styles["badge_small"])],
        ],
        colWidths=[2.85 * inch],
    )
    hist_cell.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), INK),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (0, 0), 8),
                ("BOTTOMPADDING", (0, -1), (0, -1), 8),
                ("TOPPADDING", (0, 1), (0, 1), 2),
                ("BOTTOMPADDING", (0, 1), (0, 1), 4),
            ]
        )
    )

    outer = Table([[stage_cell, hist_cell]], colWidths=[2.95 * inch, 2.95 * inch], hAlign="LEFT")
    outer.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 8)]))
    return outer


def _risk_section(prediction_results: dict[str, Any], styles: dict) -> list:
    story: list = []
    explanation = prediction_results.get("explanation", {})
    survival = prediction_results.get("survival", {})

    story.append(Paragraph("Risk &amp; Interpretability", styles["h2"]))

    waterfall_b64 = explanation.get("waterfall_plot_base64")
    if waterfall_b64:
        img_bytes = base64.b64decode(waterfall_b64)
        img_buf = io.BytesIO(img_bytes)
        img = Image(img_buf, width=6.3 * inch, height=6.3 * inch * 0.55)
        story.append(img)
        story.append(Spacer(1, 6))

    pos_feats = explanation.get("top_positive_features", [])
    neg_feats = explanation.get("top_negative_features", [])

    def feature_lines(feats: list[dict[str, Any]], arrow: str, color: colors.Color) -> list[str]:
        lines = []
        for f in feats[:5]:
            lines.append(
                f'<font color="{color.hexval()}">{arrow}</font> '
                f'{f["feature"]} <font color="{MUTED.hexval()}">(SHAP {f["shap_value"]:+.3f})</font>'
            )
        return lines or ["No significant contributors identified."]

    col1 = [Paragraph(f"<b>Increases risk of {explanation.get('explained_class', 'positive class')}</b>", styles["small"])]
    col1 += [Paragraph(line, styles["bullet"]) for line in feature_lines(pos_feats, "\u25B2", WARN)]

    col2 = [Paragraph("<b>Decreases risk</b>", styles["small"])]
    col2 += [Paragraph(line, styles["bullet"]) for line in feature_lines(neg_feats, "\u25BC", ACCENT)]

    feat_table = Table([[col1, col2]], colWidths=[3.1 * inch, 3.1 * inch])
    feat_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    story.append(feat_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Survival Trajectory", styles["h2"]))
    risk_score = survival.get("risk_score")
    median_months = survival.get("median_survival_months")
    surv_text = (
        f"Cox proportional-hazards partial risk score: <b>{_fmt_num(risk_score, 3)}</b> "
        "(higher values indicate higher relative hazard vs. the training cohort baseline). "
    )
    if median_months is not None:
        surv_text += f"Estimated median overall survival: <b>{_fmt_num(median_months, 1, ' months')}</b>."
    else:
        surv_text += "Median survival could not be estimated from the fitted curve for this input."
    story.append(Paragraph(surv_text, styles["body"]))
    return story


def generate_pdf_report(patient_data: dict[str, Any], prediction_results: dict[str, Any]) -> bytes:
    """
    Generate a formal clinical inference report.

    Parameters
    ----------
    patient_data:
        The raw (post-coercion) patient feature dict — typically
        ``prediction_results["input_features"]`` from the pipeline, or the
        original form submission.
    prediction_results:
        The full dict returned by ``ClinicalPipeline.predict()``.

    Returns
    -------
    bytes
        A complete PDF document.
    """
    styles = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=LETTER,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        title="Clinical Inference Report",
    )

    story: list = []

    # --- Header -----------------------------------------------------------
    now = datetime.now().strftime("%B %d, %Y %H:%M")
    patient_id = patient_data.get("PATIENT_ID") or patient_data.get("patient_id") or "Not specified"

    header_table = Table(
        [
            [
                Paragraph("LungInsight Clinical Decision Support", styles["title"]),
                Paragraph(f"Generated: {now}", styles["small"]),
            ],
            [
                Paragraph("Explainable AI Stage / Histology / Survival Report", styles["subtitle"]),
                Paragraph(f"Patient ID: {patient_id}", styles["small"]),
            ],
        ],
        colWidths=[4.3 * inch, 2.5 * inch],
    )
    header_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1, color=INK, spaceBefore=4, spaceAfter=6))
    story.append(
        Paragraph(
            "<b>Confidentiality notice:</b> This report contains protected health information intended "
            "for use by the treating clinical team only. It is generated by an investigational machine "
            "learning decision-support tool and is <b>not a substitute for independent clinical judgment</b>. "
            "All model outputs require review and sign-off by a qualified oncologist or pathologist.",
            styles["small"],
        )
    )
    story.append(Spacer(1, 10))

    # --- Table 1: demographics ---------------------------------------------
    story.append(Paragraph("Baseline Patient Clinical Demographics", styles["h2"]))
    story.append(_demographics_table(patient_data, styles))
    story.append(Spacer(1, 10))

    # --- Diagnostic summary --------------------------------------------------
    story.append(Paragraph("Diagnostic Summary", styles["h2"]))
    story.append(_diagnostic_summary_box(prediction_results, styles))
    story.append(Spacer(1, 4))

    stage = prediction_results["stage"]
    threshold_note = (
        f"Decision threshold for {stage.get('positive_label', 'the positive class')}: "
        f"{_fmt_pct(stage.get('decision_threshold'))} predicted probability."
    )
    story.append(Paragraph(threshold_note, styles["small"]))
    story.append(Spacer(1, 10))

    # --- Risk & interpretability + survival ----------------------------------
    story.extend(_risk_section(prediction_results, styles))
    story.append(Spacer(1, 14))

    # --- Footer / signature block --------------------------------------------
    story.append(HRFlowable(width="100%", thickness=0.75, color=RULE, spaceBefore=4, spaceAfter=10))
    sig_table = Table(
        [
            [
                Paragraph("Reviewing Oncologist — Signature: ____________________________", styles["body"]),
                Paragraph("Date: ______________", styles["body"]),
            ],
            [
                Paragraph("Reviewing Pathologist — Signature: ____________________________", styles["body"]),
                Paragraph("Date: ______________", styles["body"]),
            ],
        ],
        colWidths=[4.3 * inch, 2.5 * inch],
    )
    sig_table.setStyle(
        TableStyle([("TOPPADDING", (0, 0), (-1, -1), 10), ("LEFTPADDING", (0, 0), (-1, -1), 0)])
    )
    story.append(sig_table)
    story.append(Spacer(1, 8))
    story.append(
        Paragraph(
            "This document was produced by an automated statistical/machine-learning model trained on "
            "retrospective clinical cohort data (MSK-CHORD and related sources). Model outputs carry "
            "inherent uncertainty and reflect population-level patterns; they may not generalize to every "
            "individual patient. Do not use this report as the sole basis for a diagnostic or treatment "
            "decision.",
            styles["small"],
        )
    )

    doc.build(story)
    buf.seek(0)
    return buf.read()