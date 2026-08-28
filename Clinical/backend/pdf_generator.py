from __future__ import annotations
import base64, io
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether


def _img_from_b64(b64: str | None, width=85*mm, height=65*mm):
    if not b64: return None
    try:
        data = base64.b64decode(b64); im = Image(io.BytesIO(data)); im.drawWidth=width; im.drawHeight=height; return im
    except Exception: return None


def generate_pdf_report(patient_data: dict, prediction_results: dict) -> bytes:
    buf=io.BytesIO()
    doc=SimpleDocTemplate(buf,pagesize=A4,rightMargin=15*mm,leftMargin=15*mm,topMargin=15*mm,bottomMargin=15*mm)
    styles=getSampleStyleSheet(); styles.add(ParagraphStyle(name='Small',parent=styles['Normal'],fontSize=8,leading=10)); styles.add(ParagraphStyle(name='CenterSmall',parent=styles['Small'],alignment=TA_CENTER)); styles.add(ParagraphStyle(name='Box',parent=styles['Normal'],fontSize=11,leading=15))
    story=[]
    patient_id=patient_data.get('PATIENT_ID','Not provided')
    date=datetime.now().strftime('%d %b %Y %H:%M')
    story += [Paragraph('LUNGINSIGHT — MULTIMODAL CLINICAL & CT INFERENCE REPORT', styles['Title']), Paragraph(f'Patient ID: <b>{patient_id}</b> &nbsp;&nbsp; Submission: {date}', styles['Small']), Paragraph('CONFIDENTIAL — FOR CLINICAL REVIEW ONLY', styles['CenterSmall']), Spacer(1,6*mm)]
    demo=[['Variable','Value']]+[[k,str(v)] for k,v in patient_data.items() if k not in ('PATIENT_ID', 'SOURCE_DATASET')]
    t=Table(demo,colWidths=[75*mm,100*mm]); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#18324a')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),0.25,colors.grey),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8),('VALIGN',(0,0),(-1,-1),'TOP')]))
    story += [Paragraph('1. Demographics & Baseline Clinical Variables',styles['Heading2']),t,Spacer(1,5*mm)]
    stage=prediction_results.get('stage',{}); hist=prediction_results.get('histology',{}); surv=prediction_results.get('survival',{}); imaging=prediction_results.get('imaging',{})
    summary=[[Paragraph('<b>Predicted Stage</b>',styles['Box']),Paragraph(f"{stage.get('label','Unavailable')}<br/>Confidence: {stage.get('confidence',0)*100:.1f}%" if stage.get('confidence') is not None else 'Unavailable',styles['Box'])],[Paragraph('<b>Histology</b>',styles['Box']),Paragraph(f"{hist.get('label','Unavailable')}<br/>Confidence: {hist.get('confidence',0)*100:.1f}%" if hist.get('confidence') is not None else 'Unavailable',styles['Box'])],[Paragraph('<b>Survival Risk Score</b>',styles['Box']),Paragraph(str(round(surv['risk_score'],4)) if surv.get('risk_score') is not None else 'Unavailable',styles['Box'])],[Paragraph('<b>Imaging Nodules</b>',styles['Box']),Paragraph(str(imaging.get('stage04_nodule_count','Unavailable')),styles['Box'])]]
    st=Table(summary,colWidths=[65*mm,110*mm]); st.setStyle(TableStyle([('BOX',(0,0),(-1,-1),0.8,colors.HexColor('#18324a')),('INNERGRID',(0,0),(-1,-1),0.4,colors.lightgrey),('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#eef5fa')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]))
    story += [Paragraph('2. Diagnostic Summary',styles['Heading2']),st,Spacer(1,5*mm)]
    story += [Paragraph('3. Visual Evidence',styles['Heading2'])]
    shap=_img_from_b64(prediction_results.get('shap',{}).get('waterfall_png_b64'),85*mm,62*mm)
    ct=_img_from_b64(prediction_results.get('ct_slice_png_b64'),85*mm,62*mm)
    evidence=Table([[ct or Paragraph('CT key slice not supplied',styles['Small']), shap or Paragraph('SHAP waterfall not available',styles['Small'])]],colWidths=[88*mm,88*mm]); evidence.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.4,colors.grey),('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
    stage09 = prediction_results.get('stage09_viewer_url')
    evidence_note = f'Stage 09 CT viewer: {stage09}' if stage09 else 'Stage 09 CT viewer not available for this run.'
    story += [evidence, Paragraph(evidence_note, styles['Small']), Spacer(1,4*mm), Paragraph('Positive risk drivers',styles['Heading3']), Paragraph(', '.join(f"{x['feature']} ({x['value']:+.3f})" for x in prediction_results.get('shap',{}).get('positive',[])) or 'Not available',styles['Small']), Paragraph('Negative risk drivers',styles['Heading3']), Paragraph(', '.join(f"{x['feature']} ({x['value']:+.3f})" for x in prediction_results.get('shap',{}).get('negative',[])) or 'Not available',styles['Small']), PageBreak()]
    story += [Paragraph('4. Survival Trajectory',styles['Heading2']), Paragraph('The survival curve below is represented numerically in the dashboard. This report records the model risk score and should not be interpreted as an individualized treatment recommendation.',styles['Small']), Spacer(1,4*mm)]
    times=surv.get('times_months',[]); probs=surv.get('survival_probability',[])
    rows=[['Month','Predicted survival']]+[[f'{t:.1f}',f'{p*100:.1f}%'] for t,p in list(zip(times,probs))[::5]]
    tt=Table(rows,colWidths=[55*mm,55*mm]); tt.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#18324a')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),0.25,colors.grey),('FONTSIZE',(0,0),(-1,-1),8)])); story += [tt,Spacer(1,8*mm),Paragraph('5. Clinician Review & Sign-off',styles['Heading2']),Spacer(1,12*mm),Table([['Reviewing Oncologist / Pathologist','Date / Signature'],['________________________________','________________________________']],colWidths=[88*mm,88*mm],rowHeights=[10*mm,15*mm],style=TableStyle([('GRID',(0,0),(-1,-1),0.4,colors.grey),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('FONTSIZE',(0,0),(-1,-1),8)])),Spacer(1,8*mm),Paragraph('Clinical disclaimer: This software is an investigational decision-support system. Predictions, survival estimates, and explanations are model outputs and require review by qualified clinicians. They are not a diagnosis and must not replace pathology, radiology, staging, multidisciplinary review, or clinical judgment.',styles['Small'])]
    doc.build(story)
    return buf.getvalue()
