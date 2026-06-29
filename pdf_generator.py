"""PDF Health Report Generator using ReportLab"""
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, HRFlowable, KeepTogether)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics import renderPDF
import io
from datetime import datetime

BRAND_BLUE = colors.HexColor('#1E3A5F')
BRAND_TEAL = colors.HexColor('#00A8A8')
RISK_RED = colors.HexColor('#E74C3C')
RISK_ORANGE = colors.HexColor('#F39C12')
RISK_GREEN = colors.HexColor('#27AE60')
RISK_YELLOW = colors.HexColor('#F1C40F')
LIGHT_GRAY = colors.HexColor('#F5F7FA')
MID_GRAY = colors.HexColor('#95A5A6')

def risk_color(score):
    if score >= 70: return RISK_RED
    if score >= 40: return RISK_ORANGE
    return RISK_GREEN

def risk_label(score):
    if score >= 70: return "HIGH RISK"
    if score >= 40: return "MODERATE RISK"
    return "LOW RISK"

def draw_risk_bar(score, width=400, height=20):
    d = Drawing(width, height + 10)
    # Background bar
    bg = Rect(0, 5, width, height, fillColor=LIGHT_GRAY, strokeColor=colors.white, strokeWidth=1)
    d.add(bg)
    # Filled portion
    fill_w = int(width * score / 100)
    fill = Rect(0, 5, fill_w, height, fillColor=risk_color(score), strokeColor=None)
    d.add(fill)
    # Score text
    txt = String(fill_w + 5, 8, f"{score}%", fontSize=10, fillColor=colors.black)
    d.add(txt)
    return d

def generate_health_report(patient_info: dict, predictions: dict, shap_info: dict = None) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm,
        title="AI Health Risk Assessment Report"
    )

    styles = getSampleStyleSheet()
    story = []

    # ── Header ──────────────────────────────────────────────────────────
    header_style = ParagraphStyle('Header', fontSize=22, textColor=colors.white,
                                   alignment=TA_CENTER, spaceAfter=4, fontName='Helvetica-Bold')
    sub_header_style = ParagraphStyle('SubHeader', fontSize=11, textColor=colors.white,
                                       alignment=TA_CENTER, fontName='Helvetica')

    header_data = [[Paragraph("🏥 AI HEALTH RISK ASSESSMENT REPORT", header_style)],
                   [Paragraph("Powered by Machine Learning & Clinical Data Analysis", sub_header_style)]]
    header_table = Table(header_data, colWidths=[17*cm])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BRAND_BLUE),
        ('TOPPADDING', (0, 0), (-1, -1), 14),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 14),
        ('ROUNDEDCORNERS', [6]),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Patient Info ─────────────────────────────────────────────────────
    now = datetime.now().strftime("%B %d, %Y  %I:%M %p")
    info_style = ParagraphStyle('Info', fontSize=10, fontName='Helvetica')
    bold_style = ParagraphStyle('Bold', fontSize=10, fontName='Helvetica-Bold')

    patient_data = [
        [Paragraph("PATIENT INFORMATION", ParagraphStyle('Sec', fontSize=11, fontName='Helvetica-Bold', textColor=BRAND_BLUE))],
        [Table([
            [Paragraph("Name:", bold_style), Paragraph(patient_info.get('name', 'N/A'), info_style),
             Paragraph("Report Date:", bold_style), Paragraph(now, info_style)],
            [Paragraph("Age:", bold_style), Paragraph(str(patient_info.get('age', 'N/A')) + " years", info_style),
             Paragraph("Gender:", bold_style), Paragraph(patient_info.get('gender', 'N/A'), info_style)],
            [Paragraph("Email:", bold_style), Paragraph(patient_info.get('email', 'N/A'), info_style),
             Paragraph("Report ID:", bold_style), Paragraph(f"RPT-{datetime.now().strftime('%Y%m%d%H%M')}", info_style)],
        ], colWidths=[3*cm, 5*cm, 3.5*cm, 5.5*cm],
        style=TableStyle([
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))]
    ]
    pt_table = Table(patient_data, colWidths=[17*cm])
    pt_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), LIGHT_GRAY),
        ('BACKGROUND', (0, 1), (-1, 1), colors.white),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#DDE3EC')),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(pt_table)
    story.append(Spacer(1, 0.4*cm))

    # ── Summary Risk Dashboard ────────────────────────────────────────────
    sec_title = ParagraphStyle('SecTitle', fontSize=13, fontName='Helvetica-Bold',
                                textColor=BRAND_BLUE, spaceBefore=8, spaceAfter=6)
    story.append(Paragraph("📊 DISEASE RISK SUMMARY", sec_title))

    disease_icons = {
        'Diabetes': '🩸', 'Heart Disease': '❤️', 'Liver Disease': '🫁', 'Kidney Failure': '🫘'
    }
    summary_rows = [['Disease', 'Risk Score', 'Risk Level', 'Confidence']]
    for disease, data in predictions.items():
        score = int(data['probability'] * 100)
        label = risk_label(score)
        conf = f"{data.get('confidence', 0.85)*100:.0f}%"
        icon = disease_icons.get(disease, '⚕️')
        summary_rows.append([
            f"{icon} {disease}",
            f"{score}%",
            label,
            conf
        ])

    summary_table = Table(summary_rows, colWidths=[6*cm, 3.5*cm, 4.5*cm, 3*cm])
    ts = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BRAND_BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DDE3EC')),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('ROUNDEDCORNERS', [4]),
    ])
    # Color risk level cells
    for i, (disease, data) in enumerate(predictions.items(), 1):
        score = int(data['probability'] * 100)
        c = risk_color(score)
        ts.add('TEXTCOLOR', (2, i), (2, i), c)
        ts.add('FONTNAME', (2, i), (2, i), 'Helvetica-Bold')
    summary_table.setStyle(ts)
    story.append(summary_table)
    story.append(Spacer(1, 0.5*cm))

    # ── Individual Disease Sections ───────────────────────────────────────
    story.append(Paragraph("📋 DETAILED RISK ANALYSIS", sec_title))

    recommendations = {
        'Diabetes': {
            'low': ["Maintain healthy weight (BMI 18.5–24.9)", "Exercise 150 min/week", "Annual glucose screening"],
            'mod': ["Monitor blood sugar every 3 months", "Adopt low-glycemic diet", "Consult endocrinologist", "30 min daily exercise"],
            'high': ["Immediate HbA1c testing required", "Consult diabetologist urgently", "Start glucose monitoring at home", "Begin dietary therapy", "Consider medication evaluation"]
        },
        'Heart Disease': {
            'low': ["Annual cardiovascular checkup", "Heart-healthy Mediterranean diet", "No smoking", "Stress management"],
            'mod': ["Lipid panel every 6 months", "BP monitoring twice weekly", "Cardiac stress test", "Reduce saturated fat intake"],
            'high': ["Urgent cardiologist consultation", "Immediate ECG required", "Daily BP monitoring", "Aspirin therapy (as directed)", "Restrict strenuous activity"]
        },
        'Liver Disease': {
            'low': ["Limit alcohol to <14 units/week", "Annual liver function tests", "Maintain healthy weight", "Vaccinate for Hepatitis A & B"],
            'mod': ["Liver function panel every 3 months", "Ultrasound imaging recommended", "Strictly limit alcohol", "Avoid hepatotoxic medications"],
            'high': ["Urgent hepatologist referral", "Complete liver panel + biopsy", "Immediate alcohol cessation", "Review all medications for liver impact"]
        },
        'Kidney Failure': {
            'low': ["Annual kidney function tests (eGFR, creatinine)", "Stay well hydrated (2L water/day)", "Control BP < 130/80 mmHg"],
            'mod': ["eGFR test every 3 months", "Low-sodium, low-protein diet", "Nephrology consultation", "Strict blood pressure control"],
            'high': ["Immediate nephrology referral", "Daily creatinine monitoring", "Strict fluid and potassium restriction", "Assess dialysis readiness"]
        }
    }

    for disease, data in predictions.items():
        score = int(data['probability'] * 100)
        level = 'high' if score >= 70 else ('mod' if score >= 40 else 'low')
        rc = risk_color(score)

        # Disease header row
        disease_header = Table([[
            Paragraph(f"{disease_icons.get(disease, '⚕️')} {disease}", 
                     ParagraphStyle('DH', fontSize=12, fontName='Helvetica-Bold', textColor=colors.white)),
            Paragraph(f"{risk_label(score)}  •  {score}% Risk",
                     ParagraphStyle('DHL', fontSize=11, fontName='Helvetica-Bold', textColor=colors.white, alignment=TA_RIGHT))
        ]], colWidths=[9*cm, 8*cm])
        disease_header.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), rc),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(disease_header)

        # Risk bar
        bar_drawing = draw_risk_bar(score, width=460, height=16)
        bar_table = Table([[bar_drawing]], colWidths=[17*cm])
        bar_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), LIGHT_GRAY),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(bar_table)

        # Recommendations
        recs = recommendations.get(disease, {}).get(level, [])
        rec_items = []
        for r in recs:
            rec_items.append([Paragraph(f"✓  {r}", ParagraphStyle('Rec', fontSize=9.5, fontName='Helvetica', leftIndent=5))])

        if rec_items:
            rec_label = Table([
                [Paragraph("Recommended Actions:", ParagraphStyle('RL', fontSize=10, fontName='Helvetica-Bold', textColor=BRAND_BLUE))]
            ] + rec_items, colWidths=[17*cm])
            rec_label.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.white),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 12),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#DDE3EC')),
            ]))
            story.append(rec_label)

        story.append(Spacer(1, 0.35*cm))

    # ── SHAP Explanations ─────────────────────────────────────────────────
    if shap_info:
        story.append(Paragraph("🔍 AI PREDICTION EXPLANATION (SHAP Values)", sec_title))
        shap_text = ParagraphStyle('SHAP', fontSize=9.5, fontName='Helvetica', leading=14)
        story.append(Paragraph(
            "The AI model uses SHAP (SHapley Additive exPlanations) to explain each prediction. "
            "Positive values increase disease risk; negative values decrease it. "
            "The factors below had the most influence on your prediction:",
            shap_text))
        story.append(Spacer(1, 0.2*cm))

        for disease, factors in shap_info.items():
            if not factors:
                continue
            story.append(Paragraph(f"  {disease_icons.get(disease, '⚕️')} {disease} — Key Factors",
                                   ParagraphStyle('SF', fontSize=10, fontName='Helvetica-Bold', textColor=BRAND_BLUE, spaceBefore=6)))
            rows = [["Feature", "Impact Direction", "SHAP Value"]]
            for feat, val in factors[:5]:
                direction = "↑ Increases Risk" if val > 0 else "↓ Decreases Risk"
                rows.append([feat.replace('_', ' ').title(), direction, f"{val:+.4f}"])

            shap_table = Table(rows, colWidths=[7*cm, 6*cm, 4*cm])
            shap_ts = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), BRAND_TEAL),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DDE3EC')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, LIGHT_GRAY]),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ])
            for i, (_, val) in enumerate(factors[:5], 1):
                c = RISK_RED if val > 0 else RISK_GREEN
                shap_ts.add('TEXTCOLOR', (1, i), (1, i), c)
                shap_ts.add('FONTNAME', (1, i), (1, i), 'Helvetica-Bold')
            shap_table.setStyle(shap_ts)
            story.append(shap_table)
            story.append(Spacer(1, 0.25*cm))

    # ── Disclaimer ────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=BRAND_TEAL))
    disclaimer_style = ParagraphStyle('Disclaimer', fontSize=8, fontName='Helvetica',
                                       textColor=MID_GRAY, alignment=TA_CENTER, leading=12)
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        "⚠️  DISCLAIMER: This AI-generated report is for informational purposes only and does NOT constitute medical advice. "
        "Predictions are based on statistical models and may not reflect your actual health condition. "
        "Always consult a qualified healthcare professional before making any health decisions. "
        "This report is NOT a substitute for professional medical examination, diagnosis, or treatment.",
        disclaimer_style))
    story.append(Spacer(1, 0.15*cm))
    story.append(Paragraph(
        f"Report generated on {now}  •  HealthAI Prediction System v2.0  •  Confidential",
        disclaimer_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
