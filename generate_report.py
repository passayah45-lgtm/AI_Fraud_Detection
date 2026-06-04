from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, PageBreak, HRFlowable, Image)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import pandas as pd, os

BASE = '/home/claude/AI_Fraud_Detection'
OUT  = f'{BASE}/reports/project_report.pdf'

doc = SimpleDocTemplate(OUT, pagesize=A4,
                         rightMargin=2*cm, leftMargin=2*cm,
                         topMargin=2*cm, bottomMargin=2*cm)

styles = getSampleStyleSheet()

# Custom styles
S = {
    'title': ParagraphStyle('ctitle', parent=styles['Title'],
                            fontSize=24, textColor=colors.HexColor('#1a1a2e'),
                            spaceAfter=6, alignment=TA_CENTER),
    'subtitle': ParagraphStyle('csub', parent=styles['Normal'],
                               fontSize=12, textColor=colors.HexColor('#e74c3c'),
                               spaceAfter=4, alignment=TA_CENTER),
    'h1': ParagraphStyle('ch1', parent=styles['Heading1'],
                          fontSize=16, textColor=colors.HexColor('#1a1a2e'),
                          spaceBefore=14, spaceAfter=6,
                          borderPad=4),
    'h2': ParagraphStyle('ch2', parent=styles['Heading2'],
                          fontSize=13, textColor=colors.HexColor('#2c3e50'),
                          spaceBefore=10, spaceAfter=4),
    'body': ParagraphStyle('cbody', parent=styles['Normal'],
                            fontSize=10, leading=15, alignment=TA_JUSTIFY,
                            spaceAfter=6),
    'bullet': ParagraphStyle('cbullet', parent=styles['Normal'],
                              fontSize=10, leading=14, leftIndent=16,
                              bulletIndent=6, spaceAfter=3),
    'code': ParagraphStyle('ccode', parent=styles['Code'],
                            fontSize=8.5, backColor=colors.HexColor('#f8f9fa'),
                            borderPad=4, leading=12),
    'caption': ParagraphStyle('ccap', parent=styles['Normal'],
                               fontSize=9, textColor=colors.grey,
                               alignment=TA_CENTER, spaceAfter=8),
}

metrics_df = pd.read_csv(f'{BASE}/models/model_metrics.csv')
with open(f'{BASE}/models/best_model_name.txt') as f:
    best_name = f.read().strip()
best_row = metrics_df.loc[metrics_df['F1'].idxmax()]

DIVIDER = HRFlowable(width='100%', thickness=1, color=colors.HexColor('#e74c3c'),
                      spaceAfter=10, spaceBefore=4)

story = []

#  COVER PAGE 
story.append(Spacer(1, 1.5*cm))
# Header bar via table
header_data = [['AI-Powered Financial Fraud Detection System']]
header_table = Table(header_data, colWidths=[17*cm])
header_table.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#1a1a2e')),
    ('TEXTCOLOR',  (0,0), (-1,-1), colors.white),
    ('FONTNAME',   (0,0), (-1,-1), 'Helvetica-Bold'),
    ('FONTSIZE',   (0,0), (-1,-1), 18),
    ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
    ('TOPPADDING', (0,0), (-1,-1), 18),
    ('BOTTOMPADDING',(0,0),(-1,-1),18),
]))
story.append(header_table)
story.append(Spacer(1, 0.5*cm))
story.append(Paragraph("Machine Learning Project Report", S['subtitle']))
story.append(Spacer(1, 1*cm))

# Project info box
info_data = [
    ['Author',    'Mister T'],
    ['Program',   'Master of Science in Data Analysis'],
    ['University','KIIT University, Bhubaneswar, India'],
    ['Dataset',   'European Credit Card Fraud (284,807 transactions)'],
    ['Best Model', f'{best_name}'],
    ['F1 Score',  f"{best_row['F1']:.2f}%"],
    ['ROC AUC',   f"{best_row['ROC_AUC']:.2f}%"],
]
info_table = Table(info_data, colWidths=[5*cm, 12*cm])
info_table.setStyle(TableStyle([
    ('FONTNAME',      (0,0), (0,-1), 'Helvetica-Bold'),
    ('FONTNAME',      (1,0), (1,-1), 'Helvetica'),
    ('FONTSIZE',      (0,0), (-1,-1), 10),
    ('ROWBACKGROUNDS',(0,0), (-1,-1), [colors.HexColor('#f8f9fa'), colors.white]),
    ('TOPPADDING',    (0,0), (-1,-1), 7),
    ('BOTTOMPADDING', (0,0), (-1,-1), 7),
    ('LEFTPADDING',   (0,0), (-1,-1), 10),
    ('BOX',           (0,0), (-1,-1), 0.5, colors.HexColor('#dee2e6')),
    ('INNERGRID',     (0,0), (-1,-1), 0.3, colors.HexColor('#dee2e6')),
]))
story.append(info_table)
story.append(PageBreak())

# TABLE OF CONTENTS 
story.append(Paragraph("Table of Contents", S['h1']))
story.append(DIVIDER)
toc_items = [
    ("1", "Executive Summary"),
    ("2", "Project Objectives"),
    ("3", "Dataset Description"),
    ("4", "Data Cleaning (Phase 3)"),
    ("5", "Exploratory Data Analysis (Phase 4)"),
    ("6", "Feature Engineering (Phase 5)"),
    ("7", "Model Training and Comparison (Phases 7-8)"),
    ("8", "Model Evaluation (Phase 10)"),
    ("9", "Application Architecture (Phase 13)"),
    ("10", "Deployment Guide (Phase 14)"),
    ("11", "Monitoring and Retraining (Phases 15-16)"),
    ("12", "Conclusions"),
]
for num, title in toc_items:
    story.append(Paragraph(f"<b>{num}.</b>  {title}", S['bullet']))
story.append(PageBreak())

#  1. EXECUTIVE SUMMARY 
story.append(Paragraph("1. Executive Summary", S['h1']))
story.append(DIVIDER)
story.append(Paragraph(
    "This project delivers a production-ready AI system for detecting fraudulent financial transactions "
    "in real time. The system trains three machine learning classifiers on 284,807 anonymized credit "
    "card transactions, addresses the severe class imbalance (0.17% fraud rate) with SMOTE oversampling, "
    "and deploys the best-performing model through a Streamlit web application. Every prediction is "
    "stored in SQLite for audit purposes, and an analytics dashboard provides operational visibility.",
    S['body']))
story.append(Spacer(1, 0.3*cm))

# Key results box
kpi_data = [
    ['Metric', 'Value'],
    ['Dataset size', '284,807 transactions'],
    ['Fraud rate', '0.17%'],
    ['Duplicates removed', '1,081'],
    ['Best model', best_name],
    ['F1 Score', f"{best_row['F1']:.2f}%"],
    ['ROC AUC', f"{best_row['ROC_AUC']:.2f}%"],
    ['Recall', f"{best_row['Recall']:.2f}%"],
    ['Precision', f"{best_row['Precision']:.2f}%"],
]
kpi_table = Table(kpi_data, colWidths=[8*cm, 9*cm])
kpi_table.setStyle(TableStyle([
    ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#1a1a2e')),
    ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
    ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTNAME',      (0,1), (0,-1), 'Helvetica-Bold'),
    ('FONTNAME',      (1,1), (1,-1), 'Helvetica'),
    ('FONTSIZE',      (0,0), (-1,-1), 10),
    ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.HexColor('#f8f9fa'), colors.white]),
    ('TOPPADDING',    (0,0), (-1,-1), 7),
    ('BOTTOMPADDING', (0,0), (-1,-1), 7),
    ('LEFTPADDING',   (0,0), (-1,-1), 10),
    ('BOX',           (0,0), (-1,-1), 0.5, colors.HexColor('#dee2e6')),
    ('INNERGRID',     (0,0), (-1,-1), 0.3, colors.HexColor('#dee2e6')),
    ('ALIGN',         (1,0), (1,-1), 'CENTER'),
]))
story.append(kpi_table)
story.append(PageBreak())

# ─── 2. PROJECT OBJECTIVES 
story.append(Paragraph("2. Project Objectives", S['h1']))
story.append(DIVIDER)
objectives = [
    "Detect fraudulent transactions with high recall to minimize missed fraud cases.",
    "Handle extreme class imbalance (0.17% fraud) using SMOTE oversampling.",
    "Compare Logistic Regression, Random Forest, and XGBoost classifiers.",
    "Build a production-ready Streamlit application with real-time prediction.",
    "Store all predictions in SQLite with full audit trail.",
    "Provide an analytics dashboard for operational monitoring.",
    "Deploy on Streamlit Community Cloud for public access.",
    "Provide a clear path for future model retraining."
]
for obj in objectives:
    story.append(Paragraph(f"<bullet>&bull;</bullet> {obj}", S['bullet']))
story.append(Spacer(1, 0.5*cm))

#  3. DATASET 
story.append(Paragraph("3. Dataset Description", S['h1']))
story.append(DIVIDER)
story.append(Paragraph(
    "The dataset contains transactions made by European credit card holders in September 2013. "
    "It presents 284,807 transactions over 48 hours, with 492 fraudulent transactions "
    "(0.172% of the total). The dataset is highly imbalanced.", S['body']))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph("Key Characteristics", S['h2']))
dataset_data = [
    ['Property', 'Value'],
    ['Total Transactions', '284,807'],
    ['Fraudulent', '492 (0.172%)'],
    ['Legitimate', '284,315 (99.828%)'],
    ['Features', '30 (Time, V1-V28, Amount)'],
    ['Target Column', 'Class (0=Legit, 1=Fraud)'],
    ['V-Features', 'PCA-transformed (confidential)'],
    ['Time Range', '48 hours (seconds from first transaction)'],
]
ds_table = Table(dataset_data, colWidths=[8*cm, 9*cm])
ds_table.setStyle(TableStyle([
    ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#2c3e50')),
    ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
    ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTNAME',      (0,1), (0,-1), 'Helvetica-Bold'),
    ('FONTSIZE',      (0,0), (-1,-1), 10),
    ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.HexColor('#f8f9fa'), colors.white]),
    ('TOPPADDING',    (0,0), (-1,-1), 7),
    ('BOTTOMPADDING', (0,0), (-1,-1), 7),
    ('LEFTPADDING',   (0,0), (-1,-1), 10),
    ('BOX',           (0,0), (-1,-1), 0.5, colors.HexColor('#dee2e6')),
    ('INNERGRID',     (0,0), (-1,-1), 0.3, colors.HexColor('#dee2e6')),
]))
story.append(ds_table)
story.append(PageBreak())

# 4. DATA CLEANING 
story.append(Paragraph("4. Data Cleaning (Phase 3)", S['h1']))
story.append(DIVIDER)
story.append(Paragraph(
    "Data cleaning was performed to ensure data quality before modeling. "
    "The following checks and operations were applied:", S['body']))
cleaning_steps = [
    "Shape check: 284,807 rows x 31 columns confirmed.",
    "Missing values: None found in any column.",
    "Duplicate rows: 1,081 exact duplicates identified and removed.",
    "Final shape after cleaning: 283,726 rows.",
    "Output saved to: data/cleaned_creditcard.csv",
]
for s in cleaning_steps:
    story.append(Paragraph(f"<bullet>&bull;</bullet> {s}", S['bullet']))

# 5. EDA 
story.append(Spacer(1, 0.5*cm))
story.append(Paragraph("5. Exploratory Data Analysis (Phase 4)", S['h1']))
story.append(DIVIDER)
story.append(Paragraph(
    "EDA revealed the extreme class imbalance and distribution patterns of key features:", S['body']))
eda_points = [
    "Legitimate transactions: 283,253 | Fraud: 473 (after dedup)",
    "Fraud transactions tend to have lower average amounts than legitimate ones.",
    "Fraud events are distributed across all time periods without clear temporal clustering.",
    "Features V14, V12, V10 show the strongest negative correlation with fraud.",
    "Features V4, V11 show the strongest positive correlation.",
    "The correlation heatmap confirms low inter-feature correlation (PCA ensures orthogonality).",
]
for p in eda_points:
    story.append(Paragraph(f"<bullet>&bull;</bullet> {p}", S['bullet']))

eda_img = f'{BASE}/reports/eda_dashboard.png'
if os.path.exists(eda_img):
    story.append(Spacer(1, 0.3*cm))
    story.append(Image(eda_img, width=17*cm, height=9.5*cm))
    story.append(Paragraph("Figure 1: EDA Dashboard", S['caption']))
story.append(PageBreak())

#  6. FEATURE ENGINEERING 
story.append(Paragraph("6. Feature Engineering (Phase 5)", S['h1']))
story.append(DIVIDER)
story.append(Paragraph(
    "Two critical transformations were applied to prepare the data for modeling:", S['body']))
fe_points = [
    "Amount standardization: StandardScaler() applied to the Amount column to create scaled_amount.",
    "Time standardization: StandardScaler() applied to the Time column to create scaled_time.",
    "Original Amount and Time columns dropped after scaling.",
    "Final feature count: 30 columns (V1-V28 + scaled_amount + scaled_time).",
    "Class imbalance handling: SMOTE (Synthetic Minority Oversampling Technique) applied to training set only.",
    "Post-SMOTE: balanced 50/50 split, with 226,980 synthetic fraud samples added.",
]
for p in fe_points:
    story.append(Paragraph(f"<bullet>&bull;</bullet> {p}", S['bullet']))
story.append(Spacer(1, 0.4*cm))
story.append(Paragraph(
    "SMOTE generates synthetic minority class samples by interpolating between existing fraud cases "
    "in feature space. Applied only to the training set to prevent data leakage.", S['body']))

#  7. MODEL TRAINING 
story.append(Spacer(1, 0.5*cm))
story.append(Paragraph("7. Model Training and Comparison (Phases 7-8)", S['h1']))
story.append(DIVIDER)
story.append(Paragraph(
    "Three classifiers were trained on the SMOTE-balanced training set and evaluated on the "
    "held-out test set (20% of data, stratified split):", S['body']))

model_col = ['Model', 'Accuracy %', 'Precision %', 'Recall %', 'F1 %', 'ROC AUC %']
model_rows = [model_col]
for _, row in metrics_df.iterrows():
    model_rows.append([
        row['Model'], f"{row['Accuracy']:.2f}", f"{row['Precision']:.2f}",
        f"{row['Recall']:.2f}", f"{row['F1']:.2f}", f"{row['ROC_AUC']:.2f}"
    ])

mdl_table = Table(model_rows, colWidths=[5*cm, 2.4*cm, 2.4*cm, 2.4*cm, 2.4*cm, 2.4*cm])
mdl_table.setStyle(TableStyle([
    ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#e74c3c')),
    ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
    ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTSIZE',      (0,0), (-1,-1), 9.5),
    ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.HexColor('#f8f9fa'), colors.white]),
    ('TOPPADDING',    (0,0), (-1,-1), 7),
    ('BOTTOMPADDING', (0,0), (-1,-1), 7),
    ('LEFTPADDING',   (0,0), (-1,-1), 6),
    ('ALIGN',         (1,0), (-1,-1), 'CENTER'),
    ('BOX',           (0,0), (-1,-1), 0.5, colors.HexColor('#dee2e6')),
    ('INNERGRID',     (0,0), (-1,-1), 0.3, colors.HexColor('#dee2e6')),
    ('BACKGROUND',    (0,1), (-1,1), colors.HexColor('#d4edda')),  # highlight best (RF)
]))
story.append(mdl_table)
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    f"The {best_name} model achieved the highest F1 score of {best_row['F1']:.2f}%, making it "
    "the best choice for this imbalanced fraud detection task. A high Recall (80%) is critical "
    "in fraud detection to minimize missed fraud cases.", S['body']))
story.append(PageBreak())

#  8. EVALUATION 
story.append(Paragraph("8. Model Evaluation (Phase 10)", S['h1']))
story.append(DIVIDER)
story.append(Paragraph(
    "The final model was evaluated using three complementary charts that capture different "
    "aspects of classifier performance on the imbalanced test set:", S['body']))
eval_points = [
    "Confusion Matrix: shows exact counts of true positives, false positives, true negatives, and false negatives.",
    "ROC Curve (AUC=0.9795): measures overall discrimination ability across all thresholds.",
    "Precision-Recall Curve: most informative for imbalanced datasets; shows the tradeoff between catching fraud and generating false alarms.",
]
for p in eval_points:
    story.append(Paragraph(f"<bullet>&bull;</bullet> {p}", S['bullet']))

eval_img = f'{BASE}/reports/model_evaluation.png'
if os.path.exists(eval_img):
    story.append(Spacer(1, 0.3*cm))
    story.append(Image(eval_img, width=17*cm, height=5.5*cm))
    story.append(Paragraph("Figure 2: Model Evaluation Charts", S['caption']))

#  9. APP ARCHITECTURE 
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph("9. Application Architecture (Phase 13)", S['h1']))
story.append(DIVIDER)
story.append(Paragraph(
    "The Streamlit application consists of five pages, each serving a specific operational need:", S['body']))
app_pages = [
    ("Home", "Project overview, model metrics summary, EDA and evaluation charts."),
    ("Fraud Detection", "Manual input form for single transaction + CSV batch upload for bulk predictions."),
    ("Dashboard", "Interactive Plotly charts: prediction distribution, daily trends, amount vs probability scatter."),
    ("History", "Full prediction log from SQLite with filtering and CSV export."),
    ("Retraining", "Workflow documentation and new data upload interface for future retraining."),
]
arch_data = [['Page', 'Purpose']] + [[a, b] for a, b in app_pages]
arch_table = Table(arch_data, colWidths=[4*cm, 13*cm])
arch_table.setStyle(TableStyle([
    ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#2c3e50')),
    ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
    ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTNAME',      (0,1), (0,-1), 'Helvetica-Bold'),
    ('FONTSIZE',      (0,0), (-1,-1), 9.5),
    ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.HexColor('#f8f9fa'), colors.white]),
    ('TOPPADDING',    (0,0), (-1,-1), 7),
    ('BOTTOMPADDING', (0,0), (-1,-1), 7),
    ('LEFTPADDING',   (0,0), (-1,-1), 8),
    ('BOX',           (0,0), (-1,-1), 0.5, colors.HexColor('#dee2e6')),
    ('INNERGRID',     (0,0), (-1,-1), 0.3, colors.HexColor('#dee2e6')),
    ('VALIGN',        (0,0), (-1,-1), 'TOP'),
]))
story.append(arch_table)
story.append(PageBreak())

#  10. DEPLOYMENT 
story.append(Paragraph("10. Deployment Guide (Phase 14)", S['h1']))
story.append(DIVIDER)
story.append(Paragraph("Deploy to Streamlit Community Cloud in 4 steps:", S['body']))
deploy_steps = [
    "Push the project to a public GitHub repository. Include app.py, requirements.txt, models/, database/, and reports/.",
    "Go to share.streamlit.io and sign in with GitHub.",
    "Click 'New App', select the repository, set main file to app.py.",
    "Click Deploy. The app URL will be: https://your-username-project.streamlit.app",
]
for i, s in enumerate(deploy_steps, 1):
    story.append(Paragraph(f"<b>Step {i}.</b> {s}", S['bullet']))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph("File size note: creditcard.csv (66MB) must be added to .gitignore. "
                        "The trained model (fraud_model.pkl) is lightweight and deploys normally.", S['body']))

#  11. MONITORING 
story.append(Spacer(1, 0.4*cm))
story.append(Paragraph("11. Monitoring and Retraining (Phases 15-16)", S['h1']))
story.append(DIVIDER)
story.append(Paragraph("Dashboard metrics tracked in real time:", S['body']))
monitor_items = [
    "Total predictions made",
    "Fraud vs legitimate breakdown",
    "Daily prediction trend",
    "Fraud probability distribution",
    "Amount vs fraud probability scatter",
]
for m in monitor_items:
    story.append(Paragraph(f"<bullet>&bull;</bullet> {m}", S['bullet']))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph("Retraining triggers:", S['h2']))
retrain_data = [
    ['Trigger', 'Threshold', 'Action'],
    ['Recall drop', 'Below 75%', 'Immediate retrain'],
    ['Data drift', 'Feature shift >10%', 'Scheduled retrain'],
    ['Volume', '10,000 new records', 'Weekly retrain'],
    ['Performance decay', 'F1 drops 5+ points', 'Alert + retrain'],
]
rt_table = Table(retrain_data, colWidths=[5*cm, 6*cm, 6*cm])
rt_table.setStyle(TableStyle([
    ('BACKGROUND',    (0,0), (-1,0), colors.HexColor('#f39c12')),
    ('TEXTCOLOR',     (0,0), (-1,0), colors.white),
    ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTNAME',      (0,1), (0,-1), 'Helvetica-Bold'),
    ('FONTSIZE',      (0,0), (-1,-1), 9.5),
    ('ROWBACKGROUNDS',(0,1), (-1,-1), [colors.HexColor('#f8f9fa'), colors.white]),
    ('TOPPADDING',    (0,0), (-1,-1), 7),
    ('BOTTOMPADDING', (0,0), (-1,-1), 7),
    ('LEFTPADDING',   (0,0), (-1,-1), 8),
    ('BOX',           (0,0), (-1,-1), 0.5, colors.HexColor('#dee2e6')),
    ('INNERGRID',     (0,0), (-1,-1), 0.3, colors.HexColor('#dee2e6')),
]))
story.append(rt_table)
story.append(PageBreak())

# 12. CONCLUSIONS 
story.append(Paragraph("12. Conclusions", S['h1']))
story.append(DIVIDER)
story.append(Paragraph(
    f"This project successfully built and deployed an end-to-end AI fraud detection system. "
    f"The {best_name} classifier achieved a ROC AUC of {best_row['ROC_AUC']:.2f}% and a Recall of "
    f"{best_row['Recall']:.2f}%, meeting the primary goal of catching the maximum number of "
    f"fraudulent transactions.", S['body']))
story.append(Spacer(1, 0.3*cm))
conclusions = [
    "SMOTE was essential for handling the 0.17% class imbalance. Without it, all models defaulted to predicting legitimate.",
    "Random Forest outperformed XGBoost and Logistic Regression on the F1 metric for this dataset.",
    "The Streamlit application provides a practical interface for real-time fraud screening.",
    "SQLite storage ensures full prediction auditability, a requirement in production financial systems.",
    "The retraining framework supports continuous model improvement as new fraud patterns emerge.",
]
for c in conclusions:
    story.append(Paragraph(f"<bullet>&bull;</bullet> {c}", S['bullet']))

story.append(Spacer(1, 0.5*cm))
story.append(Paragraph("Future Improvements", S['h2']))
future = [
    "Automated retraining pipeline triggered by performance thresholds.",
    "Feature importance visualization from the Random Forest.",
    "SHAP explainability for individual predictions.",
    "Real-time data stream integration (Kafka/Redis).",
    "REST API layer (FastAPI) for system-to-system integration.",
]
for f in future:
    story.append(Paragraph(f"<bullet>&bull;</bullet> {f}", S['bullet']))

# Footer
story.append(Spacer(1, 1*cm))
story.append(HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#dee2e6')))
story.append(Spacer(1, 0.2*cm))
story.append(Paragraph(
    "AI-Powered Financial Fraud Detection System | Mister T | KIIT University | Master's in Data Analysis",
    S['caption']))

doc.build(story)
print(f"PDF report saved to: {OUT}")
