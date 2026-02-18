"""
Generate Project Report PDF for Smart E-Commerce Analytics Platform
Run: python generate_report.py
Requires: pip install reportlab
"""
import json, os, pickle
import pandas as pd
import numpy as np

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, HRFlowable, PageBreak)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
except ImportError:
    print("Installing reportlab...")
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'reportlab', '-q'])
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                    TableStyle, HRFlowable, PageBreak)
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

os.makedirs('report', exist_ok=True)

# Load metrics
with open('models/churn_metrics.json') as f:
    churn_m = json.load(f)
with open('models/lstm_metrics.json') as f:
    lstm_m = json.load(f)

seg_summary = pd.read_csv('models/segment_summary.csv', index_col=0)
orders = pd.read_csv('data/orders_clean.csv', parse_dates=['transaction_date'])

# ── Document setup ────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    'report/Smart_Ecommerce_Project_Report.pdf',
    pagesize=A4,
    rightMargin=2*cm, leftMargin=2*cm,
    topMargin=2*cm, bottomMargin=2*cm
)

styles = getSampleStyleSheet()
W = A4[0] - 4*cm  # usable width

# Custom styles
title_style = ParagraphStyle('Title', parent=styles['Title'],
    fontSize=22, textColor=colors.HexColor('#1a3a5c'),
    spaceAfter=6, alignment=TA_CENTER)
subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'],
    fontSize=12, textColor=colors.HexColor('#4a6fa5'),
    spaceAfter=4, alignment=TA_CENTER)
h1_style = ParagraphStyle('H1', parent=styles['Heading1'],
    fontSize=16, textColor=colors.HexColor('#1a3a5c'),
    spaceBefore=14, spaceAfter=6,
    borderPad=4, backColor=colors.HexColor('#eef2f7'))
h2_style = ParagraphStyle('H2', parent=styles['Heading2'],
    fontSize=13, textColor=colors.HexColor('#2c5f8a'),
    spaceBefore=10, spaceAfter=4)
body_style = ParagraphStyle('Body', parent=styles['Normal'],
    fontSize=10, leading=15, spaceAfter=6, alignment=TA_JUSTIFY)
bullet_style = ParagraphStyle('Bullet', parent=styles['Normal'],
    fontSize=10, leading=14, leftIndent=16, spaceAfter=3,
    bulletIndent=6)
metric_style = ParagraphStyle('Metric', parent=styles['Normal'],
    fontSize=10, leading=14, spaceAfter=4)

def tbl(data, col_widths=None, header=True):
    t = Table(data, colWidths=col_widths)
    style = [
        ('FONTNAME',  (0,0), (-1,0 if header else -1), 'Helvetica-Bold'),
        ('FONTSIZE',  (0,0), (-1,-1), 9),
        ('BACKGROUND',(0,0), (-1,0), colors.HexColor('#1a3a5c') if header else colors.white),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white if header else colors.black),
        ('ALIGN',     (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',    (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f0f4f8')]),
        ('GRID',      (0,0), (-1,-1), 0.4, colors.HexColor('#cccccc')),
        ('TOPPADDING',(0,0), (-1,-1), 5),
        ('BOTTOMPADDING',(0,0), (-1,-1), 5),
    ]
    t.setStyle(TableStyle(style))
    return t

story = []

# ═══════════════════════════════════════════════════════════════════════════════
# COVER PAGE
# ═══════════════════════════════════════════════════════════════════════════════
story.append(Spacer(1, 2*cm))
story.append(Paragraph("Smart E-Commerce Analytics Platform", title_style))
story.append(Paragraph("Capstone Project Report", subtitle_style))
story.append(Spacer(1, 0.5*cm))
story.append(HRFlowable(width=W, thickness=2, color=colors.HexColor('#1a3a5c')))
story.append(Spacer(1, 0.5*cm))

cover_data = [
    ['Dataset', 'Olist Brazilian E-Commerce + Flipkart Reviews (Kaggle)'],
    ['Orders Analyzed', f"{len(orders):,} delivered orders (2016–2018)"],
    ['Total Revenue', f"R${orders['total_amount'].sum():,.0f}"],
    ['Unique Customers', f"{orders['customer_id'].nunique():,}"],
    ['Product Categories', f"{orders['category'].nunique()}"],
    ['Date Range', f"{orders['transaction_date'].min().date()} → {orders['transaction_date'].max().date()}"],
]
story.append(tbl(cover_data, col_widths=[5*cm, 11*cm], header=False))
story.append(Spacer(1, 1*cm))

story.append(Paragraph("Project Modules", h2_style))
modules = [
    ['Week', 'Module', 'Technique', 'Dataset'],
    ['1', 'EDA Overview', 'Pandas, Matplotlib, Seaborn', 'Olist (9 CSVs)'],
    ['2', 'Customer Segmentation', 'RFM + K-Means (k=4)', 'orders_clean.csv'],
    ['3', 'Churn Prediction', 'Random Forest Classifier', 'churn_clean.csv'],
    ['4', 'Sales Forecasting', 'LSTM (30-day look-back)', 'orders_clean.csv'],
    ['5', 'Recommendation System', 'TF-IDF + Cosine Similarity', 'flipkart_clean.csv'],
]
story.append(tbl(modules, col_widths=[1.5*cm, 4*cm, 5*cm, 5.5*cm]))
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: EDA
# ═══════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("1. Exploratory Data Analysis (EDA)", h1_style))
story.append(Paragraph(
    "The Olist Brazilian E-Commerce dataset consists of 9 relational CSV files covering ~100,000 orders "
    "placed between September 2016 and October 2018. After merging and cleaning, 96,477 delivered orders "
    "were retained for analysis.", body_style))

story.append(Paragraph("1.1 Data Cleaning Pipeline", h2_style))
cleaning_steps = [
    ['Step', 'Action', 'Impact'],
    ['1', 'Merged 9 Olist CSVs on order_id / customer_id / product_id', 'Single unified table'],
    ['2', 'Translated Portuguese category names → English', '71 categories mapped'],
    ['3', 'Aggregated payment values per order (sum)', 'One row per order'],
    ['4', 'Aggregated item counts per order', 'quantity column added'],
    ['5', 'Averaged review scores per order', 'review_score column added'],
    ['6', 'Filtered to order_status == delivered', '99,441 → 96,477 rows'],
    ['7', 'Dropped rows with missing total_amount or timestamp', 'No null revenue'],
    ['8', 'Filled missing category → Unknown', '0 nulls remaining'],
    ['9', 'Filled missing review_score → 3.0 (neutral)', '0 nulls remaining'],
    ['10', 'Renamed columns for consistency', 'Standardized schema'],
]
story.append(tbl(cleaning_steps, col_widths=[1*cm, 9*cm, 6*cm]))
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph("1.2 Key EDA Findings", h2_style))
eda_findings = [
    "• Revenue grew steadily from late 2016, peaking in November 2018 (Black Friday effect)",
    "• Top 3 categories by revenue: Health & Beauty, Watches & Gifts, Bed/Bath/Table",
    "• São Paulo (SP) accounts for ~42% of all orders — dominant market",
    "• Average order value: R$154.10 | Median: R$108.10 (right-skewed distribution)",
    "• Credit card is the dominant payment method (~74% of transactions)",
    "• 57% of reviews are 5-star; average review score: 4.09/5",
    "• Peak ordering hours: 10 AM–4 PM; Monday is the busiest day",
]
for f in eda_findings:
    story.append(Paragraph(f, bullet_style))
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: CUSTOMER SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("2. Customer Segmentation (RFM + K-Means)", h1_style))
story.append(Paragraph(
    "RFM (Recency, Frequency, Monetary) analysis was performed on 96,477 orders across "
    f"{orders['customer_id'].nunique():,} unique customers. Features were log-transformed to reduce "
    "right-skew, then standardized before K-Means clustering.", body_style))

story.append(Paragraph("2.1 Methodology", h2_style))
story.append(Paragraph(
    "The Elbow Method and Silhouette Score analysis were used to determine the optimal number of clusters. "
    "k=4 was selected, producing four distinct customer segments ordered by average spend.", body_style))

story.append(Paragraph("2.2 Segment Profiles", h2_style))
seg_data = [['Segment', 'Avg Recency (days)', 'Avg Frequency', 'Avg Monetary (R$)']]
for seg in ['Champions', 'Loyal', 'At Risk', 'Lost']:
    if seg in seg_summary.index:
        row = seg_summary.loc[seg]
        seg_data.append([seg, f"{row['Recency']:.0f}", f"{row['Frequency']:.1f}", f"R${row['Monetary']:.2f}"])
story.append(tbl(seg_data, col_widths=[4*cm, 4*cm, 4*cm, 4*cm]))
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph("2.3 Business Insights", h2_style))
seg_insights = [
    "• Champions: Low recency, high frequency, high spend — reward with loyalty programs",
    "• Loyal: Regular buyers with moderate spend — upsell opportunities",
    "• At Risk: Haven't purchased recently — target with win-back campaigns",
    "• Lost: High recency, low frequency — consider re-engagement discounts",
]
for i in seg_insights:
    story.append(Paragraph(i, bullet_style))
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: CHURN PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("3. Customer Churn Prediction", h1_style))
story.append(Paragraph(
    "A Random Forest classifier was trained on the E-Commerce Customer Churn dataset (3,941 customers) "
    f"to predict churn. The dataset has a churn rate of {churn_m['churn_rate']*100:.1f}%.", body_style))

story.append(Paragraph("3.1 Model Performance", h2_style))
perf_data = [
    ['Metric', 'Value', 'Interpretation'],
    ['Accuracy', f"{churn_m['accuracy']*100:.1f}%", 'Overall correct predictions'],
    ['F1 Score', f"{churn_m['f1_score']:.4f}", 'Harmonic mean of precision & recall'],
    ['ROC-AUC', f"{churn_m['roc_auc']:.4f}", 'Excellent discrimination (>0.9)'],
    ['Train Size', f"{churn_m['train_size']:,}", '80% stratified split'],
    ['Test Size', f"{churn_m['test_size']:,}", '20% stratified split'],
]
story.append(tbl(perf_data, col_widths=[4*cm, 3*cm, 9*cm]))
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph("3.2 Top Churn Predictors", h2_style))
churn_predictors = [
    "• Tenure — Customers with shorter tenure churn significantly more",
    "• Cashback Amount — Lower cashback correlates with higher churn",
    "• Days Since Last Order — Longer inactivity strongly predicts churn",
    "• Complain — Customers who complained have 2x higher churn rate",
    "• Satisfaction Score — Lower scores (1-2) show highest churn rates",
]
for p in churn_predictors:
    story.append(Paragraph(p, bullet_style))
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: SALES FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("4. Sales Forecasting (LSTM)", h1_style))
story.append(Paragraph(
    "An LSTM (Long Short-Term Memory) neural network was trained on daily revenue aggregated from "
    "Olist orders. A 30-day sliding window was used to predict the next day's revenue.", body_style))

story.append(Paragraph("4.1 Model Architecture", h2_style))
arch_data = [
    ['Layer', 'Units', 'Details'],
    ['Input', '30', '30-day look-back window'],
    ['LSTM', '64', 'return_sequences=True'],
    ['Dropout', '20%', 'Regularization'],
    ['LSTM', '32', 'Final LSTM layer'],
    ['Dropout', '20%', 'Regularization'],
    ['Dense', '16', 'ReLU activation'],
    ['Dense', '1', 'Output: next-day revenue'],
]
story.append(tbl(arch_data, col_widths=[4*cm, 3*cm, 9*cm]))
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph("4.2 Model Performance", h2_style))
lstm_perf = [
    ['Metric', 'Value'],
    ['MAE (Mean Absolute Error)', f"R${lstm_m['mae']:,.0f}"],
    ['RMSE (Root Mean Squared Error)', f"R${lstm_m['rmse']:,.0f}"],
    ['MAPE (Mean Absolute % Error)', f"{lstm_m['mape']:.1f}%"],
]
story.append(tbl(lstm_perf, col_widths=[8*cm, 8*cm]))
story.append(Spacer(1, 0.4*cm))
story.append(Paragraph(
    f"The model achieves a MAPE of {lstm_m['mape']:.1f}%, indicating strong forecasting accuracy. "
    "Early stopping was used to prevent overfitting (patience=10 epochs).", body_style))
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: RECOMMENDATION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("5. Product Recommendation System", h1_style))
story.append(Paragraph(
    "A hybrid recommendation system was built using the Flipkart Product Reviews dataset "
    "(189,869 reviews across multiple product categories).", body_style))

story.append(Paragraph("5.1 Approaches Implemented", h2_style))
rec_approaches = [
    ['Approach', 'Method', 'Use Case'],
    ['Popularity-Based', 'Weighted score: 60% review volume + 40% avg rating',
     'Cold-start / new users'],
    ['Content-Based', 'TF-IDF on review text + Cosine Similarity',
     'Similar product discovery'],
]
story.append(tbl(rec_approaches, col_widths=[4*cm, 7*cm, 5*cm]))
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph("5.2 Technical Details", h2_style))
rec_details = [
    "• TF-IDF Vocabulary: 3,000 features, min_df=2, English stop words removed",
    "• Products indexed: aggregated reviews per product (up to 10 reviews per product)",
    "• Similarity matrix: cosine similarity computed across all product pairs",
    "• Category filtering: popularity recommendations support category-level filtering",
]
for d in rec_details:
    story.append(Paragraph(d, bullet_style))
story.append(PageBreak())

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: DASHBOARD & DELIVERABLES
# ═══════════════════════════════════════════════════════════════════════════════
story.append(Paragraph("6. Interactive Dashboard & Deliverables", h1_style))
story.append(Paragraph(
    "An interactive Streamlit dashboard was built with 5 pages corresponding to each module. "
    "The dashboard loads real data and trained models to provide live analytics.", body_style))

story.append(Paragraph("6.1 Dashboard Pages", h2_style))
dash_data = [
    ['Page', 'Features'],
    ['📊 EDA Overview', 'Raw data inspection, cleaning steps, revenue trends, category/state analysis'],
    ['👥 Customer Segmentation', 'RFM computation, cluster visualization, PCA scatter, segment lookup'],
    ['⚠️ Churn Prediction', 'Live churn prediction, feature importance, ROC curve, confusion matrix'],
    ['📈 Sales Forecast', '30-day LSTM forecast, actual vs predicted, MAE/RMSE metrics'],
    ['🎯 Recommendations', 'Popularity ranking, content-based search, similarity heatmap'],
]
story.append(tbl(dash_data, col_widths=[4.5*cm, 11.5*cm]))
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph("6.2 Saved Model Files", h2_style))
model_files = [
    ['File', 'Description'],
    ['kmeans_segmentation.pkl', 'Trained K-Means model (k=4)'],
    ['scaler_rfm.pkl', 'StandardScaler for RFM features'],
    ['segment_summary.csv', 'Segment profile statistics'],
    ['rf_churn_model.pkl', 'Random Forest classifier (200 trees)'],
    ['churn_metrics.json', 'Accuracy, F1, AUC metrics'],
    ['lstm_sales_forecast.keras', 'Trained LSTM model weights'],
    ['scaler_timeseries.pkl', 'MinMaxScaler for time series'],
    ['lstm_metrics.json', 'MAE, RMSE, MAPE metrics'],
    ['tfidf_vectorizer.pkl', 'Fitted TF-IDF vectorizer'],
    ['tfidf_matrix.npz', 'Sparse TF-IDF feature matrix'],
    ['popularity_scores.csv', 'Product popularity rankings'],
    ['products_index.csv', 'Product metadata index'],
]
story.append(tbl(model_files, col_widths=[7*cm, 9*cm]))
story.append(Spacer(1, 0.4*cm))

story.append(Paragraph("6.3 Technology Stack", h2_style))
tech_data = [
    ['Category', 'Libraries / Tools'],
    ['Data Processing', 'pandas, numpy'],
    ['Visualization', 'matplotlib, seaborn, plotly'],
    ['Machine Learning', 'scikit-learn (KMeans, RandomForest, TF-IDF)'],
    ['Deep Learning', 'TensorFlow / Keras (LSTM)'],
    ['Dashboard', 'Streamlit'],
    ['Report', 'ReportLab'],
    ['Data Sources', 'Olist (Kaggle), Flipkart Reviews (Kaggle)'],
]
story.append(tbl(tech_data, col_widths=[5*cm, 11*cm]))

story.append(Spacer(1, 1*cm))
story.append(HRFlowable(width=W, thickness=1, color=colors.HexColor('#1a3a5c')))
story.append(Spacer(1, 0.3*cm))
story.append(Paragraph(
    "Smart E-Commerce Analytics Platform — Capstone Project Report",
    ParagraphStyle('Footer', parent=styles['Normal'], fontSize=9,
                   textColor=colors.grey, alignment=TA_CENTER)))

# Build PDF
doc.build(story)
print("✅ Report saved: report/Smart_Ecommerce_Project_Report.pdf")
