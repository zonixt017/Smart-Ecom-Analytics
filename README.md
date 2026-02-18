# 🛒 Smart E-Commerce Analytics Platform

> **Capstone Project** — End-to-end data science platform covering EDA, customer segmentation, churn prediction, sales forecasting, and product recommendations using real Kaggle datasets.

---

## 📊 Live Dashboard

```bash
streamlit run app/streamlit_app.py
```

Open → `http://localhost:8501`

---

## 🗂️ Project Structure

```
tm_capstone/
├── app/                          # Streamlit dashboard
│   ├── streamlit_app.py          # Home page
│   ├── utils.py                  # Shared utilities
│   └── pages/
│       ├── 1_📊_EDA_Overview.py
│       ├── 2_👥_Customer_Segmentation.py
│       ├── 3_⚠️_Churn_Prediction.py
│       ├── 4_📈_Sales_Forecast.py
│       └── 5_🎯_Recommendations.py
│
├── notebooks/                    # Jupyter analysis notebooks
│   ├── 01_eda.ipynb
│   ├── 02_customer_segmentation.ipynb
│   ├── 03_churn_prediction.ipynb
│   ├── 04_sales_forecasting_lstm.ipynb
│   └── 05_recommendation_system.ipynb
│
├── data/
│   ├── olist/                    # ⬇️ Download from Kaggle (see below)
│   ├── churn/                    # ⬇️ Download from Kaggle (see below)
│   ├── flipkart/                 # ⬇️ Download from Kaggle (see below)
│   └── preprocess.py             # Run to generate *_clean.csv files
│
├── models/                       # Saved model artifacts
│   ├── churn_metrics.json
│   ├── lstm_metrics.json
│   ├── segment_summary.csv
│   ├── popularity_scores.csv
│   └── products_index.csv
│
├── report/
│   └── Smart_Ecommerce_Project_Report.pdf
│
├── save_models.py                # Train & save all models
├── generate_report.py            # Generate PDF report
├── requirements.txt
└── .gitignore
```

---

## 📦 Datasets

> **Large CSV files are not included in this repo.** Download them from Kaggle and place them in the correct folders.

| Dataset | Kaggle Link | Place in |
|---------|-------------|----------|
| Olist Brazilian E-Commerce | [kaggle.com/datasets/olistbr/brazilian-ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) | `data/olist/` |
| E-Commerce Customer Churn | [kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction) | `data/churn/` |
| Flipkart Product Reviews | [kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset](https://www.kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset) | `data/flipkart/` |

---

## 🚀 Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/tm_capstone.git
cd tm_capstone
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download datasets
Place the Kaggle CSVs in the folders shown above.

### 4. Preprocess data
```bash
python data/preprocess.py
```
This generates `data/orders_clean.csv`, `data/churn_clean.csv`, `data/flipkart_clean.csv`.

### 5. Train & save models
```bash
python save_models.py
```
Saves 14 model files to `models/`.

### 6. Launch dashboard
```bash
streamlit run app/streamlit_app.py
```

### 7. (Optional) Generate PDF report
```bash
python generate_report.py
```

---

## 🧠 Modules

### 📊 Week 1 — EDA Overview
- Loaded & merged 9 Olist CSV files
- 10-step cleaning pipeline (filter, translate, aggregate, fill nulls)
- Revenue trends, category analysis, geographic breakdown, payment & review analysis

### 👥 Week 2 — Customer Segmentation
- RFM (Recency, Frequency, Monetary) computed per customer
- Log-transform + StandardScaler preprocessing
- K-Means clustering (k=4 via Elbow + Silhouette)
- Segments: **Champions**, **Loyal**, **At Risk**, **Lost**

### ⚠️ Week 3 — Churn Prediction
- Dataset: 3,941 customers with behavioral features
- Model: Random Forest (200 trees, max_depth=10)
- **Accuracy: 92.9% | F1: 0.77 | ROC-AUC: 0.958**
- Top predictors: Tenure, Cashback Amount, Days Since Last Order

### 📈 Week 4 — Sales Forecasting (LSTM)
- Daily revenue time series from Olist orders
- LSTM architecture: 64 → 32 units, 30-day look-back window
- **MAE: R$2,396 | RMSE: R$2,876 | MAPE: 7.7%**
- 30-day future forecast with autoregressive prediction

### 🎯 Week 5 — Recommendation System
- Dataset: 189,869 Flipkart product reviews
- **Popularity-based**: weighted score (60% volume + 40% rating)
- **Content-based**: TF-IDF (3,000 features) + Cosine Similarity
- Category-level filtering supported

---

## 📈 Model Performance Summary

| Model | Metric | Value |
|-------|--------|-------|
| K-Means Segmentation | Silhouette Score | Optimal at k=4 |
| Random Forest (Churn) | ROC-AUC | **0.958** |
| Random Forest (Churn) | Accuracy | **92.9%** |
| LSTM (Sales Forecast) | MAPE | **7.7%** |
| LSTM (Sales Forecast) | MAE | R$2,396 |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Data Processing | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Machine Learning | `scikit-learn` |
| Deep Learning | `TensorFlow` / `Keras` |
| Dashboard | `Streamlit` |
| Report | `ReportLab` |

---

## 📄 Report

The full project report is available at:
```
report/Smart_Ecommerce_Project_Report.pdf
```

---

## 📋 Requirements

See [`requirements.txt`](requirements.txt) for the full list of dependencies.

```bash
pip install -r requirements.txt
```
