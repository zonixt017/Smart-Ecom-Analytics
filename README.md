# 🛒 Smart E-Commerce Analytics Platform

> **Capstone Project** — End-to-end data science platform covering EDA, customer segmentation, churn prediction, sales forecasting, and product recommendations using real Kaggle datasets.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00?logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smart-ecom-analytics.streamlit.app)

> **Deploy your own** → see [Deployment](#-deployment) section below

---

## 📊 Dashboard Preview

| EDA Overview | Customer Segmentation | Churn Prediction |
|---|---|---|
| Revenue trends, cleaning pipeline, category analysis | RFM + K-Means clusters, PCA scatter | SHAP explainability, ROC curve, live predictor |

| Sales Forecast | Recommendations |
|---|---|
| LSTM 30-day forecast, actual vs predicted | TF-IDF content-based + popularity ranking |

---

## 🗂️ Project Structure

```
Smart-Ecom-Analytics/
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
│   ├── orders_clean.csv          # 96,477 Olist delivered orders
│   ├── churn_clean.csv           # 3,941 churn records
│   ├── flipkart_clean.csv        # 189,869 product reviews
│   ├── olist/                    # ⬇️ Raw Kaggle CSVs (gitignored)
│   ├── churn/                    # ⬇️ Raw Kaggle CSVs (gitignored)
│   ├── flipkart/                 # ⬇️ Raw Kaggle CSVs (gitignored)
│   └── preprocess.py             # Regenerate *_clean.csv files
│
├── models/                       # Model metrics & summaries
│   ├── churn_metrics.json        # Accuracy, F1, AUC
│   ├── lstm_metrics.json         # MAE, RMSE, MAPE
│   ├── segment_summary.csv       # RFM segment profiles
│   ├── popularity_scores.csv     # Product rankings
│   └── products_index.csv        # Product metadata
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

> **Raw CSV files are gitignored** (too large). The cleaned versions are committed.

| Dataset | Kaggle Link | Place in |
|---------|-------------|----------|
| Olist Brazilian E-Commerce | [olistbr/brazilian-ecommerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) | `data/olist/` |
| E-Commerce Customer Churn | [ankitverma2010/ecommerce-customer-churn](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction) | `data/churn/` |
| Flipkart Product Reviews | [niraliivaghani/flipkart-product-customer-reviews-dataset](https://www.kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset) | `data/flipkart/` |

---

## 🚀 Local Setup

```bash
# 1. Clone
git clone https://github.com/zonixt017/Smart-Ecom-Analytics.git
cd Smart-Ecom-Analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Re-run preprocessing from raw Kaggle data
python data/preprocess.py

# 4. (Optional) Re-train & save models
python save_models.py

# 5. Launch dashboard
streamlit run app/streamlit_app.py
```

---

## ☁️ Deployment

### Deploy to Streamlit Cloud (free)

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Click **"New app"**
3. Connect your GitHub repo: `zonixt017/Smart-Ecom-Analytics`
4. Set **Main file path**: `app/streamlit_app.py`
5. Click **Deploy** — done! 🎉

> ⚠️ **Note on TensorFlow**: Streamlit Cloud has a 1GB memory limit. If the Sales Forecast page causes memory issues, the LSTM page gracefully falls back to a pre-computed forecast. Consider using `tensorflow-cpu` in requirements for cloud deployment.

---

## 🧠 Modules

### 📊 Week 1 — EDA Overview
- Loaded & merged **9 Olist CSV files** (100k+ orders)
- **10-step cleaning pipeline**: merge → translate → aggregate → filter → fill nulls
- Revenue trends, category analysis, geographic breakdown, payment & review analysis

### 👥 Week 2 — Customer Segmentation
- **RFM** (Recency, Frequency, Monetary) computed per customer
- Log-transform + StandardScaler preprocessing
- **K-Means clustering** (k=4 via Elbow + Silhouette Score)
- Segments: **Champions** 🏆 | **Loyal** 💙 | **At Risk** ⚠️ | **Lost** ❌

### ⚠️ Week 3 — Churn Prediction + SHAP
- Dataset: 3,941 customers with behavioral features
- Model: **Random Forest** (200 trees, max_depth=10)
- **Accuracy: 92.9% | F1: 0.77 | ROC-AUC: 0.958**
- **SHAP explainability**: global feature impact, per-customer waterfall, heatmap
- Interactive live predictor with sliders

### 📈 Week 4 — Sales Forecasting (LSTM)
- Daily revenue time series from Olist orders
- **LSTM**: 64 → 32 units, 30-day look-back, EarlyStopping
- **MAE: R$2,396 | RMSE: R$2,876 | MAPE: 7.7%**
- 30-day autoregressive future forecast

### 🎯 Week 5 — Recommendation System
- Dataset: 189,869 Flipkart product reviews
- **Popularity-based**: weighted score (60% volume + 40% rating)
- **Content-based**: TF-IDF (3,000 features) + Cosine Similarity
- Category-level filtering + similarity heatmap

---

## 📈 Model Performance Summary

| Model | Metric | Value |
|-------|--------|-------|
| K-Means Segmentation | Optimal k | **4** (Elbow + Silhouette) |
| Random Forest (Churn) | ROC-AUC | **0.958** |
| Random Forest (Churn) | Accuracy | **92.9%** |
| LSTM (Sales Forecast) | MAPE | **7.7%** |
| LSTM (Sales Forecast) | MAE | **R$2,396** |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Data Processing | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn`, `plotly` |
| Machine Learning | `scikit-learn` (KMeans, RandomForest, TF-IDF) |
| Deep Learning | `TensorFlow` / `Keras` (LSTM) |
| Explainability | `shap` (TreeExplainer) |
| Dashboard | `Streamlit` |
| Report | `ReportLab` |
| Data Sources | Olist (Kaggle), Flipkart Reviews (Kaggle) |

---

## 📄 Report

Full project report: [`report/Smart_Ecommerce_Project_Report.pdf`](report/Smart_Ecommerce_Project_Report.pdf)

---

## 📋 Requirements

```bash
pip install -r requirements.txt
```

---

## 📝 License

MIT License — feel free to use, modify, and share.
