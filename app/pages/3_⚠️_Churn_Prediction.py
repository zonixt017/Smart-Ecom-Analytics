"""
Page 3 – Churn Prediction  (E-Commerce Customer Churn Dataset)
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              confusion_matrix, roc_curve)
import os

st.set_page_config(page_title="Churn Prediction", page_icon="⚠️", layout="wide")
st.title("⚠️ Customer Churn Prediction")
st.caption("Dataset: E-Commerce Customer Churn · 3,941 customers · Features: Tenure, Satisfaction, Complaints, etc.")

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

FEAT_COLS = ['Tenure', 'WarehouseToHome', 'NumberOfDeviceRegistered',
             'SatisfactionScore', 'NumberOfAddress', 'Complain',
             'DaySinceLastOrder', 'CashbackAmount',
             'PreferedOrderCat_enc', 'MaritalStatus_enc']

@st.cache_data
def load_and_train():
    df = pd.read_csv(os.path.join(DATA_DIR, 'churn_clean.csv'))

    # Encode categoricals
    le_cat = LabelEncoder()
    le_mar = LabelEncoder()
    df['PreferedOrderCat_enc'] = le_cat.fit_transform(df['PreferedOrderCat'])
    df['MaritalStatus_enc']    = le_mar.fit_transform(df['MaritalStatus'])

    X = df[FEAT_COLS]
    y = df['Churn']

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)

    y_pred = rf.predict(X_te)
    y_prob = rf.predict_proba(X_te)[:, 1]

    metrics = {
        'Accuracy': accuracy_score(y_te, y_pred),
        'F1 Score': f1_score(y_te, y_pred),
        'ROC-AUC':  roc_auc_score(y_te, y_prob),
    }
    cm  = confusion_matrix(y_te, y_pred)
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    importances = pd.Series(rf.feature_importances_,
                            index=FEAT_COLS).sort_values(ascending=False)

    df['Churn_Prob'] = rf.predict_proba(X)[:, 1].round(4)
    return df, rf, metrics, cm, fpr, tpr, importances

df, rf, metrics, cm, fpr, tpr, importances = load_and_train()

# ── KPIs ─────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Churn Rate",  f"{df['Churn'].mean():.1%}")
k2.metric("Accuracy",    f"{metrics['Accuracy']:.3f}")
k3.metric("F1 Score",    f"{metrics['F1 Score']:.3f}")
k4.metric("ROC-AUC",     f"{metrics['ROC-AUC']:.3f}")

st.markdown("---")

# ── Confusion Matrix + ROC ────────────────────────────────────────────────────
st.subheader("📊 Model Performance")
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Active', 'Churned'],
                yticklabels=['Active', 'Churned'])
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color='steelblue', linewidth=2,
            label=f"AUC = {metrics['ROC-AUC']:.3f}")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_title('ROC Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig); plt.close()

st.markdown("---")

# ── Feature Importance ────────────────────────────────────────────────────────
st.subheader("🔑 Feature Importance")
fig, ax = plt.subplots(figsize=(10, 4))
importances.plot(kind='bar', ax=ax,
                 color=sns.color_palette('Blues_r', len(importances)))
ax.set_title('Random Forest Feature Importance')
ax.set_ylabel('Importance')
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
st.pyplot(fig); plt.close()

st.markdown("---")

# ── Churn by key features ─────────────────────────────────────────────────────
st.subheader("📉 Churn Analysis by Key Features")
col3, col4 = st.columns(2)

with col3:
    fig, ax = plt.subplots(figsize=(7, 4))
    churn_by_cat = df.groupby('PreferedOrderCat')['Churn'].mean().sort_values(ascending=False)
    churn_by_cat.plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
    ax.set_title('Churn Rate by Preferred Order Category')
    ax.set_ylabel('Churn Rate')
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

with col4:
    fig, ax = plt.subplots(figsize=(7, 4))
    churn_by_sat = df.groupby('SatisfactionScore')['Churn'].mean()
    ax.bar(churn_by_sat.index, churn_by_sat.values,
           color=['#2ecc71' if v < 0.2 else '#e67e22' if v < 0.4 else '#e74c3c'
                  for v in churn_by_sat.values],
           edgecolor='white')
    ax.set_title('Churn Rate by Satisfaction Score')
    ax.set_xlabel('Satisfaction Score (1=Low, 5=High)')
    ax.set_ylabel('Churn Rate')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ── Tenure vs Churn ───────────────────────────────────────────────────────────
col5, col6 = st.columns(2)
with col5:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[df['Churn'] == 0]['Tenure'], bins=20, alpha=0.6,
            color='steelblue', label='Active')
    ax.hist(df[df['Churn'] == 1]['Tenure'], bins=20, alpha=0.6,
            color='red', label='Churned')
    ax.set_title('Tenure Distribution by Churn')
    ax.set_xlabel('Tenure (months)')
    ax.set_ylabel('Count')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig); plt.close()

with col6:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[df['Churn'] == 0]['Churn_Prob'], bins=30, alpha=0.6,
            color='steelblue', label='Active')
    ax.hist(df[df['Churn'] == 1]['Churn_Prob'], bins=30, alpha=0.6,
            color='red', label='Churned')
    ax.set_title('Churn Probability Distribution')
    ax.set_xlabel('Churn Probability')
    ax.set_ylabel('Count')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig); plt.close()

st.markdown("---")

# ── Interactive Predictor ─────────────────────────────────────────────────────
st.subheader("🔮 Predict Churn for a Customer Profile")
st.markdown("Adjust the sliders to simulate a customer:")

c1, c2, c3 = st.columns(3)
with c1:
    tenure       = st.slider("Tenure (months)", 0, 60, 12)
    wh_to_home   = st.slider("Warehouse to Home (km)", 5, 100, 30)
    num_devices  = st.slider("Number of Devices Registered", 1, 6, 3)
with c2:
    sat_score    = st.slider("Satisfaction Score (1-5)", 1, 5, 3)
    num_address  = st.slider("Number of Addresses", 1, 10, 3)
    complain     = st.selectbox("Has Complained?", [0, 1],
                                format_func=lambda x: 'Yes' if x else 'No')
with c3:
    days_since   = st.slider("Days Since Last Order", 0, 30, 5)
    cashback     = st.slider("Cashback Amount ($)", 0, 300, 150)
    pref_cat_enc = st.selectbox("Preferred Category",
                                [0, 1, 2, 3, 4],
                                format_func=lambda x: ['Fashion','Grocery','Laptop & Acc','Mobile','Others'][x])
    marital_enc  = st.selectbox("Marital Status",
                                [0, 1, 2],
                                format_func=lambda x: ['Divorced','Married','Single'][x])

input_arr = np.array([[tenure, wh_to_home, num_devices, sat_score,
                       num_address, complain, days_since, cashback,
                       pref_cat_enc, marital_enc]])
prob = rf.predict_proba(input_arr)[0][1]
pred = "🔴 Likely to Churn" if prob > 0.5 else "🟢 Likely to Stay"

st.markdown("---")
res1, res2 = st.columns(2)
res1.metric("Churn Probability", f"{prob:.1%}")
res2.metric("Prediction", pred)

st.markdown("---")

# ── SHAP Explainability ───────────────────────────────────────────────────────
st.subheader("🧠 SHAP — Model Explainability")
st.markdown("""
**SHAP (SHapley Additive exPlanations)** shows *why* the model makes each prediction.
- **Red bars** → feature pushes prediction toward churn
- **Blue bars** → feature pushes prediction away from churn
""")

@st.cache_data
def compute_shap(_model, _X):
    import shap
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(_X)
    # For binary classification, shap_values is a list [class0, class1]
    if isinstance(shap_values, list):
        return shap_values[1], explainer.expected_value[1]
    return shap_values, explainer.expected_value

X_full = df[FEAT_COLS]
shap_vals, base_val = compute_shap(rf, X_full)

shap_tab1, shap_tab2, shap_tab3 = st.tabs(["📊 Global Feature Impact", "🔍 Single Prediction", "🌡️ SHAP Heatmap"])

with shap_tab1:
    st.markdown("**Mean |SHAP| value per feature** — higher = more influential globally")
    mean_shap = pd.Series(
        np.abs(shap_vals).mean(axis=0),
        index=FEAT_COLS
    ).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 4))
    colors_shap = ['#e74c3c' if v > mean_shap.median() else '#3498db' for v in mean_shap.values]
    ax.barh(mean_shap.index[::-1], mean_shap.values[::-1], color=colors_shap[::-1])
    ax.set_title('Mean |SHAP| Value — Global Feature Importance')
    ax.set_xlabel('Mean |SHAP| Value')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

with shap_tab2:
    st.markdown("**SHAP waterfall for a single customer** — see exactly what drives their churn risk")
    cust_idx = st.slider("Select customer index", 0, len(df)-1, 0)
    row_shap = shap_vals[cust_idx]
    row_feat = X_full.iloc[cust_idx]
    churn_prob_cust = rf.predict_proba(row_feat.values.reshape(1, -1))[0][1]

    st.metric("Churn Probability", f"{churn_prob_cust:.1%}",
              delta="High Risk" if churn_prob_cust > 0.5 else "Low Risk")

    # Waterfall chart
    shap_df = pd.DataFrame({
        'Feature': FEAT_COLS,
        'Value': row_feat.values,
        'SHAP': row_shap
    }).sort_values('SHAP', key=abs, ascending=False).head(8)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors_w = ['#e74c3c' if v > 0 else '#2ecc71' for v in shap_df['SHAP']]
    bars = ax.barh(
        [f"{r['Feature']}={r['Value']:.1f}" for _, r in shap_df.iterrows()],
        shap_df['SHAP'], color=colors_w
    )
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title(f'SHAP Waterfall — Customer #{cust_idx} (Churn Prob: {churn_prob_cust:.1%})')
    ax.set_xlabel('SHAP Value (impact on churn probability)')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

with shap_tab3:
    st.markdown("**SHAP values for top 200 customers** — patterns across the dataset")
    sample_n = min(200, len(shap_vals))
    shap_sample = shap_vals[:sample_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(shap_sample.T, aspect='auto', cmap='RdBu_r',
                   vmin=-np.percentile(np.abs(shap_sample), 95),
                   vmax= np.percentile(np.abs(shap_sample), 95))
    ax.set_yticks(range(len(FEAT_COLS)))
    ax.set_yticklabels(FEAT_COLS, fontsize=9)
    ax.set_xlabel('Customer Index')
    ax.set_title(f'SHAP Value Heatmap (first {sample_n} customers)')
    plt.colorbar(im, ax=ax, label='SHAP Value')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

st.markdown("---")

# ── High-risk customers ───────────────────────────────────────────────────────
st.subheader("🚨 Top 20 High-Risk Customers")
high_risk = (df[['Tenure', 'SatisfactionScore', 'Complain',
                 'DaySinceLastOrder', 'CashbackAmount', 'Churn', 'Churn_Prob']]
             .sort_values('Churn_Prob', ascending=False)
             .head(20)
             .reset_index(drop=True))
st.dataframe(high_risk, use_container_width=True)
