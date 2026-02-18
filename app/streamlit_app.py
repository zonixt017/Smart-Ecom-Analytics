"""
Smart E-Commerce Analytics Platform
Main Streamlit Entry Point
"""
import streamlit as st

st.set_page_config(
    page_title="Smart E-Commerce Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛒 Smart E-Commerce Analytics Platform")
st.markdown("""
Welcome to the **Smart E-Commerce Analytics Platform** — a complete end-to-end data science dashboard.

### 📌 Navigate using the sidebar:
| Page | Description |
|------|-------------|
| 📊 EDA Overview | Sales trends, top products, customer demographics |
| 👥 Customer Segmentation | RFM analysis and K-Means clusters |
| ⚠️ Churn Prediction | Predict customer churn probability |
| 📈 Sales Forecast | 30-day LSTM sales forecast |
| 🎯 Recommendations | Personalized product recommendations |

---
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("**Dataset**: 1,000 customers · 15,000 transactions · 35 products")
with col2:
    st.info("**Period**: Jan 2022 – Dec 2024")
with col3:
    st.info("**Models**: K-Means · Random Forest · LSTM · Collaborative Filtering")

st.markdown("---")
st.caption("Capstone Project | Smart E-Commerce Analytics Platform")
