"""
Page 1 – EDA Overview  (Olist Brazilian E-Commerce)
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

st.set_page_config(page_title="EDA Overview", page_icon="📊", layout="wide")
st.title("📊 EDA Overview — Brazilian E-Commerce (Olist)")
st.caption("Source: Olist Brazilian E-Commerce Public Dataset · 9 raw CSV files merged & cleaned → 96,477 delivered orders (2016–2018)")

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
OLIST_DIR = os.path.join(DATA_DIR, 'olist')

@st.cache_data
def load_raw():
    orders   = pd.read_csv(os.path.join(OLIST_DIR, 'olist_orders_dataset.csv'))
    items    = pd.read_csv(os.path.join(OLIST_DIR, 'olist_order_items_dataset.csv'))
    payments = pd.read_csv(os.path.join(OLIST_DIR, 'olist_order_payments_dataset.csv'))
    customers= pd.read_csv(os.path.join(OLIST_DIR, 'olist_customers_dataset.csv'))
    products = pd.read_csv(os.path.join(OLIST_DIR, 'olist_products_dataset.csv'))
    reviews  = pd.read_csv(os.path.join(OLIST_DIR, 'olist_order_reviews_dataset.csv'))
    return orders, items, payments, customers, products, reviews

@st.cache_data
def load_clean():
    df = pd.read_csv(os.path.join(DATA_DIR, 'orders_clean.csv'),
                     parse_dates=['transaction_date'])
    return df

orders_raw, items_raw, payments_raw, customers_raw, products_raw, reviews_raw = load_raw()
df = load_clean()

# ── KPIs ─────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Orders (clean)",  f"{len(df):,}")
k2.metric("Unique Customers",      f"{df['customer_id'].nunique():,}")
k3.metric("Total Revenue",         f"R${df['total_amount'].sum():,.0f}")
k4.metric("Avg Order Value",       f"R${df['total_amount'].mean():.2f}")
k5.metric("Product Categories",    f"{df['category'].nunique()}")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: RAW DATA INSPECTION
# ═══════════════════════════════════════════════════════════════════════════════
st.header("🔬 Step 1 — Raw Data Inspection")
st.markdown("The Olist dataset consists of **9 relational CSV files**. Here we inspect each one before cleaning.")

tab_raw1, tab_raw2, tab_raw3, tab_raw4, tab_raw5, tab_raw6 = st.tabs([
    "Orders", "Order Items", "Payments", "Customers", "Products", "Reviews"])

with tab_raw1:
    st.markdown(f"**olist_orders_dataset.csv** — {orders_raw.shape[0]:,} rows × {orders_raw.shape[1]} cols")
    st.dataframe(orders_raw.head(5), use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Data Types:**")
        dtypes_df = orders_raw.dtypes.reset_index()
        dtypes_df.columns = ['column', 'dtype']
        dtypes_df['dtype'] = dtypes_df['dtype'].astype(str)
        st.dataframe(dtypes_df, use_container_width=True)
    with col2:
        st.markdown("**Missing Values:**")
        miss = orders_raw.isnull().sum().rename('missing').reset_index()
        miss['%'] = (miss['missing'] / len(orders_raw) * 100).round(1)
        st.dataframe(miss, use_container_width=True)

with tab_raw2:
    st.markdown(f"**olist_order_items_dataset.csv** — {items_raw.shape[0]:,} rows × {items_raw.shape[1]} cols")
    st.dataframe(items_raw.head(5), use_container_width=True)
    miss = items_raw.isnull().sum().rename('missing').reset_index()
    miss['%'] = (miss['missing'] / len(items_raw) * 100).round(1)
    st.dataframe(miss, use_container_width=True)

with tab_raw3:
    st.markdown(f"**olist_order_payments_dataset.csv** — {payments_raw.shape[0]:,} rows × {payments_raw.shape[1]} cols")
    st.dataframe(payments_raw.head(5), use_container_width=True)
    st.dataframe(payments_raw['payment_type'].value_counts().reset_index(), use_container_width=True)

with tab_raw4:
    st.markdown(f"**olist_customers_dataset.csv** — {customers_raw.shape[0]:,} rows × {customers_raw.shape[1]} cols")
    st.dataframe(customers_raw.head(5), use_container_width=True)
    st.markdown(f"**Top 10 States:** {customers_raw['customer_state'].value_counts().head(10).to_dict()}")

with tab_raw5:
    st.markdown(f"**olist_products_dataset.csv** — {products_raw.shape[0]:,} rows × {products_raw.shape[1]} cols")
    st.dataframe(products_raw.head(5), use_container_width=True)
    miss = products_raw.isnull().sum().rename('missing').reset_index()
    miss['%'] = (miss['missing'] / len(products_raw) * 100).round(1)
    st.dataframe(miss[miss['missing'] > 0], use_container_width=True)

with tab_raw6:
    st.markdown(f"**olist_order_reviews_dataset.csv** — {reviews_raw.shape[0]:,} rows × {reviews_raw.shape[1]} cols")
    st.dataframe(reviews_raw.head(5), use_container_width=True)
    st.markdown(f"**Review Score Distribution:** {reviews_raw['review_score'].value_counts().sort_index().to_dict()}")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: DATA CLEANING STEPS
# ═══════════════════════════════════════════════════════════════════════════════
st.header("🧹 Step 2 — Data Cleaning & Preprocessing")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
    **Cleaning steps applied (`data/preprocess.py`):**

    | Step | Action |
    |------|--------|
    | 1 | Merged 9 Olist CSVs on `order_id` / `customer_id` / `product_id` |
    | 2 | Translated Portuguese category names → English |
    | 3 | Aggregated payment values per order (sum) |
    | 4 | Aggregated item counts per order |
    | 5 | Averaged review scores per order |
    | 6 | **Filtered** to `order_status == 'delivered'` only |
    | 7 | **Dropped** rows with missing `total_amount` or `transaction_date` |
    | 8 | Filled missing `category` → `'Unknown'` |
    | 9 | Filled missing `review_score` → `3.0` (neutral) |
    | 10 | Renamed columns for consistency (`customer_unique_id` → `customer_id`) |
    """)

with col_b:
    # Before vs After
    total_orders = len(orders_raw)
    delivered    = (orders_raw['order_status'] == 'delivered').sum()
    clean_orders = len(df)

    st.markdown("**Before vs After Cleaning:**")
    summary = pd.DataFrame({
        'Stage': ['Raw Orders', 'Delivered Only', 'After Null Drop (final)'],
        'Count': [total_orders, delivered, clean_orders],
        'Retained %': [
            '100%',
            f'{delivered/total_orders*100:.1f}%',
            f'{clean_orders/total_orders*100:.1f}%'
        ]
    })
    st.dataframe(summary, use_container_width=True)

    # Missing values in clean data
    st.markdown("**Missing values in cleaned dataset:**")
    miss_clean = df.isnull().sum().rename('missing').reset_index()
    miss_clean.columns = ['column', 'missing']
    miss_clean['%'] = (miss_clean['missing'] / len(df) * 100).round(2)
    st.dataframe(miss_clean[miss_clean['missing'] > 0]
                 if miss_clean['missing'].sum() > 0
                 else pd.DataFrame({'status': ['✅ No missing values in cleaned data']}),
                 use_container_width=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: CLEANED DATA OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
st.header("📋 Step 3 — Cleaned Dataset Overview")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Sample of cleaned orders_clean.csv:**")
    st.dataframe(df[['order_id','customer_id','transaction_date','total_amount',
                      'quantity','category','city','state','review_score']]
                 .head(10), use_container_width=True)
with col2:
    st.markdown("**Descriptive Statistics:**")
    st.dataframe(df[['total_amount','quantity','review_score']].describe().round(2),
                 use_container_width=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: EDA VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════
st.header("📊 Step 4 — Exploratory Analysis")

# ── Monthly Revenue ───────────────────────────────────────────────────────────
st.subheader("📅 Monthly Revenue Trend")
monthly = (df.groupby(df['transaction_date'].dt.to_period('M'))['total_amount']
           .sum().reset_index())
monthly['transaction_date'] = monthly['transaction_date'].dt.to_timestamp()

fig, ax = plt.subplots(figsize=(14, 4))
ax.fill_between(monthly['transaction_date'], monthly['total_amount'],
                alpha=0.3, color='steelblue')
ax.plot(monthly['transaction_date'], monthly['total_amount'],
        color='steelblue', linewidth=2, marker='o', markersize=4)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)
ax.set_title('Monthly Revenue (R$) — Olist 2016–2018')
ax.set_ylabel('Revenue (R$)')
plt.tight_layout()
st.pyplot(fig); plt.close()

# ── Category & State breakdown ────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏷️ Top 10 Categories by Revenue")
    cat_rev = (df.groupby('category')['total_amount'].sum()
               .sort_values(ascending=False).head(10))
    fig, ax = plt.subplots(figsize=(7, 5))
    cat_rev.plot(kind='barh', ax=ax, color=sns.color_palette('Blues_r', 10))
    ax.set_title('Top 10 Categories by Revenue')
    ax.set_xlabel('Revenue (R$)')
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig); plt.close()

with col2:
    st.subheader("🗺️ Top 10 States by Orders")
    state_cnt = df['state'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(7, 5))
    state_cnt.plot(kind='bar', ax=ax,
                   color=sns.color_palette('Set2', 10), edgecolor='white')
    ax.set_title('Top 10 States by Order Count')
    ax.set_ylabel('Orders')
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ── Order Value Distribution & Review Scores ──────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("💰 Order Value Distribution")
    fig, ax = plt.subplots(figsize=(7, 4))
    vals = df['total_amount'].clip(upper=df['total_amount'].quantile(0.99))
    ax.hist(vals, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_title('Order Value Distribution (99th pct cap)')
    ax.set_xlabel('Order Value (R$)')
    ax.set_ylabel('Count')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

with col4:
    st.subheader("⭐ Review Score Distribution")
    fig, ax = plt.subplots(figsize=(7, 4))
    score_counts = df['review_score'].value_counts().sort_index()
    colors = ['#d73027', '#f46d43', '#fdae61', '#a6d96a', '#1a9850']
    score_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='white')
    ax.set_title('Review Score Distribution')
    ax.set_xlabel('Score')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ── Payment types ─────────────────────────────────────────────────────────────
st.subheader("💳 Payment Type Analysis")
col5, col6 = st.columns(2)

with col5:
    pay_type = payments_raw['payment_type'].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    pay_type.plot(kind='bar', ax=ax,
                  color=sns.color_palette('Set2', len(pay_type)), edgecolor='white')
    ax.set_title('Payment Types (Raw Olist Data)')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

with col6:
    order_status = orders_raw['order_status'].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    wedges, texts, autotexts = ax.pie(
        order_status.values, labels=order_status.index,
        autopct='%1.1f%%', startangle=140,
        colors=sns.color_palette('Set3', len(order_status)))
    ax.set_title('Order Status Distribution (Raw)')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ── Day of Week & Hour ────────────────────────────────────────────────────────
st.subheader("🕐 Order Timing Analysis")
col7, col8 = st.columns(2)

with col7:
    df['day_of_week'] = df['transaction_date'].dt.day_name()
    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_cnt = df['day_of_week'].value_counts().reindex(dow_order)
    fig, ax = plt.subplots(figsize=(7, 4))
    dow_cnt.plot(kind='bar', ax=ax, color='steelblue', edgecolor='white')
    ax.set_title('Orders by Day of Week')
    ax.set_ylabel('Orders')
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

with col8:
    df['hour'] = df['transaction_date'].dt.hour
    hour_cnt = df.groupby('hour')['order_id'].count()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hour_cnt.index, hour_cnt.values, color='steelblue',
            linewidth=2, marker='o', markersize=4)
    ax.fill_between(hour_cnt.index, hour_cnt.values, alpha=0.2, color='steelblue')
    ax.set_title('Orders by Hour of Day')
    ax.set_xlabel('Hour')
    ax.set_ylabel('Orders')
    ax.set_xticks(range(0, 24))
    plt.tight_layout()
    st.pyplot(fig); plt.close()

# ── Correlation heatmap ───────────────────────────────────────────────────────
st.subheader("🔗 Correlation Heatmap")
num_cols = ['total_amount', 'quantity', 'review_score']
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            ax=ax, linewidths=0.5, square=True)
ax.set_title('Correlation — Order Features')
plt.tight_layout()
st.pyplot(fig); plt.close()
