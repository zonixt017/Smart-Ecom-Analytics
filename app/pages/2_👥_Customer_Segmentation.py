"""
Page 2 – Customer Segmentation  (RFM + K-Means on Olist orders)
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

st.set_page_config(page_title="Customer Segmentation", page_icon="👥", layout="wide")
st.title("👥 Customer Segmentation — RFM + K-Means")
st.caption("Based on 96,477 real Olist orders · RFM computed per unique customer")

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

@st.cache_data
def load_and_segment():
    df = pd.read_csv(os.path.join(DATA_DIR, 'orders_clean.csv'),
                     parse_dates=['transaction_date'])

    snapshot = df['transaction_date'].max() + pd.Timedelta(days=1)

    rfm = df.groupby('customer_id').agg(
        Recency   = ('transaction_date', lambda x: (snapshot - x.max()).days),
        Frequency = ('order_id',         'count'),
        Monetary  = ('total_amount',     'sum'),
    ).reset_index()
    rfm['Monetary'] = rfm['Monetary'].round(2)

    # Log-transform to reduce skew
    rfm_log = rfm[['Recency','Frequency','Monetary']].copy()
    rfm_log['Recency']   = np.log1p(rfm_log['Recency'])
    rfm_log['Frequency'] = np.log1p(rfm_log['Frequency'])
    rfm_log['Monetary']  = np.log1p(rfm_log['Monetary'])

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)

    # K-Means k=4
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm['Cluster'] = km.fit_predict(rfm_scaled)

    # Label clusters by mean Monetary descending
    cluster_means = rfm.groupby('Cluster')['Monetary'].mean().sort_values(ascending=False)
    label_map = {old: new for new, old in enumerate(cluster_means.index)}
    rfm['Cluster'] = rfm['Cluster'].map(label_map)

    SEG_NAMES = {0: 'Champions', 1: 'Loyal', 2: 'At Risk', 3: 'Lost'}
    rfm['Segment'] = rfm['Cluster'].map(SEG_NAMES)

    # PCA for 2D viz
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(rfm_scaled)
    rfm['PC1'] = pca_coords[:, 0]
    rfm['PC2'] = pca_coords[:, 1]

    return rfm, snapshot

rfm, snapshot = load_and_segment()

SEG_ORDER  = ['Champions', 'Loyal', 'At Risk', 'Lost']
SEG_COLORS = {'Champions': '#2ecc71', 'Loyal': '#3498db',
              'At Risk': '#e67e22', 'Lost': '#e74c3c'}

# ── KPIs ─────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Customers",  f"{len(rfm):,}")
k2.metric("Avg Recency (days)", f"{rfm['Recency'].mean():.0f}")
k3.metric("Avg Frequency",    f"{rfm['Frequency'].mean():.1f}")
k4.metric("Avg Monetary (R$)",f"R${rfm['Monetary'].mean():.2f}")

st.markdown("---")

# ── Segment distribution ──────────────────────────────────────────────────────
st.subheader("📊 Segment Distribution")
col1, col2 = st.columns(2)

with col1:
    seg_cnt = rfm['Segment'].value_counts().reindex(SEG_ORDER)
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(seg_cnt.index, seg_cnt.values,
                  color=[SEG_COLORS[s] for s in seg_cnt.index], edgecolor='white')
    for bar, val in zip(bars, seg_cnt.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:,}', ha='center', va='bottom', fontsize=10)
    ax.set_title('Customers per Segment')
    ax.set_ylabel('Count')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

with col2:
    seg_rev = rfm.groupby('Segment')['Monetary'].sum().reindex(SEG_ORDER)
    fig, ax = plt.subplots(figsize=(6, 4))
    wedges, texts, autotexts = ax.pie(
        seg_rev.values,
        labels=seg_rev.index,
        colors=[SEG_COLORS[s] for s in seg_rev.index],
        autopct='%1.1f%%', startangle=140)
    ax.set_title('Revenue Share by Segment')
    plt.tight_layout()
    st.pyplot(fig); plt.close()

st.markdown("---")

# ── RFM Box plots ─────────────────────────────────────────────────────────────
st.subheader("📦 RFM Distribution by Segment")
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, feat in zip(axes, ['Recency', 'Frequency', 'Monetary']):
    data_plot = [rfm[rfm['Segment'] == s][feat].values for s in SEG_ORDER]
    bp = ax.boxplot(data_plot, patch_artist=True, tick_labels=SEG_ORDER)
    for patch, seg in zip(bp['boxes'], SEG_ORDER):
        patch.set_facecolor(SEG_COLORS[seg])
        patch.set_alpha(0.7)
    ax.set_title(feat)
    ax.tick_params(axis='x', rotation=20)
    if feat == 'Monetary':
        ax.set_ylabel('R$')
plt.tight_layout()
st.pyplot(fig); plt.close()

st.markdown("---")

# ── PCA scatter ───────────────────────────────────────────────────────────────
st.subheader("🔵 Customer Clusters (PCA 2D)")
fig, ax = plt.subplots(figsize=(10, 6))
for seg in SEG_ORDER:
    sub = rfm[rfm['Segment'] == seg]
    ax.scatter(sub['PC1'], sub['PC2'], label=seg,
               color=SEG_COLORS[seg], alpha=0.4, s=10)
ax.set_title('K-Means Clusters (PCA projection)')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.legend(markerscale=3)
plt.tight_layout()
st.pyplot(fig); plt.close()

st.markdown("---")

# ── Segment summary table ─────────────────────────────────────────────────────
st.subheader("📋 Segment Summary")
summary = rfm.groupby('Segment').agg(
    Customers  = ('customer_id', 'count'),
    Avg_Recency= ('Recency',     'mean'),
    Avg_Freq   = ('Frequency',   'mean'),
    Avg_Monetary=('Monetary',    'mean'),
    Total_Rev  = ('Monetary',    'sum'),
).round(2).reindex(SEG_ORDER)
st.dataframe(summary, use_container_width=True)

st.markdown("---")

# ── Customer lookup ───────────────────────────────────────────────────────────
st.subheader("🔍 Customer Lookup")
seg_filter = st.selectbox("Filter by Segment", ['All'] + SEG_ORDER)
show = rfm if seg_filter == 'All' else rfm[rfm['Segment'] == seg_filter]
st.dataframe(
    show[['customer_id','Recency','Frequency','Monetary','Segment']]
    .sort_values('Monetary', ascending=False)
    .head(30)
    .reset_index(drop=True),
    use_container_width=True
)
