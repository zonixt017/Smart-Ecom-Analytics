"""
Page 5 – Product Recommendation System  (Flipkart Product Reviews)
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os

st.set_page_config(page_title="Recommendations", page_icon="🎯", layout="wide")
st.title("🎯 Product Recommendation System")
st.caption("Dataset: Flipkart Product Reviews · 189,869 reviews · Popularity-based + Content-Based Filtering")

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, 'flipkart_clean.csv'))
    df = df.dropna(subset=['product_name', 'rating'])
    df['rating'] = df['rating'].clip(1, 5)
    return df

@st.cache_data
def build_popularity(df):
    pop = (df.groupby(['product_name', 'category'])
           .agg(
               review_count=('rating', 'count'),
               avg_rating  =('rating', 'mean'),
               avg_price   =('price',  'mean'),
           ).reset_index())
    pop['avg_rating'] = pop['avg_rating'].round(2)
    pop['avg_price']  = pop['avg_price'].round(2)

    sc = MinMaxScaler()
    pop[['cnt_n', 'rat_n']] = sc.fit_transform(pop[['review_count', 'avg_rating']])
    pop['pop_score'] = (0.6 * pop['cnt_n'] + 0.4 * pop['rat_n']).round(4)
    return pop.sort_values('pop_score', ascending=False).reset_index(drop=True)

@st.cache_data
def build_content_model(df):
    # Aggregate reviews per product
    prod = (df.groupby('product_name')
            .agg(
                category    =('category',    'first'),
                avg_rating  =('rating',      'mean'),
                avg_price   =('price',       'mean'),
                review_count=('rating',      'count'),
                text        =('review_text', lambda x: ' '.join(x.dropna().astype(str)[:10]))
            ).reset_index())
    prod['avg_rating'] = prod['avg_rating'].round(2)

    # TF-IDF on review text
    tfidf = TfidfVectorizer(max_features=3000, stop_words='english', min_df=2)
    tfidf_mat = tfidf.fit_transform(prod['text'].fillna(''))
    sim_matrix = cosine_similarity(tfidf_mat, tfidf_mat)
    sim_df = pd.DataFrame(sim_matrix, index=prod['product_name'], columns=prod['product_name'])
    return prod, sim_df

df  = load_data()
pop = build_popularity(df)
prod, sim_df = build_content_model(df)

CATEGORIES = ['All'] + sorted(df['category'].dropna().unique().tolist())

# ── KPIs ─────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Reviews",   f"{len(df):,}")
k2.metric("Unique Products", f"{df['product_name'].nunique():,}")
k3.metric("Categories",      f"{df['category'].nunique()}")
k4.metric("Avg Rating",      f"{df['rating'].mean():.2f} ⭐")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🌟 Popularity-Based", "🔍 Content-Based (TF-IDF)"])

# ── Tab 1: Popularity ─────────────────────────────────────────────────────────
with tab1:
    st.subheader("🌟 Top Products by Popularity Score")
    st.caption("Score = 60% review volume + 40% average rating (both normalized)")

    col_f, col_n = st.columns([2, 1])
    with col_f:
        cat_filter = st.selectbox("Filter by Category", CATEGORIES, key='pop_cat')
    with col_n:
        n_pop = st.slider("Number of recommendations", 3, 15, 5, key='pop_n')

    df_pop = pop.copy()
    if cat_filter != 'All':
        df_pop = df_pop[df_pop['category'] == cat_filter]
    df_pop = df_pop.head(n_pop)

    # Product cards
    n_cols = min(n_pop, 5)
    cols = st.columns(n_cols)
    for i, (_, row) in enumerate(df_pop.iterrows()):
        with cols[i % n_cols]:
            price_str = f"₹{row['avg_price']:,.0f}" if pd.notna(row['avg_price']) and row['avg_price'] > 0 else "N/A"
            st.markdown(f"""
            <div style='background:#f0f4ff;border-radius:10px;padding:12px;margin:4px;text-align:center;min-height:130px'>
            <b style='font-size:12px'>{row['product_name'][:50]}</b><br>
            <small style='color:#666'>{row['category']}</small><br>
            ⭐ {row['avg_rating']:.1f} &nbsp;|&nbsp; 💬 {int(row['review_count'])}<br>
            {price_str}<br>
            <b>Score: {row['pop_score']:.3f}</b>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Charts
    col1, col2 = st.columns(2)
    with col1:
        top10 = pop.head(10)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh([n[:35] for n in top10['product_name']],
                top10['pop_score'],
                color=sns.color_palette('Blues_r', 10))
        ax.set_title('Top 10 Products by Popularity Score')
        ax.set_xlabel('Score')
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        cat_cnt = df['category'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(7, 5))
        cat_cnt.plot(kind='bar', ax=ax,
                     color=sns.color_palette('Set2', 10), edgecolor='white')
        ax.set_title('Reviews by Category (Top 10)')
        ax.set_ylabel('Review Count')
        ax.tick_params(axis='x', rotation=30)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    # Rating distribution
    st.subheader("⭐ Rating Distribution")
    fig, ax = plt.subplots(figsize=(10, 3))
    rating_cnt = df['rating'].value_counts().sort_index()
    colors = ['#d73027', '#f46d43', '#fdae61', '#a6d96a', '#1a9850']
    rating_cnt.plot(kind='bar', ax=ax, color=colors, edgecolor='white')
    ax.set_title('Rating Distribution (Flipkart Reviews)')
    ax.set_xlabel('Rating')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=0)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.subheader("📋 Full Popularity Table (Top 30)")
    st.dataframe(
        pop[['product_name', 'category', 'review_count', 'avg_rating', 'avg_price', 'pop_score']]
        .head(30),
        use_container_width=True
    )

# ── Tab 2: Content-Based ──────────────────────────────────────────────────────
with tab2:
    st.subheader("🔍 Content-Based Recommendations (TF-IDF on Review Text)")
    st.caption("Finds similar products based on what customers say in their reviews")

    all_products = sorted(prod['product_name'].tolist())
    selected = st.selectbox("Select a Product", all_products, key='cb_prod')
    n_cb = st.slider("Number of recommendations", 3, 10, 5, key='cb_n')

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Selected Product:**")
        sel_info = prod[prod['product_name'] == selected].iloc[0]
        st.markdown(f"""
        <div style='background:#e8f4f8;border-radius:10px;padding:15px'>
        <b>{selected[:60]}</b><br>
        Category: {sel_info['category']}<br>
        Avg Rating: ⭐ {sel_info['avg_rating']:.2f}<br>
        Reviews: {int(sel_info['review_count'])}<br>
        Avg Price: {'₹' + f"{sel_info['avg_price']:,.0f}" if pd.notna(sel_info['avg_price']) and sel_info['avg_price'] > 0 else 'N/A'}
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("**Similar Products (by review content):**")
        if selected in sim_df.index:
            sim_scores = sim_df[selected].drop(index=selected, errors='ignore')
            top_similar = sim_scores.sort_values(ascending=False).head(n_cb)
            recs = prod[prod['product_name'].isin(top_similar.index)].copy()
            recs['similarity'] = recs['product_name'].map(top_similar).round(4)
            recs = recs.sort_values('similarity', ascending=False)
            st.dataframe(
                recs[['product_name', 'category', 'avg_rating', 'avg_price', 'similarity']]
                .reset_index(drop=True),
                use_container_width=True
            )
        else:
            st.info("Product not found in similarity matrix.")

    # Similarity heatmap for top 8 products
    st.markdown("---")
    st.subheader("🗺️ Similarity Heatmap (Top 8 Most-Reviewed Products)")
    top8 = pop.head(8)['product_name'].tolist()
    top8 = [p for p in top8 if p in sim_df.index]
    if len(top8) >= 2:
        sub = sim_df.loc[top8, top8]
        labels = [n[:25] for n in sub.index]
        sub_plot = sub.copy()
        sub_plot.index   = labels
        sub_plot.columns = labels
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(sub_plot, annot=True, fmt='.2f', cmap='YlOrRd',
                    ax=ax, linewidths=0.5, square=True)
        ax.set_title('Product Similarity (TF-IDF Cosine)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig); plt.close()
