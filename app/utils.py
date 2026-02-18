"""
Shared utility functions for the Streamlit dashboard.
(Import this in page files as needed)
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import os

DATA_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def load_data():
    customers    = pd.read_csv(os.path.join(DATA_DIR, 'customers.csv'),    parse_dates=['join_date'])
    transactions = pd.read_csv(os.path.join(DATA_DIR, 'transactions.csv'), parse_dates=['transaction_date'])
    products     = pd.read_csv(os.path.join(DATA_DIR, 'products.csv'))
    completed    = transactions[transactions['status'] == 'Completed'].copy()
    return customers, transactions, products, completed


def compute_rfm(completed):
    snapshot_date = completed['transaction_date'].max() + pd.Timedelta(days=1)
    rfm = completed.groupby('customer_id').agg(
        Recency   = ('transaction_date', lambda x: (snapshot_date - x.max()).days),
        Frequency = ('transaction_id',   'count'),
        Monetary  = ('total_amount',     'sum'),
    ).reset_index()
    rfm['Monetary'] = rfm['Monetary'].round(2)
    return rfm, snapshot_date


def compute_popularity(completed, products):
    pop = (completed.groupby(['product_id', 'product_name', 'category'])
           .agg(purchase_count=('transaction_id', 'count'),
                total_revenue =('total_amount',   'sum'))
           .reset_index())
    pop = pop.merge(products[['product_id', 'rating']], on='product_id', how='left')
    sc  = MinMaxScaler()
    pop[['cnt_n', 'rev_n', 'rat_n']] = sc.fit_transform(
        pop[['purchase_count', 'total_revenue', 'rating']])
    pop['pop_score'] = (0.5 * pop['cnt_n'] + 0.3 * pop['rev_n'] + 0.2 * pop['rat_n']).round(4)
    return pop.sort_values('pop_score', ascending=False).reset_index(drop=True)


def build_user_item(completed):
    return (completed.groupby(['customer_id', 'product_id'])['quantity']
            .sum().unstack(fill_value=0))


def get_cf_recommendations(customer_id, user_item, item_sim_df, products, n=5, popularity=None):
    if customer_id not in user_item.index:
        if popularity is not None:
            return popularity[['product_name', 'category', 'rating', 'pop_score']].head(n)
        return pd.DataFrame()
    purchased     = user_item.loc[customer_id]
    purchased_ids = purchased[purchased > 0].index.tolist()
    scores        = pd.Series(0.0, index=item_sim_df.index)
    for pid in purchased_ids:
        if pid in item_sim_df.index:
            scores += item_sim_df[pid]
    scores   = scores.drop(labels=[p for p in purchased_ids if p in scores.index], errors='ignore')
    top_pids = scores.sort_values(ascending=False).head(n).index.tolist()
    recs     = products[products['product_id'].isin(top_pids)][
        ['product_id', 'product_name', 'category', 'price', 'rating']].copy()
    recs['score'] = recs['product_id'].map(scores)
    return recs.sort_values('score', ascending=False).reset_index(drop=True)
