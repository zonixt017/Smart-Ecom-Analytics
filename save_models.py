"""
Train and save all ML/DL models to models/ folder.
Run from project root: python save_models.py
"""
import pandas as pd
import numpy as np
import os, pickle, json
import warnings
warnings.filterwarnings('ignore')

MODELS_DIR = 'models'
DATA_DIR   = 'data'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs('report', exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Customer Segmentation — K-Means + Scaler
# ─────────────────────────────────────────────────────────────────────────────
print("Training K-Means segmentation model...")
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

df = pd.read_csv(os.path.join(DATA_DIR, 'orders_clean.csv'), parse_dates=['transaction_date'])
snapshot = df['transaction_date'].max() + pd.Timedelta(days=1)

rfm = df.groupby('customer_id').agg(
    Recency   = ('transaction_date', lambda x: (snapshot - x.max()).days),
    Frequency = ('order_id',         'count'),
    Monetary  = ('total_amount',     'sum'),
).reset_index()

rfm_log = rfm[['Recency','Frequency','Monetary']].copy()
rfm_log['Recency']   = np.log1p(rfm_log['Recency'])
rfm_log['Frequency'] = np.log1p(rfm_log['Frequency'])
rfm_log['Monetary']  = np.log1p(rfm_log['Monetary'])

scaler_rfm = StandardScaler()
rfm_scaled = scaler_rfm.fit_transform(rfm_log)

km = KMeans(n_clusters=4, random_state=42, n_init=10)
km.fit(rfm_scaled)

with open(os.path.join(MODELS_DIR, 'kmeans_segmentation.pkl'), 'wb') as f:
    pickle.dump(km, f)
with open(os.path.join(MODELS_DIR, 'scaler_rfm.pkl'), 'wb') as f:
    pickle.dump(scaler_rfm, f)

# Save segment metadata
cluster_means = rfm.copy()
cluster_means['Cluster'] = km.predict(rfm_scaled)
seg_order = cluster_means.groupby('Cluster')['Monetary'].mean().sort_values(ascending=False)
label_map = {old: new for new, old in enumerate(seg_order.index)}
cluster_means['Cluster'] = cluster_means['Cluster'].map(label_map)
SEG_NAMES = {0: 'Champions', 1: 'Loyal', 2: 'At Risk', 3: 'Lost'}
cluster_means['Segment'] = cluster_means['Cluster'].map(SEG_NAMES)

seg_summary = cluster_means.groupby('Segment')[['Recency','Frequency','Monetary']].mean().round(2)
seg_summary.to_csv(os.path.join(MODELS_DIR, 'segment_summary.csv'))
print(f"  ✅ Saved: kmeans_segmentation.pkl, scaler_rfm.pkl, segment_summary.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Churn Prediction — Random Forest
# ─────────────────────────────────────────────────────────────────────────────
print("Training Random Forest churn model...")
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

churn = pd.read_csv(os.path.join(DATA_DIR, 'churn_clean.csv'))
le_cat = LabelEncoder()
le_mar = LabelEncoder()
churn['PreferedOrderCat_enc'] = le_cat.fit_transform(churn['PreferedOrderCat'])
churn['MaritalStatus_enc']    = le_mar.fit_transform(churn['MaritalStatus'])

FEAT_COLS = ['Tenure', 'WarehouseToHome', 'NumberOfDeviceRegistered',
             'SatisfactionScore', 'NumberOfAddress', 'Complain',
             'DaySinceLastOrder', 'CashbackAmount',
             'PreferedOrderCat_enc', 'MaritalStatus_enc']

X = churn[FEAT_COLS]
y = churn['Churn']
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr)

y_pred = rf.predict(X_te)
y_prob = rf.predict_proba(X_te)[:, 1]
metrics = {
    'accuracy': round(accuracy_score(y_te, y_pred), 4),
    'f1_score': round(f1_score(y_te, y_pred), 4),
    'roc_auc':  round(roc_auc_score(y_te, y_prob), 4),
    'churn_rate': round(float(y.mean()), 4),
    'train_size': len(X_tr),
    'test_size':  len(X_te),
}

with open(os.path.join(MODELS_DIR, 'rf_churn_model.pkl'), 'wb') as f:
    pickle.dump(rf, f)
with open(os.path.join(MODELS_DIR, 'le_category.pkl'), 'wb') as f:
    pickle.dump(le_cat, f)
with open(os.path.join(MODELS_DIR, 'le_marital.pkl'), 'wb') as f:
    pickle.dump(le_mar, f)
with open(os.path.join(MODELS_DIR, 'churn_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"  ✅ Saved: rf_churn_model.pkl, le_category.pkl, le_marital.pkl, churn_metrics.json")
print(f"     Accuracy={metrics['accuracy']}, F1={metrics['f1_score']}, AUC={metrics['roc_auc']}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Recommendation System — TF-IDF + Popularity
# ─────────────────────────────────────────────────────────────────────────────
print("Building recommendation models...")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import scipy.sparse

flip = pd.read_csv(os.path.join(DATA_DIR, 'flipkart_clean.csv'))

# Popularity model
pop = (flip.groupby(['product_name', 'category'])
       .agg(review_count=('rating','count'), avg_rating=('rating','mean'), avg_price=('price','mean'))
       .reset_index())
sc = MinMaxScaler()
pop[['cnt_n','rat_n']] = sc.fit_transform(pop[['review_count','avg_rating']])
pop['pop_score'] = (0.6 * pop['cnt_n'] + 0.4 * pop['rat_n']).round(4)
pop = pop.sort_values('pop_score', ascending=False).reset_index(drop=True)
pop.to_csv(os.path.join(MODELS_DIR, 'popularity_scores.csv'), index=False)

# TF-IDF content model
prod = (flip.groupby('product_name')
        .agg(category=('category','first'), avg_rating=('rating','mean'),
             avg_price=('price','mean'), review_count=('rating','count'),
             text=('review_text', lambda x: ' '.join(x.dropna().astype(str)[:10])))
        .reset_index())
prod.to_csv(os.path.join(MODELS_DIR, 'products_index.csv'), index=False)

tfidf = TfidfVectorizer(max_features=3000, stop_words='english', min_df=2)
tfidf_mat = tfidf.fit_transform(prod['text'].fillna(''))

with open(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'), 'wb') as f:
    pickle.dump(tfidf, f)
scipy.sparse.save_npz(os.path.join(MODELS_DIR, 'tfidf_matrix.npz'), tfidf_mat)

print(f"  ✅ Saved: popularity_scores.csv, products_index.csv, tfidf_vectorizer.pkl, tfidf_matrix.npz")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Sales Forecasting — LSTM (save weights)
# ─────────────────────────────────────────────────────────────────────────────
print("Training LSTM sales forecasting model...")
try:
    from sklearn.preprocessing import MinMaxScaler as MMS
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping

    daily = (df.groupby(df['transaction_date'].dt.date)['total_amount']
             .sum().reset_index())
    daily.columns = ['date', 'revenue']
    daily['date'] = pd.to_datetime(daily['date'])
    full_range = pd.date_range(daily['date'].min(), daily['date'].max(), freq='D')
    daily = daily.set_index('date').reindex(full_range, fill_value=0).reset_index()
    daily.columns = ['date', 'revenue']
    daily['revenue_smooth'] = daily['revenue'].rolling(7, min_periods=1).mean()

    LOOK_BACK = 30
    series = daily['revenue_smooth'].values.reshape(-1, 1)
    scaler_ts = MMS()
    scaled = scaler_ts.fit_transform(series)

    X_all, y_all = [], []
    for i in range(LOOK_BACK, len(scaled)):
        X_all.append(scaled[i - LOOK_BACK:i, 0])
        y_all.append(scaled[i, 0])
    X_all = np.array(X_all).reshape(-1, LOOK_BACK, 1)
    y_all = np.array(y_all)

    split = int(len(X_all) * 0.8)
    X_tr2, X_te2 = X_all[:split], X_all[split:]
    y_tr2, y_te2 = y_all[:split], y_all[split:]

    tf.random.set_seed(42)
    model = Sequential([
        Input(shape=(LOOK_BACK, 1)),
        LSTM(64, return_sequences=True), Dropout(0.2),
        LSTM(32), Dropout(0.2),
        Dense(16, activation='relu'), Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_tr2, y_tr2, epochs=60, batch_size=32,
              validation_data=(X_te2, y_te2), callbacks=[es], verbose=0)

    model.save(os.path.join(MODELS_DIR, 'lstm_sales_forecast.keras'))
    with open(os.path.join(MODELS_DIR, 'scaler_timeseries.pkl'), 'wb') as f:
        pickle.dump(scaler_ts, f)

    y_pred2 = scaler_ts.inverse_transform(model.predict(X_te2, verbose=0)).flatten()
    y_true2 = scaler_ts.inverse_transform(y_te2.reshape(-1,1)).flatten()
    lstm_metrics = {
        'mae':  round(float(np.mean(np.abs(y_true2 - y_pred2))), 2),
        'rmse': round(float(np.sqrt(np.mean((y_true2 - y_pred2)**2))), 2),
        'mape': round(float(np.mean(np.abs((y_true2 - y_pred2)/(y_true2+1e-8)))*100), 2),
    }
    with open(os.path.join(MODELS_DIR, 'lstm_metrics.json'), 'w') as f:
        json.dump(lstm_metrics, f, indent=2)

    print(f"  ✅ Saved: lstm_sales_forecast.keras, scaler_timeseries.pkl, lstm_metrics.json")
    print(f"     MAE=R${lstm_metrics['mae']:,.0f}, RMSE=R${lstm_metrics['rmse']:,.0f}, MAPE={lstm_metrics['mape']:.1f}%")
except Exception as e:
    print(f"  ⚠️  LSTM save skipped: {e}")

print("\n✅ All models saved to models/ folder!")
print(f"Files: {os.listdir(MODELS_DIR)}")
