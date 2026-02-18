"""
Page 4 – Sales Forecasting (LSTM on Olist daily revenue)
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

st.set_page_config(page_title="Sales Forecast", page_icon="📈", layout="wide")
st.title("📈 Sales Forecasting — LSTM Deep Learning")
st.caption("Source: Olist Brazilian E-Commerce · Daily revenue aggregated from 96,477 orders (2016–2018)")

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

@st.cache_data
def load_daily():
    df = pd.read_csv(os.path.join(DATA_DIR, 'orders_clean.csv'),
                     parse_dates=['transaction_date'])
    daily = (df.groupby(df['transaction_date'].dt.date)['total_amount']
             .sum().reset_index())
    daily.columns = ['date', 'revenue']
    daily['date'] = pd.to_datetime(daily['date'])
    full_range = pd.date_range(daily['date'].min(), daily['date'].max(), freq='D')
    daily = daily.set_index('date').reindex(full_range, fill_value=0).reset_index()
    daily.columns = ['date', 'revenue']
    daily['revenue_smooth'] = daily['revenue'].rolling(7, min_periods=1).mean()
    return daily

@st.cache_data
def run_lstm(daily_df):
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping

    LOOK_BACK = 30
    series = daily_df['revenue_smooth'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)

    X_all, y_all = [], []
    for i in range(LOOK_BACK, len(scaled)):
        X_all.append(scaled[i - LOOK_BACK:i, 0])
        y_all.append(scaled[i, 0])
    X_all = np.array(X_all).reshape(-1, LOOK_BACK, 1)
    y_all = np.array(y_all)

    split = int(len(X_all) * 0.8)
    X_tr, X_te = X_all[:split], X_all[split:]
    y_tr, y_te = y_all[:split], y_all[split:]

    tf.random.set_seed(42)
    model = Sequential([
        Input(shape=(LOOK_BACK, 1)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_tr, y_tr, epochs=60, batch_size=32,
                        validation_data=(X_te, y_te),
                        callbacks=[es], verbose=0)

    y_pred_sc = model.predict(X_te, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_sc).flatten()
    y_true = scaler.inverse_transform(y_te.reshape(-1, 1)).flatten()
    test_dates = daily_df['date'].values[LOOK_BACK + split:]

    # 30-day forecast
    last_seq = scaled[-LOOK_BACK:].reshape(1, LOOK_BACK, 1)
    fc_scaled = []
    for _ in range(30):
        p = model.predict(last_seq, verbose=0)[0, 0]
        fc_scaled.append(p)
        last_seq = np.roll(last_seq, -1, axis=1)
        last_seq[0, -1, 0] = p
    forecast = scaler.inverse_transform(np.array(fc_scaled).reshape(-1, 1)).flatten()
    last_date = daily_df['date'].max()
    fc_dates  = pd.date_range(last_date + pd.Timedelta(days=1), periods=30, freq='D')

    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    return y_true, y_pred, test_dates, forecast, fc_dates, mae, rmse, mape, history.history

daily = load_daily()

# ── Check TF ─────────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    tf_available = True
except ImportError:
    tf_available = False

# ── Historical chart ──────────────────────────────────────────────────────────
st.subheader("📅 Historical Daily Revenue (Olist 2016–2018)")
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(daily['date'], daily['revenue'], alpha=0.2, color='steelblue', linewidth=0.8)
ax.plot(daily['date'], daily['revenue_smooth'], color='steelblue', linewidth=2,
        label='7-Day Moving Average')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.xticks(rotation=45)
ax.set_title('Daily Revenue with 7-Day Moving Average (R$)')
ax.set_ylabel('Revenue (R$)')
ax.legend()
plt.tight_layout()
st.pyplot(fig); plt.close()

# ── Monthly bar chart ─────────────────────────────────────────────────────────
st.subheader("📊 Monthly Revenue Summary")
monthly = (daily.groupby(daily['date'].dt.to_period('M'))['revenue']
           .sum().reset_index())
monthly['date'] = monthly['date'].astype(str)
fig, ax = plt.subplots(figsize=(14, 3))
ax.bar(monthly['date'], monthly['revenue'], color='steelblue', edgecolor='white')
ax.set_title('Monthly Revenue (R$)')
ax.set_ylabel('Revenue (R$)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(fig); plt.close()

st.markdown("---")

# ── LSTM section ──────────────────────────────────────────────────────────────
if not tf_available:
    st.warning("⚠️ TensorFlow not installed. Showing moving-average fallback forecast.")
    last_30 = daily['revenue_smooth'].values[-30:]
    avg_d = last_30.mean()
    std_d = last_30.std()
    fc_dates = pd.date_range(daily['date'].max() + pd.Timedelta(days=1), periods=30, freq='D')
    np.random.seed(42)
    forecast = np.clip(avg_d + np.random.normal(0, std_d * 0.3, 30), 0, None)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily.tail(90)['date'], daily.tail(90)['revenue_smooth'],
            color='steelblue', linewidth=2, label='Historical (7-Day MA)')
    ax.plot(fc_dates, forecast, color='red', linewidth=2,
            linestyle='--', marker='o', markersize=4, label='30-Day Forecast')
    ax.axvline(x=daily['date'].max(), color='gray', linestyle=':', linewidth=1.5)
    ax.set_title('30-Day Sales Forecast (Fallback)')
    ax.set_ylabel('Revenue (R$)')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig); plt.close()

else:
    st.subheader("🤖 LSTM Model — Training & Forecast")
    with st.spinner("Training LSTM on Olist daily revenue... (~30 seconds)"):
        y_true, y_pred, test_dates, forecast, fc_dates, mae, rmse, mape, hist = run_lstm(daily)

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Test MAE",  f"R${mae:,.0f}")
    k2.metric("Test RMSE", f"R${rmse:,.0f}")
    k3.metric("MAPE",      f"{mape:.1f}%")
    k4.metric("30-Day Avg Forecast", f"R${forecast.mean():,.0f}/day")

    # Actual vs Predicted
    st.subheader("🎯 Actual vs Predicted (Test Set)")
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(test_dates, y_true, label='Actual',    color='steelblue', linewidth=2)
    ax.plot(test_dates, y_pred, label='Predicted', color='orange',    linewidth=2, linestyle='--')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=45)
    ax.set_title('LSTM — Actual vs Predicted Revenue (R$)')
    ax.set_ylabel('Revenue (R$)')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # 30-day forecast
    st.subheader("🔮 30-Day Sales Forecast")
    fig, ax = plt.subplots(figsize=(14, 5))
    hist_tail = daily.tail(90)
    ax.plot(hist_tail['date'], hist_tail['revenue_smooth'],
            color='steelblue', linewidth=2, label='Historical (7-Day MA)')
    ax.plot(fc_dates, forecast, color='red', linewidth=2,
            linestyle='--', marker='o', markersize=4, label='30-Day Forecast')
    ax.axvline(x=daily['date'].max(), color='gray', linestyle=':',
               linewidth=1.5, label='Forecast Start')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
    plt.xticks(rotation=45)
    ax.set_title('30-Day Sales Forecast (LSTM) — R$')
    ax.set_ylabel('Revenue (R$)')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Training loss + forecast table
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(hist['loss'],     label='Train Loss')
        ax.plot(hist['val_loss'], label='Val Loss')
        ax.set_title('LSTM Training Loss')
        ax.set_xlabel('Epoch')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with col2:
        fc_df = pd.DataFrame({
            'Date': fc_dates.strftime('%Y-%m-%d'),
            'Forecast Revenue (R$)': forecast.round(2)
        })
        st.dataframe(fc_df, use_container_width=True)
