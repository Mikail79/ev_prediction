"""
train_model.py — Multivariate LSTM Model Training for EV Forecasting
=====================================================================
Trains a 2-layer LSTM on monthly WA EV registration data combined
with charging station infrastructure data (stations count, L2 ports,
DC Fast ports) for improved prediction accuracy.

Features (4 inputs):
  1. Electric Vehicle (EV) Total
  2. Stations_Cumulative (charging stations)
  3. Ports_L2_Cumulative (Level 2 ports)
  4. Ports_DCFC_Cumulative (DC Fast Charging ports)

Outputs:
  - ev_model.h5   (trained Keras model)
  - scaler.pkl    (fitted MinMaxScaler — 4 features)
  - metrics.json  (MAPE, MAE, RMSE for the dashboard)

Usage:
  python train_model.py
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# 1. Load & Prepare Data
# ──────────────────────────────────────────────────────────────
try:
    base_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_dir = os.path.abspath("")  # Fallback for Jupyter Notebooks

EV_DATA_PATH = os.path.join(base_dir, "ev_population.csv")
CHARGING_DATA_PATH = os.path.join(base_dir, "charging_stations.csv")
LOOKBACK = 12  # months
NUM_FEATURES = 4  # EV_Total, Stations, L2 Ports, DCFC Ports

print("=" * 60)
print("  EV Forecasting — Multivariate LSTM Training Pipeline")
print("=" * 60)

# --- Fetch EV Population Data ---
try:
    print("\n[DATA] Fetching EV population data from data.wa.gov API...")
    url_ev = "https://data.wa.gov/api/views/d886-d5q2/rows.csv?accessType=DOWNLOAD"
    df_ev_raw = pd.read_csv(url_ev)
    df_ev = df_ev_raw[['Date', 'Electric Vehicle (EV) Total']].copy()
    df_ev.to_csv(EV_DATA_PATH, index=False)
    print("   [OK] Live EV data fetched and backed up locally.")
except Exception as e:
    print(f"   [WARN] API fetch failed: {e}. Falling back to local data.")
    df_ev = pd.read_csv(EV_DATA_PATH)

df_ev['Date'] = pd.to_datetime(df_ev['Date'])
df_ev = df_ev.sort_values("Date").reset_index(drop=True)

# --- Fetch Charging Station Data ---
try:
    print("\n[DATA] Fetching charging station data from AFDC API...")
    from fetch_charging_data import fetch_all_stations, build_monthly_timeseries
    stations, total = fetch_all_stations()
    df_charging = build_monthly_timeseries(stations)
    df_charging.to_csv(CHARGING_DATA_PATH, index=False)
    print("   [OK] Live charging data fetched and backed up locally.")
except Exception as e:
    print(f"   [WARN] AFDC API fetch failed: {e}. Falling back to local data.")
    df_charging = pd.read_csv(CHARGING_DATA_PATH)
    df_charging['Date'] = pd.to_datetime(df_charging['Date'])

# --- Merge datasets on month ---
df_ev['merge_key'] = df_ev['Date'].dt.to_period('M')
df_charging['Date'] = pd.to_datetime(df_charging['Date'])
df_charging['merge_key'] = df_charging['Date'].dt.to_period('M')

df = pd.merge(
    df_ev[['merge_key', 'Electric Vehicle (EV) Total']],
    df_charging[['merge_key', 'Stations_Cumulative', 'Ports_L2_Cumulative', 'Ports_DCFC_Cumulative']],
    on='merge_key',
    how='inner'
)

# Restore Date column from merge_key
df['Date'] = df['merge_key'].dt.to_timestamp(how='end')
df = df.sort_values('Date').reset_index(drop=True)
df = df.drop(columns=['merge_key'])

# Reorder columns
df = df[['Date', 'Electric Vehicle (EV) Total', 'Stations_Cumulative', 'Ports_L2_Cumulative', 'Ports_DCFC_Cumulative']]

# Save merged dataset for reference
merged_path = os.path.join(base_dir, "ev_multivariate.csv")
df.to_csv(merged_path, index=False)

print(f"\n[DATA] Merged Dataset: {len(df)} monthly records, {NUM_FEATURES} features")
print(f"   Date Range : {df['Date'].min().strftime('%b %Y')} -> {df['Date'].max().strftime('%b %Y')}")
print(f"   EV Total   : {int(df['Electric Vehicle (EV) Total'].min()):,} -> {int(df['Electric Vehicle (EV) Total'].max()):,}")
print(f"   Stations   : {int(df['Stations_Cumulative'].min()):,} -> {int(df['Stations_Cumulative'].max()):,}")
print(f"   L2 Ports   : {int(df['Ports_L2_Cumulative'].min()):,} -> {int(df['Ports_L2_Cumulative'].max()):,}")
print(f"   DCFC Ports : {int(df['Ports_DCFC_Cumulative'].min()):,} -> {int(df['Ports_DCFC_Cumulative'].max()):,}")

# ──────────────────────────────────────────────────────────────
# 2. Scale (all 4 features together)
# ──────────────────────────────────────────────────────────────
feature_cols = ['Electric Vehicle (EV) Total', 'Stations_Cumulative', 'Ports_L2_Cumulative', 'Ports_DCFC_Cumulative']
values = df[feature_cols].values.astype(float)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# ──────────────────────────────────────────────────────────────
# 3. Create Sequences (Multivariate)
# ──────────────────────────────────────────────────────────────
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback : i, :])  # All features
        y.append(data[i, 0])  # Target: EV Total only (first column)
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, LOOKBACK)
# X shape: (samples, lookback, num_features) = (samples, 12, 4)

# Train/Test split — last 20% for testing
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"\n[SPLIT] Train/Test Split:")
print(f"   Training : {len(X_train)} samples")
print(f"   Testing  : {len(X_test)} samples")
print(f"   Lookback : {LOOKBACK} months")
print(f"   Features : {NUM_FEATURES} (EV Total, Stations, L2 Ports, DCFC Ports)")
print(f"   Input Shape : {X_train.shape}")

# ──────────────────────────────────────────────────────────────
# 4. Build Multivariate LSTM Model
# ──────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

tf.random.set_seed(42)
np.random.seed(42)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(LOOKBACK, NUM_FEATURES)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1),
])

model.compile(optimizer="adam", loss="mse")

print("\n[MODEL] Multivariate LSTM Architecture:")
model.summary()

# ──────────────────────────────────────────────────────────────
# 5. Train
# ──────────────────────────────────────────────────────────────
print("\n[TRAIN] Training started...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1,
)

# ──────────────────────────────────────────────────────────────
# 6. Evaluate
# ──────────────────────────────────────────────────────────────
predictions_scaled = model.predict(X_test)

# To inverse-transform, we need to reconstruct the full feature array
# Create dummy arrays for the other 3 features (they won't affect the inverse of column 0)
dummy = np.zeros((len(predictions_scaled), NUM_FEATURES))
dummy[:, 0] = predictions_scaled.flatten()
predictions = scaler.inverse_transform(dummy)[:, 0]

dummy_actual = np.zeros((len(y_test), NUM_FEATURES))
dummy_actual[:, 0] = y_test
actuals = scaler.inverse_transform(dummy_actual)[:, 0]

# MAPE
mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
# MAE
mae = np.mean(np.abs(actuals - predictions))
# RMSE
rmse = np.sqrt(np.mean((actuals - predictions) ** 2))

print("\n" + "=" * 60)
print("  [RESULTS] Evaluation Results (Multivariate LSTM)")
print("=" * 60)
print(f"   MAPE : {mape:.2f}%")
print(f"   MAE  : {mae:,.0f}")
print(f"   RMSE : {rmse:,.0f}")
print(f"   {'[OK] MAPE < 10% -- Target Met!' if mape < 10 else '[WARN] MAPE >= 10% -- Consider tuning.'}")

# ──────────────────────────────────────────────────────────────
# 7. Save Artifacts
# ──────────────────────────────────────────────────────────────
try:
    save_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    save_dir = os.path.abspath("")

model_path = os.path.join(save_dir, "ev_model.h5")
scaler_path = os.path.join(save_dir, "scaler.pkl")
metrics_path = os.path.join(save_dir, "metrics.json")

model.save(model_path)
joblib.dump(scaler, scaler_path)

# Save metrics + latest charging station count for the dashboard
latest_stations = int(df['Stations_Cumulative'].iloc[-1])
latest_ports = int(df['Ports_L2_Cumulative'].iloc[-1] + df['Ports_DCFC_Cumulative'].iloc[-1])

metrics = {
    "mape": round(mape, 2),
    "mae": round(mae, 2),
    "rmse": round(rmse, 2),
    "latest_stations": latest_stations,
    "latest_ports": latest_ports,
    "num_features": NUM_FEATURES,
    "model_type": "Multivariate LSTM",
}
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n[SAVE] Saved Artifacts:")
print(f"   Model   -> {model_path}")
print(f"   Scaler  -> {scaler_path}")
print(f"   Metrics -> {metrics_path}")
print(f"   Merged  -> {merged_path}")
print("\n[DONE] Training complete. Run `streamlit run app.py` to launch the dashboard.")
