"""
train_model.py — LSTM Model Training for EV Registration Forecasting
=====================================================================
Trains a 2-layer LSTM on monthly WA EV registration data.
Outputs:
  - ev_model.h5   (trained Keras model)
  - scaler.pkl    (fitted MinMaxScaler)
  - metrics.json  (MAPE, RMSE for the dashboard)

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

DATA_PATH = os.path.join(base_dir, "ev_population.csv")
LOOKBACK = 12  # months

print("=" * 60)
print("  EV Registration Forecasting — Model Training Pipeline")
print("=" * 60)

df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.sort_values("Date").reset_index(drop=True)
values = df["Electric Vehicle (EV) Total"].values.astype(float).reshape(-1, 1)

print(f"\n[DATA] Dataset: {len(values)} monthly records")
print(f"   Range : {df['Date'].min().strftime('%b %Y')} -> {df['Date'].max().strftime('%b %Y')}")
print(f"   Min   : {int(values.min()):,}")
print(f"   Max   : {int(values.max()):,}")

# ──────────────────────────────────────────────────────────────
# 2. Scale
# ──────────────────────────────────────────────────────────────
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# ──────────────────────────────────────────────────────────────
# 3. Create Sequences
# ──────────────────────────────────────────────────────────────
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback : i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, LOOKBACK)
X = X.reshape(X.shape[0], X.shape[1], 1)  # (samples, timesteps, features)

# Train/Test split — last 20% for testing
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"\n[SPLIT] Train/Test Split:")
print(f"   Training : {len(X_train)} samples")
print(f"   Testing  : {len(X_test)} samples")
print(f"   Lookback : {LOOKBACK} months")

# ──────────────────────────────────────────────────────────────
# 4. Build LSTM Model
# ──────────────────────────────────────────────────────────────
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

tf.random.set_seed(42)
np.random.seed(42)

model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1),
])

model.compile(optimizer="adam", loss="mse")

print("\n[MODEL] Architecture:")
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
predictions = scaler.inverse_transform(predictions_scaled)
actuals = scaler.inverse_transform(y_test.reshape(-1, 1))

# MAPE
mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
# RMSE
rmse = np.sqrt(np.mean((actuals - predictions) ** 2))

print("\n" + "=" * 60)
print("  [RESULTS] Evaluation Results")
print("=" * 60)
print(f"   MAPE : {mape:.2f}%")
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

metrics = {"mape": round(mape, 2), "rmse": round(rmse, 2)}
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\n[SAVE] Saved Artifacts:")
print(f"   Model   -> {model_path}")
print(f"   Scaler  -> {scaler_path}")
print(f"   Metrics -> {metrics_path}")
print("\n[DONE] Training complete. Run `streamlit run app.py` to launch the dashboard.")
