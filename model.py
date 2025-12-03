# GTFS Mobility Demand Forecasting Using LSTM
# -------------------------------------------
# Author: Jeswitha Shivani
# Description:
# This script loads GTFS stop_times data, extracts hourly mobility patterns,
# builds a time-series dataset, trains an LSTM model, and predicts hourly demand.

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# ----------------------------
# 1. Load GTFS Stop Times Data
# ----------------------------
stop_times = pd.read_csv("stop_times.txt")

# ----------------------------
# 2. Clean Time Format
# ----------------------------
# Keep only valid HH:MM:SS rows
stop_times = stop_times[stop_times["arrival_time"].str.match(r'^\d{1,2}:\d{2}:\d{2}$')]

# ----------------------------
# 3. Extract Hour
# ----------------------------
stop_times["hour"] = stop_times["arrival_time"].str.split(":").str[0].astype(int)

# Normalize extended GTFS hours (24–31 → 0–7)
stop_times["hour"] = stop_times["hour"] % 24

# ----------------------------
# 4. Compute Hourly Mobility Demand
# ----------------------------
hourly_counts = (
    stop_times.groupby("hour")
    .size()
    .reset_index(name="total_trips")
    .sort_values("hour")
)

# Extract single-day pattern
one_day = hourly_counts["total_trips"].values

# Repeat for 90 days to create time-series for LSTM
series = np.tile(one_day, 90)

# ----------------------------
# 5. Normalize Data
# ----------------------------
scaler = MinMaxScaler()
scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

# ----------------------------
# 6. Create LSTM Training Windows
# ----------------------------
seq_len = 24  # use 24 hours to predict next hour
X, y = [], []

for i in range(len(scaled) - seq_len):
    X.append(scaled[i:i + seq_len])
    y.append(scaled[i + seq_len])

X = np.array(X).reshape(-1, seq_len, 1)
y = np.array(y)

# ----------------------------
# 7. Build LSTM Model
# ----------------------------
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(seq_len, 1)),
    LSTM(32),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.summary()

# ----------------------------
# 8. Train the Model
# ----------------------------
history = model.fit(
    X, y,
    epochs=20,
    batch_size=32,
    validation_split=0.1
)

# ----------------------------
# 9. Predict Next Hour
# ----------------------------
last_sequence = scaled[-seq_len:].reshape(1, seq_len, 1)
pred_scaled = model.predict(last_sequence)
predicted_value = scaler.inverse_transform(pred_scaled)[0][0]

print("Predicted next hour mobility:", predicted_value)

# ----------------------------
# 10. Plot Actual vs Predicted
# ----------------------------
y_pred = model.predict(X)
y_pred = scaler.inverse_transform(y_pred)
actual = scaler.inverse_transform(y.reshape(-1, 1))

plt.figure(figsize=(12, 5))
plt.plot(actual[:200], label="Actual")
plt.plot(y_pred[:200], label="Predicted")
plt.xlabel("Time (Hours)")
plt.ylabel("Mobility Demand (Total Trips)")
plt.title("LSTM Mobility Forecasting")
plt.legend()
plt.savefig("actual_vs_predicted.png")
plt.show()
