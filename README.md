# gtfs-lstm-mobility-forecasting
Deep learning model that forecasts hourly mobility demand using GTFS stop_times and LSTM networks.
1. Project Motive
 This project aims to forecast future public transport mobility demand using AI.
 Understanding hourly mobility behavior helps optimize transport planning, fleet management, EV
 charging, and traffic engineering.
 LSTM neural networks are ideal for modeling sequential temporal patterns such as daily mobility
 cycles.
 2. Why GTFS Data?
 GTFS (General Transit Feed Specification) is the universal public transport standard used by Google
 Maps, transit agencies, and mobility companies.
 stop_times.txt contains detailed arrival and departure times, enabling extraction of hourly demand
 patterns.
 3. Data Processing Steps- Loaded stop_times.txt into pandas.- Cleaned arrival_time using regex to keep valid HH:MM:SS times.- Extracted 'hour' from arrival_time.- Normalized extended GTFS hours (24–31) to 0–23 using modulo 24.- Computed the number of trips per hour to form a daily mobility pattern.- Repeated the 24-hour pattern across multiple days to generate a long time series for LSTM training.
 4. LSTM Preparation- Scaled dataset to 0–1 using MinMaxScaler.- Created 24-hour input windows to predict the next hour.- Reshaped data into (samples, timesteps, features) structure required by LSTM.
 5. LSTM Model Architecture- Two stacked LSTM layers with 64 and 32 units.
- Dense output layer for numerical prediction.- Loss function: Mean Squared Error (MSE).- Optimizer: Adam.
 6. Model Training
 The model was trained for 20 epochs with a batch size of 32.
 Validation split of 10% ensured monitoring of overfitting.
 7. Forecasting
 Using the last 24 hours of data, the model predicts the next hour’s mobility.
 Predictions are inverse-transformed back into original scale.
 8. Visualization
 A plot of “Actual vs Predicted” mobility shows near-perfect overlap, proving the LSTM accurately
 learned daily mobility cycles.
 Conclusion
 This project demonstrates the full pipeline for mobility demand forecasting using GTFS:
 data cleaning, feature engineering, time-series modeling, LSTM training, evaluation,
 and visualization. It is suitable for automotive AI roles, mobility engineering.
