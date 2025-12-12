import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Paths
MODEL_PATH = r"D:\DeepLearning🤖\Stock-Price-Prediction-Fnn\models\fnn_stock_model.keras"
SCALER_X_PATH = r"D:\DeepLearning🤖\Stock-Price-Prediction-Fnn\scaler_X.pkl"
SCALER_Y_PATH = r"D:\DeepLearning🤖\Stock-Price-Prediction-Fnn\scaler_y.pkl"
DATA_PATH = r"D:\DeepLearning🤖\Stock-Price-Prediction-Fnn\merged_data\merged_stock_data.csv"

# Load model & scalers
model = tf.keras.models.load_model(MODEL_PATH)
scaler_X = joblib.load(SCALER_X_PATH)
scaler_y = joblib.load(SCALER_Y_PATH)

# Load data
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

FEATURES = ["open", "high", "low", "close", "volume"]
TARGET = ["close"]

# Prepare features and target
X = scaler_X.transform(df[FEATURES])
y_true = scaler_y.transform(df[TARGET])  # scaled

# Split test set (last 20%)
split = int(len(X) * 0.8)
X_test = X[split:]
y_test_scaled = y_true[split:]

# Predict
y_pred_scaled = model.predict(X_test)

# Inverse transform to original scale
y_test = scaler_y.inverse_transform(y_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n📌 Model Evaluation on Test Set:")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# Plot actual vs predicted
plt.figure(figsize=(12,6))
plt.plot(df['Date'][split:], y_test, label="Actual Close")
plt.plot(df['Date'][split:], y_pred, label="Predicted Close")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title("Actual vs Predicted Close Price")
plt.legend()
plt.show()
