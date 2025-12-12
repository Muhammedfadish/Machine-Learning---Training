import tensorflow as tf
import joblib
import pandas as pd
import numpy as np

# Path to model
MODEL_PATH = r"D:\DeepLearning🤖\Stock-Price-Prediction-Fnn\models\fnn_stock_model.keras"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load scalers
scaler_X = joblib.load(r"D:\DeepLearning🤖\Stock-Price-Prediction-Fnn\scaler_X.pkl")
scaler_y = joblib.load(r"D:\DeepLearning🤖\Stock-Price-Prediction-Fnn\scaler_y.pkl")

# Load dataset
DATA_PATH = r"D:\DeepLearning🤖\Stock-Price-Prediction-Fnn\merged_data\merged_stock_data.csv"
df = pd.read_csv(DATA_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

FEATURES = ["open", "high", "low", "close", "volume"]

# Last row for prediction
latest_row = df.iloc[-1]
raw_input = np.array([latest_row[f] for f in FEATURES]).reshape(1, -1)
X_scaled = scaler_X.transform(raw_input)

# Predict
y_scaled_pred = model.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_scaled_pred)

print("\n📌 Last Date in data:", latest_row["Date"].date())
print("💰 Predicted next-day close price:", round(float(y_pred[0][0]), 2))
