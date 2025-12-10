# src/predict_cli.py

import json
import numpy as np
import pandas as pd
from joblib import load
from tensorflow.keras.models import load_model


# Load trained components
scaler = load("models/heart_scaler.joblib")
model = load_model("models/heart_fnn_model.h5")

with open("models/heart_class_mapping.json", "r") as f:
    class_map = json.load(f)

# Feature order MUST match training
feature_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]


def get_float_input(prompt):
    while True:
        try:
            return float(input(prompt))
        except:
            print("Please enter a valid number.")


def main():
    print("\n=== Heart Disease Severity Prediction ===\n")

    patient = {}
    for feature in feature_names:
        value = get_float_input(f"Enter value for {feature}: ")
        patient[feature] = value

    # Convert to model input
    X = np.array([patient[f] for f in feature_names]).reshape(1, -1)
    X_scaled = scaler.transform(X)

    # Predict
    preds = model.predict(X_scaled)
    class_id = int(np.argmax(preds))
    severity = class_map[str(class_id)] if str(class_id) in class_map else class_map[class_id]

    print("\nPredicted Severity:", severity)
    print("Probabilities:", preds[0])


if __name__ == "__main__":
    main()
