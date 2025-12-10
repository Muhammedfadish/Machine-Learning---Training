# src/train_fnn.py

import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from joblib import dump
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# -------------------------------------
# 1. Load processed heart.csv
# -------------------------------------
df = pd.read_csv("data\heart.csv")
print("Dataset Shape:", df.shape)
print(df.head())


# -------------------------------------
# 2. Separate Features & Target
# -------------------------------------
X = df.drop(columns=["num"])
y = df["num"].astype(int)

num_classes = len(y.unique())
print("\nClasses:", sorted(y.unique()))


# -------------------------------------
# 3. Train-Test Split
# -------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# -------------------------------------
# 4. Feature Scaling (Standardization)
# -------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

os.makedirs("models", exist_ok=True)
dump(scaler, "models/heart_scaler.joblib")

print("\nScaler saved.")


# -------------------------------------
# 5. Handle Class Imbalance
# -------------------------------------
classes = np.unique(y_train)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}
print("\nClass Weights:", class_weight_dict)


# -------------------------------------
# 6. Build FNN Model (Multi-Class)
# -------------------------------------
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),

    Dense(32, activation="relu"),
    Dropout(0.3),

    Dense(num_classes, activation="softmax")  # softmax for multiclass
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# -------------------------------------
# 7. Train Model with EarlyStopping
# -------------------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=16,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=1
)


# -------------------------------------
# 8. Evaluate Model
# -------------------------------------
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy:.4f}")


# -------------------------------------
# 9. Save Model + Class Mapping
# -------------------------------------
model.save("models/heart_fnn_model.h5")

class_desc = {
    0: "No heart disease",
    1: "Mild heart disease",
    2: "Moderate heart disease",
    3: "Severe heart disease",
    4: "Very severe heart disease"
}

with open("models/heart_class_mapping.json", "w") as f:
    json.dump(class_desc, f, indent=4)

print("\nModel and class mapping saved successfully.")
