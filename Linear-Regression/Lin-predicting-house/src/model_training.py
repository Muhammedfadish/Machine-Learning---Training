# model_training.py
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Paths
DATA_PATH = r"D:\Machine Learning(ML)\Questions\predicting House Linear\data\cleaned_House_price.csv"
MODEL_PATH = r"D:\Machine Learning(ML)\Questions\predicting House Linear\src\house_price_model.pkl"

# Load data
df = pd.read_csv(DATA_PATH)
print("Loaded:", df.shape)
print(df.head())

# Features and target
# We created Age in preprocessing; if not present compute it here:
if 'Age' not in df.columns and 'YearBuilt' in df.columns:
    df['Age'] = 2025 - df['YearBuilt']
    df = df.drop(columns=['YearBuilt'])

feature_cols = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'Age', 'Location', 'Condition', 'Garage']
X = df[feature_cols]
y = df['Price']

# Optional: log-transform target to stabilize variance
y_log = np.log1p(y)   # use log(1 + price) to handle zeros

# Split
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

# Column types
numeric_features = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'Age']
categorical_features = ['Location', 'Condition', 'Garage']  # Garage is 'Yes'/'No' or 0/1, OneHot works either way

# ColumnTransformer: OneHot encode categoricals, scale numerics
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])


# Choose model: RidgeCV or RandomForest
# Using RidgeCV (regularized linear) on log-target:
ridge = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5)

pipeline = Pipeline(steps=[
    ('pre', preprocessor),
    ('model', ridge)
])

# Train
pipeline.fit(X_train, y_train_log)
print("Pipeline trained.")

# Evaluate on test set (remember to invert log)
y_pred_log = pipeline.predict(X_test)
y_pred = np.expm1(y_pred_log)  # invert log1p
y_test = np.expm1(y_test_log)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MSE: {mse:.2f}")
print(f"Test R2: {r2:.4f}")

# Cross-validation (on log target)
cv_scores = cross_val_score(pipeline, X, y_log, cv=5, scoring='r2')
print("5-fold CV R2 scores:", cv_scores)
print("CV R2 mean:", cv_scores.mean())

# Plot actual vs predicted (small plot)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted (test set)")
plt.show()

# Save pipeline
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)
print("Saved pipeline to", MODEL_PATH)
