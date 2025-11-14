# train_model.py
from sklearn.linear_model import LinearRegression
import joblib

def train_and_save_model(X_train, y_train, save_path='models/linear_regression_model.pkl'):
    # Initialize model
    model = LinearRegression()

    # Train model
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, save_path)

    print(f"✅ Model trained and saved at {save_path}")
    return model