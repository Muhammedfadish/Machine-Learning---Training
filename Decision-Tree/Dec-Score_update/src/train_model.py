# train_model.py
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

def train_and_save_model(X_train, y_train, save_path='models/decision_tree_model.pkl'):
    # Create 'models' directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
        print("📁 Created 'models' directory.")
    else:
        print("📁 'models' directory already exists.")

    # Initialize and train model
    model = DecisionTreeClassifier(criterion='entropy', random_state=42)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, save_path)
    print(f"✅ Model trained and saved at {save_path}")

    return model
