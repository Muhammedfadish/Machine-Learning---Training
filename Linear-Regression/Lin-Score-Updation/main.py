# main.py

from data_preprocessing import load_and_preprocess_data
from train_model import train_and_save_model
from predict import make_prediction
import os  # ✅ Import here at the top

def main():
    # ✅ Step 0: Create the 'models' directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
        print("📁 Created 'models' directory.")
    else:
        print("📁 'models' directory already exists.")

    # ✅ Step 1: Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data('score_updated.csv')

    # ✅ Step 2: Train model and save it in the 'models' folder
    model = train_and_save_model(X_train, y_train)

    # ✅ Step 3: Make a new prediction
    make_prediction(18)  # Predict marks for 10 hours of study

if __name__ == "__main__":
    main()
