# test.py (updated)
import joblib
import pandas as pd
import numpy as np

MODEL_PATH = r"D:\Machine Learning(ML)\Questions\predicting House Linear\src\house_price_model.pkl"
pipeline = joblib.load(MODEL_PATH)
print("Model pipeline loaded.")

# If you removed YearBuilt and replaced with Age at preprocessing, ask Age; else ask YearBuilt and compute Age
while True:
    try:
        Area = float(input("Area (in sq ft): "))
        Bedrooms = int(input("Bedrooms: "))
        Bathrooms = int(input("Bathrooms: "))
        Floors = int(input("Floors: "))
        # compute Age from YearBuilt or accept Age:
        year_input = input("Enter Year Built (or type 'age' to enter Age directly): ").strip()
        if year_input.lower() == 'age':
            Age = int(input("Enter Age (years): "))
        else:
            YearBuilt = int(year_input)
            Age = 2025 - YearBuilt

        Location = input("Location (e.g., Downtown): ").strip()
        Condition = input("Condition (e.g., Excellent): ").strip()
        Garage = input("Garage (Yes/No): ").strip()

        df = pd.DataFrame([[Area, Bedrooms, Bathrooms, Floors, Age, Location, Condition, Garage]],
                          columns=['Area','Bedrooms','Bathrooms','Floors','Age','Location','Condition','Garage'])

        # pipeline predicts log(price), so returns log; but our pipeline here was trained on log target, yes
        y_pred_log = pipeline.predict(df)[0]
        y_pred = np.expm1(y_pred_log)
        print(f"\n💰 Predicted House Price: ₹{y_pred:,.2f}")

    except Exception as e:
        print("Error:", e)

    if input("\nPredict another? (yes/no): ").lower() != 'yes':
        break
