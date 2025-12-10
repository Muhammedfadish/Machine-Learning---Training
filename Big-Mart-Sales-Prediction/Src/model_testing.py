import numpy as np
import pickle

MODEL_PATH = r"D:\Machine Learning\Big_Mart_Sales_Prediction\Models\model.pkl"

# Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("\n🚀 BigMart Sales Predictor Ready!")
print("----------------------------------")

# ----- USER INPUT -----

item_weight = float(input("Item Weight: "))
item_fat = int(input("Item Fat Content (0=Low Fat, 1=Regular): "))
item_visibility = float(input("Item Visibility: "))
item_type = int(input("Item Type Code (0–15): "))
item_mrp = float(input("Item MRP: "))
outlet_year = int(input("Outlet Establishment Year: "))
outlet_size = int(input("Outlet Size (0=Small,1=Medium,2=High): "))
outlet_location = int(input("Outlet Location (0=Tier1,1=Tier2,2=Tier3): "))
outlet_type = int(input("Outlet Type (0-3): "))

# Create array
input_data = np.array([
    item_weight,
    item_fat,
    item_visibility,
    item_type,
    item_mrp,
    outlet_year,
    outlet_size,
    outlet_location,
    outlet_type
]).reshape(1, -1)

# Predict
prediction = model.predict(input_data)[0]

print("\n📊 Predicted Sales:", round(prediction, 2))
print("----------------------------------")
