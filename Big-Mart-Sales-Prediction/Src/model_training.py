import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

DATA_PATH = r"D:\Machine Learning\Big_Mart_Sales_Prediction\Data\Cleaned_Train.csv"
MODEL_PATH = r"D:\Machine Learning\Big_Mart_Sales_Prediction\Models\model.pkl"

# Load cleaned data
data = pd.read_csv(DATA_PATH)

# Encode categorical columns
categorical_cols = [
    'Item_Fat_Content',
    'Item_Type',
    'Outlet_Size',
    'Outlet_Location_Type',
    'Outlet_Type'
]

encoder = LabelEncoder()

for col in categorical_cols:
    data[col] = encoder.fit_transform(data[col].astype(str))

# SPLIT X AND Y  
# (No dropping ID columns, because they do NOT exist)
X = data.drop("Item_Outlet_Sales", axis=1)
Y = data["Item_Outlet_Sales"]

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Train XGBoost model
model = XGBRegressor()
model.fit(X_train, Y_train)

# Evaluate
pred_test = model.predict(X_test)
r2 = r2_score(Y_test, pred_test)

print(f"📈 XGBoost R² Score: {r2}")

# Save model
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("🎉 Model saved at:", MODEL_PATH)
