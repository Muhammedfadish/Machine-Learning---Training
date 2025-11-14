# data_preprocessing.py
import pandas as pd
import os

raw_path = r"D:\Machine Learning(ML)\Questions\predicting House Linear\data\House Price Prediction Dataset.csv"
cleaned_path = r"D:\Machine Learning(ML)\Questions\predicting House Linear\data\cleaned_House_price.csv"
df = pd.read_csv(raw_path)

# drop Id if present
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

# drop missing rows (or handle them)
df = df.dropna()

# create Age feature
CURRENT_YEAR = 2025
df['Age'] = CURRENT_YEAR - df['YearBuilt']
# you may drop YearBuilt if you want
df = df.drop(columns=['YearBuilt'])

# Save cleaned
os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
df.to_csv(cleaned_path, index=False)
print("Saved cleaned data with Age ->", cleaned_path)
