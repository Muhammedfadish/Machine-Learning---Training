import pandas as pd

# Correct file path: use double \\ or a raw string
df = pd.read_csv(r"D:\Machine Learning(ML)\Questions\Loan Eligibility Prediction\data\loan_approval_dataset (1).csv")

# 🧽 Step 1: Remove spaces from column names
df.columns = df.columns.str.strip()

# Step 2: Show cleaned column names
print(df.columns)


# Now, you’ll select only the columns that match your project goal.
# Keep only the important columns
df = df[['cibil_score', 'income_annum', 'self_employed',
         'residential_assets_value', 'loan_amount', 'loan_term']]

df.rename(columns={
    'cibil_score': 'Credit_Score',
    'income_annum': 'Income',
    'self_employed': 'Employment_Status',
    'residential_assets_value': 'Assets',
    'loan_amount': 'Loan_Amount',
    'loan_term': 'Loan_Term'
}, inplace=True)

# Save the cleaned dataset
df.to_csv("cleaned_loan_data.csv", index=False)
print("✅ Cleaned dataset saved successfully!")

