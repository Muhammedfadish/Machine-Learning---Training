# ===============================================
# 🧠 TEST LOAN ELIGIBILITY MODEL
# ===============================================

import joblib
import numpy as np

# STEP 1: Load the trained model
model_path = r"D:\Machine Learning(ML)\Questions\Loan Eligibility Prediction\loan_eligibility_model.pkl"
model = joblib.load(model_path)
print("✅ Model loaded successfully!\n")

# STEP 2: Get input from the user
print("Please enter the following details:\n")
credit_score = int(input("Credit Score: "))
income = int(input("Annual Income: "))
employment_status = int(input("Employment Status (0 = Not Self-Employed, 1 = Self-Employed): "))
assets = int(input("Total Asset Value: "))
loan_amount = int(input("Loan Amount: "))
loan_term = int(input("Loan Term (in months): "))

# STEP 3: Prepare input data
input_data = np.array([[credit_score, income, employment_status, assets, loan_amount, loan_term]])

# STEP 4: Make prediction
prediction = model.predict(input_data)

# STEP 5: Display result
if prediction[0] == 1:
    print("\n✅ The customer is **ELIGIBLE** for the loan.")
else:
    print("\n❌ The customer is **NOT ELIGIBLE** for the loan.")
