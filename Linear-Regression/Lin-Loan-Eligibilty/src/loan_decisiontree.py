# ===============================================
# 🧠 LOAN ELIGIBILITY PREDICTION - DECISION TREE
# ===============================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree
import matplotlib.pyplot as plt
import joblib

# STEP 1: Load cleaned dataset
data = pd.read_csv(r"D:\Machine Learning(ML)\Questions\Loan Eligibility Prediction\data\cleaned_loan_data.csv")

# STEP 2: Encode categorical column
label_encoder = LabelEncoder()
data['Employment_Status'] = label_encoder.fit_transform(data['Employment_Status'])

# STEP 3: Check data
print("✅ Data loaded successfully!")
print(data.head())

# ✅ STEP 4: Add Target Column (Loan_Status)
# You MUST have a target column. If not, create temporarily for testing.
# Replace this dummy column with your actual 'loan_status' column later.
import numpy as np
data['Loan_Status'] = np.random.choice([0, 1], size=len(data))  # 0 = Not Eligible, 1 = Eligible

# ✅ Print updated dataset
print("\n📊 Data after adding Loan_Status:")
print(data.head())

# STEP 5: Split into X (features) and y (target)
X = data[['Credit_Score', 'Income', 'Employment_Status', 'Assets', 'Loan_Amount', 'Loan_Term']]
y = data['Loan_Status']

print("\n🎯 Target Column (y):")
print(y.head())

print(data.columns)
# STEP 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# STEP 7: Train Decision Tree
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, y_train)

# STEP 8: Predictions
y_pred = model.predict(X_test)

# STEP 9: Evaluate Performance
print("\n✅ Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# STEP 10: Visualize Tree
plt.figure(figsize=(12,6))
tree.plot_tree(model, feature_names=X.columns, class_names=['Not Eligible', 'Eligible'], filled=True)
plt.show()

# STEP 11: Save Model
joblib.dump(model, "loan_eligibility_model.pkl")
print("\n💾 Model saved as 'loan_eligibility_model.pkl'")


