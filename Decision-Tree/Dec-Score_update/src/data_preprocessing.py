# data_preprocessing.py
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data():
    # Load the dataset
    data = sns.load_dataset('iris')
    print("✅ Dataset Loaded Successfully!")
    print(data.head())

    # Encode target variable
    le = LabelEncoder()
    data['species'] = le.fit_transform(data['species'])

    # Split features and labels
    X = data.drop('species', axis=1)
    y = data['species']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n✅ Data Preprocessing Complete!")
    print(f"Training Samples: {len(X_train)}, Testing Samples: {len(X_test)}")

    return X_train, X_test, y_train, y_test, le
