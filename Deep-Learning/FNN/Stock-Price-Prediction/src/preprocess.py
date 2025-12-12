import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(csv_path=r"D:\DeepLearning🤖\Stock-Price-Prediction-Fnn\merged_data\merged_stock_data.csv"):
    df = pd.read_csv(csv_path)

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # Use lowercase if CSV contains lowercase names
    features = ["open", "high", "low", "close", "volume"]
    target = ["close"]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X = scaler_X.fit_transform(df[features])
    y = scaler_y.fit_transform(df[target])

    print("✅ Preprocessing completed!")
    return X, y, scaler_X, scaler_y


if __name__ == "__main__":
    X, y, scaler_X, scaler_y = preprocess_data()
    print("X shape:", X.shape)
    print("y shape:", y.shape)
