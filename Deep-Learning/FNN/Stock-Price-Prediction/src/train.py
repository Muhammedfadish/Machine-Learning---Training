import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from preprocess import preprocess_data
import numpy as np
import joblib

def train_fnn():

    # Load processed data
    X, y, scaler_X, scaler_y = preprocess_data()

    # Split dataset (80% train, 20% test)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build FNN Model
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
        layers.Dropout(0.2),

        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),

        layers.Dense(32, activation='relu'),

        layers.Dense(1)  # predicting close price
    ])

    # Compile the model with proper metric objects
    model.compile(
    optimizer='adam',
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.MeanAbsoluteError()]  # use class instead of 'mse'
)

    # Early Stopping (Stop if model stops improving)
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,   
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # Old:
    # model.save("fnn_stock_model.h5")

   # Save model as .keras (Keras 3 compatible)
    model.save(r"D:\DeepLearning🤖\Stock-Price-Prediction-Fnn\models\fnn_stock_model.keras")

    # Save scalers using joblib
    joblib.dump(scaler_X, "scaler_X.pkl")
    joblib.dump(scaler_y, "scaler_y.pkl")

    print("\n🎉 Model trained and saved successfully!")
    print("📌 Best Epoch:", len(history.history['loss']))

if __name__ == "__main__":
    train_fnn()
