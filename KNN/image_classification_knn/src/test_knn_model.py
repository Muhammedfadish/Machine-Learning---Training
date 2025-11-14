import cv2
import numpy as np
import joblib

# Load model, scaler, and encoder
knn = joblib.load("models/knn_fruit.pkl")
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/label_encoder.pkl")

# Input image from user
img_path = input("Enter image path: ")
img = cv2.imread(img_path)

if img is None:
    print("Image not found!")
    exit()

# Preprocess image (same steps as training)
img = cv2.resize(img, (64, 64))
img_flat = img.flatten().reshape(1, -1)
img_scaled = scaler.transform(img_flat)

# Predict class
pred = knn.predict(img_scaled)[0]
pred_label = encoder.inverse_transform([pred])[0]

print(f"\n🧠 Predicted Class: {pred_label}")

# Display image
cv2.imshow("Input Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

