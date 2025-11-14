import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# =========================
# STEP 1: Load Dataset
# =========================
data_dir = "dataset/Training"
data = []
labels = []

print("Loading images from dataset...")

for label in os.listdir(data_dir):
    folder = os.path.join(data_dir, label)
    if not os.path.isdir(folder):
        continue
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (64, 64))   # Resize all to 64x64
        data.append(img.flatten())        # Flatten image to 1D array
        labels.append(label)

print("Total images loaded:", len(data))

# Convert to arrays
X = np.array(data)
y = np.array(labels)

# =========================
# STEP 2: Encode Labels
# =========================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# =========================
# STEP 3: Scale Features
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# STEP 4: Split Train/Test
# =========================
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# =========================
# STEP 5: Train KNN Model
# =========================
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print("Model training completed!")

# =========================
# STEP 6: Evaluate Model
# =========================
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(acc * 100, 2), "%")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.title("Confusion Matrix - Fruit Classification")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("visuals/confusion_matrix.png")
plt.show()

# =========================
# STEP 7: Save Model & Scaler
# =========================
os.makedirs("models", exist_ok=True)
joblib.dump(knn, "models/knn_fruit.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(encoder, "models/label_encoder.pkl")

print("Model, Scaler, and Encoder saved successfully!")
