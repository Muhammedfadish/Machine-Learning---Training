# main.py
from data_preprocessing import load_and_preprocess_data
from train_model import train_and_save_model
from predict import make_prediction
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree
import matplotlib.pyplot as plt

def main():
    # Step 1: Load and preprocess data
    X_train, X_test, y_train, y_test, le = load_and_preprocess_data()

    # Step 2: Train model
    model = train_and_save_model(X_train, y_train)

    # Step 3: Test model
    y_pred = model.predict(X_test)
    print("\n📊 Model Evaluation Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

    # Step 4: Visualize Decision Tree
    plt.figure(figsize=(15, 10))
    tree.plot_tree(model, feature_names=X_train.columns, class_names=le.classes_, filled=True, rounded=True)
    plt.title("🌳 Decision Tree Visualization")
    plt.show()

    # Step 5: Predict new data
    new_flower = [[4.3, 5.8, 2.0, 0.6]]
    make_prediction(new_flower, label_encoder=le)

if __name__ == "__main__":
    main()
