import os
import shutil

# =============== CONFIGURATION ===============
# Path to your main dataset folder
SOURCE_DIR = r"D:\Machine Learning(ML)\Image_Recognising\image_classification_knn\dataset\Test"  # <-- change this to your dataset folder
# Path to save merged dataset
DEST_DIR = r"D:\Machine Learning(ML)\Image_Recognising\image_classification_knn\dataset\Testing"         # <-- change this if you want

# Fruits you want to keep
TARGET_FRUITS = ["apple", "banana", "orange"]
# =============================================

# Create destination folders if not exist
for fruit in TARGET_FRUITS:
    os.makedirs(os.path.join(DEST_DIR, fruit), exist_ok=True)

# Walk through dataset
for root, dirs, files in os.walk(SOURCE_DIR):
    folder_name = os.path.basename(root).lower()  # current folder name in lowercase

    # Check if this folder name belongs to any target fruit
    for fruit in TARGET_FRUITS:
        if fruit in folder_name:  # match partial names like 'apple_red', 'banana_ripen'
            print(f"📂 Found {fruit} folder: {root}")

            # Copy each image to the destination fruit folder
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    src_path = os.path.join(root, file)
                    dest_path = os.path.join(DEST_DIR, fruit, file)

                    # Avoid overwriting files with same name
                    if os.path.exists(dest_path):
                        name, ext = os.path.splitext(file)
                        dest_path = os.path.join(DEST_DIR, fruit, f"{name}_copy{ext}")

                    shutil.copy2(src_path, dest_path)

print("\n✅ All selected fruits merged successfully!")
print(f"Merged dataset saved at: {DEST_DIR}")
