import pandas as pd
import os

folder_path = r"D:\DeepLearning🤖\Stock-Price-Prediction-Fnn\data"

df_list = []

for file in os.listdir(folder_path):
    if file.endswith('.csv'):
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)

        # Convert column names to lowercase
        df.columns = df.columns.str.lower().str.strip()

        # Rename 'date' to 'Date'
        if 'date' in df.columns:
            df.rename(columns={'date': 'Date'}, inplace=True)
        else:
            print("❌ No 'date' column in:", file)
            continue

        df_list.append(df)
        print("✔ Added:", file)

# Combine all dataframes
merged_df = pd.concat(df_list, ignore_index=True)

# Convert Date to datetime
merged_df['Date'] = pd.to_datetime(merged_df['Date'])

# Sort by Date
merged_df = merged_df.sort_values(by='Date')

# Save final merged file
merged_df.to_csv("merged_stock_data.csv", index=False)

print("\n✅ All files merged successfully!")
