import pandas as pd
from sklearn.preprocessing import LabelEncoder

def clean_data():
    path = path = r"D:\Machine Learning\Big_Mart_Sales_Prediction\data\train_v9rqX0R.csv"

    save_path = "D:/Machine Learning/Big_Mart_Sales_Prediction/Data/Cleaned_Train.csv"


    data = pd.read_csv(path)

    # Missing values
    data['Item_Weight'].fillna(data['Item_Weight'].mean(), inplace=True)

    mode_outlet_size = data.pivot_table(values='Outlet_Size',
                                        columns='Outlet_Type',
                                        aggfunc=lambda x: x.mode()[0])

    missing = data['Outlet_Size'].isnull()
    data.loc[missing, 'Outlet_Size'] = data.loc[missing, 'Outlet_Type'].apply(lambda x: mode_outlet_size[x])

    # Fat content cleanup
    data.replace({
        'Item_Fat_Content': {
            'low fat': 'Low Fat',
            'LF': 'Low Fat',
            'reg': 'Regular'
        }
    }, inplace=True)

    # Encoding
    encoder = LabelEncoder()
    categorical = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size',
                   'Outlet_Location_Type', 'Outlet_Type']

    for col in categorical:
        data[col] = encoder.fit_transform(data[col].astype(str))

    # Drop IDs
    data.drop(columns=['Item_Identifier', 'Outlet_Identifier'], inplace=True)

    # Save cleaned data
    data.to_csv(save_path, index=False)
    print("✅ Cleaned data saved to:", save_path)


# Run automatically
if __name__ == "__main__":
    clean_data()
