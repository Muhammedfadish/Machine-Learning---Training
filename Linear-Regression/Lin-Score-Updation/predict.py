import joblib
import pandas as pd

def make_prediction(hours, model_path='models/linear_regression_model.pkl'):
    model = joblib.load(model_path)
    input_data = pd.DataFrame({'Hours': [hours]})
    prediction = model.predict(input_data)[0]
    print(f"🎯 Predicted marks for {hours} hours of study: {prediction:.2f}")
    return prediction
