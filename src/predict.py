import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "model", "model.pkl")
scaler_path = os.path.join(BASE_DIR, "model", "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

def predict(data):
    data = np.array(data).reshape(1, -1)
    data = scaler.transform(data)

    prediction = model.predict(data)
    probability = model.predict_proba(data)

    return prediction[0], probability[0][1]