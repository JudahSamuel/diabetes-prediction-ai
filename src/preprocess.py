import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data():
    df = pd.read_csv("data/diabetes_binary_health.csv")
    return df

def preprocess(df):
    # Target (binary)
    y = df["Diabetes_012"].apply(lambda x: 1 if x >= 1 else 0)

    # Selected features (6 inputs)
    X = df[[
        "Age",
        "BMI",
        "PhysActivity",
        "HighBP",
        "HighChol",
        "CholCheck"
    ]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler