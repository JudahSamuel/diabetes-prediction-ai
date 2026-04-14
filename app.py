import streamlit as st
from src.predict import predict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Page config
st.set_page_config(page_title="Diabetes Predictor", layout="centered")

# Title
st.title("🧠 AI-Based Diabetes Prediction System")
st.markdown("### Enter patient health details")

# Sidebar inputs
st.sidebar.header("Patient Details")

age = st.sidebar.number_input("Age", 1, 120, 25)
bmi = st.sidebar.number_input("BMI", 10.0, 50.0, 22.0)

phys = st.sidebar.selectbox(
    "Physical Activity",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

bp = st.sidebar.selectbox(
    "High Blood Pressure",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

chol = st.sidebar.selectbox(
    "High Cholesterol",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

glucose = st.sidebar.selectbox(
    "Regular Cholesterol Check (Glucose Proxy)",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

# Load dataset once
df = pd.read_csv("data/diabetes_binary_health.csv")

# -------------------------------
# 🔮 PREDICTION
# -------------------------------
if st.sidebar.button("Predict"):

    data = [age, bmi, phys, bp, chol, glucose]

    result, prob = predict(data)

    st.subheader("Prediction Result")

    if result == 1:
        st.error(f"🔴 High Risk of Diabetes ({prob*100:.2f}%)")
        st.write("⚠️ Suggestion: Exercise regularly, monitor BP & sugar levels")
    else:
        st.success(f"🟢 Low Risk ({prob*100:.2f}%)")
        st.write("✅ Maintain healthy lifestyle")

    st.subheader("Risk Level")

    if prob > 0.7:
        st.warning("HIGH RISK")
    elif prob > 0.4:
        st.info("MEDIUM RISK")
    else:
        st.success("LOW RISK")

    # -------------------------------
    # 📊 USER INPUT VISUALIZATION
    # -------------------------------
    st.subheader("📊 Your Input Summary")

    user_data = {
        "Age": age,
        "BMI": bmi,
        "PhysActivity": phys,
        "HighBP": bp,
        "HighChol": chol,
        "CholCheck": glucose
    }

    st.bar_chart(user_data)

    # -------------------------------
    # 📊 COMPARISON WITH DATASET
    # -------------------------------
    st.subheader("📊 Comparison with Dataset Average")

    avg_values = df[[
        "Age", "BMI", "PhysActivity",
        "HighBP", "HighChol", "CholCheck"
    ]].mean()

    comparison_df = pd.DataFrame({
        "Your Values": list(user_data.values()),
        "Average": avg_values
    }, index=user_data.keys())

    st.bar_chart(comparison_df)

# -------------------------------
# 📊 DATASET VISUALIZATION
# -------------------------------
st.subheader("📊 Dataset Correlation Heatmap")

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), ax=ax)
st.pyplot(fig)

# -------------------------------
# 🔥 FEATURE IMPORTANCE
# -------------------------------
st.subheader("🔥 Feature Importance")

model = joblib.load("model/model.pkl")

features = ["Age", "BMI", "PhysActivity", "HighBP", "HighChol", "CholCheck"]

importance = model.feature_importances_

fig2, ax2 = plt.subplots()
ax2.barh(features, importance)
ax2.set_xlabel("Importance")
ax2.set_title("Feature Importance")
st.pyplot(fig2)