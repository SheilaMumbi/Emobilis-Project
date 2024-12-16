import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model and scaler
model = joblib.load("/content/logreg_chd_model.pkl")
scaler = joblib.load("/content/scaler.pkl")

# Streamlit app
st.title("Cardio Guard")
st.write("This app predicts the likelihood of developing coronary heart disease based on user input.")
st.image("/content/How-Does-the-Cardiovascular-System-Work.webp")
# User inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
sex_male = st.selectbox("Sex", options=["Female", "Male"])
cigs_per_day = st.number_input("Cigarettes Per Day", min_value=0, max_value=100, value=0, step=1)
tot_chol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=400, value=200, step=1)
sys_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80, max_value=200, value=120, step=1)
glucose = st.number_input("Glucose Level (mg/dL)", min_value=50, max_value=300, value=100, step=1)

# Process the inputs
sex_male = 1 if sex_male == "Male" else 0

# Combine inputs into an array
user_data = np.array([[age, sex_male, cigs_per_day, tot_chol, sys_bp, glucose]])

# Scale the input using the saved scaler
scaled_data = scaler.transform(user_data)

# Prediction
if st.button("Predict"):
    prediction = model.predict(scaled_data)
    prediction_prob = model.predict_proba(scaled_data)[:, 1]
    custom_threshold = 0.3
    if model.predict_proba(scaled_data)[0][1] > custom_threshold:
        st.error(f"High Risk of CHD! Probability: {prediction_prob[0]:.2f}")
    else:
        st.success(f"Low Risk of CHD. Probability: {prediction_prob[0]:.2f}")


st.write("### Summary of Your Inputs")
st.write(f"**Age**: {age}, **Sex**: {'Male' if sex_male else 'Female'}, **Cigs/Day**: {cigs_per_day}")
st.write(f"**Cholesterol**: {tot_chol}, **BP**: {sys_bp}, **Glucose**: {glucose}")

st.write("### Disclaimer")
st.write("This prediction is based on statistical modeling and should not be used as a substitute for professional medical advice.")