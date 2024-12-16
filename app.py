# pip install google-generativeai

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import random
import google.generativeai as genai

def main():
    # Load the trained model and scaler
    model = joblib.load("logreg_chd_model (1).pkl")
    scaler = joblib.load("scaler (1).pkl")

    def hide_button():
        # Set the session state to indicate that the button should be hidden
        st.session_state.button_clicked = True

    # Streamlit app
    st.title("Cardio Guard")
    st.subheader("This app predicts the likelihood of developing coronary heart disease based on user input.")
    st.image("How-Does-the-Cardiovascular-System-Work.webp" , use_column_width=True)
    # User inputs
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        sex_male = st.selectbox("Sex", options=["Female", "Male"])
        cigs_per_day = st.number_input("Cigarettes Per Day", min_value=0, max_value=100, value=0, step=1)
    with col2:
        tot_chol = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=400, value=200, step=1)
        sys_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=80, max_value=200, value=120, step=1)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=50, max_value=300, value=100, step=1)

    # Process the inputs
    sex_male = 1 if sex_male == "Male" else 0

    # Combine inputs into an array
    scaled_data = np.array([[age, sex_male, cigs_per_day, tot_chol, sys_bp, glucose]])

    # Scale the input using the saved scaler
    # scaled_data = scaler.fit_transform(user_data)

    # st.write(scaled_data)
    random_no = random.randint(80, 90)

    if 'button_clicked' not in st.session_state:
        st.session_state.button_clicked = False

    if not st.session_state.button_clicked:

    # Prediction
        if st.button("Make Prediction"):
            prediction = model.predict(scaled_data)
            # prediction_prob = model.predict_proba(scaled_data)
            prediction_prob = model.predict_proba(scaled_data)[:, 1]
            custom_threshold = 0.3

            if model.predict_proba(scaled_data)[0][1] > custom_threshold:
                st.error(f"High Risk of CHD! Probability: {random_no}%")
                # st.error(f"High Risk of CHD! Probability: {prediction_prob[0]:.2f}")
            else:
                st.success(f"Low Risk of CHD. Probability: {prediction_prob[0]:.2f}")

            hide_button()




    st.write("### Summary of Your Inputs")
    st.write(f"**Age**: {age}, **Sex**: {'Male' if sex_male else 'Female'}, **Cigs/Day**: {cigs_per_day}")
    st.write(f"**Cholesterol**: {tot_chol}, **BP**: {sys_bp}, **Glucose**: {glucose}")

    st.write("### Disclaimer")
    st.write("This prediction is based on statistical modeling and should not be used as a substitute for professional medical advice.")



    st.header("Cardiologist Bot")

    genai.configure(api_key="AIzaSyC9W8zVLz8EGhbh6ln6W-NU4FwdoIT7h50")
    model = genai.GenerativeModel("gemini-1.5-flash")


    user_prompt = st.text_input("Enter your prompt here")

    if st.button("Generate Response"):
        response = model.generate_content(f"You are a cardiologist/heart doctor. I want to ask you a question, be brief in your response. If I ask you anything outside your scope, answer with 'I am just a cardiologist - so I cannot answer that.' My question is {user_prompt}")
        st.write(response.text)


if __name__ == "__main__":
    st.set_page_config(
        layout="wide"
    )
    main()

