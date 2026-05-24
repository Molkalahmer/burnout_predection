import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("burnout_model.pkl")
scaler = joblib.load("scaler.pkl")

# Title
st.title("Burnout Prediction")

# Inputs
age = st.number_input("Age", 18, 65)

experience_years = st.number_input("Experience Years", 0, 40)

work_hours_per_week = st.number_input("Work Hours Per Week", 0, 100)

stress_level = st.slider("Stress Level", 0, 10)

sleep_hours = st.slider("Sleep Hours", 0, 12)

work_life_balance = st.slider("Work Life Balance", 0, 10)

# Prediction button
if st.button("Predict"):

    # Create array
    data = np.array([[
        age,
        experience_years,
        work_hours_per_week,
        stress_level,
        sleep_hours,
        work_life_balance
    ]])

    # Scaling
    data_scaled = scaler.transform(data)

    # Prediction
    prediction = model.predict(data_scaled)

    # Display result
    st.success(f"Predicted Burnout Level: {prediction[0]}")
