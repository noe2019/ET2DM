import streamlit as st
import requests

# FastAPI backend URL
FASTAPI_URL = "http://localhost:8001/predict"  # Update if running FastAPI on a different host/port

st.title('Early Diabetes Prediction!')

# Create input fields
RIAGENDR = st.number_input("Sex", min_value=1, max_value=2, step=1)
RIDAGEYR = st.number_input("Age", min_value=21.0, max_value=120.0, step=0.1)
RACE = st.number_input("Race", min_value=1, max_value=4, step=1)
COUPLE = st.number_input("Marital Status", min_value=1, max_value=3, step=1)
SMOKER = st.number_input("Smoker", min_value=0, max_value=1, step=1)
EDUC = st.number_input("Education Level", min_value=1, max_value=3, step=1)
COVERED_INSURANCE = st.number_input("Covered Insurance", min_value=0, max_value=1, step=1)
FAT = st.number_input("Fat", min_value=1, max_value=3, step=1)
Abdobesity = st.number_input("Abdominal Obesity", min_value=0, max_value=1, step=1)
TOTAL_ACCULTURATION_SCORE = st.number_input("Total Acculturation Score", min_value=1, max_value=3, step=1)
POVERTIES = st.number_input("Poverty status", min_value=0, max_value=1, step=1)
HTN = st.number_input("Hypertension", min_value=0, max_value=1, step=1)

# Button to send data to FastAPI for prediction
if st.button("Predict"):
    # Create payload for FastAPI
    payload = {
        "RIAGENDR": RIAGENDR,
        "RIDAGEYR": RIDAGEYR,
        "RACE": RACE,
        "COUPLE": COUPLE,
        "SMOKER": SMOKER,
        "EDUC": EDUC,
        "COVERED_INSURANCE": COVERED_INSURANCE,
        "FAT": FAT,
        "Abdobesity": Abdobesity,
        "TOTAL_ACCULTURATION_SCORE": TOTAL_ACCULTURATION_SCORE,
        "POVERTIES": POVERTIES,
        "HTN": HTN
    }
    
    # Send POST request to FastAPI
    response = requests.post(FASTAPI_URL, json=payload)

    # Check response
    if response.status_code == 200:
        # Extract prediction result and display risk level
        predicted_class = response.json().get("predicted_class")
        if predicted_class:
            st.write(f"Prediction: {predicted_class}")
        else:
            st.write("Unexpected response format.")
    else:
        st.write("Error:", response.text)