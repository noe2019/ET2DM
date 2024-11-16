import streamlit as st
import requests

# FastAPI backend URL
FASTAPI_URL = "http://localhost:8001/predict"  # Update if running FastAPI on a different host/port

st.title('Early Diabetes Prediction!')

# Create input fields with user-friendly labels and dropdowns
age = st.number_input("Age (years)", min_value=21.0, max_value=120.0, step=0.1)

race_mapping = {
    "Non-Hispanic White": 1,
    "Non-Hispanic Black": 2,
    "Mexican American": 3,
    "Other Hispanic/Other Race": 4
}
race = st.selectbox("Race/Ethnicity", options=race_mapping.keys())

education_mapping = {
    "Less than High School": 1,
    "High School Graduate": 2,
    "Some College/College Graduate": 3
}
education = st.selectbox("Education Level", options=education_mapping.keys())

marital_status_mapping = {
    "Single": 1,
    "Married/Cohabiting": 2,
    "Divorced/Separated/Widowed": 3
}
marital_status = st.selectbox("Marital Status", options=marital_status_mapping.keys())

acculturation_score_mapping = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}
acculturation_score = st.selectbox("Total Acculturation Score", options=acculturation_score_mapping.keys())

dietary_fat_mapping = {
    "Low": 1,
    "Medium": 2,
    "High": 3
}
dietary_fat = st.selectbox("Dietary Fat Intake", options=dietary_fat_mapping.keys())

poverty_status_mapping = {
    "Above Poverty Level": 0,
    "Below Poverty Level": 1
}
poverty_status = st.selectbox("Poverty Status", options=poverty_status_mapping.keys())

hypertension_mapping = {
    "No": 0,
    "Yes": 1
}
hypertension = st.selectbox("Hypertension (High Blood Pressure)", options=hypertension_mapping.keys())

gender_mapping = {
    "Male": 1,
    "Female": 2
}
gender = st.selectbox("Sex", options=gender_mapping.keys())

smoking_status_mapping = {
    "Non-Smoker": 0,
    "Smoker": 1
}
smoking_status = st.selectbox("Smoking Status", options=smoking_status_mapping.keys())

# Button to send data to FastAPI for prediction
if st.button("Predict"):
    # Map selected options back to numeric values for backend compatibility
    payload = {
        "RIDAGEYR": age,
        "RACE": race_mapping[race],
        "EDUC": education_mapping[education],
        "COUPLE": marital_status_mapping[marital_status],
        "TOTAL_ACCULTURATION_SCORE_v2": acculturation_score_mapping[acculturation_score],
        "FAT": dietary_fat_mapping[dietary_fat],
        "POVERTIES": poverty_status_mapping[poverty_status],
        "HTN": hypertension_mapping[hypertension],
        "RIAGENDR": gender_mapping[gender],
        "SMOKER": smoking_status_mapping[smoking_status],
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