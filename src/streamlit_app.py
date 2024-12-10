import streamlit as st
import requests
from datetime import datetime

# Set up the FastAPI endpoint URL
BASE_URL = "http://127.0.0.1:8001"

st.title("Flood Prediction System")
st.write(
    """
    This app interacts with the Flood Prediction API to predict:
    - Flood Severity
    - Economic Impact
    - Water Level
    """
)

# User inputs
st.header("Enter Input Data")
date_input = st.date_input("Select Date", datetime.now().date())
time_input = st.time_input("Select Time", datetime.now().time())
datetime_input = datetime.combine(date_input, time_input)
precipitation_mm = st.number_input("Precipitation (mm)", min_value=0.0, value=150.0)
dam_discharge_m3_s = st.number_input("Dam Discharge (m³/s)", min_value=0.0, value=1250.0)
state_affected = st.selectbox(
    "State Affected",
    ["Benue", "Kogi", "Adamawa", "Taraba", "Delta", "Cross River", "Anambra", "Bayelsa"]
)
rainfall_mm = st.number_input("Rainfall (mm)", min_value=0.0, value=200.0)

# Submit button
if st.button("Submit for Prediction"):
    # Prepare input data for API
    input_data = {
        "datetime": datetime_input.isoformat(),
        "precipitation_mm": precipitation_mm,
        "dam_discharge_m3_s": dam_discharge_m3_s,
        "state_affected": state_affected,
        "rainfall_mm": rainfall_mm
    }

    # Call the FastAPI endpoints
    try:
        # Flood Severity Prediction
        severity_response = requests.post(f"{BASE_URL}/predict/flood_severity", json=input_data)
        severity_result = severity_response.json()

        # Economic Impact Prediction
        economic_response = requests.post(f"{BASE_URL}/predict/economic_impact", json=input_data)
        economic_result = economic_response.json()

        # Water Level Prediction
        water_level_response = requests.post(f"{BASE_URL}/predict/water_level", json=input_data)
        water_level_result = water_level_response.json()

        # Display results
        st.subheader("Prediction Results")
        st.write("**Flood Severity Prediction**")
        st.json(severity_result)

        st.write("**Economic Impact Prediction**")
        st.json(economic_result)

        st.write("**Water Level Prediction**")
        st.json(water_level_result)

    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while connecting to the API: {e}")