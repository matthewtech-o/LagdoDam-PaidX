from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from datetime import datetime
import pandas as pd
from src.feature_engineering import generate_features


app = FastAPI()

# Load the pre-trained models
severity_model = joblib.load('models/flood_severity_model.pkl') 
impact_model = joblib.load('models/economic_impact_model.pkl') 
water_model = joblib.load('models/water_level_model.pkl') 

# # Define the request schema with the updated input features
# # class PredictionInput(BaseModel):
# #     year: int
# #     volume_variation_m3s: float
# #     dam_discharge_m_s: float
# #     rolling_precipitation_std: float
# #     rolling_water_level_mean: float
# #     rolling_water_level_std: float
# #     States_Affected: str
# #     Year_y: float
# #     Displacement_People: float
# #     Flood_Duration_Months: float
# #     Benue_River_Discharge_m3_s: float
# #     Dam_Release_m3_s: float
# #     Notification_Days_Before_Release: float
# #     Economic_Loss_Naira_Billion: float
# #     Farmland_Affected_Hectares: float
# #     rolling_discharge_mean: float
# #     Rainfall_mm: float
# #     rolling_discharge_std: float
# #     month: int
# #     Food_Security_Impact_percent: float
# #     is_rainy_season: int
# #     Dasin_Hausa_Dam_Status: str
# #     Months_of_Occurrence: str
# #     surface_area_m3s: float
# #     lag_discharge: float
# #     Year_x: float
# #     precipitation_mm: float
# #     lag_precipitation: float
# #     lag_water_level: float
# #     rolling_precipitation_mean: float
# #     day: int
# #     Agricultural_Damage_Hectares: float
# #     Infrastructure_Damage_Billion: float

# # Define the list of features that are common across all models
# model_features = [
#     'year', 'volume_variation_m3s', 'dam_discharge_m_s', 'rolling_precipitation_std',
#     'rolling_water_level_mean', 'rolling_water_level_std', 'States_Affected', 'Year_y',
#     'Displacement_People', 'Flood_Duration_Months', 'Benue_River_Discharge_m3_s',
#     'Dam_Release_m3_s', 'Notification_Days_Before_Release', 'Economic_Loss_Naira_Billion',
#     'Farmland_Affected_Hectares', 'rolling_discharge_mean', 'Rainfall_mm', 'rolling_discharge_std',
#     'month', 'Food_Security_Impact_percent', 'is_rainy_season', 'Dasin_Hausa_Dam_Status',
#     'Months_of_Occurrence', 'surface_area_m3s', 'lag_discharge', 'Year_x', 'precipitation_mm',
#     'lag_precipitation', 'lag_water_level', 'rolling_precipitation_mean', 'day',
#     'Agricultural_Damage_Hectares', 'Infrastructure_Damage_Billion'
# ]

# # Preprocessing mappings
# months_mapping = {"July": 1, "August": 2, "September": 3, "October": 4}
# states_mapping = {
#     'Benue': 1, 'Kogi': 2, 'Adamawa': 3, 'Taraba': 4, 'Delta': 5, 
#     'Cross River': 6, 'Anambra': 7, 'Bayelsa': 8
# }
# dam_status_mapping = {"Not started": 0, "Delayed": 1, "Construction Started": 2}

# # Route for predictions
# def make_prediction(input_data: PredictionInput, model, target: str):
#     input_df = pd.DataFrame([input_data.dict()])
#     try:
#         # Preprocess categorical values
#         input_df['Dasin_Hausa_Dam_Status'] = input_df['Dasin_Hausa_Dam_Status'].map(dam_status_mapping)
#         input_df['Months_of_Occurrence'] = input_df['Months_of_Occurrence'].map(months_mapping)
#         input_df['States_Affected'] = input_df['States_Affected'].map(states_mapping)
        
#         # Select only the relevant features for prediction
#         input_df = input_df[model_features]
        
#         # Make prediction
#         prediction = model.predict(input_df)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error in {target} prediction: {e}")
#     return {target: prediction[0]}

# Input schema
class UserInput(BaseModel):
    datetime: datetime
    precipitation_mm: float
    dam_discharge_m3_s: float
    state_affected: str
    rainfall_mm: float

# Preprocessing pipeline
def preprocess_user_input(user_input: UserInput):
    """
    Preprocess user input to generate the full feature set.
    """
    # Convert input to a dictionary
    user_dict = user_input.dict()

    # Extract date components from datetime
    user_dict['year'] = user_dict['datetime'].year
    user_dict['month'] = user_dict['datetime'].month
    user_dict['day'] = user_dict['datetime'].day

    # Convert updated dictionary to DataFrame
    user_data = pd.DataFrame([user_dict])

    # Drop the original datetime field
    user_data.drop(columns=["datetime"], inplace=True)

    # Generate all features
    full_features = generate_features(user_data)

    # Drop unnecessary columns
    if 'state_affected' in full_features.columns:
        full_features.drop(columns=['state_affected'], inplace=True)

    return full_features

@app.get("/")
def welcome():
    return {"message":"Welcome to the Flood Prediction API!"}

@app.post("/predict/flood_severity")
def predict_flood_severity(input_data: UserInput):
    try:
        # Preprocess input data and make a prediction
        features = preprocess_user_input(input_data)
        severity = severity_model.predict(features)[0]

        # Map severity levels to messages
        severity_messages = {
            "No flood": "No flood is expected. No precautions necessary.",
            "Moderate": "Flood severity is moderate. Stay alert and monitor weather updates.",
            "Severe": "Flood severity is severe. Prepare for immediate action and evacuation if required.",
            "Minor": "Flood severity is minor. Exercise caution in low-lying areas."
        }

        message = severity_messages.get(severity, "Unknown severity level. Please verify the input data.")
        return {"severity_prediction": severity, "message": message}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in flood severity prediction: {e}")

@app.post("/predict/economic_impact")
def predict_economic_impact(input_data: UserInput):
    try:
        # Preprocess input data and make a prediction
        features = preprocess_user_input(input_data)
        impact = impact_model.predict(features)[0]

        # Interpret the GDP loss prediction
        if impact < 120:
            message = "Economic impact is low."
        elif 120 <= impact < 160:
            message = "Economic impact is moderate. Potential for significant damage."
        elif 160 <= impact < 200:
            message = "Economic impact is high. Major losses anticipated."
        else:
            message = "Economic impact is severe. Widespread damage expected."

        return {"economic_impact_prediction": impact, "message": message}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in economic impact prediction: {e}")

@app.post("/predict/water_level")
def predict_water_level(input_data: UserInput):
    try:
        # Preprocess input data and make a prediction
        features = preprocess_user_input(input_data)
        water_level = water_model.predict(features)[0]  # Assuming the model returns an array

        # Interpret the water level prediction
        if water_level < 210:
            message = "Water level is within normal range."
        elif 210 <= water_level < 213:
            message = "Water level is elevated. Exercise caution in flood-prone areas."
        elif 213 <= water_level < 215:
            message = "Water level is high. Prepare for possible flooding."
        else:
            message = "Water level is critical. Immediate action required."

        return {"water_level_prediction": water_level, "message": message}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in water level prediction: {e}")

# Run the FastAPI server
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8002)
