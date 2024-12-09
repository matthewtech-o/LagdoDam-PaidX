from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load the pre-trained models (replace with your actual paths to models)
severity_model = joblib.load('LagdoDam-PaidX/models/flood_severity_model.pkl')  # Replace with actual model path
impact_model = joblib.load('LagdoDam-PaidX/models/economic_impact_model.pkl')  # Replace with actual model path
water_model = joblib.load('LagdoDam-PaidX/models/water_level_model.pkl')  # Replace with actual model path

# Define the request schema with the updated input features
class PredictionInput(BaseModel):
    year: int
    volume_variation_m3s: float
    dam_discharge_m_s: float
    rolling_precipitation_std: float
    rolling_water_level_mean: float
    rolling_water_level_std: float
    States_Affected: str
    Year_y: float
    Displacement_People: float
    Flood_Duration_Months: float
    Benue_River_Discharge_m3_s: float
    Dam_Release_m3_s: float
    Notification_Days_Before_Release: float
    Economic_Loss_Naira_Billion: float
    Farmland_Affected_Hectares: float
    rolling_discharge_mean: float
    Rainfall_mm: float
    rolling_discharge_std: float
    month: int
    Food_Security_Impact_percent: float
    is_rainy_season: int
    Dasin_Hausa_Dam_Status: str
    Months_of_Occurrence: str
    surface_area_m3s: float
    lag_discharge: float
    Year_x: float
    precipitation_mm: float
    lag_precipitation: float
    lag_water_level: float
    rolling_precipitation_mean: float
    day: int
    Agricultural_Damage_Hectares: float
    Infrastructure_Damage_Billion: float

# Define the list of features that are common across all models
model_features = [
    'year', 'volume_variation_m3s', 'dam_discharge_m_s', 'rolling_precipitation_std',
    'rolling_water_level_mean', 'rolling_water_level_std', 'States_Affected', 'Year_y',
    'Displacement_People', 'Flood_Duration_Months', 'Benue_River_Discharge_m3_s',
    'Dam_Release_m3_s', 'Notification_Days_Before_Release', 'Economic_Loss_Naira_Billion',
    'Farmland_Affected_Hectares', 'rolling_discharge_mean', 'Rainfall_mm', 'rolling_discharge_std',
    'month', 'Food_Security_Impact_percent', 'is_rainy_season', 'Dasin_Hausa_Dam_Status',
    'Months_of_Occurrence', 'surface_area_m3s', 'lag_discharge', 'Year_x', 'precipitation_mm',
    'lag_precipitation', 'lag_water_level', 'rolling_precipitation_mean', 'day',
    'Agricultural_Damage_Hectares', 'Infrastructure_Damage_Billion'
]

# Preprocessing mappings
months_mapping = {"July": 1, "August": 2, "September": 3, "October": 4}
states_mapping = {
    'Benue': 1, 'Kogi': 2, 'Adamawa': 3, 'Taraba': 4, 'Delta': 5, 
    'Cross River': 6, 'Anambra': 7, 'Bayelsa': 8
}
dam_status_mapping = {"Not started": 0, "Delayed": 1, "Construction Started": 2}

# Route for predictions
def make_prediction(input_data: PredictionInput, model, target: str):
    input_df = pd.DataFrame([input_data.dict()])
    try:
        # Preprocess categorical values
        input_df['Dasin_Hausa_Dam_Status'] = input_df['Dasin_Hausa_Dam_Status'].map(dam_status_mapping)
        input_df['Months_of_Occurrence'] = input_df['Months_of_Occurrence'].map(months_mapping)
        input_df['States_Affected'] = input_df['States_Affected'].map(states_mapping)
        
        # Select only the relevant features for prediction
        input_df = input_df[model_features]
        
        # Make prediction
        prediction = model.predict(input_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in {target} prediction: {e}")
    return {target: prediction[0]}

@app.get("/")
def welcome():
    return {"message":"Welcome to the Flood Prediction API!

@app.post("/predict/flood_severity")
def predict_flood_severity(input_data: PredictionInput):
    return make_prediction(input_data, severity_model, "flood_severity")

@app.post("/predict/economic_impact")
def predict_economic_impact(input_data: PredictionInput):
    return make_prediction(input_data, impact_model, "economic_impact")

@app.post("/predict/water_level")
def predict_water_level(input_data: PredictionInput):
    return make_prediction(input_data, water_model, "water_level")

# Run the FastAPI server
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
