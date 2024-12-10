import pandas as pd
import numpy as np
from datetime import datetime

def create_rolling_features(data, window=7):
    """
    Create rolling window statistics (mean, std, min, max) for relevant columns.
    Window size is set to 7 by default, which can be adjusted.
    """
    data['rolling_precipitation_mean'] = data['precipitation_mm'].rolling(window=window).mean()
    data['rolling_precipitation_std'] = data['precipitation_mm'].rolling(window=window).std()
    data['rolling_discharge_mean'] = data['dam_discharge_(m_/s)'].rolling(window=window).mean()
    data['rolling_discharge_std'] = data['dam_discharge_(m_/s)'].rolling(window=window).std()
    data['rolling_water_level_mean'] = data['water_level_m'].rolling(window=window).mean()
    data['rolling_water_level_std'] = data['water_level_m'].rolling(window=window).std()
    
    return data

def create_lag_features(data, lag=1):
    """
    Create lag features (previous time steps).
    Lag of 1 means the value from the previous day.
    """
    data['lag_precipitation'] = data['precipitation_mm'].shift(lag)
    data['lag_discharge'] = data['dam_discharge_(m_/s)'].shift(lag)
    data['lag_water_level'] = data['water_level_m'].shift(lag)
    
    return data

def create_seasonal_features(data):
    """
    Create seasonal features based on the month of the year (Rainy or Dry season).
    """
    data['is_rainy_season'] = data['month'].apply(lambda x: 1 if 5 <= x <= 10 else 0)  # May to October is rainy season
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)  # Sine transform of the month
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)  # Cosine transform of the month
    
    return data

def create_interaction_features(data):
    """
    Create interaction features like precipitation and discharge, which may affect flood severity.
    """
    data['precipitation_discharge_interaction'] = data['precipitation_mm'] * data['dam_discharge_(m_/s)']
    data['precipitation_water_level_interaction'] = data['precipitation_mm'] * data['water_level_m']
    
    return data

def feature_engineering(data, output_file=None):
    """
    Apply all feature engineering steps (rolling features, lag features, seasonal, and interaction features).
    """
    data = create_rolling_features(data)
    data = create_lag_features(data)
    data = create_seasonal_features(data)
    data = create_interaction_features(data)
    
    # Drop rows with missing values caused by lag and rolling calculations
    data.dropna(inplace=True)

    # Optionally save the feature-engineered data
    if output_file:
        data.to_csv(output_file, index=False)
    
    return data

def generate_features(data):
    """
    Derive all required features based on minimal user input, including date-derived features.
    """
    # Map categorical features
    state_mapping = {
        'Benue': 1, 'Kogi': 2, 'Adamawa': 3, 'Taraba': 4,
        'Delta': 5, 'Cross River': 6, 'Anambra': 7, 'Bayelsa': 8
    }
    data['States_Affected'] = data['state_affected'].map(state_mapping).fillna(0)

    # Example defaults for missing features
    data['volume_variation_m3s'] = 0.0
    data['rolling_precipitation_std'] = 0.0
    data['rolling_water_level_mean'] = 210.0
    data['rolling_water_level_std'] = 2.0
    data['Year_y'] = data['year']
    data['Displacement_People'] = 1000.0
    data['Flood_Duration_Months'] = 2.0
    data['Benue_River_Discharge_m3_s'] = 1200.0
    data['Dam_Release_m3_s'] = data['dam_discharge_m3_s']
    data['Notification_Days_Before_Release'] = 3.0
    data['Economic_Loss_Naira_Billion'] = 150.0
    data['Farmland_Affected_Hectares'] = 5000.0
    data['rolling_discharge_mean'] = data['dam_discharge_m3_s'].rolling(window=3, min_periods=1).mean()
    data['rolling_discharge_std'] = data['dam_discharge_m3_s'].rolling(window=3, min_periods=1).std().fillna(0)
    data['Food_Security_Impact_percent'] = 70.0
    data['is_rainy_season'] = 1
    data['Dasin_Hausa_Dam_Status'] = 0  # "Not started"
    data['Months_of_Occurrence'] = data['month']  # Use extracted month
    data['surface_area_m3s'] = 500.0
    data['lag_discharge'] = data['dam_discharge_m3_s']
    data['Year_x'] = data['year']
    data['lag_precipitation'] = 140.0
    data['lag_water_level'] = 208.0
    data['rolling_precipitation_mean'] = data['precipitation_mm'].rolling(window=3, min_periods=1).mean()
    data['Agricultural_Damage_Hectares'] = 3000.0
    data['Infrastructure_Damage_Billion'] = 20.0

    # Ensure all required features are included
    required_features = [
        "year", "volume_variation_m3s", "dam_discharge_m3_s", "rolling_precipitation_std",
        "rolling_water_level_mean", "rolling_water_level_std", "States_Affected", "Year_y",
        "Displacement_People", "Flood_Duration_Months", "Benue_River_Discharge_m3_s",
        "Dam_Release_m3_s", "Notification_Days_Before_Release", "Economic_Loss_Naira_Billion",
        "Farmland_Affected_Hectares", "rolling_discharge_mean", "Rainfall_mm",
        "rolling_discharge_std", "month", "Food_Security_Impact_percent", "is_rainy_season",
        "Dasin_Hausa_Dam_Status", "Months_of_Occurrence", "surface_area_m3s", "lag_discharge",
        "Year_x", "precipitation_mm", "lag_precipitation", "lag_water_level",
        "rolling_precipitation_mean", "day", "Agricultural_Damage_Hectares", "Infrastructure_Damage_Billion"
    ]
    for feature in required_features:
        if feature not in data.columns:
            data[feature] = 0  # Default placeholder for missing features

    return data[required_features]