import pandas as pd
import numpy as np

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