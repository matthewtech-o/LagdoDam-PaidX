import joblib
from data_preprocessing import load_data, merge_datasets, preprocess_data, split_data, scale_features
from feature_engineering import feature_engineering
from model_training import train_flood_severity_model, train_economic_impact_model, train_water_level_model, evaluate_classification_model, evaluate_regression_model, print_feature_importance

def main():
    # Define file paths for the datasets
    file_paths = {
        'lagdo_dam_precipitation_data': 'data/raw/lagdo_dam_precipitation_data.csv',
        'lagdo_dam_discharge_data': 'data/raw/lagdo_dam_discharge_data.csv',
        'lagdo_dam_flood_severity_data': 'data/raw/lagdo_dam_flood_severity_data.csv',
        'lagdo_dam_economic_impact_data': 'data/raw/lagdo_dam_economic_impact_data.csv',
        'lagdo_Dam_Flood_Data': 'data/raw/lagdo_Dam_Flood_Data between 1982 to 2024.csv',
        'dahiti_lagdo_Dam_SurfaceArea': 'data/raw/dahiti_lagdo_Dam_SurfaceArea.csv',
        'dahiti_lagdo_Dam_VolumeVariation': 'data/raw/dahiti_lagdo_Dam_VolumeVariation.csv',
        'dahiti_lagdo_dam_water_levels': 'data/raw/dahiti_lagdo_dam_water_levels.csv'
    }
    
    # Load data
    data = load_data(file_paths)
    
    # Merge datasets
    merged_data = merge_datasets(data)
    
    # Preprocess data and save it to a CSV file
    if merged_data is not None:  # Ensure that merging was successful
        preprocessed_data = preprocess_data(merged_data, output_file='data/New2/merged_data.csv')
    
        # Apply feature engineering and save the engineered data
        engineered_data = feature_engineering(preprocessed_data, output_file='data/New2/engineered_data.csv')

        # Split data for flood severity, economic impact, and water level
        target_columns = ['flood_severity', 'GDP Loss (Billion)', 'water_level_m']  # List of target columns
        X_train, X_test, y_train, y_test = split_data(engineered_data, target_columns)  # Pass the list of target columns

        # Save X_train features to a CSV file
        X_train.to_csv('data/New2/X_train_features.csv', index=False)  # Save as CSV file
        print("\nX_train features have been saved successfully as 'X_train_features.csv'.")

        # Print the X_train and y_train DataFrames
        print("\nX_train (first 5 rows):")
        print(X_train.head())  # Display first 5 rows of X_train
    
        # Scale features
        X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
        # Train models
        severity_model = train_flood_severity_model(X_train_scaled, y_train['flood_severity'])

        print_feature_importance(severity_model, top_n=33)

        impact_model = train_economic_impact_model(X_train_scaled, y_train['GDP Loss (Billion)'])
        water_model = train_water_level_model(X_train_scaled, y_train['water_level_m'])

        # Save trained models
        joblib.dump(severity_model, 'models/flood_severity_model.pkl')
        joblib.dump(impact_model, 'models/economic_impact_model.pkl')
        joblib.dump(water_model, 'models/water_level_model.pkl')
        print("Models have been saved successfully!")
    
        # Evaluate models
        print("Flood Severity Model Evaluation:")
        evaluate_classification_model(severity_model, X_test_scaled, y_test['flood_severity'])
    
        print("\nEconomic Impact Model Evaluation:")
        evaluate_regression_model(impact_model, X_test_scaled, y_test['GDP Loss (Billion)'])
    
        print("\nWater Level Prediction Model Evaluation:")
        evaluate_regression_model(water_model, X_test_scaled, y_test['water_level_m'])

if __name__ == "__main__":
    main()
