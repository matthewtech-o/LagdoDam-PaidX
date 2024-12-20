import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, make_scorer
import numpy as np
import pandas as pd

def train_flood_severity_model(X_train, y_train):
    """Train a Random Forest model for flood severity prediction."""
    features = ['precipitation_mm', 'dam_discharge_(m_/s)', 'Year_x', 'Agricultural Damage (Hectares)',
                'Infrastructure Damage (Billion)', 'Food Security Impact (%)', 'Year_y', 
                'Month(s) of Occurrence', 'Flood Duration (Months)', 'States Affected',
                'Economic Loss (₦ Billion)', 'Displacement (People)', 'Farmland Affected (Hectares)',
                'Dam Release (m³/s)', 'Rainfall (mm)', 'Benue River Discharge (m³/s)',
                'Notification Days Before Release', 'Dasin Hausa Dam Status', 'surface_area_m3s',
                'volume_variation_m3s', 'year', 'month', 'day', 'rolling_precipitation_mean',
                'rolling_precipitation_std', 'rolling_discharge_mean', 'rolling_discharge_std',
                'rolling_water_level_mean', 'rolling_water_level_std', 'lag_precipitation',
                'lag_discharge', 'lag_water_level', 'is_rainy_season', 'month_sin', 'month_cos',
                'precipitation_discharge_interaction', 'precipitation_water_level_interaction']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Use provided feature names or generate if not available
    if hasattr(X_train, 'columns'):
        feature_names = X_train.columns.tolist()
    else:
        feature_names = features[:X_train.shape[1]]
    
    feature_importances = model.feature_importances_
    model.feature_importances_dict = dict(zip(feature_names, feature_importances))
    
    return model


def cross_validate_classification_model(X, y):
    """Perform k-fold cross-validation for classification model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    
    print("Cross-Validation Accuracy Scores:", cv_scores)
    print(f"Mean Accuracy: {np.mean(cv_scores):.4f}")
    print(f"Standard Deviation: {np.std(cv_scores):.4f}")


def print_feature_importance(model, top_n=10):
    """Print feature importances for the model."""
    if hasattr(model, 'feature_importances_dict'):
        sorted_features = sorted(model.feature_importances_dict.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop Feature Importances:")
        for feature, importance in sorted_features[:top_n]:
            print(f"- {feature}: {importance:.4f}")
    else:
        print("No feature importances found.")

def train_economic_impact_model(X_train, y_train):
    """Train a Random Forest Regressor for economic impact prediction."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    if hasattr(X_train, 'columns'):
        feature_names = X_train.columns.tolist()
    else:
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    
    feature_importances = model.feature_importances_
    model.feature_importances_dict = dict(zip(feature_names, feature_importances))
    
    return model

def train_water_level_model(X_train, y_train):
    """Train a Random Forest Regressor for water level prediction."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    if hasattr(X_train, 'columns'):
        feature_names = X_train.columns.tolist()
    else:
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    
    feature_importances = model.feature_importances_
    model.feature_importances_dict = dict(zip(feature_names, feature_importances))
    
    return model

def evaluate_classification_model(model, X_test, y_test):
    """Evaluate classification model performance."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")


def cross_validate_regression_model(X, y):
    """Perform k-fold cross-validation for regression model."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)  # Negative MAE
    
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring=mae_scorer)
    print("Cross-Validation MAE Scores:", -cv_scores)  # Convert back to positive
    print(f"Mean MAE: {-np.mean(cv_scores):.4f}")
    print(f"Standard Deviation: {np.std(-cv_scores):.4f}")

def evaluate_regression_model(model, X_test, y_test):
    """Evaluate regression model performance."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae:.2f}")
