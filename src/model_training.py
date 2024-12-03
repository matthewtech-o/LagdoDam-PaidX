import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

def train_flood_severity_model(X_train, y_train):
    """Train a Random Forest model for flood severity prediction."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_economic_impact_model(X_train, y_train):
    """Train a Random Forest Regressor for economic impact prediction."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_water_level_model(X_train, y_train):
    """Train a Random Forest Regressor for water level prediction."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_classification_model(model, X_test, y_test):
    """Evaluate classification model performance."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

def evaluate_regression_model(model, X_test, y_test):
    """Evaluate regression model performance."""
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {mae:.2f}")