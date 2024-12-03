import pandas as pd

def save_predictions(predictions, file_name):
    """Save model predictions to a CSV file."""
    pd.DataFrame(predictions, columns=['Predictions']).to_csv(file_name, index=False)

def load_csv(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)