import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

def load_data(file_paths):
    """Loads multiple CSV files into a dictionary of DataFrames."""
    data = {}
    for file_name, file_path in file_paths.items():
        # Specify errors='replace' to ignore invalid characters
        data[file_name] = pd.read_csv(file_path)
    return data

def merge_datasets(data):
    """Merge all datasets on 'datetime'."""
    
    # Print column names to debug
    for file_name in data:
        print(f"Columns in {file_name}: {data[file_name].columns}")
    
    # Ensure 'datetime' is created from 'Year' or 'year' (if it exists)
    for file_name in ['lagdo_dam_precipitation_data', 'lagdo_dam_discharge_data', 'lagdo_dam_flood_severity_data', 'lagdo_dam_economic_impact_data']:
        if 'datetime' not in data[file_name].columns:
            if 'Year' in data[file_name].columns:  # Handling 'Year' instead of 'year'
                data[file_name]['datetime'] = pd.to_datetime(data[file_name]['Year'], format='%Y', dayfirst=True)
            elif 'year' in data[file_name].columns:  # Handle lowercase 'year'
                data[file_name]['datetime'] = pd.to_datetime(data[file_name]['year'], format='%Y', dayfirst=True)
            else:
                # If there's no 'year' or 'Year' column, raise an error or handle it differently
                print(f"Warning: 'year' or 'Year' column missing in {file_name}. Cannot create 'datetime'.")
                return None  # Returning None to indicate failure in merging
    
    # Ensure 'lagdo_Dam_Flood_Data' has a 'datetime' column
    if 'datetime' not in data['lagdo_Dam_Flood_Data'].columns:
        if 'Year' in data['lagdo_Dam_Flood_Data'].columns:
            data['lagdo_Dam_Flood_Data']['datetime'] = pd.to_datetime(data['lagdo_Dam_Flood_Data']['Year'], format='%Y', dayfirst=True)
    
    # Convert 'datetime' column to datetime64[ns] format for all datasets
    for file_name in data:
        if 'datetime' in data[file_name].columns:
            try:
                # Explicitly specify dayfirst=True to prevent parsing warning
                data[file_name]['datetime'] = pd.to_datetime(data[file_name]['datetime'], errors='coerce', dayfirst=True)
            except Exception as e:
                print(f"Error parsing datetime in {file_name}: {e}")
    
    # Optimize memory by converting categorical columns to 'category'
    categorical_columns = ['State', 'flood_severity', 'affected_regions', 'Month(s) of Occurrence']
    for file_name in data:
        for col in categorical_columns:
            if col in data[file_name].columns:
                data[file_name][col] = data[file_name][col].astype('category')
    
    # Drop unnecessary columns (e.g., 'error' columns) before merging to save memory
    for file_name in data:
        if 'error' in data[file_name].columns:
            data[file_name].drop(columns=['error'], inplace=True)
    
    # Now perform the merge using the 'datetime' column
    merged_data = data['lagdo_dam_precipitation_data']
    merged_data = merged_data.merge(data['lagdo_dam_discharge_data'], on='datetime', how='outer')
    merged_data = merged_data.merge(data['lagdo_dam_flood_severity_data'], on='datetime', how='outer')
    merged_data = merged_data.merge(data['lagdo_dam_economic_impact_data'], on='datetime', how='outer')
    merged_data = merged_data.merge(data['lagdo_Dam_Flood_Data'], on='datetime', how='outer')
    merged_data = merged_data.merge(data['dahiti_lagdo_Dam_SurfaceArea'], on='datetime', how='outer')
    merged_data = merged_data.merge(data['dahiti_lagdo_Dam_VolumeVariation'], on='datetime', how='outer')
    merged_data = merged_data.merge(data['dahiti_lagdo_dam_water_levels'], on='datetime', how='outer')
    
    return merged_data

def preprocess_data(data, output_file=None):
    """Clean and preprocess the merged data."""
    
    # Handle missing values using forward fill and backward fill
    data.ffill(inplace=True)  # Forward fill to propagate the last valid value forward
    data.bfill(inplace=True)  # Backward fill to propagate the next valid value backward
    
    # Convert datetime column to pandas datetime (if necessary)
    data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')  # Ensure that datetime is properly formatted
    
    # Extract date-related features (e.g., year, month, day)
    data['year'] = data['datetime'].dt.year
    data['month'] = data['datetime'].dt.month
    data['day'] = data['datetime'].dt.day
    
    # Drop columns that aren't useful for the model (e.g., 'datetime', 'Flood Severity', 'States Affected')
    columns_to_drop = ['datetime', 'Flood Severity', 'affected_regions', 'State']
    data.drop(columns=[col for col in columns_to_drop if col in data.columns], inplace=True)  # Drop only if columns exist
    
    # Optionally save the preprocessed data
    if output_file:
        data.to_csv(output_file, index=False)
        print(f"Preprocessed data saved to {output_file}")
    
    return data


def split_data(data, target_columns):
    """Splits the data into train and test sets for multiple target variables."""
    X = data.drop(columns=target_columns)  # Drop all target columns
    y = data[target_columns]  # Use target_columns as the dependent variable
    
    # Train-test split
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    # Now split y for each target
    y_train_dict = {}
    y_test_dict = {}
    
    for target in target_columns:
        y_train_dict[target], y_test_dict[target] = train_test_split(y[target], test_size=0.2, random_state=42)
    
    # Return all the splits as separate variables
    return X_train, X_test, y_train_dict, y_test_dict

def scale_features(X_train, X_test):
    """Scales the features using StandardScaler and encodes categorical columns."""
    
    # Define the categorical columns with custom mappings
    custom_mappings = {
        "Flood Severity": {"No flood": 0, "Moderate": 1, "Severe": 2, "Minor": 3},
        "Dasin Hausa Dam Status": {"Not started": 0, "Delayed": 1, "Construction Started": 2},
        "Month(s) of Occurrence": {"July": 1, "August": 2, "September": 3, "October": 4},
        "States Affected": {
            'Benue': 1,
            'Kogi': 2,
            'Adamawa': 3,
            'Taraba': 4,
            'Delta': 5,
            'Cross River': 6,
            'Anambra': 7,
            'Bayelsa': 8
        }
    }

    # Identify categorical columns
    categorical_columns = ['flood_severity', 'Dasin Hausa Dam Status', 'Month(s) of Occurrence', 'States Affected']
    
    # Separate numerical columns
    numerical_columns = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Ensure categorical columns exist in the dataset before applying transformations
    categorical_columns = [col for col in categorical_columns if col in X_train.columns]

    # Custom transformer function to map categorical columns to their specified values
    def map_categorical_values(df):
        for col, mapping in custom_mappings.items():
            if col in df.columns:
                # Set categories first, to allow 'Unknown'
                df[col] = df[col].astype('category').cat.add_categories(['Unknown'])
                # Fill NaN with 'Unknown' and then apply the mapping
                df[col] = df[col].fillna('Unknown').map(mapping).astype(int, errors='ignore')  # Handle NaN values safely
        return df

    # Create a ColumnTransformer with custom mappings for categorical features and standard scaling for numerical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_columns),
            ('cat', Pipeline(steps=[
                ('mapper', FunctionTransformer(map_categorical_values)),  # Apply the custom mapping
                ('imputer', SimpleImputer(strategy='most_frequent'))  # Handle any missing values in categorical columns
            ]), categorical_columns)
        ])

    # Create a pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    # Fit and transform the train data
    X_train_scaled = pipeline.fit_transform(X_train)

    # Apply the same transformation to test data
    X_test_scaled = pipeline.transform(X_test)

    return X_train_scaled, X_test_scaled