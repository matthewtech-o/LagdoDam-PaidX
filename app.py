from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained models (replace with your actual paths to models)
severity_model = joblib.load('models/flood_severity_model.pkl')
impact_model = joblib.load('models/economic_impact_model.pkl')
water_model = joblib.load('models/water_level_model.pkl')

# Route for flood severity prediction
@app.route('/predict/flood_severity', methods=['POST'])
def predict_flood_severity():
    # Get data from POST request
    data = request.get_json(force=True)
    
    # Convert input data to DataFrame (assuming it comes as a dictionary)
    input_data = pd.DataFrame([data])
    
    # Make prediction
    prediction = severity_model.predict(input_data)
    
    # Return the prediction as JSON response
    return jsonify({'flood_severity': prediction[0]})

# Route for economic impact prediction
@app.route('/predict/economic_impact', methods=['POST'])
def predict_economic_impact():
    # Get data from POST request
    data = request.get_json(force=True)
    
    # Convert input data to DataFrame
    input_data = pd.DataFrame([data])
    
    # Make prediction
    prediction = impact_model.predict(input_data)
    
    # Return the prediction as JSON response
    return jsonify({'economic_impact': prediction[0]})

# Route for water level prediction
@app.route('/predict/water_level', methods=['POST'])
def predict_water_level():
    # Get data from POST request
    data = request.get_json(force=True)
    
    # Convert input data to DataFrame
    input_data = pd.DataFrame([data])
    
    # Make prediction
    prediction = water_model.predict(input_data)
    
    # Return the prediction as JSON response
    return jsonify({'water_level': prediction[0]})

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)