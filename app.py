from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load your trained models
try:
    random_forest_model = joblib.load('random_forest_model.joblib')
    gradient_boosting_model = joblib.load('gradient_boosting_model.joblib')
except Exception as e:
    print("Error loading models:", e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    # Process input data
    engine_temperature = data.get('engine_temperature')
    battery_voltage = data.get('battery_voltage')
    vibration_level = data.get('vibration_level')
    fuel_efficiency = data.get('fuel_efficiency')
    mileage = data.get('mileage')

    # Prepare input for prediction
    input_data = pd.DataFrame([[engine_temperature, battery_voltage, vibration_level, fuel_efficiency, mileage]],
                              columns=['engine_temperature', 'battery_voltage', 'vibration', 'fuel_efficiency', 'mileage'])

    rf_prediction, rf_confidence, gb_prediction, gb_confidence = get_predictions(input_data)

    warnings = []
    if battery_voltage < 12.0:
        warnings.append("🔋 Battery voltage is low! Please check the battery.")
    if engine_temperature > 120.0:
        warnings.append("🔥 Engine temperature is too high! Please check the cooling system.")
    if vibration_level > 0.1:
        warnings.append("💥 Excessive vibration detected! Please inspect the vehicle for issues.")
    if fuel_efficiency < 10.0:
        warnings.append("⛽ Fuel efficiency is low! Consider checking the engine or air filters.")
    if mileage > 200000:
        warnings.append("⚠️ High mileage — potential wear and tear! Consider a full inspection.")

    return jsonify({'warnings': warnings, 'rf_confidence': float(rf_confidence), 'gb_confidence': float(gb_confidence)})

def get_predictions(input_data):
    rf_prediction = random_forest_model.predict(input_data)
    rf_confidence = random_forest_model.predict_proba(input_data)[:, 1]
    gb_prediction = gradient_boosting_model.predict(input_data)
    gb_confidence = gradient_boosting_model.predict_proba(input_data)[:, 1]
    return rf_prediction[0], rf_confidence[0], gb_prediction[0], gb_confidence[0]

if __name__ == '__main__':
    app.run(debug=True)
