from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load model
model = joblib.load("model.pkl")

# Home route
@app.route('/')
def home():
    return "API Running"

# Prediction route (manual input)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        spo2 = data['spo2']
        temperature = data['temperature']
        heart_rate = data['heart_rate']

        prediction = model.predict([[spo2, temperature, heart_rate]])

        return jsonify({'status': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)})

# Dataset + prediction route (FOR DASHBOARD)
@app.route('/data', methods=['GET'])
def get_data():
    try:
        data = pd.read_csv("data.csv")

        # Clean column names
        data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")

        # Predict
        predictions = model.predict(data[['spo2', 'temperature', 'heart_rate']])
        data['status'] = predictions

        return data.to_json(orient='records')

    except Exception as e:
        return jsonify({'error': str(e)})

# IMPORTANT for Render
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
