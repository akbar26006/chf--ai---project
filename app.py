from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

model = joblib.load("model.pkl")

@app.route('/')
def home():
    return "API Running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    spo2 = data['spo2']
    temperature = data['temperature']
    heart_rate = data['heart_rate']

    prediction = model.predict([[spo2, temperature, heart_rate]])

    return jsonify({'status': prediction[0]})
