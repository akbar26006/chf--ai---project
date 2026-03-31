from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return "API Running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    spo2 = data['spo2']
    temperature = data['temperature']
    heartrate = data['heartrate']

    prediction = model.predict([[spo2, temperature, heartrate]])

    return jsonify({'status': prediction[0]})
