from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model and encoder
model_path = os.path.join("models", "disease_predictor.pkl")
print("Loading model from:", model_path)
model = joblib.load(model_path)

encoder = joblib.load(os.path.join("models", "label_encoder.pkl"))

# Define the symptom order
SYMPTOM_ORDER = ['fever', 'cough', 'chest_pain', 'fatigue', 'weight_loss', 'shortness_of_breath',
                 'nausea', 'vomiting', 'headache', 'dizziness', 'palpitations', 'abdominal_pain',
                 'joint_pain', 'muscle_pain', 'diarrhea', 'constipation', 'blood_in_stool',
                 'skin_rash', 'night_sweats', 'loss_of_appetite', 'swelling', 'yellow_skin',
                 'back_pain', 'vision_problems', 'bleeding']

@app.route('/')
def home():
    return "Disease Prediction API Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    inputs = [int(data.get(symptom, 0)) for symptom in SYMPTOM_ORDER]
    prediction = model.predict([inputs])
    disease = encoder.inverse_transform(prediction)[0]
    return jsonify({'predicted_disease': disease})

@app.route('/test')
def test_page():
    return render_template('test.html')

if __name__ == '__main__':
    app.run(debug=True)

