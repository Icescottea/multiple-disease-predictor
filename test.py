import pandas as pd
import joblib

# Load model
model = joblib.load("models/rf_disease_model.pkl")

# Load symptom order from training CSV
df = pd.read_csv("dataset/multiple_disease_dataset.csv")  # or your actual training file path
SYMPTOM_ORDER = list(df.columns[:-1])  # all except the 'disease' column

# Define test input (some known symptoms from your new dataset)
test_symptoms = ['fever', 'cough', 'fatigue', 'nausea', 'abdominal pain']

# Encode symptom vector
input_vector = [1 if symptom in test_symptoms else 0 for symptom in SYMPTOM_ORDER]

# Predict
prediction = model.predict([input_vector])[0]
print("Predicted Disease:", prediction)
