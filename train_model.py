import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# File paths
DATASET_PATH = os.path.join("dataset", "multiple_disease_dataset.csv")
MODEL_PATH = os.path.join("models", "disease_predictor.pkl")
ENCODER_PATH = os.path.join("models", "label_encoder.pkl")

# Load the dataset
df = pd.read_csv(DATASET_PATH)

# Split features and label
X = df.drop('disease', axis=1)
y = df['disease']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save label encoder
joblib.dump(le, ENCODER_PATH)
print(f"Label encoder saved to {ENCODER_PATH}")

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
