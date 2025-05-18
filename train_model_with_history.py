import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import random

# Load dataset
df = pd.read_csv('dataset/multiple_disease_dataset.csv')

# Add fake past disease flags (randomized for now)
df['prev_ischemic'] = [random.randint(0,1) for _ in range(len(df))]
df['prev_chronic'] = [random.randint(0,1) for _ in range(len(df))]
df['prev_tb'] = [random.randint(0,1) for _ in range(len(df))]
df['prev_cirrhosis'] = [random.randint(0,1) for _ in range(len(df))]
df['prev_cancer'] = [random.randint(0,1) for _ in range(len(df))]

# Define features and label
X = df.drop('disease', axis=1)
y = df['disease']

# Encode label
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save label encoder
joblib.dump(le, 'models/label_encoder.pkl')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'models/disease_predictor.pkl')
print("✅ Model retrained and saved with 30 features.")
