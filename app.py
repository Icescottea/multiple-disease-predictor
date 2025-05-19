from flask import Flask, request, jsonify, render_template, redirect, url_for, session, Response
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from io import StringIO
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import numpy as np
import os
import csv
import requests
import json

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mdps.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Load model
model_path = os.path.join("models", "rf_disease_model.pkl")
print("Loading model from:", model_path)
model = joblib.load(model_path)

# Load symptoms dynamically from JSON
with open("static/data/symptoms.json") as f:
    SYMPTOM_ORDER = sorted([s.lower() for s in json.load(f)])

PRIORITY_DISEASES = ['ischemic heart disease', 'chronic heart disease', 'tuberculosis', 'cirrhosis of the liver', 'cancer']

SPECIALTY_KEYWORDS = {
    'ischemic heart disease': 'cardiology',
    'chronic heart disease': 'cardiology',
    'tuberculosis': 'respiratory',
    'cirrhosis of the liver': 'gastroenterology',
    'cancer': 'oncology',
}

GOOGLE_API_KEY = "AIzaSyC8fLOaXHdFKFLFoGImanFjPMtpPqHSjfk"

def get_hospitals_from_google(location, disease):
    keyword = SPECIALTY_KEYWORDS.get(disease.lower(), "hospital")
    query = f"{keyword} hospital in {location}"

    url = f"https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": query,
        "key": GOOGLE_API_KEY
    }

    res = requests.get(url, params=params)
    if res.status_code == 200:
        data = res.json()
        results = data.get('results', [])[:5]
        return [{
            "name": r.get("name"),
            "address": r.get("formatted_address"),
            "rating": r.get("rating", "N/A"),
            "place_id": r.get("place_id")
        } for r in results]
    return []

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    location = db.Column(db.String(100), nullable=True)
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(10), nullable=True)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class MedicalHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    disease = db.Column(db.String(100), nullable=False)
    symptoms_json = db.Column(db.Text, nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)

class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        location = request.form['location']
        age = int(request.form['age'])
        gender = request.form['gender']

        if User.query.filter_by(username=username).first():
            return "Username already taken", 409

        new_user = User(
            username=username,
            email=email,
            location=location,
            age=age,
            gender=gender,
            password_hash=generate_password_hash(password)
        )
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session['user_id'] = user.id
            session['username'] = user.username
            return redirect(url_for('home_page'))
        return "Invalid credentials", 401
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input received'}), 400

    input_data = [1 if symptom in data and data[symptom] == 1 else 0 for symptom in SYMPTOM_ORDER]

    if 'user_id' in session:
        history = MedicalHistory.query.filter_by(user_id=session['user_id']).order_by(MedicalHistory.date.desc()).limit(3).all()
        past_diseases = [h.disease.lower() for h in history]
        history_vector = [1 if d in past_diseases else 0 for d in PRIORITY_DISEASES]
    else:
        history_vector = [0] * len(PRIORITY_DISEASES)

    full_input = input_data + history_vector
    try:
        prediction = model.predict([full_input])[0]
        return jsonify({'predicted_disease': prediction})
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({'error': str(e)}), 500

@app.route('/symptom-checker', methods=['GET', 'POST'])
def symptom_checker():
    prediction = None
    user_gender = None
    user_age = None
    user_logged_in = 'username' in session

    if user_logged_in:
        user = User.query.get(session['user_id'])
        user_gender = user.gender
        user_age = user.age

    if request.method == 'POST':
        symptom_data = request.get_json()
        age = symptom_data.get('age', user_age)
        gender = symptom_data.get('gender', user_gender)

        input_data = [int(symptom_data.get(sym, 0)) for sym in SYMPTOM_ORDER]

        if user_logged_in:
            history = MedicalHistory.query.filter_by(user_id=session['user_id']).order_by(MedicalHistory.date.desc()).limit(3).all()
            past_diseases = [h.disease.lower() for h in history]
            history_vector = [1 if d in past_diseases else 0 for d in PRIORITY_DISEASES]
        else:
            history_vector = [0] * len(PRIORITY_DISEASES)

        full_input = input_data + history_vector
        pred = model.predict([full_input])[0]
        prediction = pred

        if user_logged_in:
            record = MedicalHistory(
                user_id=session['user_id'],
                disease=prediction,
                symptoms_json=json.dumps(dict(zip(SYMPTOM_ORDER, input_data)))
            )
            db.session.add(record)
            db.session.commit()

        return jsonify({'predicted_disease': prediction})

    return render_template(
        'symptom_checker.html',
        symptoms=SYMPTOM_ORDER,
        prediction=prediction,
        user_logged_in=user_logged_in,
        user_gender=user_gender,
        user_age=user_age
    )

@app.route('/recommend-hospital', methods=['POST'])
def recommend_hospital():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    disease = request.form.get('disease')
    user = User.query.get(session['user_id'])
    location = user.location
    hospitals = get_hospitals_from_google(location, disease)
    return render_template('hospital_results.html', hospitals=hospitals, disease=disease, location=location)

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user_id = session['user_id']
    user = User.query.get(user_id)
    history = MedicalHistory.query.filter_by(user_id=user_id).order_by(MedicalHistory.date.desc()).all()
    for rec in history:
        rec.symptoms = json.loads(rec.symptoms_json)
    return render_template('profile.html', history=history, username=session['username'], location=user.location)
    
@app.route('/admin-login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        admin = Admin.query.filter_by(username=username).first()
        if admin and admin.check_password(password):
            session['is_admin'] = True
            return redirect(url_for('admin_dashboard'))
        else:
            return "Unauthorized", 401
    return render_template('admin_login.html')

@app.route('/admin-dashboard')
def admin_dashboard():
    if not session.get('is_admin'):
        return redirect(url_for('admin_login'))

    users = User.query.all()
    records = MedicalHistory.query.order_by(MedicalHistory.date.desc()).all()
    return render_template('admin_dashboard.html', users=users, records=records)

@app.route('/admin-logout')
def admin_logout():
    session.pop('is_admin', None)
    return redirect(url_for('admin_login'))

@app.route('/admin/delete-user/<int:user_id>')
def delete_user(user_id):
    if not session.get('is_admin'):
        return redirect(url_for('admin_login'))

    # Also delete user’s history
    MedicalHistory.query.filter_by(user_id=user_id).delete()
    User.query.filter_by(id=user_id).delete()
    db.session.commit()
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/delete-record/<int:record_id>')
def delete_record(record_id):
    if not session.get('is_admin'):
        return redirect(url_for('admin_login'))

    MedicalHistory.query.filter_by(id=record_id).delete()
    db.session.commit()
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/stats')
def admin_stats():
    if not session.get('is_admin'):
        return redirect(url_for('admin_login'))

    from collections import Counter
    records = MedicalHistory.query.all()
    disease_counts = Counter([r.disease for r in records])
    top_diseases = disease_counts.most_common(5)
    return render_template('admin_stats.html', top_diseases=top_diseases)

@app.route('/admin/export-csv')
def export_csv():
    if not session.get('is_admin'):
        return redirect(url_for('admin_login'))

    records = MedicalHistory.query.order_by(MedicalHistory.date.desc()).all()

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Record ID', 'User ID', 'Disease', 'Symptoms JSON', 'Date'])

    for r in records:
        writer.writerow([r.id, r.user_id, r.disease, r.symptoms_json, r.date])

    output.seek(0)
    return Response(output, mimetype='text/csv',
                    headers={"Content-Disposition": "attachment;filename=medical_history.csv"})

if __name__ == '__main__':
    app.run(debug=True)

