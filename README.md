# LedeMekai – Multiple Disease Prediction System  

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-07405E?style=for-the-badge&logo=sqlite&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-323330?style=for-the-badge&logo=javascript&logoColor=F7DF1E)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)

## 📖 About  

**LedeMekai** is a web-based multiple disease prediction system designed to help users identify potential health conditions based on their symptoms.  
It addresses the healthcare access gap in developing regions by offering preliminary, accessible diagnostic support for critical diseases such as ischemic heart disease, chronic kidney disease, tuberculosis, cirrhosis of the liver, and cancer.  

The system uses a trained **Random Forest** machine learning model to provide the top three most probable diseases with confidence scores, records predictions in the user’s profile, and recommends nearby hospitals based on predicted conditions and the user’s location.  

## ✨ Key Features  

- **Symptom-Based Predictions:** Users enter demographic data and select symptoms via a responsive interface.  
- **Machine Learning Model:** Trained Random Forest classifier outputs top three probable diseases with confidence scores.  
- **User Profiles:** Stores previous predictions for easy access and tracking.  
- **Hospital Recommendations:** Suggests nearby hospitals based on predicted conditions and user location.  
- **Scrum Methodology:** Developed iteratively using requirement analysis, database modeling, and testing cycles.  
- **Responsive Design:** HTML, CSS, and JavaScript frontend optimized for desktops and mobile devices.  
- **External API Integration:** Location-based hospital search integrated with the system.  

## 🛠️ Technology Stack  

### Backend  
- **Language:** Python  
- **Framework:** Flask  
- **Machine Learning:** Random Forest (scikit-learn)  
- **Database:** SQLite  
- **External APIs:** Location-based hospital recommendations  

### Frontend  
- **HTML5, CSS3, JavaScript** for responsive, user-centered design  

### Development & Tools  
- **Version Control:** Git & GitHub  
- **Methodology:** Scrum-based iterative development  
- **Testing:** Unit and integration testing for accuracy and usability  

## 🚀 Getting Started  

Follow these steps to set up the project locally for development and testing.

### Prerequisites  
- Python 3.10+  
- pip (Python package manager)  
- SQLite3  

### Backend Setup  

1. **Clone Repository**  
    ```bash
    git clone https://your-repository-url/ledemekai.git
    cd ledemekai
    ```

2. **Create a Virtual Environment & Activate**  
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**  
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Application**  
    ```bash
    flask run
    ```
    The application will be available at `http://127.0.0.1:5000`.

### Environment Variables  

Create a `.env` file to configure your settings:  
```env
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=your_secret_key
MAPS_API_KEY=your_location_api_key
```

## 📋 System Workflow

User Registration/Login: Users create profiles to save prediction history.

Symptom Entry: Demographic and symptom data entered via the web interface.

Prediction Engine: Random Forest model processes inputs and outputs top three disease probabilities with confidence scores.

Hospital Recommendation: System suggests nearby hospitals using location APIs.

Result Storage: Predictions saved to the user’s profile for future reference.

## 🔐 Security Features

Secure password hashing and session management.

Input validation and sanitization to prevent XSS and SQL injection.

Environment variables for sensitive keys.

## 📊 Evaluation

The system was tested extensively with unit and integration testing, demonstrating strong model accuracy and interface usability.

## 👥 Default User Roles

User: Enter symptoms, view predictions, access hospital recommendations.

Admin (optional): Manage datasets, monitor model performance, and oversee system usage.

## 📝 License

This project is open source under the MIT License – see the LICENSE.txt
