from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

# Load models and utilities
models_and_utils = joblib.load("models_and_utils.joblib")
knn = models_and_utils["knn"]
dt = models_and_utils["decision_tree"]
logreg = models_and_utils["logistic_regression"]
scaler = models_and_utils["scaler"]

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age = float(request.form['age'])
        cholesterol = float(request.form['cholesterol'])
        resting_bp = float(request.form['resting_bp'])
        max_hr = float(request.form['max_hr'])
        exercise_angina = int(request.form['exercise_angina'])  # Assume 1 for Yes, 0 for No
        oldpeak = float(request.form['oldpeak'])
        st_slope = int(request.form['st_slope'])  # 0, 1, or 2
        fasting_bs = int(request.form['fasting_bs'])  # Assume 1 for Yes, 0 for No

        # Prepare input for the model
        new_person = pd.DataFrame({
            'Age': [age],
            'Cholesterol': [cholesterol],
            'RestingBP': [resting_bp],
            'MaxHR': [max_hr],
            'ExerciseAngina': [exercise_angina],
            'Oldpeak': [oldpeak],
            'ST_Slope': [st_slope],
            'FastingBS': [fasting_bs]
        })

        # Scale the input
        new_person_scaled = scaler.transform(new_person)

        # Predictions and probabilities
        knn_prediction = knn.predict(new_person_scaled)[0]
        knn_probabilities = knn.predict_proba(new_person_scaled)[0]

        dt_prediction = dt.predict(new_person_scaled)[0]
        dt_probabilities = dt.predict_proba(new_person_scaled)[0]

        logreg_prediction = logreg.predict(new_person_scaled)[0]
        logreg_probabilities = logreg.predict_proba(new_person_scaled)[0]

        # Format predictions
        results = {
            "KNN": {
                "prediction": "Heart Disease" if knn_prediction == 1 else "No Heart Disease",
                "probabilities": {
                    "No Heart Disease": f"{knn_probabilities[0] * 100:.2f}%",
                    "Heart Disease": f"{knn_probabilities[1] * 100:.2f}%"
                }
            },
            "Decision Tree": {
                "prediction": "Heart Disease" if dt_prediction == 1 else "No Heart Disease",
                "probabilities": {
                    "No Heart Disease": f"{dt_probabilities[0] * 100:.2f}%",
                    "Heart Disease": f"{dt_probabilities[1] * 100:.2f}%"
                }
            },
            "Logistic Regression": {
                "prediction": "Heart Disease" if logreg_prediction == 1 else "No Heart Disease",
                "probabilities": {
                    "No Heart Disease": f"{logreg_probabilities[0] * 100:.2f}%",
                    "Heart Disease": f"{logreg_probabilities[1] * 100:.2f}%"
                }
            }
        }

        # Render results as cards
        return render_template('results.html', results=results)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
