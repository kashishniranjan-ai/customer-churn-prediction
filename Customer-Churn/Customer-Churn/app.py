"""
app.py  –  Flask backend for Customer Churn Prediction
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load model ───────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(BASE_DIR, "churn_model.pkl")
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ── Exact column order the model was trained on ──────────────────────────────
MODEL_COLUMNS = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
    'gender_Male',
    'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year',
    'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
]

# ── Helper: preprocess form data exactly as get_dummies(drop_first=True) ─────
def preprocess(form: dict) -> pd.DataFrame:
    """
    Convert raw HTML form values into a single-row DataFrame whose columns
    match MODEL_COLUMNS exactly (same encoding as pandas.get_dummies with
    drop_first=True used during training).
    """
    row = {col: 0 for col in MODEL_COLUMNS}

    # ── Numeric features ─────────────────────────────────────────────────────
    row['SeniorCitizen']  = int(form.get('SeniorCitizen', 0))
    row['tenure']         = float(form.get('tenure', 0))
    row['MonthlyCharges'] = float(form.get('MonthlyCharges', 0))
    row['TotalCharges']   = float(form.get('TotalCharges', 0))

    # ── Binary / one-hot features ─────────────────────────────────────────────
    # gender  (reference: Female)
    if form.get('gender') == 'Male':
        row['gender_Male'] = 1

    # Partner  (reference: No)
    if form.get('Partner') == 'Yes':
        row['Partner_Yes'] = 1

    # Dependents  (reference: No)
    if form.get('Dependents') == 'Yes':
        row['Dependents_Yes'] = 1

    # PhoneService  (reference: No)
    if form.get('PhoneService') == 'Yes':
        row['PhoneService_Yes'] = 1

    # MultipleLines  (reference: No)
    ml = form.get('MultipleLines', 'No')
    if ml == 'No phone service':
        row['MultipleLines_No phone service'] = 1
    elif ml == 'Yes':
        row['MultipleLines_Yes'] = 1

    # InternetService  (reference: DSL)
    isp = form.get('InternetService', 'DSL')
    if isp == 'Fiber optic':
        row['InternetService_Fiber optic'] = 1
    elif isp == 'No':
        row['InternetService_No'] = 1

    # OnlineSecurity  (reference: No)
    os_ = form.get('OnlineSecurity', 'No')
    if os_ == 'No internet service':
        row['OnlineSecurity_No internet service'] = 1
    elif os_ == 'Yes':
        row['OnlineSecurity_Yes'] = 1

    # OnlineBackup  (reference: No)
    ob = form.get('OnlineBackup', 'No')
    if ob == 'No internet service':
        row['OnlineBackup_No internet service'] = 1
    elif ob == 'Yes':
        row['OnlineBackup_Yes'] = 1

    # DeviceProtection  (reference: No)
    dp = form.get('DeviceProtection', 'No')
    if dp == 'No internet service':
        row['DeviceProtection_No internet service'] = 1
    elif dp == 'Yes':
        row['DeviceProtection_Yes'] = 1

    # TechSupport  (reference: No)
    ts = form.get('TechSupport', 'No')
    if ts == 'No internet service':
        row['TechSupport_No internet service'] = 1
    elif ts == 'Yes':
        row['TechSupport_Yes'] = 1

    # StreamingTV  (reference: No)
    stv = form.get('StreamingTV', 'No')
    if stv == 'No internet service':
        row['StreamingTV_No internet service'] = 1
    elif stv == 'Yes':
        row['StreamingTV_Yes'] = 1

    # StreamingMovies  (reference: No)
    smv = form.get('StreamingMovies', 'No')
    if smv == 'No internet service':
        row['StreamingMovies_No internet service'] = 1
    elif smv == 'Yes':
        row['StreamingMovies_Yes'] = 1

    # Contract  (reference: Month-to-month)
    ct = form.get('Contract', 'Month-to-month')
    if ct == 'One year':
        row['Contract_One year'] = 1
    elif ct == 'Two year':
        row['Contract_Two year'] = 1

    # PaperlessBilling  (reference: No)
    if form.get('PaperlessBilling') == 'Yes':
        row['PaperlessBilling_Yes'] = 1

    # PaymentMethod  (reference: Bank transfer (automatic))
    pm = form.get('PaymentMethod', 'Bank transfer (automatic)')
    if pm == 'Credit card (automatic)':
        row['PaymentMethod_Credit card (automatic)'] = 1
    elif pm == 'Electronic check':
        row['PaymentMethod_Electronic check'] = 1
    elif pm == 'Mailed check':
        row['PaymentMethod_Mailed check'] = 1

    return pd.DataFrame([row], columns=MODEL_COLUMNS)


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        form = request.form.to_dict()
        X = preprocess(form)

        prediction   = int(model.predict(X)[0])
        probability  = float(model.predict_proba(X)[0][1])

        result = {
            "prediction":  prediction,
            "will_churn":  prediction == 1,
            "probability": round(probability * 100, 2),
            "label":       "Customer Will Churn" if prediction == 1 else "Customer Will Stay",
            "status":      "danger" if prediction == 1 else "success",
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/feature-importances")
def feature_importances():
    path = os.path.join(BASE_DIR, "static", "feature_importances.json")
    with open(path) as f:
        data = json.load(f)
    # Return top-15 sorted
    sorted_data = dict(sorted(data.items(), key=lambda x: x[1], reverse=True)[:15])
    return jsonify(sorted_data)


@app.route("/confusion-matrix")
def confusion_matrix_data():
    path = os.path.join(BASE_DIR, "static", "confusion_matrix.json")
    with open(path) as f:
        data = json.load(f)
    return jsonify(data)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
