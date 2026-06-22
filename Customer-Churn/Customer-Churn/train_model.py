"""
train_model.py
--------------
Trains a churn prediction model on synthetic Telco-like data and saves
churn_model.pkl.  Run this once before starting app.py.

    python train_model.py
"""

import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import json

# ── 1. Generate synthetic Telco Customer Churn data ──────────────────────────
np.random.seed(42)
N = 5000

def rand_choice(options, size, p=None):
    return np.random.choice(options, size=size, p=p)

data = pd.DataFrame({
    "gender":           rand_choice(["Male", "Female"], N),
    "SeniorCitizen":    rand_choice([0, 1], N, p=[0.84, 0.16]),
    "Partner":          rand_choice(["Yes", "No"], N),
    "Dependents":       rand_choice(["Yes", "No"], N, p=[0.3, 0.7]),
    "tenure":           np.random.randint(1, 73, N),
    "PhoneService":     rand_choice(["Yes", "No"], N, p=[0.9, 0.1]),
    "MultipleLines":    rand_choice(["Yes", "No", "No phone service"], N, p=[0.42, 0.48, 0.10]),
    "InternetService":  rand_choice(["DSL", "Fiber optic", "No"], N, p=[0.34, 0.44, 0.22]),
    "OnlineSecurity":   rand_choice(["Yes", "No", "No internet service"], N, p=[0.28, 0.50, 0.22]),
    "OnlineBackup":     rand_choice(["Yes", "No", "No internet service"], N, p=[0.34, 0.44, 0.22]),
    "DeviceProtection": rand_choice(["Yes", "No", "No internet service"], N, p=[0.34, 0.44, 0.22]),
    "TechSupport":      rand_choice(["Yes", "No", "No internet service"], N, p=[0.29, 0.49, 0.22]),
    "StreamingTV":      rand_choice(["Yes", "No", "No internet service"], N, p=[0.38, 0.40, 0.22]),
    "StreamingMovies":  rand_choice(["Yes", "No", "No internet service"], N, p=[0.39, 0.39, 0.22]),
    "Contract":         rand_choice(["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.21, 0.24]),
    "PaperlessBilling": rand_choice(["Yes", "No"], N, p=[0.59, 0.41]),
    "PaymentMethod":    rand_choice(
                            ["Electronic check", "Mailed check",
                             "Bank transfer (automatic)", "Credit card (automatic)"],
                            N, p=[0.34, 0.23, 0.22, 0.21]),
    "MonthlyCharges":   np.round(np.random.uniform(18, 118, N), 2),
    "TotalCharges":     np.round(np.random.uniform(18, 8680, N), 2),
})

# Churn label – realistic business logic
churn_prob = (
    0.05
    + 0.30 * (data["Contract"] == "Month-to-month")
    + 0.15 * (data["InternetService"] == "Fiber optic")
    + 0.10 * (data["tenure"] < 12)
    - 0.10 * (data["tenure"] > 48)
    + 0.08 * (data["PaymentMethod"] == "Electronic check")
    - 0.08 * (data["OnlineSecurity"] == "Yes")
    - 0.06 * (data["TechSupport"] == "Yes")
    + 0.05 * (data["SeniorCitizen"] == 1)
    - 0.04 * (data["Partner"] == "Yes")
)
churn_prob = churn_prob.clip(0.02, 0.90)
data["Churn"] = (np.random.rand(N) < churn_prob).astype(int)

print(f"Churn rate: {data['Churn'].mean():.2%}")

# ── 2. One-hot encode (same as pandas.get_dummies(drop_first=True)) ───────────
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

cat_cols = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

X = pd.get_dummies(data.drop("Churn", axis=1), columns=cat_cols, drop_first=True)

# Ensure all model columns present, fill missing with 0
for col in MODEL_COLUMNS:
    if col not in X.columns:
        X[col] = 0
X = X[MODEL_COLUMNS]

y = data["Churn"]

# ── 3. Train / test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 4. Train XGBoost ─────────────────────────────────────────────────────────
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")
print(f"Confusion matrix:\n{cm}")

# ── 5. Save model and metadata ───────────────────────────────────────────────
with open("churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Feature importances for the frontend chart
importances = dict(zip(MODEL_COLUMNS, model.feature_importances_.tolist()))
with open("static/feature_importances.json", "w") as f:
    json.dump(importances, f)

# Confusion matrix values for the frontend
cm_data = {
    "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
    "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    "accuracy": round(acc, 4),
}
with open("static/confusion_matrix.json", "w") as f:
    json.dump(cm_data, f)

print("✅  Saved: churn_model.pkl, static/feature_importances.json, static/confusion_matrix.json")
