import joblib
import pandas as pd
from pathlib import Path


MODEL_PATH = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "fraud_model_v2.joblib"
)

model = joblib.load(MODEL_PATH)


FEATURE_COLUMNS = [
    "Time",
    *[f"V{i}" for i in range(1, 29)],
    "Amount",
]


def predict(features: dict):
    X = pd.DataFrame([features], columns=FEATURE_COLUMNS)

    probability = model.predict_proba(X)[0][1]

    prediction = int(probability >= 0.5)

    return prediction, probability

def get_risk_level(probability: float) -> str:
    if probability >= 0.75:
        return "HIGH"

    if probability >= 0.30:
        return "MEDIUM"

    return "LOW"

def predict(features: dict):
    X = pd.DataFrame([features], columns=FEATURE_COLUMNS)

    probability = float(model.predict_proba(X)[0][1])

    prediction = probability >= 0.5
    risk_level = get_risk_level(probability)

    return {
        "is_fraud": prediction,
        "fraud_probability": probability,
        "risk_level": risk_level,
    }