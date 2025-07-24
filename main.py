from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Fraud Detection API", version="1.0")

# Load model
model = joblib.load("model/xgboost_fraud_model.pkl")

# Request schema
class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.post("/predict")
def predict_fraud(data: Transaction):
    input_array = np.array([[getattr(data, f"V{i}") for i in range(1, 29)] + [data.Amount]])
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1]
    return {
        "fraud": bool(prediction),
        "probability": round(float(probability), 4)
    }
