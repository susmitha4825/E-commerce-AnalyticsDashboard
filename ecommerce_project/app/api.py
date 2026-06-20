"""
app/api.py  — FastAPI backend for E-commerce Order Prediction
Run: uvicorn app.api:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import os

# ── Load saved model artifacts ──────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(__file__))

with open(os.path.join(BASE, "models", "loan_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE, "models", "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(BASE, "models", "feature_names.pkl"), "rb") as f:
    feature_names = pickle.load(f)

imputer_path = os.path.join(BASE, "models", "imputer.pkl")
imputer = pickle.load(open(imputer_path, "rb")) if os.path.exists(imputer_path) else None

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="E-commerce Order Predictor",
    description="Predicts whether an order will be Delivered or Returned/Cancelled",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request schema ────────────────────────────────────────────────────────────
class OrderInput(BaseModel):
    customer_age: float       # e.g. 28
    quantity: float           # e.g. 2
    unit_price: float         # e.g. 1500.0
    discount_pct: float       # e.g. 10.0
    discount_amount: float    # e.g. 150.0
    revenue: float            # e.g. 2850.0
    month: int                # 1–12
    category_enc: int         # encoded category (0–5)
    city_enc: int             # encoded city (0–9)
    customer_gender_enc: int  # 0 = Female, 1 = Male
    payment_method_enc: int   # 0–4
    product_enc: int          # encoded product

# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "E-commerce Order Predictor API is running!"}


@app.get("/features")
def get_features():
    """Return expected feature names in order."""
    return {"features": feature_names}


@app.post("/predict")
def predict(order: OrderInput):
    """
    Predict order outcome.
    Returns: prediction label + confidence probability.
    """
    features = np.array([[
        order.customer_age,
        order.quantity,
        order.unit_price,
        order.discount_pct,
        order.discount_amount,
        order.revenue,
        order.month,
        order.category_enc,
        order.city_enc,
        order.customer_gender_enc,
        order.payment_method_enc,
        order.product_enc,
    ]])

    if imputer is not None:
        features = imputer.transform(features)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]

    label = "Delivered / Shipped" if prediction == 1 else "Returned / Cancelled"
    confidence = round(float(max(probability)) * 100, 2)

    return {
        "prediction": int(prediction),
        "label": label,
        "confidence": confidence,
        "probabilities": {
            "returned_cancelled": round(float(probability[0]) * 100, 2),
            "delivered_shipped":  round(float(probability[1]) * 100, 2),
        }
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
