"""FastAPI service for customer churn predictions.

Run:
    uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.modeling.features import (
    DEFAULT_CUSTOMER,
    customer_to_feature_frame,
    risk_label,
)

MODEL_PATH = Path("models/churn_model.joblib")
METRICS_PATH = Path("reports/training_metrics.json")
API_VERSION = "1.0.0"

app = FastAPI(
    title="Customer Churn Intelligence API",
    description="Predict churn risk with SHAP explanations.",
    version=API_VERSION,
)


class CustomerPayload(BaseModel):
    customerID: Optional[str] = Field(default="DEMO-0001")
    gender: str = Field(default="Male")
    SeniorCitizen: int = Field(default=0)
    Partner: str = Field(default="No")
    Dependents: str = Field(default="No")
    tenure: int = Field(default=12)
    PhoneService: str = Field(default="Yes")
    MultipleLines: str = Field(default="No")
    InternetService: str = Field(default="Fiber optic")
    OnlineSecurity: str = Field(default="No")
    OnlineBackup: str = Field(default="No")
    DeviceProtection: str = Field(default="No")
    TechSupport: str = Field(default="No")
    StreamingTV: str = Field(default="Yes")
    StreamingMovies: str = Field(default="Yes")
    Contract: str = Field(default="Month-to-month")
    PaperlessBilling: str = Field(default="Yes")
    PaymentMethod: str = Field(default="Electronic check")
    MonthlyCharges: float = Field(default=79.85)
    TotalCharges: float = Field(default=958.20)


@lru_cache(maxsize=1)
def load_bundle() -> Dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run: python -m src.modeling.train"
        )
    return joblib.load(MODEL_PATH)


def top_shap_features(
    model,
    features: pd.DataFrame,
    feature_columns: List[str],
    top_k: int = 3,
) -> List[Dict[str, float]]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    row_shap = np.array(shap_values).reshape(-1)
    ranked = sorted(
        zip(feature_columns, row_shap),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    return [
        {"feature": name, "shap_value": round(float(value), 4)}
        for name, value in ranked[:top_k]
    ]


@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        bundle = load_bundle()
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_version": bundle.get("version", API_VERSION),
            "api_version": API_VERSION,
        }
    except FileNotFoundError:
        return {
            "status": "degraded",
            "model_loaded": False,
            "model_version": None,
            "api_version": API_VERSION,
            "message": "Model artifact missing. Run training pipeline first.",
        }


@app.post("/predict")
def predict(payload: CustomerPayload) -> Dict[str, Any]:
    try:
        bundle = load_bundle()
        model = bundle["model"]
        feature_columns: List[str] = bundle["feature_columns"]
        indexer_labels = bundle["indexer_labels"]

        features = customer_to_feature_frame(
            payload.model_dump(),
            feature_columns=feature_columns,
            indexer_labels=indexer_labels,
        )
        probability = float(model.predict_proba(features)[0, 1])
        shap_top = top_shap_features(model, features, feature_columns, top_k=3)

        return {
            "customerID": payload.customerID,
            "churn_probability": round(probability, 4),
            "risk_label": risk_label(probability),
            "top_shap_features": shap_top,
            "model_version": bundle.get("version", API_VERSION),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.get("/batch-stats")
def batch_stats() -> Dict[str, Any]:
    bundle: Optional[Dict[str, Any]] = None
    try:
        bundle = load_bundle()
    except FileNotFoundError:
        bundle = None

    metrics: Dict[str, Any] = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    elif bundle and "metrics" in bundle:
        metrics = bundle["metrics"]

    if not metrics:
        raise HTTPException(
            status_code=404,
            detail="No training metrics found. Run: python -m src.modeling.train",
        )

    return {
        "model_version": bundle.get("version") if bundle else None,
        "metrics": metrics,
        "registered_model": "ChurnClassifier",
        "registry_stage": "Staging",
        "sample_customer": DEFAULT_CUSTOMER,
    }
