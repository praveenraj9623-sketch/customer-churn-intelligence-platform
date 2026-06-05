"""Inference helpers used by tests and local scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd

from src.modeling.features import (
    DEFAULT_CUSTOMER,
    customer_to_feature_frame,
    risk_label,
)

MODEL_PATH = Path("models/churn_model.joblib")


def load_model(model_path: Path = MODEL_PATH) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run: python -m src.modeling.train"
        )
    return joblib.load(model_path)


def predict_customer(customer: Dict[str, Any], model_path: Path = MODEL_PATH) -> Dict[str, Any]:
    bundle = load_model(model_path)
    features = customer_to_feature_frame(
        customer,
        feature_columns=bundle["feature_columns"],
        indexer_labels=bundle["indexer_labels"],
    )
    probability = float(bundle["model"].predict_proba(features)[0, 1])
    return {
        "customerID": customer.get("customerID", DEFAULT_CUSTOMER["customerID"]),
        "churn_probability": round(probability, 4),
        "risk_segment": risk_label(probability),
        "model_version": bundle.get("version"),
    }


def predict_customers(customers: pd.DataFrame, model_path: Path = MODEL_PATH) -> pd.DataFrame:
    rows = []
    for record in customers.to_dict(orient="records"):
        rows.append(predict_customer(record, model_path=model_path))
    return pd.DataFrame(rows)
