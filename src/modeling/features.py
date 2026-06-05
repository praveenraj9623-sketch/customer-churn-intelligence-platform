"""Shared feature engineering for inference (matches Spark pipeline output)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

INDEXER_PATH = Path("data/processed/indexer_labels.json")
TARGET_COL = "churn_label"

CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_group",
]

NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "charge_per_month",
]

DEFAULT_CUSTOMER: Dict[str, Any] = {
    "customerID": "DEMO-0001",
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 79.85,
    "TotalCharges": 958.20,
}


def load_indexer_labels(path: Path = INDEXER_PATH) -> Dict[str, List[str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Indexer labels not found at {path}. Run: python -m src.processing.spark_pipeline"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def tenure_group_label(tenure: int) -> str:
    if tenure <= 12:
        return "0-12"
    if tenure <= 24:
        return "13-24"
    if tenure <= 48:
        return "25-48"
    return "49+"


def encode_category(value: str, labels: List[str]) -> float:
    """StringIndexer uses alphabetical label order in Spark."""
    text = str(value)
    if text in labels:
        return float(labels.index(text))
    return float(len(labels))


def engineer_features(customer: Dict[str, Any]) -> Dict[str, Any]:
    row = {**DEFAULT_CUSTOMER, **customer}
    tenure = int(pd.to_numeric(row.get("tenure", 0), errors="coerce") or 0)
    monthly = float(pd.to_numeric(row.get("MonthlyCharges", 0), errors="coerce") or 0)
    total = float(pd.to_numeric(row.get("TotalCharges", 0), errors="coerce") or 0)

    row["tenure"] = tenure
    row["SeniorCitizen"] = int(pd.to_numeric(row.get("SeniorCitizen", 0), errors="coerce") or 0)
    row["MonthlyCharges"] = monthly
    row["TotalCharges"] = total
    row["tenure_group"] = tenure_group_label(tenure)
    row["charge_per_month"] = total / tenure if tenure > 0 else monthly
    return row


def customer_to_feature_frame(
    customer: Dict[str, Any],
    feature_columns: List[str],
    indexer_labels: Dict[str, List[str]],
) -> pd.DataFrame:
    row = engineer_features(customer)
    encoded: Dict[str, float] = {}

    for col in NUMERIC_COLUMNS:
        encoded[col] = float(row[col])

    for col in CATEGORICAL_COLUMNS:
        encoded[col] = encode_category(row.get(col, DEFAULT_CUSTOMER.get(col, "")), indexer_labels[col])

    frame = pd.DataFrame([encoded])
    return frame[feature_columns]


def risk_label(probability: float) -> str:
    if probability < 0.33:
        return "Low"
    if probability < 0.66:
        return "Medium"
    return "High"
