import pandas as pd

from src.modeling.features import DEFAULT_CUSTOMER, engineer_features
from src.modeling.predict import predict_customer


def test_engineer_features_adds_derived_columns():
    row = engineer_features({"tenure": 6, "MonthlyCharges": 50.0, "TotalCharges": 300.0})
    assert row["tenure_group"] == "0-12"
    assert row["charge_per_month"] == 50.0


def test_predict_customer_returns_probability():
    result = predict_customer(DEFAULT_CUSTOMER)
    assert "churn_probability" in result
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["risk_segment"] in {"Low", "Medium", "High"}
