"""Evaluate saved churn model and create model interpretation files.

Run:
    python -m src.modeling.evaluate
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from src.modeling.train import RAW_DATA_PATH, TARGET, load_dataset

MODEL_PATH = Path("models/churn_model.joblib")
METRICS_PATH = Path("reports/evaluation_metrics.json")
FEATURE_IMPORTANCE_PATH = Path("reports/feature_importance.csv")


def evaluate_model() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python -m src.modeling.train")

    df = load_dataset(RAW_DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].map({"No": 0, "Yes": 1})

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "confusion_matrix": cm,
        "classification_report": report,
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Feature importance after one-hot encoding.
    try:
        preprocessor = model.named_steps["preprocessor"]
        xgb_model = model.named_steps["model"]
        feature_names = preprocessor.get_feature_names_out()
        importances = xgb_model.feature_importances_
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)
        importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    except Exception as exc:
        print(f"Feature importance export skipped: {exc}")

    print(json.dumps({"roc_auc": metrics["roc_auc"], "confusion_matrix": cm}, indent=2))
    print(f"Evaluation metrics saved to: {METRICS_PATH}")
    return metrics


if __name__ == "__main__":
    evaluate_model()
