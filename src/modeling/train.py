"""Train XGBoost churn model with MLflow tracking.

Run:
    python -m src.modeling.train
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.modeling.features import INDEXER_PATH, TARGET_COL

PROCESSED_PATH = Path("data/processed/churn_features.parquet")
MODEL_PATH = Path("models/churn_model.joblib")
METRICS_PATH = Path("reports/training_metrics.json")
SHAP_PATH = Path("reports/shap_summary.png")
MODEL_VERSION = "churn_xgboost_v1"
REGISTERED_MODEL_NAME = "ChurnClassifier"

HYPERPARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
}


def load_processed_data(path: Path = PROCESSED_PATH) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found: {path}. Run: python -m src.processing.spark_pipeline"
        )
    df = pd.read_parquet(path)
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])
    return X, y


def train_model() -> dict:
    load_dotenv()
    X, y = load_processed_data()
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(**HYPERPARAMS)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("telco_churn_xgboost")

    with mlflow.start_run(run_name="churn_xgboost_v1") as run:
        for key, value in HYPERPARAMS.items():
            mlflow.log_param(key, value)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "auc": float(roc_auc_score(y_test, y_proba)),
        }
        for key, value in metrics.items():
            mlflow.log_metric(key, round(value, 4))

        SHAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        sample = X_test.sample(min(500, len(X_test)), random_state=42)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, sample, show=False)
        plt.tight_layout()
        plt.savefig(SHAP_PATH, dpi=120, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(str(SHAP_PATH))

        mlflow.xgboost.log_model(model, artifact_path="model")
        model_uri = f"runs:/{run.info.run_id}/model"
        registered = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
        client = MlflowClient()
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=registered.version,
            stage="Staging",
        )

        indexer_labels = json.loads(INDEXER_PATH.read_text(encoding="utf-8"))
        bundle = {
            "model": model,
            "feature_columns": feature_columns,
            "indexer_labels": indexer_labels,
            "version": MODEL_VERSION,
            "metrics": {k: round(v, 4) for k, v in metrics.items()},
            "shap_background": sample,
        }

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH))

        METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        METRICS_PATH.write_text(json.dumps(bundle["metrics"], indent=2), encoding="utf-8")
        mlflow.log_artifact(str(METRICS_PATH))

    print("Training complete")
    print(json.dumps(bundle["metrics"], indent=2))
    print(f"Model saved to: {MODEL_PATH}")
    return bundle["metrics"]


def main() -> None:
    train_model()


if __name__ == "__main__":
    main()
