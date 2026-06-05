"""Airflow DAG for weekly customer churn retraining."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from mlflow.tracking import MlflowClient

METRICS_PATH = Path("reports/training_metrics.json")
REGISTERED_MODEL_NAME = "ChurnClassifier"
AUC_THRESHOLD = 0.75

DEFAULT_ARGS = {
    "owner": "data-science-team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def task_run_spark_pipeline() -> None:
    from src.processing.spark_pipeline import main

    main()


def task_train_model() -> None:
    from src.modeling.train import main

    main()


def task_validate_model() -> None:
    if not METRICS_PATH.exists():
        raise RuntimeError(f"Metrics file missing: {METRICS_PATH}")

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    auc = float(metrics.get("auc", 0))
    if auc <= AUC_THRESHOLD:
        raise ValueError(f"Model AUC {auc:.4f} is below threshold {AUC_THRESHOLD}")


def task_promote_model() -> None:
    import os

    client = MlflowClient(tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        raise RuntimeError(f"No registered versions found for {REGISTERED_MODEL_NAME}")

    latest = sorted(versions, key=lambda item: int(item.version), reverse=True)[0]
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=latest.version,
        stage="Production",
    )
    print(f"Promoted {REGISTERED_MODEL_NAME} v{latest.version} to Production")


with DAG(
    dag_id="weekly_churn_retraining",
    default_args=DEFAULT_ARGS,
    description="Weekly Spark ETL, model training, validation, and MLflow promotion",
    start_date=datetime(2026, 1, 1),
    schedule="0 0 * * 0",
    catchup=False,
    tags=["churn", "ml", "xgboost"],
) as dag:

    run_spark_pipeline = PythonOperator(
        task_id="run_spark_pipeline",
        python_callable=task_run_spark_pipeline,
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=task_train_model,
    )

    validate_model = PythonOperator(
        task_id="validate_model",
        python_callable=task_validate_model,
    )

    promote_model = PythonOperator(
        task_id="promote_model",
        python_callable=task_promote_model,
    )

    run_spark_pipeline >> train_model >> validate_model >> promote_model
