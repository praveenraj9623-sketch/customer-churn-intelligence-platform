"""PySpark ETL and feature engineering for Telco Customer Churn.

Run:
    python -m src.processing.spark_pipeline
"""
from __future__ import annotations

import json
from pathlib import Path

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, StringIndexerModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, trim, when

RAW_PATH = Path("data/raw/telco_churn.csv")
PROCESSED_PATH = Path("data/processed/churn_features.parquet")
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


def build_spark_session() -> SparkSession:
    return (
        SparkSession.builder.appName("customer-churn-etl").master("local[*]").getOrCreate()
    )


def run_spark_pipeline(
    raw_path: Path = RAW_PATH,
    processed_path: Path = PROCESSED_PATH,
    indexer_path: Path = INDEXER_PATH,
) -> None:
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")

    spark = build_spark_session()
    df = spark.read.csv(str(raw_path), header=True, inferSchema=True)

    df = df.withColumn(
        "TotalCharges",
        when(trim(col("TotalCharges")) == "", None).otherwise(col("TotalCharges").cast("double")),
    )
    df = df.fillna({"TotalCharges": 0.0})

    df = df.withColumn(TARGET_COL, when(col("Churn") == "Yes", 1).otherwise(0))
    df = df.drop("Churn", "customerID")

    df = df.withColumn(
        "tenure_group",
        when(col("tenure") <= 12, "0-12")
        .when(col("tenure") <= 24, "13-24")
        .when(col("tenure") <= 48, "25-48")
        .otherwise("49+"),
    )
    df = df.withColumn(
        "charge_per_month",
        when(col("tenure") > 0, col("TotalCharges") / col("tenure")).otherwise(col("MonthlyCharges")),
    )

    indexers = [
        StringIndexer(inputCol=column, outputCol=f"{column}_idx", handleInvalid="keep")
        for column in CATEGORICAL_COLUMNS
    ]
    pipeline = Pipeline(stages=indexers)
    model = pipeline.fit(df)
    indexed_df = model.transform(df)

    for column in CATEGORICAL_COLUMNS:
        indexed_df = indexed_df.drop(column).withColumnRenamed(f"{column}_idx", column)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    # Pandas parquet write avoids winutils/HADOOP_HOME requirements on Windows.
    indexed_df.toPandas().to_parquet(processed_path, index=False)

    labels: dict[str, list[str]] = {}
    for stage in model.stages:
        if isinstance(stage, StringIndexerModel):
            labels[stage.getInputCol()] = list(stage.labelsArray[0])

    indexer_path.parent.mkdir(parents=True, exist_ok=True)
    indexer_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")

    spark.stop()
    print(f"Processed features saved to: {processed_path}")
    print(f"Indexer labels saved to: {indexer_path}")


def main() -> None:
    run_spark_pipeline()


if __name__ == "__main__":
    main()
