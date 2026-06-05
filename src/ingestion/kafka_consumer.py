"""Kafka consumer that scores events via FastAPI and stores results in MongoDB.

Run API:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Start infrastructure:
    docker compose up -d zookeeper kafka mongodb

Run:
    python -m src.ingestion.kafka_consumer
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv
from kafka import KafkaConsumer
from kafka.errors import KafkaError
from pymongo import MongoClient
from pymongo.errors import PyMongoError

load_dotenv()

BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
INPUT_TOPIC = os.getenv("KAFKA_TOPIC_CUSTOMER_EVENTS", "customer-events")
API_PREDICT_URL = os.getenv("API_PREDICT_URL", "http://127.0.0.1:8000/predict")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "churn_platform")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "churn_predictions")
REQUEST_TIMEOUT = int(os.getenv("API_REQUEST_TIMEOUT", "30"))


def json_deserializer(data: bytes) -> dict:
    return json.loads(data.decode("utf-8"))


def score_event(event: dict, predict_url: str) -> dict:
    payload = {k: v for k, v in event.items() if k != "Churn"}
    response = requests.post(predict_url, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def run_consumer() -> None:
    consumer: KafkaConsumer | None = None
    mongo_client: MongoClient | None = None

    try:
        consumer = KafkaConsumer(
            INPUT_TOPIC,
            bootstrap_servers=BOOTSTRAP_SERVERS,
            value_deserializer=json_deserializer,
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            group_id="churn-prediction-consumer",
            consumer_timeout_ms=None,
        )
        mongo_client = MongoClient(MONGO_URI)
        collection = mongo_client[MONGO_DB][MONGO_COLLECTION]

        print(f"Listening on topic '{INPUT_TOPIC}', storing in '{MONGO_COLLECTION}'")

        for message in consumer:
            event = message.value
            customer_id = event.get("customerID", "unknown")
            try:
                prediction = score_event(event, API_PREDICT_URL)
                document = {
                    "customerID": customer_id,
                    "input_event": event,
                    "prediction": prediction,
                    "timestamp": datetime.now(timezone.utc),
                }
                collection.insert_one(document)
                print(
                    f"Stored prediction for {customer_id}: "
                    f"prob={prediction.get('churn_probability')} "
                    f"risk={prediction.get('risk_label')}"
                )
            except requests.RequestException as exc:
                print(f"API error for {customer_id}: {exc}", file=sys.stderr)
            except PyMongoError as exc:
                print(f"MongoDB error for {customer_id}: {exc}", file=sys.stderr)
            except Exception as exc:
                print(f"Processing error for {customer_id}: {exc}", file=sys.stderr)

    except KafkaError as exc:
        print(f"Kafka consumer error: {exc}", file=sys.stderr)
        raise
    finally:
        if consumer is not None:
            consumer.close()
        if mongo_client is not None:
            mongo_client.close()


def main() -> None:
    run_consumer()


if __name__ == "__main__":
    main()
