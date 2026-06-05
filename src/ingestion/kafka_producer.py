"""Kafka producer simulating real-time customer events.

Start Kafka:
    docker compose up -d zookeeper kafka

Run:
    python -m src.ingestion.kafka_producer
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from kafka import KafkaProducer
from kafka.errors import KafkaError

load_dotenv()

RAW_DATA_PATH = Path(os.getenv("RAW_DATA_PATH", "data/raw/telco_churn.csv"))
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC_CUSTOMER_EVENTS", "customer-events")
DELAY_SECONDS = float(os.getenv("KAFKA_PRODUCER_DELAY", "0.5"))


def json_serializer(data: dict) -> bytes:
    return json.dumps(data, default=str).encode("utf-8")


def run_producer(
    raw_path: Path = RAW_DATA_PATH,
    bootstrap_servers: str = BOOTSTRAP_SERVERS,
    topic: str = TOPIC,
    delay_seconds: float = DELAY_SECONDS,
) -> None:
    if not raw_path.exists():
        raise FileNotFoundError(f"Dataset not found: {raw_path}")

    df = pd.read_csv(raw_path)
    producer: KafkaProducer | None = None

    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=json_serializer,
            retries=5,
            request_timeout_ms=30000,
        )
        print(f"Publishing {len(df)} events to topic '{topic}' ({delay_seconds}s delay)")

        for index, row in df.iterrows():
            event = row.fillna("").to_dict()
            future = producer.send(topic, event)
            future.get(timeout=10)
            print(f"[{index + 1}/{len(df)}] Sent customerID={event.get('customerID')}")
            time.sleep(delay_seconds)

        producer.flush()
        print("Producer finished successfully")
    except KafkaError as exc:
        print(f"Kafka error: {exc}", file=sys.stderr)
        raise
    except Exception as exc:
        print(f"Producer failed: {exc}", file=sys.stderr)
        raise
    finally:
        if producer is not None:
            producer.close()


def main() -> None:
    run_producer()


if __name__ == "__main__":
    main()
