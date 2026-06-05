# Customer Churn Platform

A production-style customer churn analytics project for Cursor / VS Code.

This project includes:

- Telco churn dataset in `data/raw/telco_churn.csv`
- PySpark ETL pipeline
- XGBoost training with MLflow tracking
- Evaluation metrics and optional SHAP explanation output
- FastAPI prediction endpoint
- Streamlit business dashboard
- Kafka producer/consumer for simulated real-time customer events
- Django admin skeleton for customer records
- Airflow DAG skeleton for weekly retraining

---

## 1. Open in Cursor

1. Extract the ZIP file.
2. Open Cursor.
3. Click **File → Open Folder**.
4. Select the `customer-churn-platform` folder.
5. Open Cursor terminal.

---

## 2. Create virtual environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Mac/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Note: Airflow and PySpark are heavy packages. If installation is slow, first install only the core packages for model + dashboard:
>
> `pip install pandas numpy scikit-learn xgboost shap fastapi uvicorn streamlit plotly python-dotenv joblib mlflow`

---

## 3. Train the churn model

```bash
python -m src.modeling.train
```

This creates:

- `models/churn_model.joblib`
- `reports/training_metrics.json`
- `models/sample_customer.json`

---

## 4. Evaluate the model

```bash
python -m src.modeling.evaluate
```

This creates:

- `reports/evaluation_metrics.json`
- `reports/feature_importance.csv`

---

## 5. Run Streamlit dashboard

```bash
streamlit run app.py
```

Open the local URL shown in terminal, usually:

`http://localhost:8501`

---

## 6. Run FastAPI prediction API

```bash
uvicorn src.api.main:app --reload
```

Open:

`http://127.0.0.1:8000/docs`

Sample endpoint:

`POST /predict`

---

## 7. Optional: Run infrastructure

Start Kafka, MongoDB, and MLflow server:

```bash
docker compose up -d
```

MLflow UI:

`http://localhost:5000`

---

## 8. Optional: Kafka real-time simulation

Terminal 1:

```bash
python -m src.ingestion.kafka_consumer
```

Terminal 2:

```bash
python -m src.ingestion.kafka_producer
```

---

## 9. Project explanation for interview

This is a production-style churn prediction platform. It starts from raw customer data, performs cleaning and feature engineering, trains a machine learning model, evaluates churn risk, exposes predictions through an API, shows business insights in a dashboard, and simulates real-time scoring using Kafka.

Business use case:

- Identify customers likely to churn.
- Help retention teams prioritize high-risk customers.
- Understand important churn drivers such as contract type, tenure, monthly charges, and service usage.
- Turn model predictions into actionable retention decisions.

---

## Important note

This project is designed as a strong fresher-level portfolio project. Do not claim it as company work experience. Present it as academic/project-based data-domain experience.
