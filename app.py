"""Premium Streamlit dashboard for Customer Churn Intelligence Platform.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix

from src.modeling.features import (
    DEFAULT_CUSTOMER,
    customer_to_feature_frame,
    risk_label,
)

load_dotenv()

DATA_PATH = Path("data/raw/telco_churn.csv")
METRICS_PATH = Path("reports/training_metrics.json")
MODEL_PATH = Path("models/churn_model.joblib")
SHAP_PATH = Path("reports/shap_summary.png")
API_PREDICT_URL = os.getenv("API_PREDICT_URL", "http://127.0.0.1:8000/predict")
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

RISK_ORDER = ["Low", "Medium", "High"]
RISK_COLORS = {"Low": "#16a34a", "Medium": "#f59e0b", "High": "#dc2626"}
CHURN_COLORS = {"No": "#2563eb", "Yes": "#ef4444"}
PRIMARY = "#ef4444"
DARK = "#111827"
MUTED = "#6b7280"
CARD_BG = "#ffffff"
PAGE_BG = "#f8fafc"

st.set_page_config(
    page_title="Churn Intelligence Platform",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# UI helpers
# -----------------------------
def inject_css() -> None:
    """Apply modern dashboard styling."""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(180deg, {PAGE_BG} 0%, #ffffff 42%);
        }}
        .block-container {{
            padding-top: 2.2rem;
            padding-bottom: 3rem;
            max-width: 1320px;
        }}
        div[data-testid="stMetric"] {{
            background: {CARD_BG};
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 18px 18px 14px 18px;
            box-shadow: 0 12px 32px rgba(15, 23, 42, 0.06);
        }}
        div[data-testid="stMetric"] label {{
            color: {MUTED};
            font-size: 0.86rem;
            font-weight: 700;
        }}
        div[data-testid="stMetricValue"] {{
            color: {DARK};
            font-weight: 800;
        }}
        .hero-card {{
            border-radius: 26px;
            padding: 30px 34px;
            margin-bottom: 18px;
            background:
                radial-gradient(circle at top right, rgba(239, 68, 68, 0.18), transparent 26%),
                linear-gradient(135deg, #111827 0%, #1f2937 52%, #991b1b 100%);
            color: white;
            box-shadow: 0 18px 50px rgba(15, 23, 42, 0.18);
        }}
        .hero-title {{
            font-size: 2.2rem;
            line-height: 1.12;
            font-weight: 900;
            letter-spacing: -0.04em;
            margin: 0;
        }}
        .hero-subtitle {{
            margin-top: 10px;
            color: #e5e7eb;
            font-size: 1rem;
        }}
        .stack-pill {{
            display: inline-block;
            padding: 7px 10px;
            margin: 14px 8px 0 0;
            border-radius: 999px;
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.18);
            color: #fff;
            font-size: 0.82rem;
            font-weight: 700;
        }}
        .section-note {{
            padding: 14px 16px;
            border-radius: 16px;
            background: #fff7ed;
            border: 1px solid #fed7aa;
            color: #9a3412;
            font-size: 0.92rem;
            margin: 10px 0 16px 0;
        }}
        .success-card {{
            border-radius: 18px;
            padding: 18px 20px;
            background: #ecfdf5;
            border: 1px solid #bbf7d0;
            color: #14532d;
            font-size: 1.02rem;
            font-weight: 750;
        }}
        .warning-card {{
            border-radius: 18px;
            padding: 18px 20px;
            background: #fff7ed;
            border: 1px solid #fed7aa;
            color: #9a3412;
            font-size: 1.02rem;
            font-weight: 750;
        }}
        .danger-card {{
            border-radius: 18px;
            padding: 18px 20px;
            background: #fef2f2;
            border: 1px solid #fecaca;
            color: #991b1b;
            font-size: 1.02rem;
            font-weight: 750;
        }}
        .small-muted {{
            color: {MUTED};
            font-size: 0.88rem;
        }}
        .stTabs [data-baseweb="tab-list"] {{
            gap: 12px;
            border-bottom: 1px solid #e5e7eb;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 999px 999px 0 0;
            padding: 12px 16px;
            font-weight: 800;
        }}
        .stButton > button {{
            border-radius: 14px;
            min-height: 44px;
            font-weight: 800;
            box-shadow: 0 10px 24px rgba(239, 68, 68, 0.16);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Customer Churn Intelligence Platform</div>
            <div class="hero-subtitle">
                Executive churn analytics, risk segmentation, live prediction API scoring, MLflow model tracking, and SHAP explainability.
            </div>
            <span class="stack-pill">PySpark ETL</span>
            <span class="stack-pill">XGBoost</span>
            <span class="stack-pill">MLflow</span>
            <span class="stack-pill">FastAPI</span>
            <span class="stack-pill">Streamlit</span>
            <span class="stack-pill">SHAP</span>
            <span class="stack-pill">Kafka-ready</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def polish_chart(fig: go.Figure, height: int = 420, show_legend: bool = True) -> go.Figure:
    """Apply consistent Plotly styling."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=24, r=24, t=70, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(font=dict(size=18, color=DARK, family="Arial Black"), x=0.02, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=show_legend,
        font=dict(color="#334155", size=12),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eef2f7", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eef2f7", zeroline=False)
    return fig


def risk_badge_html(label: str, probability: float) -> str:
    cls = "success-card"
    if label == "Medium":
        cls = "warning-card"
    elif label == "High":
        cls = "danger-card"
    return f"""
    <div class="{cls}">
        Risk: {label} · Churn probability: {probability:.1%}
    </div>
    """


# -----------------------------
# Data / model helpers
# -----------------------------
@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["ChurnLabel"] = df["Churn"].map({"No": 0, "Yes": 1})
    return df


@st.cache_resource(show_spinner=False)
def load_model_bundle() -> dict[str, Any] | None:
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def load_metrics() -> dict[str, float]:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    bundle = load_model_bundle()
    if bundle and "metrics" in bundle:
        return bundle["metrics"]
    return {}


@st.cache_data(show_spinner="Scoring all customers for risk segmentation...")
def score_customer_risks(sample_size: int | None = None) -> pd.DataFrame:
    """Score either full dataset or an optional sample.

    Default is the full Telco dataset. This avoids showing only 1,500 customers
    in the risk segment page when the uploaded dataset has 7,043 customers.
    """
    bundle = load_model_bundle()
    if bundle is None:
        return pd.DataFrame()

    raw_df = load_raw_data()
    if sample_size is None:
        df = raw_df.copy()
    else:
        df = raw_df.sample(n=min(sample_size, len(raw_df)), random_state=42)

    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    indexer_labels = bundle["indexer_labels"]
    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        customer = row.to_dict()
        features = customer_to_feature_frame(customer, feature_columns, indexer_labels)
        prob = float(model.predict_proba(features)[0, 1])
        rows.append(
            {
                "customerID": customer.get("customerID"),
                "churn_probability": prob,
                "risk_label": risk_label(prob),
                "Contract": customer.get("Contract"),
                "tenure": customer.get("tenure"),
                "MonthlyCharges": customer.get("MonthlyCharges"),
                "InternetService": customer.get("InternetService"),
            }
        )
    return pd.DataFrame(rows)


def feature_importance_df(bundle: dict[str, Any], top_n: int = 15) -> pd.DataFrame:
    model = bundle["model"]
    importance = model.feature_importances_
    columns = bundle["feature_columns"]
    return (
        pd.DataFrame({"feature": columns, "importance": importance})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )


def local_predict(customer: dict[str, Any], bundle: dict[str, Any] | None) -> dict[str, Any] | None:
    """Fallback predictor when FastAPI is not running."""
    if bundle is None:
        return None
    features = customer_to_feature_frame(customer, bundle["feature_columns"], bundle["indexer_labels"])
    prob = float(bundle["model"].predict_proba(features)[0, 1])
    top_features = (
        feature_importance_df(bundle, top_n=3)
        .rename(columns={"importance": "shap_value"})
        .to_dict(orient="records")
    )
    return {"churn_probability": prob, "risk_label": risk_label(prob), "top_shap_features": top_features}


def normalize_prediction_result(result: dict[str, Any] | None, bundle: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Normalize different FastAPI/local prediction response schemas.

    Older API versions may return `risk_segment` instead of `risk_label`, and
    some versions may return `probability` instead of `churn_probability`.
    This helper prevents KeyError crashes in the dashboard.
    """
    if not result or not isinstance(result, dict):
        return None

    probability = (
        result.get("churn_probability")
        if result.get("churn_probability") is not None
        else result.get("probability")
    )
    if probability is None:
        probability = result.get("prediction_probability") or result.get("churn_prob")

    if probability is None:
        return None

    probability = float(probability)

    # Accept both labels used across the project: risk_label and risk_segment.
    label = result.get("risk_label") or result.get("risk_segment") or risk_label(probability)

    # Normalize the top-driver explanation to a table-friendly format.
    drivers = result.get("top_shap_features") or result.get("top_drivers") or result.get("drivers") or []
    if not drivers and bundle is not None:
        drivers = (
            feature_importance_df(bundle, top_n=3)
            .rename(columns={"importance": "shap_value"})
            .to_dict(orient="records")
        )

    return {
        "churn_probability": probability,
        "risk_label": str(label),
        "top_shap_features": drivers,
        "raw_response": result,
    }


# -----------------------------
# Chart helpers
# -----------------------------
def make_contract_bar(df: pd.DataFrame) -> go.Figure:
    churn_contract = (
        df.groupby("Contract", as_index=False)["ChurnLabel"]
        .mean()
        .assign(**{"Churn Rate %": lambda x: x["ChurnLabel"] * 100})
    )
    order = ["Month-to-month", "One year", "Two year"]
    churn_contract["Contract"] = pd.Categorical(churn_contract["Contract"], categories=order, ordered=True)
    churn_contract = churn_contract.sort_values("Contract")
    fig = px.bar(
        churn_contract,
        x="Contract",
        y="Churn Rate %",
        text="Churn Rate %",
        title="Churn rate by contract type",
        color="Contract",
        color_discrete_map={
            "Month-to-month": "#ef4444",
            "One year": "#f59e0b",
            "Two year": "#16a34a",
        },
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_line_width=0)
    fig.update_layout(yaxis_title="Churn rate", xaxis_title="Contract")
    return polish_chart(fig, height=430, show_legend=False)


def make_monthly_charge_hist(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="MonthlyCharges",
        color="Churn",
        nbins=42,
        marginal="box",
        opacity=0.78,
        title="Monthly charges distribution by churn status",
        color_discrete_map=CHURN_COLORS,
    )
    fig.update_layout(barmode="overlay", xaxis_title="Monthly charges", yaxis_title="Customer count")
    return polish_chart(fig, height=430)


def make_tenure_scatter(df: pd.DataFrame) -> go.Figure:
    plot_df = df.sample(n=min(1800, len(df)), random_state=42)
    fig = px.scatter(
        plot_df,
        x="tenure",
        y="MonthlyCharges",
        color="Churn",
        size="TotalCharges",
        size_max=14,
        opacity=0.68,
        title="Tenure vs monthly charges, sized by lifetime value",
        color_discrete_map=CHURN_COLORS,
        hover_data=["customerID", "Contract", "InternetService", "TotalCharges"],
    )
    fig.update_layout(xaxis_title="Tenure in months", yaxis_title="Monthly charges")
    return polish_chart(fig, height=520)


def make_churn_donut(df: pd.DataFrame) -> go.Figure:
    counts = df["Churn"].value_counts().reset_index()
    counts.columns = ["Churn", "customers"]
    fig = go.Figure(
        data=[
            go.Pie(
                labels=counts["Churn"],
                values=counts["customers"],
                hole=0.58,
                pull=[0.03 if x == "Yes" else 0 for x in counts["Churn"]],
                marker=dict(colors=[CHURN_COLORS.get(x, "#64748b") for x in counts["Churn"]], line=dict(color="#ffffff", width=4)),
                textinfo="label+percent",
                textfont=dict(size=13, color="white"),
                hovertemplate="%{label}<br>Customers: %{value:,}<br>Share: %{percent}<extra></extra>",
            )
        ]
    )
    fig.add_annotation(
        text=f"{df['ChurnLabel'].mean() * 100:.1f}%<br><span style='font-size:12px;color:#64748b'>churn</span>",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=26, color=DARK),
    )
    fig.update_layout(title="Churn composition", annotations=fig.layout.annotations)
    return polish_chart(fig, height=430)


def make_risk_donut(risk_counts: pd.DataFrame, total_customers: int) -> go.Figure:
    risk_counts = risk_counts.copy()
    risk_counts["risk_label"] = pd.Categorical(risk_counts["risk_label"], categories=RISK_ORDER, ordered=True)
    risk_counts = risk_counts.sort_values("risk_label")
    pulls = [0.02, 0.05, 0.10]

    # Plotly does not have a true 3D pie chart. This is a clean 3D-style donut:
    # thick ring, pulled high-risk slice, strong border, and center annotation.
    fig = go.Figure(
        data=[
            go.Pie(
                labels=risk_counts["risk_label"],
                values=risk_counts["count"],
                hole=0.48,
                sort=False,
                pull=pulls,
                rotation=120,
                marker=dict(
                    colors=[RISK_COLORS.get(x, "#64748b") for x in risk_counts["risk_label"]],
                    line=dict(color="#ffffff", width=5),
                ),
                textinfo="label+percent",
                textfont=dict(size=14, color="white"),
                hovertemplate="Risk: %{label}<br>Customers: %{value:,}<br>Share: %{percent}<extra></extra>",
            )
        ]
    )
    fig.add_annotation(
        text=f"{total_customers:,}<br><span style='font-size:12px;color:#64748b'>customers scored</span>",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=24, color=DARK),
    )
    fig.update_layout(title="3D-style risk segment distribution", annotations=fig.layout.annotations)
    return polish_chart(fig, height=520)


def make_feature_importance_bar(importance_df: pd.DataFrame, title: str = "Top churn drivers") -> go.Figure:
    plot_df = importance_df.sort_values("importance", ascending=True)
    fig = px.bar(
        plot_df,
        x="importance",
        y="feature",
        orientation="h",
        text="importance",
        title=title,
        color="importance",
        color_continuous_scale=["#dbeafe", "#2563eb", "#ef4444"],
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside", marker_line_width=0)
    fig.update_layout(coloraxis_showscale=False, xaxis_title="Importance", yaxis_title="Feature")
    return polish_chart(fig, height=520, show_legend=False)


def make_gauge(probability: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=probability * 100,
            delta={"reference": 50, "suffix": " pts", "increasing": {"color": "#dc2626"}, "decreasing": {"color": "#16a34a"}},
            number={"suffix": "%", "font": {"size": 56, "color": DARK}},
            title={"text": "Churn probability", "font": {"size": 20, "color": DARK}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748b"},
                "bar": {"color": PRIMARY, "thickness": 0.22},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#e5e7eb",
                "steps": [
                    {"range": [0, 33], "color": "#dcfce7"},
                    {"range": [33, 66], "color": "#fef3c7"},
                    {"range": [66, 100], "color": "#fee2e2"},
                ],
                "threshold": {"line": {"color": "#991b1b", "width": 4}, "thickness": 0.8, "value": probability * 100},
            },
        )
    )
    return polish_chart(fig, height=420, show_legend=False)


def make_confusion_matrix(cm: Any) -> go.Figure:
    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Customers"),
        x=["No churn", "Churn"],
        y=["No churn", "Churn"],
        title="Confusion matrix - full processed dataset",
        color_continuous_scale="Blues",
    )
    fig.update_traces(textfont=dict(size=18, color="white"))
    return polish_chart(fig, height=460, show_legend=False)


# -----------------------------
# Pages
# -----------------------------
def business_overview_tab(df: pd.DataFrame) -> None:
    st.subheader("📊 Business Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Churn rate", f"{df['ChurnLabel'].mean() * 100:.1f}%")
    c3.metric("Avg monthly charge", f"${df['MonthlyCharges'].mean():.2f}")
    c4.metric("Monthly revenue at risk", f"${df.loc[df['Churn'] == 'Yes', 'MonthlyCharges'].sum():,.0f}")

    st.markdown(
        """
        <div class="section-note">
            Business insight: month-to-month contract customers show the highest churn exposure. Retention teams should prioritize contract upgrades, support quality, and high monthly charge customers.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.plotly_chart(make_contract_bar(df), use_container_width=True)
    with right:
        st.plotly_chart(make_churn_donut(df), use_container_width=True)

    left2, right2 = st.columns([1, 1], gap="large")
    with left2:
        st.plotly_chart(make_monthly_charge_hist(df), use_container_width=True)
    with right2:
        payment_churn = (
            df.groupby("PaymentMethod", as_index=False)["ChurnLabel"]
            .mean()
            .assign(churn_rate=lambda x: x["ChurnLabel"] * 100)
            .sort_values("churn_rate", ascending=True)
        )
        fig = px.bar(
            payment_churn,
            x="churn_rate",
            y="PaymentMethod",
            orientation="h",
            text="churn_rate",
            title="Churn rate by payment method",
            color="churn_rate",
            color_continuous_scale=["#dcfce7", "#f59e0b", "#ef4444"],
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, xaxis_title="Churn rate", yaxis_title="Payment method")
        st.plotly_chart(polish_chart(fig, height=430, show_legend=False), use_container_width=True)

    st.plotly_chart(make_tenure_scatter(df), use_container_width=True)


def risk_segments_tab(bundle: dict[str, Any] | None, df: pd.DataFrame) -> None:
    st.subheader("🎯 Risk Segments")
    if bundle is None:
        st.warning("Train the model first: `python -m src.modeling.train`")
        return

    scored = score_customer_risks(sample_size=None)
    if scored.empty:
        st.info("No scored customers available yet.")
        return

    total_scored = len(scored)
    risk_counts = scored["risk_label"].value_counts().reindex(RISK_ORDER, fill_value=0).reset_index()
    risk_counts.columns = ["risk_label", "count"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers scored", f"{total_scored:,}")
    k2.metric("High risk", f"{int(risk_counts.loc[risk_counts['risk_label'] == 'High', 'count'].iloc[0]):,}")
    k3.metric("Medium risk", f"{int(risk_counts.loc[risk_counts['risk_label'] == 'Medium', 'count'].iloc[0]):,}")
    k4.metric("Avg model probability", f"{scored['churn_probability'].mean():.1%}")

    st.markdown(
        """
        <div class="section-note">
            The risk segment view now scores the full uploaded Telco dataset, not only a sample. High-risk customers should be prioritized for retention calls, offers, and support intervention.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([0.9, 1.1], gap="large")
    with left:
        st.plotly_chart(make_risk_donut(risk_counts, total_scored), use_container_width=True)
    with right:
        segment_stats = (
            scored.groupby("risk_label")
            .agg(
                customers=("customerID", "count"),
                avg_probability=("churn_probability", "mean"),
                avg_tenure=("tenure", "mean"),
                avg_monthly_charge=("MonthlyCharges", "mean"),
            )
            .reindex(RISK_ORDER)
            .reset_index()
        )
        st.dataframe(
            segment_stats,
            use_container_width=True,
            hide_index=True,
            column_config={
                "risk_label": st.column_config.TextColumn("Risk segment"),
                "customers": st.column_config.NumberColumn("Customers", format="%d"),
                "avg_probability": st.column_config.ProgressColumn("Avg churn probability", min_value=0, max_value=1, format="%.1%"),
                "avg_tenure": st.column_config.NumberColumn("Avg tenure", format="%.1f"),
                "avg_monthly_charge": st.column_config.NumberColumn("Avg monthly charge", format="$%.2f"),
            },
        )

        high_risk = scored.sort_values("churn_probability", ascending=False).head(12)
        st.markdown("**Top high-risk customers to prioritize**")
        st.dataframe(
            high_risk[["customerID", "churn_probability", "Contract", "tenure", "MonthlyCharges", "InternetService"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "churn_probability": st.column_config.ProgressColumn("Churn probability", min_value=0, max_value=1, format="%.1%"),
                "MonthlyCharges": st.column_config.NumberColumn("Monthly charge", format="$%.2f"),
            },
        )

    if SHAP_PATH.exists():
        st.image(str(SHAP_PATH), caption="Top churn drivers - SHAP summary", use_column_width=True)
    else:
        st.info("SHAP plot will appear after training.")

    top_drivers = feature_importance_df(bundle).head(10)
    st.plotly_chart(make_feature_importance_bar(top_drivers, "Top 10 churn drivers - feature importance"), use_container_width=True)


def live_predictions_tab(bundle: dict[str, Any] | None) -> None:
    st.subheader("⚡ Live Predictions")
    st.markdown(
        """
        <div class="section-note">
            Enter customer profile details and score the customer using the FastAPI prediction service. If the API is not running, the dashboard will try local model fallback.
        </div>
        """,
        unsafe_allow_html=True,
    )

    customer = DEFAULT_CUSTOMER.copy()

    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        st.markdown("**Customer profile**")
        customer["gender"] = st.selectbox("Gender", ["Male", "Female"])
        customer["SeniorCitizen"] = st.selectbox("Senior citizen", [0, 1])
        customer["Partner"] = st.selectbox("Partner", ["Yes", "No"])
        customer["Dependents"] = st.selectbox("Dependents", ["Yes", "No"])
        customer["tenure"] = st.slider("Tenure (months)", 0, 72, 12)
    with col2:
        st.markdown("**Service details**")
        customer["InternetService"] = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
        customer["Contract"] = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        customer["PaymentMethod"] = st.selectbox(
            "Payment method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        customer["PaperlessBilling"] = st.selectbox("Paperless billing", ["Yes", "No"])
    with col3:
        st.markdown("**Billing and support**")
        customer["MonthlyCharges"] = st.number_input("Monthly charges", min_value=0.0, value=79.85, step=5.0)
        customer["TotalCharges"] = st.number_input("Total charges", min_value=0.0, value=958.20, step=50.0)
        customer["PhoneService"] = st.selectbox("Phone service", ["Yes", "No"])
        customer["TechSupport"] = st.selectbox("Tech support", ["Yes", "No", "No internet service"])

    clicked = st.button("Score customer", type="primary", use_container_width=True)
    if not clicked:
        return

    result = None
    api_used = True
    try:
        response = requests.post(API_PREDICT_URL, json=customer, timeout=20)
        response.raise_for_status()
        result = response.json()
    except requests.RequestException:
        api_used = False
        result = local_predict(customer, bundle)

    if result is None:
        st.error("Prediction failed. Start API with `uvicorn src.api.main:app --port 8000` or train/load the local model.")
        return

    normalized = normalize_prediction_result(result, bundle)
    if normalized is None:
        st.error(
            "Prediction returned an unexpected response format. "
            "Open http://127.0.0.1:8000/docs and test /predict once, or restart FastAPI."
        )
        with st.expander("Show raw prediction response"):
            st.json(result)
        return

    probability = normalized["churn_probability"]
    label = normalized["risk_label"]
    st.markdown(risk_badge_html(label, probability), unsafe_allow_html=True)
    st.caption("Prediction source: FastAPI" if api_used else "Prediction source: local model fallback")

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.plotly_chart(make_gauge(probability), use_container_width=True)
    with right:
        drivers = pd.DataFrame(normalized.get("top_shap_features", []))
        if not drivers.empty:
            if "importance" in drivers.columns and "shap_value" not in drivers.columns:
                drivers = drivers.rename(columns={"importance": "shap_value"})
            drivers["impact_direction"] = drivers["shap_value"].apply(lambda x: "Increases churn risk" if x > 0 else "Reduces churn risk")
            st.markdown("**Top drivers for this prediction**")
            st.dataframe(
                drivers[["feature", "shap_value", "impact_direction"]],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "feature": st.column_config.TextColumn("Feature"),
                    "shap_value": st.column_config.NumberColumn("Impact value", format="%.4f"),
                    "impact_direction": st.column_config.TextColumn("Interpretation"),
                },
            )
        else:
            st.info("Top driver explanation is unavailable for this prediction.")


def model_performance_tab(bundle: dict[str, Any] | None, metrics: dict[str, float]) -> None:
    st.subheader("🧪 Model Performance")
    if not metrics:
        st.warning("No metrics yet. Run `python -m src.modeling.train`")
        return

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
    m2.metric("AUC", f"{metrics.get('auc', 0):.3f}")
    m3.metric("F1", f"{metrics.get('f1', 0):.3f}")
    m4.metric("Precision", f"{metrics.get('precision', 0):.3f}")
    m5.metric("Recall", f"{metrics.get('recall', 0):.3f}")

    with st.expander("How to explain these metrics in an interview", expanded=False):
        st.markdown(
            """
            - **AUC** shows how well the model separates churners from non-churners across thresholds.
            - **Precision** tells how many predicted churn customers were actually churn customers.
            - **Recall** tells how many actual churn customers the model captured.
            - **F1** balances precision and recall.
            - For churn prevention, recall is important because missing likely churners can hurt revenue.
            """
        )

    if bundle is None:
        return

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.plotly_chart(make_feature_importance_bar(feature_importance_df(bundle), "Model feature importance"), use_container_width=True)
    with right:
        processed_path = Path("data/processed/churn_features.parquet")
        if processed_path.exists():
            pdata = pd.read_parquet(processed_path)
            X = pdata.drop(columns=["churn_label"])
            y = pdata["churn_label"].astype(int)
            preds = bundle["model"].predict(X)
            cm = confusion_matrix(y, preds)
            st.plotly_chart(make_confusion_matrix(cm), use_container_width=True)

    st.markdown("**Latest MLflow runs**")
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        runs = mlflow.search_runs(experiment_names=["telco_churn_xgboost"], max_results=5)
        if runs.empty:
            st.info("No MLflow runs found yet.")
        else:
            display_cols = [
                c
                for c in runs.columns
                if c.startswith("metrics.") or c.startswith("params.") or c == "run_id"
            ]
            st.dataframe(runs[display_cols], use_container_width=True, hide_index=True)
    except Exception as exc:
        st.info(f"MLflow table unavailable: {exc}")


# -----------------------------
# Main app
# -----------------------------
def main() -> None:
    inject_css()
    hero()

    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=62)
        st.markdown("### Churn Control Center")
        st.caption("Local portfolio dashboard")
        st.divider()
        st.markdown("**Runtime status**")
        st.write("✅ Streamlit dashboard")
        st.write("✅ Trained model" if MODEL_PATH.exists() else "⚠️ Model missing")
        st.write("✅ Processed data" if Path("data/processed/churn_features.parquet").exists() else "⚠️ Processed data missing")
        st.write("✅ Metrics" if METRICS_PATH.exists() else "⚠️ Metrics missing")
        st.divider()
        st.caption("API URL")
        st.code(API_PREDICT_URL, language="text")

    df = load_raw_data()
    metrics = load_metrics()
    bundle = load_model_bundle()

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Business Overview", "🎯 Risk Segments", "⚡ Live Predictions", "🧪 Model Performance"]
    )

    with tab1:
        business_overview_tab(df)
    with tab2:
        risk_segments_tab(bundle, df)
    with tab3:
        live_predictions_tab(bundle)
    with tab4:
        model_performance_tab(bundle, metrics)


if __name__ == "__main__":
    main()
