"""Streamlit UI for the Bank Marketing classifier.

This version rebuilds the interface and logic to better align with the
metrics saved from the training notebook (ROC-AUC, F1, tuned threshold).
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.inspection import permutation_importance

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "bank-additional.csv"
MODEL_PATH = BASE_DIR / "artifacts" / "final_model.pkl"
METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"
LOG_PATH = BASE_DIR / "user_inputs_log.csv"

st.set_page_config(page_title="Bank Term Deposit Predictor", layout="wide")


# ---------------------------------------------------------------------
# Data/model loading
# ---------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics() -> Dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep=";")
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != "y"]
    df[cat_cols] = df[cat_cols].replace("unknown", np.nan)
    df["was_contacted"] = (df["pdays"] != 999).astype(int)
    df["pdays"] = df["pdays"].replace(999, -1)
    df["is_retired"] = (df["job"] == "retired").astype(int)
    df["eco_index"] = df["euribor3m"] * df["cons.conf.idx"]
    return df


@st.cache_data
def compute_bounds(df: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    bounds: Dict[str, Tuple[float, float]] = {}
    for col in df.select_dtypes(exclude=["object"]).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return bounds


@st.cache_data
def compute_summary(df: pd.DataFrame) -> Dict[str, float]:
    yes_rate = df["y"].map({"no": 0, "yes": 1}).mean()
    feature_count = df.shape[1] - 1
    return {"records": len(df), "yes_rate": yes_rate, "feature_count": feature_count}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def pick_threshold(metrics: Dict) -> float:
    if not metrics:
        return 0.5
    return float(
        metrics.get("threshold")
        or metrics.get("test", {}).get("threshold")
        or metrics.get("f1_threshold")
        or 0.5
    )


def build_input() -> pd.DataFrame:
    st.sidebar.header("Client configuration")

    # Client profile
    st.sidebar.subheader("Profile")
    age = st.sidebar.slider("Age", 18, 100, value=30)
    job = st.sidebar.selectbox(
        "Job",
        [
            "admin.",
            "blue-collar",
            "entrepreneur",
            "housemaid",
            "management",
            "retired",
            "self-employed",
            "services",
            "student",
            "technician",
            "unemployed",
            "unknown",
        ],
    )
    marital = st.sidebar.selectbox("Marital status", ["divorced", "married", "single", "unknown"])
    education = st.sidebar.selectbox(
        "Education",
        [
            "basic.4y",
            "basic.6y",
            "basic.9y",
            "high.school",
            "illiterate",
            "professional.course",
            "university.degree",
            "unknown",
        ],
    )

    # Credit status
    st.sidebar.subheader("Credit status")
    default = st.sidebar.selectbox("In default?", ["no", "yes", "unknown"])
    housing = st.sidebar.selectbox("Housing loan", ["no", "yes", "unknown"])
    loan = st.sidebar.selectbox("Personal loan", ["no", "yes", "unknown"])

    # Campaign details
    st.sidebar.subheader("Campaign")
    contact = st.sidebar.selectbox("Contact type", ["cellular", "telephone"])
    month = st.sidebar.selectbox("Last contact month", ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
    day_of_week = st.sidebar.selectbox("Last contact day", ["mon", "tue", "wed", "thu", "fri"])
    duration = st.sidebar.number_input("Last call duration (s)", min_value=0, value=200, step=10)
    campaign = st.sidebar.number_input("Contacts in current campaign", min_value=1, value=1, step=1)
    pdays = st.sidebar.number_input("Days since prior contact (999 = none)", min_value=0, value=999, step=1)
    previous = st.sidebar.number_input("Previous contacts", min_value=0, value=0, step=1)
    poutcome = st.sidebar.selectbox("Outcome of previous campaign", ["failure", "nonexistent", "success"])

    # Macro indicators
    st.sidebar.subheader("Economy")
    emp_var_rate = st.sidebar.number_input("Employment variation rate", value=-1.8, step=0.1, format="%.3f")
    cons_price_idx = st.sidebar.number_input("Consumer price index", value=93.0, step=0.1, format="%.3f")
    cons_conf_idx = st.sidebar.number_input("Consumer confidence index", value=-40.0, step=0.1, format="%.3f")
    euribor3m = st.sidebar.number_input("Euribor 3m", value=1.0, step=0.1, format="%.3f")
    nr_employed = st.sidebar.number_input("Number of employees", value=5000.0, step=10.0, format="%.1f")

    df = pd.DataFrame(
        {
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "month": month,
            "day_of_week": day_of_week,
            "duration": duration,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome,
            "emp.var.rate": emp_var_rate,
            "cons.price.idx": cons_price_idx,
            "cons.conf.idx": cons_conf_idx,
            "euribor3m": euribor3m,
            "nr.employed": nr_employed,
        },
        index=[0],
    )
    return df


def align_features(df_in: pd.DataFrame, bounds: Dict[str, Tuple[float, float]]) -> Tuple[pd.DataFrame, list]:
    df = df_in.replace("unknown", np.nan).copy()
    flagged = []
    for col, (lower, upper) in bounds.items():
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        val = df.loc[0, col]
        if pd.isna(val):
            continue
        if val < lower or val > upper:
            flagged.append(col)
            df.loc[0, col] = float(np.clip(val, lower, upper))

    df["was_contacted"] = (df["pdays"] != 999).astype(int)
    df["pdays"] = df["pdays"].replace(999, -1)
    df["is_retired"] = (df["job"] == "retired").astype(int)
    df["eco_index"] = df["euribor3m"] * df["cons.conf.idx"]
    return df, flagged


def predict(model, features: pd.DataFrame, threshold: float) -> Dict[str, float]:
    proba = model.predict_proba(features)[0]
    prob_no, prob_yes = map(float, proba)
    return {
        "prob_yes": prob_yes,
        "prob_no": prob_no,
        "label": int(prob_yes >= threshold),
    }


def render_header(summary: Dict, metrics: Dict):
    st.title("Bank Term Deposit Prediction")
    st.caption("Single-record scoring with the tuned classifier and decision threshold.")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records", f"{summary['records']:,}")
    col2.metric("Yes rate", f"{summary['yes_rate']:.1%}")
    roc_auc = metrics.get("test", {}).get("roc_auc") or metrics.get("roc_auc")
    f1 = metrics.get("test", {}).get("f1") or metrics.get("f1")
    thr = metrics.get("threshold") or metrics.get("test", {}).get("threshold")
    col3.metric("ROC-AUC", f"{roc_auc:.3f}" if roc_auc else "—")
    col4.metric("F1 (test)", f"{f1:.3f}" if f1 else "—")
    if thr:
        st.info(f"Decision threshold from training: {float(thr):.2f}")
    st.divider()


# ---------------------------------------------------------------------
# App execution
# ---------------------------------------------------------------------
if not MODEL_PATH.exists():
    st.error("Missing model artifact. Run the notebook to generate artifacts/final_model.pkl.")
    st.stop()
if not DATA_PATH.exists():
    st.error("Dataset bank-additional.csv not found next to the app.")
    st.stop()

model = load_model()
metrics = load_metrics()
threshold = pick_threshold(metrics)
df = load_data()
bounds = compute_bounds(df)
summary = compute_summary(df)

render_header(summary, metrics)

tab_predict, tab_model, tab_data = st.tabs(["Predict", "Model card", "Data preview"])

with tab_predict:
    user_df = build_input()
    features, flagged = align_features(user_df, bounds)

    st.subheader("Prepared features")
    st.dataframe(features.T.rename(columns={0: "value"}), hide_index=False, width="stretch")

    left, right = st.columns([0.35, 0.65], gap="large")
    with left:
        st.markdown("**Decision threshold**")
        thr_input = st.slider("Threshold", 0.0, 1.0, value=float(threshold), step=0.01)
        run = st.button("Run prediction", type="primary")
    with right:
        if run:
            result = predict(model, features, thr_input)
            verdict = "Client likely subscribes" if result["label"] == 1 else "Client unlikely to subscribe"
            st.success(verdict)
            c1, c2 = st.columns(2)
            c1.metric("P(yes)", f"{result['prob_yes']:.1%}")
            c2.metric("P(no)", f"{result['prob_no']:.1%}")
            st.progress(result["prob_yes"], text="Subscription probability")
            chart_df = pd.DataFrame(
                {"Outcome": ["No", "Yes"], "Probability": [result["prob_no"], result["prob_yes"]]}
            ).set_index("Outcome")
            st.bar_chart(chart_df)
            if flagged:
                st.warning("Clipped out-of-range numeric fields: " + ", ".join(flagged))
            features.assign(
                prediction=int(result["label"]),
                prob_yes=result["prob_yes"],
                prob_no=result["prob_no"],
                threshold=thr_input,
            ).to_csv(LOG_PATH, mode="a", header=not LOG_PATH.exists(), index=False)
            st.caption(f"Appended record to {LOG_PATH.name}")
        else:
            st.info("Choose parameters and click Run prediction.")

with tab_model:
    st.subheader("Metrics")
    if metrics:
        st.json(metrics)
    else:
        st.write("No metrics.json found; run the notebook to generate it.")

    st.markdown("**Permutation importance (sampled)**")
    try:
        sample = df.sample(frac=0.35, random_state=42)
        X = sample.drop(columns=["y"])
        y = sample["y"].map({"no": 0, "yes": 1})
        perm = permutation_importance(
            model, X, y, n_repeats=6, random_state=42, scoring="roc_auc"
        )
        feature_names = model.named_steps["preprocess"].get_feature_names_out()
        selected_mask = model.named_steps["selector"].get_support()
        selected = feature_names[selected_mask]
        n = min(len(selected), len(perm.importances_mean))
        imp_df = (
            pd.DataFrame({"feature": selected[:n], "importance": perm.importances_mean[:n]})
            .sort_values("importance", ascending=False)
            .head(15)
        )
        st.dataframe(imp_df, width="stretch")
    except Exception as exc:
        st.warning(f"Could not compute permutation importance: {exc}")

with tab_data:
    st.subheader("Sample rows")
    st.dataframe(df.head(), width="stretch")
    st.markdown("Target distribution")
    class_counts = df["y"].value_counts(normalize=True).rename("proportion")
    st.bar_chart(class_counts)

st.markdown("---")
st.caption("Rebuilt Streamlit UI powered by the tuned scikit-learn pipeline.")
