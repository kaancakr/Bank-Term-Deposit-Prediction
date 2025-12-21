import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.inspection import permutation_importance

# ----------------------------------------------------------------------
# Paths and caching helpers
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "bank-additional.csv"
MODEL_PATH = BASE_DIR / "artifacts" / "final_model.pkl"
METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"


st.set_page_config(page_title="Bank Term Deposit Predictor", page_icon="🏦", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, sep=";")
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    cat_cols.remove("y")
    df[cat_cols] = df[cat_cols].replace("unknown", np.nan)
    # Derive was_contacted; keep pdays for model
    df["was_contacted"] = (df["pdays"] != 999).astype(int)
    df["pdays"] = df["pdays"].replace(999, -1)
    return df


@st.cache_data
def compute_bounds(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(exclude=["object"]).columns
    bounds = {}
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return bounds


@st.cache_data
def compute_summary(df: pd.DataFrame):
    yes_rate = df["y"].map({"no": 0, "yes": 1}).mean()
    feature_count = df.shape[1] - 1  # exclude target
    return {
        "records": len(df),
        "yes_rate": yes_rate,
        "feature_count": feature_count,
    }


@st.cache_data
def compute_feature_importance(_model, df: pd.DataFrame):
    X = df.drop(columns=["y"])
    y = df["y"].map({"no": 0, "yes": 1})
    # Limit for responsiveness
    sample = X.sample(frac=0.4, random_state=42)
    y_sample = y.loc[sample.index]
    perm = permutation_importance(
        _model, sample, y_sample, n_repeats=8, scoring="roc_auc", random_state=42
    )
    feature_names = _model.named_steps["preprocess"].get_feature_names_out()
    selected_mask = _model.named_steps["selector"].get_support()
    selected_features = feature_names[selected_mask]
    importances = (
        pd.DataFrame({"feature": selected_features, "importance": perm.importances_mean})
        .sort_values("importance", ascending=False)
        .head(15)
    )
    return importances


# ----------------------------------------------------------------------
# Load resources
# ----------------------------------------------------------------------
if not MODEL_PATH.exists():
    st.error("Model artifact not found. Please run the notebook to train and save the model.")
    st.stop()

model = load_model()
metrics = load_metrics()
decision_threshold = 0.5
if metrics:
    decision_threshold = float(metrics.get("threshold", decision_threshold))

if not DATA_PATH.exists():
    st.error("Dataset not found next to the app. Please add bank-additional.csv.")
    st.stop()

df = load_data()
numeric_bounds = compute_bounds(df)
summary_stats = compute_summary(df)

# ----------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------
st.title("🏦 Bank Term Deposit Prediction App")
st.markdown(
    """
Use this tool to estimate whether a client will subscribe to a term deposit based on
their profile, contact history, and macro‑economic indicators.
"""
)

with st.container():
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Records", f"{summary_stats['records']:,}")
    col_b.metric("Yes Rate", f"{summary_stats['yes_rate']:.1%}")
    col_c.metric("Features", f"{summary_stats['feature_count']}")
    col_d.metric("F1 (test)", f"{metrics.get('test', {}).get('f1', metrics.get('f1', '—'))}")
st.divider()


def user_input_features():
    st.sidebar.title("🔧 Input Panel")
    st.sidebar.caption("Configure a single client scenario.")

    st.sidebar.subheader("Client Profile")
    age = st.sidebar.slider("Age", 18, 100, 30)
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
    marital = st.sidebar.selectbox("Marital Status", ["divorced", "married", "single", "unknown"])
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

    st.sidebar.subheader("Financial Status")
    default = st.sidebar.selectbox("Credit in Default?", ["no", "yes", "unknown"])
    housing = st.sidebar.selectbox("Housing Loan?", ["no", "yes", "unknown"])
    loan = st.sidebar.selectbox("Personal Loan?", ["no", "yes", "unknown"])

    st.sidebar.subheader("Last Contact Info")
    contact = st.sidebar.selectbox("Contact Communication Type", ["cellular", "telephone"])
    month = st.sidebar.selectbox(
        "Last Contact Month", ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    )
    day_of_week = st.sidebar.selectbox("Last Contact Day", ["mon", "tue", "wed", "thu", "fri"])
    duration = st.sidebar.number_input(
        "Duration of last call (seconds)", min_value=0, value=200, help="If duration is 0, target is usually no."
    )

    st.sidebar.subheader("Campaign History")
    campaign = st.sidebar.number_input("Number of contacts (current campaign)", min_value=1, value=1)
    pdays = st.sidebar.number_input(
        "Days since last contact (previous)", value=999, help="999 means client was not previously contacted"
    )
    previous = st.sidebar.number_input("Number of contacts (previous)", min_value=0, value=0)
    poutcome = st.sidebar.selectbox("Outcome of previous campaign", ["failure", "nonexistent", "success"])

    st.sidebar.subheader("Economic Indicators")
    emp_var_rate = st.sidebar.number_input("Employment Variation Rate", value=-1.8)
    cons_price_idx = st.sidebar.number_input("Consumer Price Index", value=93.0)
    cons_conf_idx = st.sidebar.number_input("Consumer Confidence Index", value=-40.0)
    euribor3m = st.sidebar.number_input("Euribor 3 Month Rate", value=1.0)
    nr_employed = st.sidebar.number_input("Number of Employees", value=5000.0)

    return pd.DataFrame(
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


def sanitize_input(df_in: pd.DataFrame, bounds: dict):
    cleaned = df_in.replace("unknown", np.nan).copy()
    flagged = []
    for col, (lower, upper) in bounds.items():
        if col not in cleaned.columns:
            continue
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
        val = cleaned.loc[0, col]
        if pd.isna(val):
            continue
        if val < lower or val > upper:
            flagged.append(col)
            cleaned.loc[0, col] = float(np.clip(val, lower, upper))
    return cleaned, flagged


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
tab_predict, tab_eda = st.tabs(["🔮 Predict", "📊 EDA & Feature Insights"])

with tab_predict:
    st.markdown("Adjust inputs on the left, then run a prediction below.")
    st.caption("Values labeled 'unknown' are treated as missing and handled by the model.")
    st.divider()
    input_df = user_input_features()
    # Derive was_contacted from pdays to match training pipeline
    input_df["was_contacted"] = (input_df["pdays"] != 999).astype(int)
    input_df["pdays"] = input_df["pdays"].replace(999, -1)

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Client Parameters")
        df_display = input_df.T.reset_index()
        df_display.columns = ["Feature", "Value"]
        df_display["Value"] = df_display["Value"].astype(str)
        st.dataframe(df_display, width="stretch", hide_index=True)

    with right:
        st.subheader("Prediction Result")
        predict_clicked = st.button("Run Prediction", type="primary")

        if predict_clicked:
            cleaned_df, flagged_cols = sanitize_input(input_df, numeric_bounds)
            if flagged_cols:
                st.warning(
                    "Some numeric fields were outside expected range and were clipped: "
                    + ", ".join(flagged_cols)
                )

            proba = model.predict_proba(cleaned_df)[0]
            prob_no, prob_yes = map(float, proba)
            pred_label = int(prob_yes >= decision_threshold)

            verdict = "likely to SUBSCRIBE" if pred_label == 1 else "unlikely to subscribe"
            st.success(f"Client is {verdict} (threshold={decision_threshold:.2f}).")

            col_a, col_b = st.columns(2)
            col_a.metric("Probability of YES", f"{prob_yes:.1%}")
            col_b.metric("Probability of NO", f"{prob_no:.1%}")

            st.progress(prob_yes, text="Subscription probability")

            st.markdown("#### Probability Breakdown")
            proba_df = pd.DataFrame({"Outcome": ["No", "Yes"], "Probability": [prob_no, prob_yes]}).set_index(
                "Outcome"
            )
            st.bar_chart(proba_df)

            log_path = BASE_DIR / "user_inputs_log.csv"
            cleaned_df.assign(prediction=int(pred_label), prob_yes=prob_yes, prob_no=prob_no).to_csv(
                log_path, mode="a", header=not log_path.exists(), index=False
            )
            st.caption("Inputs and prediction appended to user_inputs_log.csv")
        else:
            st.info("Click Run Prediction to generate a score for the configured client.")

with tab_eda:
    st.subheader("Dataset overview")
    class_counts = df["y"].value_counts(normalize=True).rename("proportion")
    col1, col2 = st.columns([0.65, 0.35], gap="large")
    with col1:
        st.write("Class balance")
        st.bar_chart(class_counts)
    with col2:
        st.write("Quick stats")
        st.metric("Records", f"{summary_stats['records']:,}")
        st.metric("Yes Rate", f"{summary_stats['yes_rate']:.1%}")
        if metrics:
            roc_auc = metrics.get("test", {}).get("roc_auc") or metrics.get("roc_auc")
            f1 = (
                metrics.get("test", {}).get("f1")
                or metrics.get("f1")
                or metrics.get("cv_f1")
                or metrics.get("f1_default")
            )
            if roc_auc is not None:
                st.metric("ROC-AUC", f"{roc_auc:.3f}")
            if f1 is not None:
                st.metric("F1 Score", f"{f1:.3f}")
            if metrics.get("threshold") is not None:
                st.metric("Decision Threshold", f"{metrics['threshold']:.2f}")

    if metrics:
        st.write("Model metrics")
        st.json(metrics)

    st.markdown("### Top permutation importances")
    try:
        importances = compute_feature_importance(model, df)
        st.dataframe(importances, width="stretch")
    except Exception as exc:
        st.warning(f"Could not compute feature importances: {exc}")

    st.markdown("### Sample of raw data")
    st.dataframe(df.head(), width="stretch")


st.markdown("---")
with st.expander("About this model"):
    st.markdown(
        """
        - Trained on the Bank Marketing dataset (Moro et al., 2014)
        - Includes preprocessing (imputation, scaling, one-hot encoding) + feature selection
        - Hyperparameters tuned via randomized search optimizing ROC-AUC
        """
    )
