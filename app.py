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
def compute_feature_importance(model, df: pd.DataFrame):
    X = df.drop(columns=["y"])
    y = df["y"].map({"no": 0, "yes": 1})
    # Limit for responsiveness
    sample = X.sample(frac=0.4, random_state=42)
    y_sample = y.loc[sample.index]
    perm = permutation_importance(
        model, sample, y_sample, n_repeats=8, scoring="roc_auc", random_state=42
    )
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    selected_mask = model.named_steps["selector"].get_support()
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

if not DATA_PATH.exists():
    st.error("Dataset not found next to the app. Please add bank-additional.csv.")
    st.stop()

df = load_data()
numeric_bounds = compute_bounds(df)

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
    input_df = user_input_features()

    left, right = st.columns(2)
    with left:
        st.subheader("Client Parameters")
        df_display = input_df.T.reset_index()
        df_display.columns = ["Feature", "Value"]
        df_display["Value"] = df_display["Value"].astype(str)
        st.dataframe(df_display, width="stretch")

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

            prediction = model.predict(cleaned_df)
            proba = model.predict_proba(cleaned_df)[0]
            prob_no, prob_yes = map(float, proba)

            verdict = "likely to SUBSCRIBE" if prediction[0] == 1 else "unlikely to subscribe"
            st.write(f"**Client is {verdict}.**")

            col_a, col_b = st.columns(2)
            col_a.metric("Probability of YES", f"{prob_yes:.1%}")
            col_b.metric("Probability of NO", f"{prob_no:.1%}")

            st.markdown("#### Probability Breakdown")
            proba_df = pd.DataFrame({"Outcome": ["No", "Yes"], "Probability": [prob_no, prob_yes]}).set_index(
                "Outcome"
            )
            st.bar_chart(proba_df)

            log_path = BASE_DIR / "user_inputs_log.csv"
            cleaned_df.assign(prediction=int(prediction[0]), prob_yes=prob_yes, prob_no=prob_no).to_csv(
                log_path, mode="a", header=not log_path.exists(), index=False
            )

with tab_eda:
    st.subheader("Dataset overview")
    st.write("Class balance")
    class_counts = df["y"].value_counts(normalize=True).rename("proportion")
    st.bar_chart(class_counts)

    if metrics:
        st.write("Model metrics")
        st.json(metrics)

    st.markdown("### Top permutation importances")
    try:
        importances = compute_feature_importance(model, df)
        st.dataframe(importances, use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not compute feature importances: {exc}")

    st.markdown("### Sample of raw data")
    st.dataframe(df.head(), use_container_width=True)


st.markdown("---")
with st.expander("About this model"):
    st.markdown(
        """
        - Trained on the Bank Marketing dataset (Moro et al., 2014)
        - Includes preprocessing (imputation, scaling, one-hot encoding) + feature selection
        - Hyperparameters tuned via randomized search optimizing ROC-AUC
        """
    )
