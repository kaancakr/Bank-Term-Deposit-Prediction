import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# Page Configuration
st.set_page_config(
    page_title="Bank Term Deposit Predictor",
    page_icon="🏦",
    layout="wide"
)

# 1. Load the trained model
# The model file 'final_model.pkl' must be in the same directory.
# If you used a specific name in your notebook, update it here.
try:
    model = joblib.load('final_model.pkl')
except FileNotFoundError:
    st.error("Model file 'final_model.pkl' not found. Please run the Jupyter Notebook to train and save the model first.")
    st.stop()

# 1b. Load training data to compute numeric bounds for outlier filtering
DATA_PATH = Path(__file__).parent / "bank-additional.csv"
numeric_bounds = {}

if DATA_PATH.exists():
    try:
        train_df = pd.read_csv(DATA_PATH, sep=";")
        numeric_cols = train_df.select_dtypes(exclude=["object"]).columns
        bounds = {}
        for col in numeric_cols:
            q1 = train_df[col].quantile(0.25)
            q3 = train_df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            bounds[col] = (lower, upper)
        numeric_bounds = bounds
    except Exception as e:
        st.warning(f"Could not compute bounds from dataset: {e}")
else:
    st.warning("bank-additional.csv not found; outlier filtering will be skipped.")

# 2. App Title and Description
st.title("🏦 Bank Term Deposit Prediction App")
st.markdown(
    """
    Use this tool to **estimate whether a client will subscribe to a term deposit** based on
    their profile, contact history, and macro‑economic indicators.

    - Adjust inputs from the **left sidebar**
    - Click **“Predict Subscription”** to see the predicted outcome and probabilities
    - Use the results to support **targeting and campaign planning** decisions
    """
)

st.markdown("---")

# 3. Sidebar - User Input Features
st.sidebar.title("🔧 Input Panel")
st.sidebar.markdown(
    """
    Configure a **single client scenario** below.

    Fields are grouped as:
    - Client profile
    - Financial status
    - Last contact information
    - Campaign history
    - Economic indicators
    """
)

def user_input_features():
    # --- Bank Client Data ---
    st.sidebar.subheader("Client Profile")
    age = st.sidebar.slider('Age', 18, 100, 30)
    job = st.sidebar.selectbox('Job', 
        ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
         'retired', 'self-employed', 'services', 'student', 'technician', 
         'unemployed', 'unknown'])
    marital = st.sidebar.selectbox('Marital Status', 
        ['divorced', 'married', 'single', 'unknown'])
    education = st.sidebar.selectbox('Education', 
        ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 
         'professional.course', 'university.degree', 'unknown'])
    
    # --- Financial Status ---
    st.sidebar.subheader("Financial Status")
    default = st.sidebar.selectbox('Credit in Default?', ['no', 'yes', 'unknown'])
    housing = st.sidebar.selectbox('Housing Loan?', ['no', 'yes', 'unknown'])
    loan = st.sidebar.selectbox('Personal Loan?', ['no', 'yes', 'unknown'])

    # --- Last Contact Info ---
    st.sidebar.subheader("Last Contact Info")
    contact = st.sidebar.selectbox('Contact Communication Type', ['cellular', 'telephone'])
    month = st.sidebar.selectbox('Last Contact Month', 
        ['mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    day_of_week = st.sidebar.selectbox('Last Contact Day', 
        ['mon', 'tue', 'wed', 'thu', 'fri'])
    duration = st.sidebar.number_input('Duration of last call (seconds)', min_value=0, value=200, 
        help="Important: If duration is 0, the target is usually 'no'.")

    # --- Campaign Info ---
    st.sidebar.subheader("Campaign History")
    campaign = st.sidebar.number_input('Number of contacts (current campaign)', min_value=1, value=1)
    pdays = st.sidebar.number_input('Days since last contact (previous)', value=999, 
        help="999 means client was not previously contacted")
    previous = st.sidebar.number_input('Number of contacts (previous)', min_value=0, value=0)
    poutcome = st.sidebar.selectbox('Outcome of previous campaign', 
        ['failure', 'nonexistent', 'success'])
    
    # --- Socio-Economic Indicators ---
    st.sidebar.subheader("Economic Indicators")
    emp_var_rate = st.sidebar.number_input('Employment Variation Rate', value=-1.8)
    cons_price_idx = st.sidebar.number_input('Consumer Price Index', value=93.0)
    cons_conf_idx = st.sidebar.number_input('Consumer Confidence Index', value=-40.0)
    euribor3m = st.sidebar.number_input('Euribor 3 Month Rate', value=1.0)
    nr_employed = st.sidebar.number_input('Number of Employees', value=5000.0)

    # Combine all inputs into a dictionary
    data = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'month': month,
        'day_of_week': day_of_week,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome,
        'emp.var.rate': emp_var_rate,
        'cons.price.idx': cons_price_idx,
        'cons.conf.idx': cons_conf_idx,
        'euribor3m': euribor3m,
        'nr.employed': nr_employed
    }
    # Convert to DataFrame (single row)
    features = pd.DataFrame(data, index=[0])
    return features

def sanitize_input(df: pd.DataFrame, bounds: dict):
    """Clip numeric values to IQR-based bounds; return cleaned df and list of flagged cols."""
    if not bounds:
        return df.copy(), []
    cleaned = df.copy()
    flagged = []
    for col, (lower, upper) in bounds.items():
        if col not in cleaned.columns:
            continue
        # Ensure numeric dtype to avoid pandas dtype warning when clipping
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
        val = cleaned.loc[0, col]
        if pd.isna(val):
            continue
        if val < lower or val > upper:
            flagged.append(col)
            cleaned.loc[0, col] = float(np.clip(val, lower, upper))
    return cleaned, flagged

# Get input from the user
input_df = user_input_features()

# 4. Main Panel - Display Prediction
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Client Parameters")
    st.caption("All selected / entered values for the current client.")
    # Reshape for a clean two-column table to avoid mixed-type Arrow issues
    df_display = input_df.T.reset_index()
    df_display.columns = ["Feature", "Value"]
    df_display["Value"] = df_display["Value"].astype(str)
    st.dataframe(df_display, width="stretch")

with col2:
    st.subheader("Prediction Result")
    st.caption("Model output based on the trained classification pipeline.")

    predict_clicked = st.button('🔮 Predict Subscription', width="stretch")

    if predict_clicked:
        # Filter outliers in numeric inputs (clip to IQR bounds)
        cleaned_df, flagged_cols = sanitize_input(input_df, numeric_bounds)

        if flagged_cols:
            st.warning(
                "Some numeric fields were outside expected range and were clipped: "
                + ", ".join(flagged_cols)
            )

        # Make prediction
        # Make prediction
        prediction = model.predict(cleaned_df)
        prediction_proba = model.predict_proba(cleaned_df)

        prob_no = float(prediction_proba[0][0])
        prob_yes = float(prediction_proba[0][1])

        # Display main verdict
        if prediction[0] == 1:
            st.success("✅ The client is **likely to SUBSCRIBE** to the term deposit.")
        else:
            st.error("❌ The client is **unlikely to subscribe** to the term deposit.")

        # Key probability metrics
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            st.metric("Probability of YES", f"{prob_yes:.1%}")
        with mcol2:
            st.metric("Probability of NO", f"{prob_no:.1%}")

        # Simple bar chart for probabilities
        st.markdown("#### Probability Breakdown")
        proba_df = pd.DataFrame(
            {"Outcome": ["No", "Yes"], "Probability": [prob_no, prob_yes]}
        ).set_index("Outcome")
        st.bar_chart(proba_df)

        # Persist sanitized input for audit/logging
        log_path = Path(__file__).parent / "user_inputs_log.csv"
        cleaned_df.assign(prediction=int(prediction[0]), prob_yes=prob_yes, prob_no=prob_no).to_csv(
            log_path, mode="a", header=not log_path.exists(), index=False
        )

# 5. Additional information / help
st.markdown("---")
with st.expander("ℹ️ How to interpret the prediction", expanded=False):
    st.markdown(
        """
        - **Probability of YES** shows how confident the model is that the client will subscribe.
        - **Probability of NO** shows the confidence that the client will not subscribe.
        - Use these probabilities comparatively: higher **YES** probability suggests a better
          candidate for targeted marketing.
        """
    )

with st.expander("📊 About this model"):
    st.markdown(
        """
        - Trained on the **Bank Marketing** dataset by Moro et al. (2014)
        - Built for the ADA 442 Statistical Learning project
        - Uses a scikit‑learn pipeline with preprocessing + classification model
        """
    )
