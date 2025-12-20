# ADA 442 Project – Bank Term Deposit Prediction

This project implements the final assignment for **ADA 442 Statistical Learning | Classification** (Author: Dr. Hakan Emekci, v23.01.07, created 11 Nov 2025) based on the Bank Marketing dataset by Moro et al. (2014) \(`https://archive.ics.uci.edu/ml/datasets/Bank+Marketing`\).

The repo contains:
- `bank-additional.csv` – 10% subset of the Bank Marketing dataset (20 input features).
- `project.ipynb` – Jupyter notebook for data cleaning, preprocessing, feature engineering, model selection, and training.
- `final_model.pkl` – Trained model pipeline saved from the notebook (used by the Streamlit app).
- `app.py` – Streamlit app for deploying the final model.

## Setup

Create and activate a virtual environment (recommended), then install dependencies:

```bash
cd /Users/kaancakir/ADA442Project
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Workflow

1. **Open and run the notebook**
   - Launch Jupyter:
     ```bash
     jupyter notebook
     ```
   - Open `project.ipynb`.
   - Run all cells in order.
    - The notebook will:
     - Load `bank-additional.csv` (10% subset, 4119 rows, 20 inputs).
     - Perform data cleaning, preprocessing (encoding + scaling), and basic feature engineering.
     - Build a `scikit-learn` pipeline with preprocessing + model.
     - Train and compare **≥3 models** (e.g., logistic regression, random forest, gradient boosting / neural net).
     - Select the best model using evaluation metrics (e.g., F1, ROC-AUC) and **hyperparameter tuning**.
     - Save the best pipeline to `final_model.pkl` using `joblib.dump`.

2. **Run the Streamlit app**

   After `final_model.pkl` is created in the same folder as `app.py`, run:

   ```bash
   cd /Users/kaancakir/ADA442Project
   streamlit run app.py
   ```

   This will open the **Bank Term Deposit Prediction App** in your browser. Use the sidebar controls to enter client information and press **“Predict Subscription”** to see the model’s prediction and probabilities.

## What process was applied (and why)

- **Data cleaning:** Replace `"unknown"` with missing values, clip numeric outliers using IQR bounds to keep inputs realistic for the model and UI.
- **Preprocessing pipeline:** One-hot encode categoricals and scale numeric features so tree-based and linear models receive comparable feature magnitudes.
- **Feature selection:** Model-based selector retains the most informative engineered features, reducing noise and overfitting risk.
- **Model comparison:** Evaluated multiple classifiers (logistic regression, random forest, gradient boosting, etc.) on ROC-AUC/F1 to balance ranking quality and minority-class recall.
- **Hyperparameter tuning:** Randomized/ grid search on key parameters (trees, depth, learning rate) to squeeze extra lift without overfitting.
- **Final choice:** Gradient Boosting selected for best ROC-AUC with competitive F1 on the hold-out split; saved as `final_model.pkl` for deployment.
- **Deployment:** Streamlit app wraps the pipeline, validates inputs (clips out-of-range numerics, treats `"unknown"` as missing), and logs user runs for traceability.

## Notes

- If you get an error like `EOFError` or “file not found” when loading `final_model.pkl`, rerun the notebook to regenerate and save the model.
- Make sure your notebook clearly documents (per assignment requirements):
  - **Data cleaning** and handling data quality issues.
  - **Preprocessing**: categorical encoding, numerical scaling.
  - **Feature engineering / selection** where appropriate.
  - **Model selection**: compare at least three models with metrics.
  - **Hyperparameter tuning** (e.g., `GridSearchCV`) for the selected model.
  - **Evaluation** on held-out test data.
  - **Pipeline creation** covering the full process.
- Deliverables (zip as `Group_0XX.zip`): `project.ipynb`, `presentation.ppt` (≤5 slides), `report.pdf` (≤2 pages, include Streamlit cloud URL).
- Grading weights: Data Cleaning 10%, Preprocessing 10%, Feature Engineering 10%, Model Selection 10%, Pipeline 15%, Presentation 15%, Deployment 25%.
- Deadline per spec: **21 Dec 2025, 11:59 PM**. No late submissions.
- Include the Streamlit app URL and cite sources properly. Academic honesty rules apply.


