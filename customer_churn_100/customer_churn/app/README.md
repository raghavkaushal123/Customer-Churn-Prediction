# Customer Churn Intelligence Dashboard

## Overview

This Streamlit application deploys the customer churn analytics project.

The dashboard includes:

- Overview KPIs
- Exploratory analysis
- Customer segmentation
- Model comparison
- High-risk customer prioritization
- Explainability and SHAP insights
- Business impact calculations
- Batch CSV scoring
- Single-customer what-if churn prediction

## Requirements

Install the required packages from the project root:

```bash
pip install -r requirements.txt
```

## Run locally

From the project root folder, run:

```bash
streamlit run app/app.py
```

The application opens in a browser, normally at:

```text
http://localhost:8501
```

## Required project files

The app requires these locations:

```text
data/
charts/
models/
```

Important deployed model files:

```text
models/best_churn_model.joblib
models/calibrated_churn_probability_model.joblib
```

## Batch scoring input

For batch scoring, upload a CSV with customer attributes consistent with the original customer churn dataset. Include the service, contract, billing, payment, tenure, and charge fields used for modeling.

## Risk tiers

- Low Risk: churn probability below 0.35
- Medium Risk: churn probability from 0.35 to 0.65
- High Risk: churn probability above 0.65