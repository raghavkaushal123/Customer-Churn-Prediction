# Customer Churn Prediction & Retention Analytics

An end-to-end machine learning and data analytics system for predicting customer churn, performing customer segmentation, evaluating model performance, explaining predictions (XAI), and generating data-driven business retention strategies.

---

## 📌 Project Overview

This repository contains full implementation workflows for predicting customer churn using telecom / customer subscription dataset:
- **Phase 1: Data Understanding & Ingestion**
- **Phase 2: Data Cleaning & Preprocessing**
- **Phase 3: Exploratory Data Analysis (EDA) & Visualizations**
- **Phase 4: Feature Engineering & Domain Indicators**
- **Phase 5: Customer Segmentation (K-Means Clustering & Silhouette Analysis)**
- **Phase 6: Model Development (Baseline vs Enhanced Models)**
- **Phase 7: Comprehensive Model Evaluation (ROC, PR Curves, Threshold Tradeoffs)**
- **Phase 8: Explainable AI (SHAP & Calibrated Risk Scoring)**
- **Phase 9: Business Retention Strategy & Financial Impact Quantification**
- **Interactive Web App**: Web interface for churn prediction and analytics.

---

## 📁 Repository Structure

```
.
├── customer_churn_100/
│   └── customer_churn/
│       ├── app/                         # Interactive Web Application
│       ├── charts/                      # Generated Analytical Charts & Figures
│       ├── data/                        # Processed Datasets & Risk Profiles
│       ├── models/                      # Saved Model Artifacts (.joblib)
│       ├── churn_features.py            # Feature definitions & transformations
│       ├── phase1_data_understanding.py
│       ├── phase2_data_cleaning.py
│       ├── phase3_eda.py
│       ├── phase4_feature_engineering.py
│       ├── phase5_customer_segmentation.py
│       ├── phase6_model_development.py
│       ├── phase7_model_evaluation.py
│       ├── phase8_churn_probability_xai.py
│       ├── phase9_business_recommendations.py
│       └── requirements.txt
├── churn_project/                       # Baseline pipeline implementation
├── figures/                             # Architecture & workflow diagrams
├── figures_v2/                          # Pipeline lineage & defect diagrams
├── Report_*.pdf / *.docx                # Executive Reports & Implementation Documents
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

Ensure Python 3.8+ is installed:

```bash
cd customer_churn_100/customer_churn
pip install -r requirements.txt
```

---

## 📊 Analytics & Machine Learning Pipeline

1. **Exploratory Analysis & Feature Engineering**: Run `phase1` through `phase4` scripts to clean data and generate predictive customer retention signals.
2. **Segmentation & Clustering**: Run `phase5_customer_segmentation.py` to identify customer profiles based on tenure and usage patterns.
3. **Model Training & XAI**: Run `phase6` through `phase8` scripts to train XGBoost/RandomForest models, generate SHAP explanations, and calibrate churn probabilities.
4. **Business Action Plan**: Run `phase9_business_recommendations.py` to estimate financial impact and prioritized retention targets.

---

## 📄 Documentation & Reports

Detailed technical documentation and executive reports are included as PDF and DOCX files in the root directory.
