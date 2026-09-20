import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from churn_features import (
    MODEL_EXCLUDED_COLUMNS,
    aggregate_shap_by_feature,
    assign_shap_drivers,
    build_model_matrix,
)

warnings.filterwarnings("ignore")

INPUT_FILE = "data/Customer-Churn-Segmented.csv"
CHART_FOLDER = "charts"
DATA_FOLDER = "data"
MODEL_FOLDER = "models"

RANDOM_STATE = 42
# A 5% hold-out left 353 records with only 94 churners, an interval on
# ROC-AUC of roughly +/-0.045 -- wider than the gap between the top models,
# so they could not be ranked. 20% gives 1409 records and ~374 churners.
TEST_SIZE = 0.20
CALIBRATION_CV_FOLDS = 5

os.makedirs(CHART_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

print("=" * 75)
print("PHASE 8: CHURN PROBABILITY PREDICTION AND EXPLAINABLE AI")
print("=" * 75)

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv(INPUT_FILE)

df["ChurnFlag"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

customer_ids = df["customerID"].copy()

# The feature matrix is built by the shared helper so every phase, and the
# dashboard, assemble identical columns.
#
# CustomerSegment is excluded. Cluster is kept instead and cast to a nominal
# code: the two carry the same information now that segment names are derived
# from centroid geometry, and passing the human-readable name as well would
# duplicate the signal. Cluster is cast to a string so the one-hot encoder
# treats it as a category rather than scaling it as though cluster 3 were
# three times cluster 1.
X = build_model_matrix(df)

columns_to_drop = MODEL_EXCLUDED_COLUMNS

y = df["ChurnFlag"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

print("\n1. DATA PREPARATION")
print(f"Split ratio: {1 - TEST_SIZE:.0%} train / {TEST_SIZE:.0%} test")
print(f"Training records: {len(X_train)}")
print(f"Test records: {len(X_test)}")
print(f"Churners in test set: {int(y_test.sum())}")
print(f"Training churn rate: {y_train.mean() * 100:.2f}%")
print(f"Test churn rate: {y_test.mean() * 100:.2f}%")

# ---------------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------------
numeric_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_transformer, numeric_features),
        ("categorical", categorical_transformer, categorical_features)
    ]
)

negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()

scale_pos_weight = negative_count / positive_count

# ---------------------------------------------------------------
# 3. Tuned XGBoost model settings from Phase 6
# ---------------------------------------------------------------
def create_xgb_model():
    return XGBClassifier(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        reg_alpha=0.05,
        reg_lambda=2,
        scale_pos_weight=scale_pos_weight * 1.15,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

base_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", create_xgb_model())
    ]
)

# ---------------------------------------------------------------
# 4. Calibrate probabilities using Platt scaling
# ---------------------------------------------------------------
print("\n2. PROBABILITY CALIBRATION")
print("Applying Platt scaling using sigmoid calibration.")

calibrated_model = CalibratedClassifierCV(
    estimator=base_pipeline,
    method="sigmoid",
    cv=CALIBRATION_CV_FOLDS
)

calibrated_model.fit(X_train, y_train)

test_probabilities = calibrated_model.predict_proba(X_test)[:, 1]

test_roc_auc = roc_auc_score(y_test, test_probabilities)
test_brier_score = brier_score_loss(y_test, test_probabilities)

print(f"Calibrated test ROC-AUC: {test_roc_auc:.4f}")
print(f"Calibrated Brier score: {test_brier_score:.4f}")

# ---------------------------------------------------------------
# 5. Calibration curve
# ---------------------------------------------------------------
observed_fraction, mean_predicted_probability = calibration_curve(
    y_test,
    test_probabilities,
    n_bins=10,
    strategy="quantile"
)

plt.figure(figsize=(8, 6))

plt.plot(
    mean_predicted_probability,
    observed_fraction,
    marker="o",
    linewidth=2,
    label="Calibrated XGBoost"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="black",
    label="Perfect Calibration"
)

plt.title("Calibration Curve: Predicted vs Observed Churn")
plt.xlabel("Mean Predicted Churn Probability")
plt.ylabel("Observed Churn Frequency")
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(CHART_FOLDER, "phase8_calibration_curve.png"),
    bbox_inches="tight"
)

plt.close()

print("Saved: charts/phase8_calibration_curve.png")

# ---------------------------------------------------------------
# 6. Train calibrated model on full dataset
# Generate probabilities for all customers
# ---------------------------------------------------------------
print("\n3. CUSTOMER CHURN PROBABILITY PREDICTIONS")

calibrated_model.fit(X, y)

all_probabilities = calibrated_model.predict_proba(X)[:, 1]

risk_tiers = pd.cut(
    all_probabilities,
    bins=[-0.01, 0.35, 0.65, 1.00],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

risk_predictions = pd.DataFrame({
    "customerID": customer_ids,
    "ChurnProbability": all_probabilities.round(4),
    "RiskTier": risk_tiers,
    "ObservedChurn": df["Churn"],
    "CustomerSegment": df["CustomerSegment"],
    "tenure": df["tenure"],
    "MonthlyCharges": df["MonthlyCharges"],
    "Contract": df["Contract"],
    "PaymentMethod": df["PaymentMethod"],
    "InternetService": df["InternetService"],
    "TechSupport": df["TechSupport"]
})

print("\nRisk-tier distribution:")
print(risk_predictions["RiskTier"].value_counts().to_string())

print("\nAverage predicted churn probability by risk tier:")
print(
    risk_predictions.groupby(
        "RiskTier",
        observed=False
    )["ChurnProbability"].mean().round(4).to_string()
)

risk_predictions.to_csv(
    os.path.join(
        DATA_FOLDER,
        "phase8_customer_churn_risk_predictions.csv"
    ),
    index=False
)

print("Saved: data/phase8_customer_churn_risk_predictions.csv")

# Risk tier chart
risk_summary = (
    risk_predictions["RiskTier"]
    .value_counts()
    .reindex(["Low Risk", "Medium Risk", "High Risk"])
    .reset_index()
)

risk_summary.columns = ["RiskTier", "Customers"]

plt.figure(figsize=(8, 5))

ax = sns.barplot(
    data=risk_summary,
    x="RiskTier",
    y="Customers",
    hue="RiskTier",
    palette={
        "Low Risk": "green",
        "Medium Risk": "orange",
        "High Risk": "crimson"
    },
    legend=False
)

plt.title("Customer Distribution by Churn Risk Tier")
plt.xlabel("Risk Tier")
plt.ylabel("Customer Count")

for container in ax.containers:
    ax.bar_label(container, padding=3)

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_FOLDER,
        "phase8_risk_tier_distribution.png"
    ),
    bbox_inches="tight"
)

plt.close()

print("Saved: charts/phase8_risk_tier_distribution.png")

# ---------------------------------------------------------------
# 7. Train final non-calibrated XGBoost for SHAP explanations
# ---------------------------------------------------------------
print("\n4. GLOBAL FEATURE IMPORTANCE")

final_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", create_xgb_model())
    ]
)

final_pipeline.fit(X_train, y_train)

fitted_preprocessor = final_pipeline.named_steps["preprocessor"]
fitted_xgb_model = final_pipeline.named_steps["model"]

feature_names = fitted_preprocessor.get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": fitted_xgb_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).head(20)

print("\nTop 20 global churn drivers:")
print(importance_df.round(4).to_string(index=False))

importance_df.to_csv(
    os.path.join(
        DATA_FOLDER,
        "phase8_global_feature_importance.csv"
    ),
    index=False
)

plt.figure(figsize=(11, 8))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature",
    color="crimson"
)

plt.title("Top 20 Global Churn Drivers")
plt.xlabel("XGBoost Feature Importance")
plt.ylabel("Feature")
plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_FOLDER,
        "phase8_global_feature_importance.png"
    ),
    bbox_inches="tight"
)

plt.close()

print("Saved: charts/phase8_global_feature_importance.png")

# ---------------------------------------------------------------
# 8. SHAP global summary
# ---------------------------------------------------------------
print("\n5. SHAP EXPLAINABLE AI")

X_test_transformed = fitted_preprocessor.transform(X_test)

if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()

sample_size = min(300, len(X_test_transformed))

sample_indices = np.random.RandomState(RANDOM_STATE).choice(
    len(X_test_transformed),
    size=sample_size,
    replace=False
)

X_shap_sample = X_test_transformed[sample_indices]

explainer = shap.TreeExplainer(fitted_xgb_model)

shap_values = explainer.shap_values(X_shap_sample)

plt.figure()

shap.summary_plot(
    shap_values,
    X_shap_sample,
    feature_names=feature_names,
    show=False,
    max_display=20
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_FOLDER,
        "phase8_shap_global_summary.png"
    ),
    bbox_inches="tight"
)

plt.close()

print("Saved: charts/phase8_shap_global_summary.png")

# ---------------------------------------------------------------
# 9. Local SHAP reason codes for highest-risk customers
# ---------------------------------------------------------------
print("\n6. LOCAL SHAP REASON CODES")

high_risk_customers = risk_predictions[
    risk_predictions["RiskTier"] == "High Risk"
].sort_values(
    by="ChurnProbability",
    ascending=False
).head(5)

high_risk_ids = high_risk_customers["customerID"].tolist()

high_risk_rows = df[
    df["customerID"].isin(high_risk_ids)
].copy()

high_risk_rows = high_risk_rows.set_index(
    "customerID"
).loc[high_risk_ids].reset_index()

# Built through the same helper as the training matrix. Constructing it by
# hand here would leave Cluster as an integer, and the fitted preprocessor
# expects the nominal string form it was trained on.
X_high_risk = build_model_matrix(high_risk_rows)

X_high_risk_transformed = fitted_preprocessor.transform(X_high_risk)

if hasattr(X_high_risk_transformed, "toarray"):
    X_high_risk_transformed = X_high_risk_transformed.toarray()

high_risk_shap_values = explainer.shap_values(
    X_high_risk_transformed
)

reason_code_rows = []

for row_position, (_, customer) in enumerate(high_risk_rows.iterrows()):
    customer_shap_values = high_risk_shap_values[row_position]

    top_feature_indexes = np.argsort(
        np.abs(customer_shap_values)
    )[-3:][::-1]

    reasons = []

    for feature_index in top_feature_indexes:
        feature_name = feature_names[feature_index]
        shap_impact = customer_shap_values[feature_index]

        direction = (
            "increased" if shap_impact > 0 else "reduced"
        )

        reasons.append(
            f"{feature_name} {direction} churn risk"
        )

    customer_probability = risk_predictions.loc[
        risk_predictions["customerID"] == customer["customerID"],
        "ChurnProbability"
    ].iloc[0]

    customer_risk_tier = risk_predictions.loc[
        risk_predictions["customerID"] == customer["customerID"],
        "RiskTier"
    ].iloc[0]

    reason_code_rows.append({
        "customerID": customer["customerID"],
        "ChurnProbability": customer_probability,
        "RiskTier": customer_risk_tier,
        "ReasonCode1": reasons[0] if len(reasons) > 0 else "Not available",
        "ReasonCode2": reasons[1] if len(reasons) > 1 else "Not available",
        "ReasonCode3": reasons[2] if len(reasons) > 2 else "Not available"
    })

reason_codes_df = pd.DataFrame(reason_code_rows)

print("\nHigh-risk customer SHAP reason codes:")
print(reason_codes_df.to_string(index=False))

reason_codes_df.to_csv(
    os.path.join(
        DATA_FOLDER,
        "phase8_high_risk_customer_reason_codes.csv"
    ),
    index=False
)

print("Saved: data/phase8_high_risk_customer_reason_codes.csv")

# ---------------------------------------------------------------
# 9b. Per-customer churn drivers for every customer
#
# Phase 9 needs a driver for each targeted customer. It previously used a
# fixed rule cascade that tested contract type first; because every
# high-risk customer is on a month-to-month contract, all of them landed in
# one category. SHAP gives each customer's own attribution instead. One-hot
# columns are summed back to their source feature, then grouped into the
# actionable levers defined in churn_features.LEVER_GROUPS.
# ---------------------------------------------------------------
print("\n6b. PER-CUSTOMER SHAP DRIVERS")

X_all_transformed = fitted_preprocessor.transform(X)

if hasattr(X_all_transformed, "toarray"):
    X_all_transformed = X_all_transformed.toarray()

all_shap_values = explainer.shap_values(X_all_transformed)

feature_shap = aggregate_shap_by_feature(
    all_shap_values,
    feature_names,
    X.columns.tolist()
)

driver_df = assign_shap_drivers(feature_shap)
driver_df.insert(0, "customerID", customer_ids.values)

driver_df.to_csv(
    os.path.join(DATA_FOLDER, "phase8_customer_risk_drivers.csv"),
    index=False
)

high_risk_driver_ids = risk_predictions.loc[
    risk_predictions["RiskTier"] == "High Risk", "customerID"
]

print("Primary lever among High Risk customers:")
print(
    driver_df[driver_df["customerID"].isin(high_risk_driver_ids)]
    ["PrimaryLever"].value_counts().to_string()
)
print("Saved: data/phase8_customer_risk_drivers.csv")

# ---------------------------------------------------------------
# 10. Save calibrated probability model
# ---------------------------------------------------------------
joblib.dump(
    calibrated_model,
    os.path.join(
        MODEL_FOLDER,
        "calibrated_churn_probability_model.joblib"
    )
)

print("\n7. OUTPUT FILES")
print("Saved: models/calibrated_churn_probability_model.joblib")
print("Saved: data/phase8_customer_churn_risk_predictions.csv")
print("Saved: data/phase8_global_feature_importance.csv")
print("Saved: data/phase8_high_risk_customer_reason_codes.csv")
print("Saved: charts/phase8_calibration_curve.png")
print("Saved: charts/phase8_risk_tier_distribution.png")
print("Saved: charts/phase8_global_feature_importance.png")
print("Saved: charts/phase8_shap_global_summary.png")

print("\n8. PHASE 8 SUMMARY")
print("- Generated calibrated churn probabilities for every customer.")
print("- Assigned Low, Medium, and High churn-risk tiers.")
print("- Verified calibration using calibration curve and Brier score.")
print("- Produced global XGBoost feature importance.")
print("- Applied SHAP for global and customer-level explanations.")
print("- Generated reason codes for the five highest-risk customers.")