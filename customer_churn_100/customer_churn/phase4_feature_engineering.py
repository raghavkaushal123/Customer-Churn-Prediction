import os

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from churn_features import ENGINEERED_FEATURE_COLUMNS, add_engineered_features

INPUT_FILE = "data/Customer-Churn-Cleaned.csv"
OUTPUT_FILE = "data/Customer-Churn-Model-Ready.csv"

# Unscaled, un-encoded engineered features. This is the artefact phase 5 merges
# into the segmented dataset so the features actually reach the models. The
# scaled OUTPUT_FILE above is retained for inspection only: phases 6 to 8 must
# receive raw features, because they fit their own scaler inside a
# leakage-safe ColumnTransformer.
FEATURES_FILE = "data/Customer-Churn-Features-Unscaled.csv"
ARTIFACTS_FILE = "models/feature_engineering_artifacts.joblib"

MODEL_FOLDER = "models"

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"

os.makedirs(MODEL_FOLDER, exist_ok=True)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)

print("=" * 70)
print("PHASE 4: FEATURE ENGINEERING")
print("=" * 70)

# Load cleaned dataset
df = pd.read_csv(INPUT_FILE)

print("\n1. INITIAL DATASET")
print(f"Initial shape: {df.shape}")

# ---------------------------------------------------------------
# 2. Build every engineered feature through the shared module
#
# The same function is used by the Streamlit dashboard when it scores an
# uploaded file, so a customer is described identically at training time and at
# prediction time. The returned artefacts hold the ARPU quartile edges used to
# assign ValueTier, which must be reused rather than recomputed on new data.
# ---------------------------------------------------------------
df, feature_artifacts = add_engineered_features(df)

joblib.dump(feature_artifacts, ARTIFACTS_FILE)

print("\n2. FEATURE ENGINEERING ARTEFACTS")
print(f"ARPU quartile edges fitted and saved to: {ARTIFACTS_FILE}")

# ---------------------------------------------------------------
# 3. Encode target variable
# ---------------------------------------------------------------
df["ChurnFlag"] = df[TARGET_COLUMN].map({
    "No": 0,
    "Yes": 1
})

# ---------------------------------------------------------------
# 4. Display engineered feature summary
# ---------------------------------------------------------------
engineered_columns = [
    "TenureBand",
    "ARPU",
    "ChargeRatio",
    "ServiceCount",
    "ContractRiskFlag",
    "ValueTier",
    "AutoPayFlag",
    "ChurnFlag"
]

print("\n3. ENGINEERED FEATURES CREATED")
print(engineered_columns)

print("\n4. ENGINEERED FEATURE SAMPLE")
print(df[engineered_columns].head(10).to_string(index=False))

print("\n5. ENGINEERED FEATURE DATA TYPES")
print(df[engineered_columns].dtypes.to_string())

print("\n6. ENGINEERED FEATURE MISSING VALUES")
print(df[engineered_columns].isna().sum().to_string())

print("\n7. SERVICE COUNT DISTRIBUTION")
print(df["ServiceCount"].value_counts().sort_index().to_string())

print("\n8. VALUE TIER DISTRIBUTION")
print(df["ValueTier"].value_counts().sort_index().to_string())

# ---------------------------------------------------------------
# 5. Export the unscaled engineered features
#
# This is the file phase 5 merges into the segmented dataset. It deliberately
# holds raw values: phases 6 to 8 fit their own scaler and encoder inside a
# ColumnTransformer that sees only the training rows of each fold, so handing
# them pre-scaled columns would both double-scale the data and leak the test
# distribution into training.
# ---------------------------------------------------------------
unscaled_features = df[[ID_COLUMN] + ENGINEERED_FEATURE_COLUMNS].copy()
unscaled_features.to_csv(FEATURES_FILE, index=False)

print("\n9. UNSCALED FEATURE EXPORT")
print(f"Rows exported: {len(unscaled_features)}")
print(f"Feature columns: {ENGINEERED_FEATURE_COLUMNS}")
print(f"Saved: {FEATURES_FILE}")

# ---------------------------------------------------------------
# 6. Drop identifier and original target before model preparation
# Keep ChurnFlag as the machine-learning target
# ---------------------------------------------------------------
model_df = df.drop(columns=[ID_COLUMN, TARGET_COLUMN])

# ---------------------------------------------------------------
# 7. One-hot encode categorical variables
# ---------------------------------------------------------------
categorical_columns = model_df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

model_df = pd.get_dummies(
    model_df,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)

print("\n10. CATEGORICAL ENCODING")
print(f"Categorical columns encoded: {len(categorical_columns)}")
print(categorical_columns)

# ---------------------------------------------------------------
# 8. Standardize numerical features
# Do not standardize ChurnFlag because it is the target
# ---------------------------------------------------------------
numeric_columns = model_df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

numeric_columns_to_scale = [
    column for column in numeric_columns
    if column != "ChurnFlag"
]

scaler = StandardScaler()

model_df[numeric_columns_to_scale] = scaler.fit_transform(
    model_df[numeric_columns_to_scale]
)

print("\n11. NUMERICAL STANDARDIZATION")
print(f"Numerical columns standardized: {len(numeric_columns_to_scale)}")

# ---------------------------------------------------------------
# 9. Final validation and save
# ---------------------------------------------------------------
print("\n12. FINAL MODEL-READY DATASET")
print(f"Final shape: {model_df.shape}")
print(f"Total missing values: {model_df.isna().sum().sum()}")

print("\nTarget class distribution:")
print(model_df["ChurnFlag"].value_counts().to_string())

model_df.to_csv(OUTPUT_FILE, index=False)

print("\n13. OUTPUT")
print(f"Model-ready dataset saved successfully: {OUTPUT_FILE}")

print("\n14. PHASE 4 SUMMARY")
print("- TenureBand captures non-linear early-life churn risk.")
print("- ARPU estimates average revenue per active month.")
print("- ChargeRatio identifies potential recent-price-change behaviour.")
print("- ServiceCount represents add-on service adoption.")
print("- ContractRiskFlag identifies month-to-month contract exposure.")
print("- ValueTier prioritizes customers by revenue value.")
print("- AutoPayFlag captures automatic versus manual payment behaviour.")
print("- Categorical features were one-hot encoded.")
print("- Numerical features were standardized using StandardScaler.")
print(f"- Unscaled engineered features exported to {FEATURES_FILE}")
print("  so phase 5 can merge them and phases 6-8 can train on them.")
print(f"- ARPU quartile edges persisted to {ARTIFACTS_FILE}")
print("  so unseen customers are tiered with the training boundaries.")