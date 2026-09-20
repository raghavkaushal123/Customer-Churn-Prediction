import pandas as pd
from sklearn.preprocessing import StandardScaler

INPUT_FILE = "data/Customer-Churn-Cleaned.csv"
OUTPUT_FILE = "data/Customer-Churn-Model-Ready.csv"

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"

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
# 2. Tenure Band
# ---------------------------------------------------------------
tenure_bins = [-1, 6, 12, 24, 48, 72]
tenure_labels = [
    "0-6 months",
    "7-12 months",
    "13-24 months",
    "25-48 months",
    "49-72 months"
]

df["TenureBand"] = pd.cut(
    df["tenure"],
    bins=tenure_bins,
    labels=tenure_labels
)

# ---------------------------------------------------------------
# 3. ARPU: Average Revenue Per User
# Guarded for customers with zero tenure
# ---------------------------------------------------------------
df["ARPU"] = 0.0

non_zero_tenure = df["tenure"] > 0

df.loc[non_zero_tenure, "ARPU"] = (
    df.loc[non_zero_tenure, "TotalCharges"] /
    df.loc[non_zero_tenure, "tenure"]
)

# ---------------------------------------------------------------
# 4. Charge Ratio
# MonthlyCharges / average accumulated monthly charge
# Guarded against zero TotalCharges and zero tenure
# ---------------------------------------------------------------
df["ChargeRatio"] = 0.0

valid_charge_ratio = (
    (df["tenure"] > 0) &
    (df["TotalCharges"] > 0)
)

df.loc[valid_charge_ratio, "ChargeRatio"] = (
    df.loc[valid_charge_ratio, "MonthlyCharges"] /
    (
        df.loc[valid_charge_ratio, "TotalCharges"] /
        df.loc[valid_charge_ratio, "tenure"]
    )
)

# ---------------------------------------------------------------
# 5. Service Count
# Number of subscribed add-on services
# ---------------------------------------------------------------
service_columns = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

df["ServiceCount"] = (
    df[service_columns]
    .eq("Yes")
    .sum(axis=1)
)

# ---------------------------------------------------------------
# 6. Contract Risk Flag
# 1 = month-to-month contract, 0 = annual/two-year contract
# ---------------------------------------------------------------
df["ContractRiskFlag"] = (
    df["Contract"] == "Month-to-month"
).astype(int)

# ---------------------------------------------------------------
# 7. Value Tier
# Quartile ranking based on ARPU
# ---------------------------------------------------------------
df["ValueTier"] = pd.qcut(
    df["ARPU"],
    q=4,
    labels=["Low", "Medium", "High", "Premium"],
    duplicates="drop"
)

# ---------------------------------------------------------------
# 8. Auto-Pay Flag
# 1 = automatic payment, 0 = manual payment
# ---------------------------------------------------------------
automatic_payment_methods = [
    "Bank transfer (automatic)",
    "Credit card (automatic)"
]

df["AutoPayFlag"] = (
    df["PaymentMethod"]
    .isin(automatic_payment_methods)
).astype(int)

# ---------------------------------------------------------------
# 9. Encode target variable
# ---------------------------------------------------------------
df["ChurnFlag"] = df[TARGET_COLUMN].map({
    "No": 0,
    "Yes": 1
})

# ---------------------------------------------------------------
# 10. Display engineered feature summary
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

print("\n2. ENGINEERED FEATURES CREATED")
print(engineered_columns)

print("\n3. ENGINEERED FEATURE SAMPLE")
print(df[engineered_columns].head(10).to_string(index=False))

print("\n4. ENGINEERED FEATURE DATA TYPES")
print(df[engineered_columns].dtypes.to_string())

print("\n5. ENGINEERED FEATURE MISSING VALUES")
print(df[engineered_columns].isna().sum().to_string())

print("\n6. SERVICE COUNT DISTRIBUTION")
print(df["ServiceCount"].value_counts().sort_index().to_string())

print("\n7. VALUE TIER DISTRIBUTION")
print(df["ValueTier"].value_counts().sort_index().to_string())

# ---------------------------------------------------------------
# 11. Drop identifier and original target before model preparation
# Keep ChurnFlag as the machine-learning target
# ---------------------------------------------------------------
model_df = df.drop(columns=[ID_COLUMN, TARGET_COLUMN])

# ---------------------------------------------------------------
# 12. One-hot encode categorical variables
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

print("\n8. CATEGORICAL ENCODING")
print(f"Categorical columns encoded: {len(categorical_columns)}")
print(categorical_columns)

# ---------------------------------------------------------------
# 13. Standardize numerical features
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

print("\n9. NUMERICAL STANDARDIZATION")
print(f"Numerical columns standardized: {len(numeric_columns_to_scale)}")

# ---------------------------------------------------------------
# 14. Final validation and save
# ---------------------------------------------------------------
print("\n10. FINAL MODEL-READY DATASET")
print(f"Final shape: {model_df.shape}")
print(f"Total missing values: {model_df.isna().sum().sum()}")

print("\nTarget class distribution:")
print(model_df["ChurnFlag"].value_counts().to_string())

model_df.to_csv(OUTPUT_FILE, index=False)

print("\n11. OUTPUT")
print(f"Model-ready dataset saved successfully: {OUTPUT_FILE}")

print("\n12. PHASE 4 SUMMARY")
print("- TenureBand captures non-linear early-life churn risk.")
print("- ARPU estimates average revenue per active month.")
print("- ChargeRatio identifies potential recent-price-change behaviour.")
print("- ServiceCount represents add-on service adoption.")
print("- ContractRiskFlag identifies month-to-month contract exposure.")
print("- ValueTier prioritizes customers by revenue value.")
print("- AutoPayFlag captures automatic versus manual payment behaviour.")
print("- Categorical features were one-hot encoded.")
print("- Numerical features were standardized using StandardScaler.")