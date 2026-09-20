import pandas as pd

INPUT_FILE = "data/Customer-Churn.csv"
OUTPUT_FILE = "data/Customer-Churn-Cleaned.csv"

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

print("=" * 70)
print("PHASE 2: DATA CLEANING")
print("=" * 70)

# Load dataset
df = pd.read_csv(INPUT_FILE)

print("\n1. INITIAL DATASET")
print(f"Initial shape: {df.shape}")

# Replace blank or whitespace-only strings with missing values
df = df.replace(r"^\s*$", pd.NA, regex=True)

print("\n2. MISSING VALUES AFTER BLANK-TO-NULL CONVERSION")
print(df.isna().sum().to_string())

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

blank_total_charges = df["TotalCharges"].isna().sum()
zero_tenure_missing = (
    (df["tenure"] == 0) &
    (df["TotalCharges"].isna())
).sum()

print("\n3. TOTALCHARGES CLEANING")
print(f"Missing TotalCharges after conversion: {blank_total_charges}")
print(f"Missing TotalCharges with zero tenure: {zero_tenure_missing}")

# For newly joined customers with tenure 0, TotalCharges should be 0
df.loc[
    (df["tenure"] == 0) & (df["TotalCharges"].isna()),
    "TotalCharges"
] = 0.0

remaining_missing = df["TotalCharges"].isna().sum()

if remaining_missing > 0:
    median_total_charges = df["TotalCharges"].median()
    df["TotalCharges"] = df["TotalCharges"].fillna(median_total_charges)
    print(f"Remaining missing TotalCharges filled using median: {median_total_charges:.2f}")
else:
    print("All missing TotalCharges values handled using 0.0 for zero-tenure customers.")

# Check duplicate full records and duplicate customer IDs
full_duplicate_count = df.duplicated().sum()
customer_id_duplicate_count = df.duplicated(subset=[ID_COLUMN]).sum()

print("\n4. DUPLICATE CHECK")
print(f"Duplicate complete records: {full_duplicate_count}")
print(f"Duplicate customer IDs: {customer_id_duplicate_count}")

df = df.drop_duplicates()

if df.duplicated(subset=[ID_COLUMN]).sum() > 0:
    df = df.drop_duplicates(subset=[ID_COLUMN], keep="first")

print(f"Shape after duplicate removal: {df.shape}")

# Validate numerical ranges
print("\n5. NUMERICAL RANGE VALIDATION")
print(f"Tenure range: {df['tenure'].min()} to {df['tenure'].max()} months")
print(f"MonthlyCharges range: {df['MonthlyCharges'].min():.2f} to {df['MonthlyCharges'].max():.2f}")
print(f"TotalCharges range: {df['TotalCharges'].min():.2f} to {df['TotalCharges'].max():.2f}")

invalid_tenure = ((df["tenure"] < 0) | (df["tenure"] > 72)).sum()
invalid_monthly_charges = (df["MonthlyCharges"] < 0).sum()
invalid_total_charges = (df["TotalCharges"] < 0).sum()

print(f"Invalid tenure records: {invalid_tenure}")
print(f"Negative MonthlyCharges records: {invalid_monthly_charges}")
print(f"Negative TotalCharges records: {invalid_total_charges}")

# Detect and cap outliers using IQR
print("\n6. OUTLIER DETECTION AND CAPPING USING IQR")

numeric_columns = ["tenure", "MonthlyCharges", "TotalCharges"]

for column in numeric_columns:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    outlier_count = (
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ).sum()

    df[column] = df[column].clip(
        lower=lower_bound,
        upper=upper_bound
    )

    print(
        f"{column}: "
        f"Q1={q1:.2f}, Q3={q3:.2f}, "
        f"IQR={iqr:.2f}, "
        f"Lower={lower_bound:.2f}, Upper={upper_bound:.2f}, "
        f"Outliers capped={outlier_count}"
    )

# Standardize service-related categorical labels
print("\n7. CATEGORICAL LABEL STANDARDIZATION")

internet_dependent_columns = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

for column in internet_dependent_columns:
    df[column] = df[column].replace("No internet service", "No")

df["MultipleLines"] = df["MultipleLines"].replace("No phone service", "No")

print("Standardized 'No internet service' to 'No' in internet-related service columns.")
print("Standardized 'No phone service' to 'No' in MultipleLines.")

# Final quality checks
print("\n8. FINAL DATA QUALITY CHECK")
print(f"Final shape: {df.shape}")
print("\nFinal missing values:")
print(df.isna().sum().to_string())

print("\nFinal data types:")
print(df.dtypes.to_string())

print("\nFinal unique values for standardized columns:")
for column in ["MultipleLines"] + internet_dependent_columns:
    print(f"{column}: {sorted(df[column].dropna().unique())}")

# Save cleaned dataset
df.to_csv(OUTPUT_FILE, index=False)

print("\n9. OUTPUT")
print(f"Cleaned dataset saved successfully: {OUTPUT_FILE}")

print("\n10. PHASE 2 SUMMARY")
print("- Blank strings were converted to missing values.")
print("- TotalCharges was converted from text to numeric.")
print("- Missing TotalCharges for zero-tenure customers were replaced with 0.0.")
print("- Duplicate records were checked and removed if present.")
print("- Numerical variables were validated and IQR outliers were capped.")
print("- Service-related categorical labels were standardized.")