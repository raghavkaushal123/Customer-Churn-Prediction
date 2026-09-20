import pandas as pd

FILE_PATH = "data/Customer-Churn.csv"
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 150)

# Load dataset
df = pd.read_csv(FILE_PATH)

print("=" * 70)
print("PHASE 1: DATA COLLECTION AND UNDERSTANDING")
print("=" * 70)

# 1. Dataset shape
print("\n1. DATASET SHAPE")
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

# 2. Sample records
print("\n2. FIRST FIVE RECORDS")
print(df.head().to_string(index=False))

# 3. Data types
print("\n3. COLUMN DATA TYPES")
print(df.dtypes.to_string())

# 4. Missing values
print("\n4. MISSING VALUES")
print(df.isnull().sum().to_string())

# 5. Cardinality summary only
print("\n5. CATEGORICAL VARIABLE CARDINALITY")
categorical_columns = df.select_dtypes(include="object").columns

cardinality_summary = pd.DataFrame({
    "Column": categorical_columns,
    "Unique_Values": [df[column].nunique() for column in categorical_columns],
    "Example_Values": [
        ", ".join(map(str, df[column].dropna().unique()[:5]))
        for column in categorical_columns
    ]
})

print(cardinality_summary.to_string(index=False))

# 6. Target variable
print("\n6. TARGET VARIABLE: CHURN")
churn_counts = df[TARGET_COLUMN].value_counts()
churn_percentages = (df[TARGET_COLUMN].value_counts(normalize=True) * 100).round(2)

print("\nChurn count:")
print(churn_counts.to_string())

print("\nChurn distribution (%):")
print(churn_percentages.to_string())

churn_rate = (df[TARGET_COLUMN] == "Yes").mean() * 100
print(f"\nBaseline churn rate: {churn_rate:.2f}%")

# 7. Identifier and predictors
predictor_columns = [
    column for column in df.columns
    if column not in [ID_COLUMN, TARGET_COLUMN]
]

print("\n7. IDENTIFIER AND PREDICTORS")
print(f"Identifier column: {ID_COLUMN}")
print(f"Target column: {TARGET_COLUMN} (Yes = 1, No = 0)")
print(f"Number of predictor columns: {len(predictor_columns)}")
print("Predictors:")
print(", ".join(predictor_columns))

# 8. TotalCharges data-quality check
print("\n8. TOTALCHARGES DATA QUALITY CHECK")
total_charges_numeric = pd.to_numeric(df["TotalCharges"], errors="coerce")
invalid_total_charges = total_charges_numeric.isna().sum()

print(f"Invalid or blank TotalCharges values: {invalid_total_charges}")

if invalid_total_charges > 0:
    print("\nSample records with blank TotalCharges:")
    print(
        df.loc[
            total_charges_numeric.isna(),
            [ID_COLUMN, "tenure", "MonthlyCharges", "TotalCharges"]
        ].head().to_string(index=False)
    )

# 9. Summary
print("\n9. PHASE 1 SUMMARY")
print(f"- Dataset contains {df.shape[0]} customer records and {df.shape[1]} attributes.")
print(f"- The churn target has a baseline rate of {churn_rate:.2f}%.")
print(f"- {ID_COLUMN} is an identifier and will be excluded from predictive modeling.")
print(f"- {TARGET_COLUMN} will later be encoded as Yes = 1 and No = 0.")
print("- TotalCharges contains blank values and requires numeric conversion in Phase 2.")