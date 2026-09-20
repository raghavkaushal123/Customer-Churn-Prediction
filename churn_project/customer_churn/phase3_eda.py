import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_FILE = "data/Customer-Churn-Cleaned.csv"
OUTPUT_FOLDER = "charts"

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 120

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

df["ChurnFlag"] = df["Churn"].map({"No": 0, "Yes": 1})

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

print("=" * 70)
print("PHASE 3: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# ---------------------------------------------------------------
# 1. Univariate analysis
# ---------------------------------------------------------------
print("\n1. UNIVARIATE ANALYSIS")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.countplot(
    data=df,
    x="Churn",
    hue="Churn",
    legend=False,
    ax=axes[0, 0]
)
axes[0, 0].set_title("Customer Churn Distribution")
axes[0, 0].set_xlabel("Churn Status")
axes[0, 0].set_ylabel("Customer Count")

sns.histplot(
    data=df,
    x="tenure",
    bins=30,
    kde=True,
    ax=axes[0, 1],
    color="steelblue"
)
axes[0, 1].set_title("Tenure Distribution")
axes[0, 1].set_xlabel("Tenure (Months)")

sns.histplot(
    data=df,
    x="MonthlyCharges",
    bins=30,
    kde=True,
    ax=axes[1, 0],
    color="darkorange"
)
axes[1, 0].set_title("Monthly Charges Distribution")
axes[1, 0].set_xlabel("Monthly Charges")

contract_order = ["Month-to-month", "One year", "Two year"]

sns.countplot(
    data=df,
    x="Contract",
    hue="Contract",
    order=contract_order,
    legend=False,
    ax=axes[1, 1]
)
axes[1, 1].set_title("Contract Type Distribution")
axes[1, 1].set_xlabel("Contract Type")
axes[1, 1].tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_FOLDER, "phase3_univariate_analysis.png"),
    bbox_inches="tight"
)
plt.close()

print("Saved: charts/phase3_univariate_analysis.png")


# ---------------------------------------------------------------
# Helper function for bivariate churn-rate analysis
# ---------------------------------------------------------------
def create_churn_rate_chart(column, title, filename, order=None):
    summary = (
        df.groupby(column, observed=False)["ChurnFlag"]
        .agg(["mean", "count"])
        .reset_index()
    )

    summary["ChurnRate"] = (summary["mean"] * 100).round(2)
    summary = summary.rename(columns={"count": "Customers"})
    summary = summary[[column, "Customers", "ChurnRate"]]

    if order is not None:
        summary[column] = pd.Categorical(
            summary[column],
            categories=order,
            ordered=True
        )
        summary = summary.sort_values(column)

    print(f"\nChurn rate by {column}:")
    print(summary.to_string(index=False))

    plt.figure(figsize=(9, 5))

    ax = sns.barplot(
        data=summary,
        x=column,
        y="ChurnRate",
        order=order,
        color="teal"
    )

    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Churn Rate (%)")
    plt.ylim(0, max(summary["ChurnRate"]) + 10)
    plt.xticks(rotation=15, ha="right")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", padding=3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_FOLDER, filename),
        bbox_inches="tight"
    )
    plt.close()

    print(f"Saved: charts/{filename}")


# ---------------------------------------------------------------
# 2. Bivariate analysis
# ---------------------------------------------------------------
print("\n2. BIVARIATE ANALYSIS")

create_churn_rate_chart(
    "Contract",
    "Churn Rate by Contract Type",
    "phase3_churn_by_contract.png",
    order=contract_order
)

create_churn_rate_chart(
    "PaymentMethod",
    "Churn Rate by Payment Method",
    "phase3_churn_by_payment_method.png"
)

create_churn_rate_chart(
    "InternetService",
    "Churn Rate by Internet Service",
    "phase3_churn_by_internet_service.png"
)

create_churn_rate_chart(
    "SeniorCitizen",
    "Churn Rate by Senior Citizen Status",
    "phase3_churn_by_senior_citizen.png",
    order=[0, 1]
)

create_churn_rate_chart(
    "TechSupport",
    "Churn Rate by Tech Support Subscription",
    "phase3_churn_by_tech_support.png",
    order=["No", "Yes"]
)


# ---------------------------------------------------------------
# 3. Correlation analysis
# ---------------------------------------------------------------
print("\n3. CORRELATION ANALYSIS")

numeric_columns = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "ChurnFlag"
]

correlation_matrix = df[numeric_columns].corr()

print("\nCorrelation matrix:")
print(correlation_matrix.round(3).to_string())

plt.figure(figsize=(9, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5
)

plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_FOLDER, "phase3_correlation_heatmap.png"),
    bbox_inches="tight"
)
plt.close()

print("Saved: charts/phase3_correlation_heatmap.png")


# ---------------------------------------------------------------
# 4. Behavioural analysis
# ---------------------------------------------------------------
print("\n4. BEHAVIOURAL ANALYSIS")

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="tenure",
    y="MonthlyCharges",
    hue="Churn",
    alpha=0.65,
    palette={"No": "steelblue", "Yes": "crimson"}
)

plt.title("Tenure vs Monthly Charges by Churn Status")
plt.xlabel("Tenure (Months)")
plt.ylabel("Monthly Charges")
plt.legend(title="Churn")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_FOLDER, "phase3_tenure_monthlycharges_churn.png"),
    bbox_inches="tight"
)
plt.close()

print("Saved: charts/phase3_tenure_monthlycharges_churn.png")


# ---------------------------------------------------------------
# 5. Cohort analysis
# ---------------------------------------------------------------
print("\n5. COHORT ANALYSIS: CHURN RATE BY TENURE BAND")

cohort_summary = (
    df.groupby("TenureBand", observed=False)["ChurnFlag"]
    .agg(["mean", "count"])
    .reset_index()
)

cohort_summary["ChurnRate"] = (
    cohort_summary["mean"] * 100
).round(2)

cohort_summary = cohort_summary.rename(
    columns={"count": "Customers"}
)

cohort_summary = cohort_summary[
    ["TenureBand", "Customers", "ChurnRate"]
]

print(cohort_summary.to_string(index=False))

plt.figure(figsize=(10, 5))

ax = sns.barplot(
    data=cohort_summary,
    x="TenureBand",
    y="ChurnRate",
    color="purple"
)

plt.title("Churn Rate by Tenure Band")
plt.xlabel("Tenure Band")
plt.ylabel("Churn Rate (%)")
plt.ylim(0, max(cohort_summary["ChurnRate"]) + 10)

for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", padding=3)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_FOLDER, "phase3_churn_by_tenure_band.png"),
    bbox_inches="tight"
)
plt.close()

print("Saved: charts/phase3_churn_by_tenure_band.png")

print("\n6. PHASE 3 OUTPUT SUMMARY")
print("All EDA charts have been saved successfully in the charts folder.")
print("Use the terminal tables and chart files for the Phase 3 report.")