import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from churn_features import LEVER_ACTIONS, SHARED_ACTION

INPUT_FILE = "data/phase8_customer_churn_risk_predictions.csv"

# Per-customer SHAP levers written by phase 8. These replace the fixed rule
# cascade that previously assigned every high-risk customer the same driver.
DRIVERS_FILE = "data/phase8_customer_risk_drivers.csv"

# Customers in their first year are in the highest-risk part of the lifecycle
# (phase 3: 52.94% churn in months 0-6, 35.89% in months 7-12).
ONBOARDING_WINDOW_MONTHS = 12
OUTPUT_FOLDER = "data"
CHART_FOLDER = "charts"

# Business assumptions: can be changed later for dashboard analysis
EXPECTED_REMAINING_TENURE_MONTHS = 12
CAMPAIGN_SUCCESS_RATE = 0.30
COST_PER_RETENTION_OFFER = 15.00
EVALUATION_FILE = "data/phase7_model_evaluation_results.csv"
CLEANED_FILE = "data/Customer-Churn-Cleaned.csv"

# Fallbacks used only if the upstream files are missing.
FALLBACK_MODEL_RECALL = 0.7766
FALLBACK_BASELINE_CHURN_RATE = 0.2654

HIGH_RISK_THRESHOLD = 0.65

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CHART_FOLDER, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

print("=" * 75)
print("PHASE 9: BUSINESS RECOMMENDATION AND REVENUE IMPACT")
print("=" * 75)

# ---------------------------------------------------------------
# 1. Load Phase 8 risk predictions and the measured model performance
#
# Recall and the baseline churn rate are read from the upstream artefacts
# rather than hard-coded. Hard-coding them means any change to the train-test
# split leaves this financial model quoting a recall the classifier no longer
# achieves.
# ---------------------------------------------------------------
df = pd.read_csv(INPUT_FILE)

if os.path.exists(EVALUATION_FILE):
    evaluation = pd.read_csv(EVALUATION_FILE)
    best_row = evaluation.sort_values("ROC_AUC", ascending=False).iloc[0]

    MODEL_RECALL = float(best_row["Recall"])
    BEST_MODEL_NAME = str(best_row["Model"])
    recall_source = f"phase 7 hold-out result for {BEST_MODEL_NAME}"
else:
    MODEL_RECALL = FALLBACK_MODEL_RECALL
    BEST_MODEL_NAME = "unknown"
    recall_source = "fallback constant (phase 7 results not found)"

if os.path.exists(CLEANED_FILE):
    cleaned = pd.read_csv(CLEANED_FILE, usecols=["Churn"])
    BASELINE_CHURN_RATE = float((cleaned["Churn"] == "Yes").mean())
    churn_source = "measured on the cleaned dataset"
else:
    BASELINE_CHURN_RATE = FALLBACK_BASELINE_CHURN_RATE
    churn_source = "fallback constant (cleaned dataset not found)"

print("\n1. INPUT DATA")
print(f"Customers loaded: {len(df)}")
print(f"Model recall: {MODEL_RECALL:.4f} ({recall_source})")
print(f"Baseline churn rate: {BASELINE_CHURN_RATE:.4f} ({churn_source})")
print(f"High-risk threshold: p(churn) > {HIGH_RISK_THRESHOLD}")
print(f"Campaign success rate: {CAMPAIGN_SUCCESS_RATE * 100:.0f}%")
print(f"Cost per retention offer: ${COST_PER_RETENTION_OFFER:.2f}")
print(
    "Expected remaining customer lifetime: "
    f"{EXPECTED_REMAINING_TENURE_MONTHS} months"
)

# ---------------------------------------------------------------
# 2. Customer value and revenue-at-risk calculations
# ---------------------------------------------------------------
df["ExpectedRemainingTenure"] = EXPECTED_REMAINING_TENURE_MONTHS

df["CustomerValue"] = (
    df["MonthlyCharges"] *
    df["ExpectedRemainingTenure"]
)

df["RevenueAtRisk"] = (
    df["ChurnProbability"] *
    df["CustomerValue"]
)

df["TargetForRetention"] = (
    df["ChurnProbability"] > HIGH_RISK_THRESHOLD
).astype(int)

df["ExpectedSavedRevenue"] = (
    df["RevenueAtRisk"] *
    CAMPAIGN_SUCCESS_RATE *
    df["TargetForRetention"]
)

df["CampaignCost"] = (
    df["TargetForRetention"] *
    COST_PER_RETENTION_OFFER
)

df["ExpectedNetBenefit"] = (
    df["ExpectedSavedRevenue"] -
    df["CampaignCost"]
)

# ---------------------------------------------------------------
# 3. Assign per-customer drivers and retention actions from SHAP
#
# The previous version used a rule cascade that tested contract type first.
# Every high-risk customer is on a month-to-month contract, so all 732 of
# them received the same driver and the same action, and the driver chart
# collapsed to a single bar.
#
# Contract term is now treated as what it is for this cohort: a shared base
# condition. Every month-to-month customer receives the contract-migration
# offer as a base action. On top of that, each customer receives the lever
# with the largest SHAP contribution to their own predicted risk, which is
# what differentiates one targeted customer from another.
# ---------------------------------------------------------------
if not os.path.exists(DRIVERS_FILE):
    raise SystemExit(
        f"Required input not found: {DRIVERS_FILE}\n"
        "Run phase8_churn_probability_xai.py before phase 9."
    )

drivers = pd.read_csv(DRIVERS_FILE)
df = df.merge(
    drivers[["customerID", "PrimaryLever", "SecondaryLever",
             "SharedDriverContribution", "PrimaryLeverContribution"]],
    on="customerID",
    how="left"
)

missing_drivers = df["PrimaryLever"].isna().sum()
if missing_drivers > 0:
    raise SystemExit(
        f"{missing_drivers} customers have no SHAP driver. Re-run phase 8."
    )

df["SharedDriver"] = df["Contract"].eq("Month-to-month").map({
    True: "Month-to-month contract",
    False: "None"
})

df["LifecycleStage"] = df["tenure"].le(ONBOARDING_WINDOW_MONTHS).map({
    True: "Onboarding window (0-12 months)",
    False: "Established (13+ months)"
})

df["DominantRiskDriver"] = df["PrimaryLever"]


def assign_retention_action(row):
    lever_action = LEVER_ACTIONS.get(
        row["PrimaryLever"],
        "Personalised retention review"
    )

    if row["SharedDriver"] == "Month-to-month contract":
        return f"{SHARED_ACTION} + {lever_action}"

    return lever_action


df["RecommendedRetentionAction"] = df.apply(
    assign_retention_action,
    axis=1
)

# ---------------------------------------------------------------
# 4. High-risk customer prioritization
# ---------------------------------------------------------------
high_risk_customers = df[
    df["TargetForRetention"] == 1
].copy()

high_risk_customers = high_risk_customers.sort_values(
    by="RevenueAtRisk",
    ascending=False
)

high_risk_customers["PriorityRank"] = range(
    1,
    len(high_risk_customers) + 1
)

high_risk_output_columns = [
    "PriorityRank",
    "customerID",
    "ChurnProbability",
    "RiskTier",
    "CustomerSegment",
    "tenure",
    "MonthlyCharges",
    "Contract",
    "PaymentMethod",
    "InternetService",
    "TechSupport",
    "LifecycleStage",
    "SharedDriver",
    "CustomerValue",
    "RevenueAtRisk",
    "ExpectedSavedRevenue",
    "CampaignCost",
    "ExpectedNetBenefit",
    "DominantRiskDriver",
    "SecondaryLever",
    "RecommendedRetentionAction"
]

high_risk_customers = high_risk_customers[
    high_risk_output_columns
]

high_risk_customers.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "phase9_high_risk_retention_targets.csv"
    ),
    index=False
)

print("\n2. HIGH-RISK CUSTOMER PRIORITIZATION")
print(f"High-risk customers targeted: {len(high_risk_customers)}")

print("\nTop 10 high-risk retention targets:")
print(
    high_risk_customers[
        [
            "PriorityRank",
            "customerID",
            "ChurnProbability",
            "CustomerValue",
            "RevenueAtRisk",
            "DominantRiskDriver"
        ]
    ].head(10).round(2).to_string(index=False)
)

print("Saved: data/phase9_high_risk_retention_targets.csv")

# ---------------------------------------------------------------
# 5. Financial impact model
# ---------------------------------------------------------------
customer_count = len(df)
baseline_churners = customer_count * BASELINE_CHURN_RATE

customers_targeted = len(high_risk_customers)
revenue_at_risk = high_risk_customers["RevenueAtRisk"].sum()
expected_saved_revenue = high_risk_customers[
    "ExpectedSavedRevenue"
].sum()
campaign_cost = high_risk_customers["CampaignCost"].sum()
net_benefit = high_risk_customers["ExpectedNetBenefit"].sum()

retention_roi = (
    (net_benefit / campaign_cost) * 100
    if campaign_cost > 0 else 0
)

identified_churners = baseline_churners * MODEL_RECALL

expected_retained_customers = (
    identified_churners *
    CAMPAIGN_SUCCESS_RATE
)

projected_churn_rate = (
    BASELINE_CHURN_RATE *
    (1 - (MODEL_RECALL * CAMPAIGN_SUCCESS_RATE))
)

financial_summary = pd.DataFrame({
    "Metric": [
        "Customers in Base",
        "Baseline Churn Rate",
        "Baseline Churners per Cycle",
        "Model Recall",
        "High-Risk Customers Targeted",
        "Average Customer Value",
        "Total Revenue at Risk",
        "Expected Saved Revenue",
        "Campaign Cost",
        "Net Benefit",
        "Retention ROI (%)",
        "Expected Retained Customers",
        "Projected Churn Rate"
    ],
    "Value": [
        customer_count,
        BASELINE_CHURN_RATE,
        baseline_churners,
        MODEL_RECALL,
        customers_targeted,
        df["CustomerValue"].mean(),
        revenue_at_risk,
        expected_saved_revenue,
        campaign_cost,
        net_benefit,
        retention_roi,
        expected_retained_customers,
        projected_churn_rate
    ]
})

financial_summary.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "phase9_financial_impact_summary.csv"
    ),
    index=False
)

print("\n3. FINANCIAL IMPACT SUMMARY")
print(financial_summary.round(2).to_string(index=False))

print("Saved: data/phase9_financial_impact_summary.csv")

# ---------------------------------------------------------------
# 6. Recommendation mapping summary
# ---------------------------------------------------------------
recommendation_summary = (
    high_risk_customers.groupby(
        ["DominantRiskDriver", "RecommendedRetentionAction"],
        dropna=False
    )
    .agg(
        CustomersTargeted=("customerID", "count"),
        TotalRevenueAtRisk=("RevenueAtRisk", "sum"),
        ExpectedNetBenefit=("ExpectedNetBenefit", "sum")
    )
    .reset_index()
    .sort_values(
        by="TotalRevenueAtRisk",
        ascending=False
    )
)

recommendation_summary.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "phase9_recommendation_mapping.csv"
    ),
    index=False
)

print("\n4. RECOMMENDATION MAPPING")
print(recommendation_summary.round(2).to_string(index=False))

print("Saved: data/phase9_recommendation_mapping.csv")

# ---------------------------------------------------------------
# 7. Revenue-at-risk by customer segment chart
# ---------------------------------------------------------------
segment_impact = (
    high_risk_customers.groupby(
        "CustomerSegment",
        dropna=False
    )
    .agg(
        Customers=("customerID", "count"),
        RevenueAtRisk=("RevenueAtRisk", "sum"),
        ExpectedNetBenefit=("ExpectedNetBenefit", "sum")
    )
    .reset_index()
    .sort_values(
        by="RevenueAtRisk",
        ascending=False
    )
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=segment_impact,
    x="RevenueAtRisk",
    y="CustomerSegment",
    color="crimson"
)

plt.title("Revenue at Risk by High-Risk Customer Segment")
plt.xlabel("Revenue at Risk ($)")
plt.ylabel("Customer Segment")
plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_FOLDER,
        "phase9_revenue_at_risk_by_segment.png"
    ),
    bbox_inches="tight"
)

plt.close()

print("Saved: charts/phase9_revenue_at_risk_by_segment.png")

# ---------------------------------------------------------------
# 8. Risk-driver priority chart
# ---------------------------------------------------------------
driver_impact = (
    high_risk_customers.groupby(
        "DominantRiskDriver",
        dropna=False
    )
    .agg(
        Customers=("customerID", "count"),
        RevenueAtRisk=("RevenueAtRisk", "sum")
    )
    .reset_index()
    .sort_values(
        by="RevenueAtRisk",
        ascending=False
    )
)

plt.figure(figsize=(11, 6))

sns.barplot(
    data=driver_impact,
    x="RevenueAtRisk",
    y="DominantRiskDriver",
    color="darkorange"
)

plt.title("Revenue at Risk by Dominant Churn Driver")
plt.xlabel("Revenue at Risk ($)")
plt.ylabel("Dominant Risk Driver")
plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_FOLDER,
        "phase9_revenue_at_risk_by_driver.png"
    ),
    bbox_inches="tight"
)

plt.close()

print("Saved: charts/phase9_revenue_at_risk_by_driver.png")

# ---------------------------------------------------------------
# 9. Financial value chain chart
# ---------------------------------------------------------------
impact_chart_data = pd.DataFrame({
    "Stage": [
        "Revenue at Risk",
        "Expected Saved Revenue",
        "Campaign Cost",
        "Net Benefit"
    ],
    "Amount": [
        revenue_at_risk,
        expected_saved_revenue,
        campaign_cost,
        net_benefit
    ]
})

plt.figure(figsize=(10, 6))

ax = sns.barplot(
    data=impact_chart_data,
    x="Stage",
    y="Amount",
    hue="Stage",
    palette=[
        "crimson",
        "seagreen",
        "steelblue",
        "darkgreen"
    ],
    legend=False
)

plt.title("Retention Campaign Financial Impact")
plt.xlabel("Financial Stage")
plt.ylabel("Amount ($)")
plt.xticks(rotation=10)

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="$%.0f",
        padding=3
    )

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_FOLDER,
        "phase9_retention_financial_impact.png"
    ),
    bbox_inches="tight"
)

plt.close()

print("Saved: charts/phase9_retention_financial_impact.png")

# ---------------------------------------------------------------
# 10. Final summary
# ---------------------------------------------------------------
print("\n5. PHASE 9 SUMMARY")
print(
    f"- Targeted {customers_targeted} customers with "
    f"predicted churn probability above {HIGH_RISK_THRESHOLD:.2f}."
)
print(f"- Total revenue at risk: ${revenue_at_risk:,.2f}")
print(f"- Expected saved revenue: ${expected_saved_revenue:,.2f}")
print(f"- Campaign cost: ${campaign_cost:,.2f}")
print(f"- Expected net benefit: ${net_benefit:,.2f}")
print(f"- Estimated retention ROI: {retention_roi:,.2f}%")
print(
    f"- Projected churn rate after campaign: "
    f"{projected_churn_rate * 100:.2f}%"
)
print(
    "- Use the high-risk retention target file to prioritize "
    "customer outreach."
)