import os

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from churn_features import (
    ENGINEERED_FEATURE_COLUMNS,
    SEGMENTATION_COLUMNS,
    add_service_count,
    name_segments_from_centroids,
)

INPUT_FILE = "data/Customer-Churn-Cleaned.csv"

# Unscaled engineered features written by phase 4. Merging them here is what
# puts ARPU, ChargeRatio, TenureBand, ValueTier and AutoPayFlag in front of the
# models: phases 6 to 8 read the segmented file below, so a feature that is not
# merged into it never reaches a classifier. ContractRiskFlag is merged too, as
# a readable business attribute, but churn_features.MODEL_EXCLUDED_COLUMNS
# keeps it out of the model matrix because it duplicates the Contract column.
FEATURES_FILE = "data/Customer-Churn-Features-Unscaled.csv"

OUTPUT_FILE = "data/Customer-Churn-Segmented.csv"
CHART_FOLDER = "charts"
MODEL_FOLDER = "models"

# The fitted scaler, K-Means model and segment names. Persisted so the
# dashboard can assign a cluster to a new customer with the trained centroids
# instead of re-deriving segments from a hand-written rule, which would make
# the serving-time label disagree with the training-time label.
SEGMENTATION_MODEL_FILE = "models/customer_segmentation_model.joblib"

RANDOM_STATE = 42
OPTIMAL_K = 4

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 120

os.makedirs(CHART_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

print("=" * 70)
print("PHASE 5: CUSTOMER SEGMENTATION")
print("=" * 70)

# Load cleaned dataset
df = pd.read_csv(INPUT_FILE)

# Create engagement feature: number of active add-on services
df = add_service_count(df)

# Churn flag is used ONLY to profile the finished clusters for the report.
# It is never a clustering input and, after the correction described in
# section 5 below, it is never an input to the segment naming rule either.
df["ChurnFlag"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# RFM-style input features for clustering
segmentation_columns = SEGMENTATION_COLUMNS

X = df[segmentation_columns].copy()

print("\n1. SEGMENTATION FEATURES")
print(segmentation_columns)

print("\nFeature summary before standardization:")
print(X.describe().round(2).to_string())

# Standardize clustering dimensions
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Test multiple K values
print("\n2. K-MEANS CLUSTER SELECTION")

k_values = range(2, 7)
inertias = []
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(X_scaled)

    inertias.append(kmeans.inertia_)
    silhouette_scores.append(
        silhouette_score(X_scaled, cluster_labels)
    )

results = pd.DataFrame({
    "Clusters": list(k_values),
    "Inertia": inertias,
    "SilhouetteScore": silhouette_scores
})

print("\nElbow and silhouette-score results:")
print(results.round(4).to_string(index=False))

# Create elbow chart
plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    inertias,
    marker="o",
    linewidth=2,
    color="steelblue"
)

plt.axvline(
    x=OPTIMAL_K,
    color="crimson",
    linestyle="--",
    label=f"Selected K = {OPTIMAL_K}"
)

plt.title("Elbow Method for Customer Segmentation")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Within-Cluster Sum of Squares / Inertia")
plt.xticks(list(k_values))
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(CHART_FOLDER, "phase5_elbow_method.png"),
    bbox_inches="tight"
)

plt.close()

print("Saved: charts/phase5_elbow_method.png")

# Create silhouette-score chart
plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    silhouette_scores,
    marker="o",
    linewidth=2,
    color="darkgreen"
)

plt.axvline(
    x=OPTIMAL_K,
    color="crimson",
    linestyle="--",
    label=f"Selected K = {OPTIMAL_K}"
)

plt.title("Silhouette Score by Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(list(k_values))
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(CHART_FOLDER, "phase5_silhouette_scores.png"),
    bbox_inches="tight"
)

plt.close()

print("Saved: charts/phase5_silhouette_scores.png")

print(
    f"\nSelected optimal K: {OPTIMAL_K} "
    "(chosen using elbow analysis, silhouette validation, "
    "and business interpretability)."
)

# Train final segmentation model using K = 4
final_kmeans = KMeans(
    n_clusters=OPTIMAL_K,
    random_state=RANDOM_STATE,
    n_init=10
)

df["Cluster"] = final_kmeans.fit_predict(X_scaled)

# Profile each cluster
cluster_profile = (
    df.groupby("Cluster")
    .agg(
        Customers=("customerID", "count"),
        MeanTenure=("tenure", "mean"),
        MeanMonthlyCharges=("MonthlyCharges", "mean"),
        MeanServiceCount=("ServiceCount", "mean"),
        MeanTotalCharges=("TotalCharges", "mean"),
        ChurnRate=("ChurnFlag", "mean")
    )
    .reset_index()
)

cluster_profile["CustomerShare"] = (
    cluster_profile["Customers"] / len(df) * 100
)

cluster_profile["ChurnRate"] = (
    cluster_profile["ChurnRate"] * 100
)

cluster_profile = cluster_profile[
    [
        "Cluster",
        "Customers",
        "CustomerShare",
        "MeanTenure",
        "MeanMonthlyCharges",
        "MeanServiceCount",
        "MeanTotalCharges",
        "ChurnRate"
    ]
].round(2)

print("\n3. CLUSTER PROFILE")
print(cluster_profile.to_string(index=False))

# ---------------------------------------------------------------
# 5. Name each segment from its centroid geometry only
#
# The previous implementation selected a name by comparing each cluster's
# OBSERVED churn rate against the median churn rate across clusters, then
# passed that name to phases 6 to 8 as a model feature. That leaked the
# target: the label summarised the very outcome the classifier was asked to
# predict, and it could not be reproduced for a customer whose outcome is
# unknown.
#
# Naming now depends only on where the centroid sits in tenure and monthly
# charges, both of which are known for any customer at prediction time. Churn
# rate is still reported per cluster below, but it is an output of the
# analysis rather than an input to the label.
# ---------------------------------------------------------------
centroids = (
    df.groupby("Cluster")[SEGMENTATION_COLUMNS]
    .mean()
)

segment_names = name_segments_from_centroids(centroids)

print("\n4. SEGMENT NAMING (CENTROID-BASED, TARGET-FREE)")
print("Naming inputs: mean tenure and mean monthly charges per cluster.")
print("Churn rate is NOT used, so the label carries no target information.")

for cluster_id in sorted(segment_names):
    print(
        f"Cluster {cluster_id}: {segment_names[cluster_id]}  "
        f"(tenure={centroids.loc[cluster_id, 'tenure']:.2f}, "
        f"monthly={centroids.loc[cluster_id, 'MonthlyCharges']:.2f}, "
        f"services={centroids.loc[cluster_id, 'ServiceCount']:.2f})"
    )

# Append customer segment labels
df["CustomerSegment"] = df["Cluster"].map(segment_names)
cluster_profile["CustomerSegment"] = cluster_profile["Cluster"].map(
    segment_names
)

print("\n5. BUSINESS-READABLE SEGMENTS")
print(
    cluster_profile[
        [
            "Cluster",
            "CustomerSegment",
            "Customers",
            "CustomerShare",
            "MeanTenure",
            "MeanMonthlyCharges",
            "MeanServiceCount",
            "MeanTotalCharges",
            "ChurnRate"
        ]
    ].to_string(index=False)
)

# Create cluster comparison chart
profile_melted = cluster_profile.melt(
    id_vars=["CustomerSegment"],
    value_vars=[
        "MeanTenure",
        "MeanMonthlyCharges",
        "MeanServiceCount",
        "ChurnRate"
    ],
    var_name="Metric",
    value_name="Value"
)

plt.figure(figsize=(14, 7))

sns.barplot(
    data=profile_melted,
    x="CustomerSegment",
    y="Value",
    hue="Metric"
)

plt.title("Customer Segment Profile Comparison")
plt.xlabel("Customer Segment")
plt.ylabel("Average Value / Churn Rate")
plt.xticks(rotation=15, ha="right")
plt.legend(title="Metric")
plt.tight_layout()

plt.savefig(
    os.path.join(CHART_FOLDER, "phase5_cluster_profiles.png"),
    bbox_inches="tight"
)

plt.close()

print("Saved: charts/phase5_cluster_profiles.png")

# Create scatter plot of segments
plt.figure(figsize=(11, 6))

sns.scatterplot(
    data=df,
    x="tenure",
    y="MonthlyCharges",
    hue="CustomerSegment",
    size="ServiceCount",
    sizes=(25, 180),
    alpha=0.70
)

plt.title("Customer Segments: Tenure vs Monthly Charges")
plt.xlabel("Tenure (Months)")
plt.ylabel("Monthly Charges")
plt.legend(
    title="Segment / Service Count",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    os.path.join(CHART_FOLDER, "phase5_customer_segments.png"),
    bbox_inches="tight"
)

plt.close()

print("Saved: charts/phase5_customer_segments.png")

# ---------------------------------------------------------------
# 6. Persist the fitted segmentation model
#
# The scaler and the K-Means object are saved together with the name map so
# that new data is assigned a cluster by the trained centroids. Without this
# the dashboard has to guess a segment from a hand-written rule, and the
# feature the model sees at prediction time no longer matches the feature it
# was trained on.
# ---------------------------------------------------------------
segmentation_artifacts = {
    "scaler": scaler,
    "kmeans": final_kmeans,
    "segment_names": segment_names,
    "segmentation_columns": SEGMENTATION_COLUMNS,
    "optimal_k": OPTIMAL_K,
}

joblib.dump(segmentation_artifacts, SEGMENTATION_MODEL_FILE)

print("\n6. SEGMENTATION MODEL PERSISTED")
print(f"Saved: {SEGMENTATION_MODEL_FILE}")

# ---------------------------------------------------------------
# 7. Merge the phase 4 engineered features
#
# Phases 6 to 8 read this file. Any engineered feature not merged here is
# computed by phase 4 and then silently discarded, which is what previously
# kept ARPU, ChargeRatio, TenureBand, ContractRiskFlag, ValueTier and
# AutoPayFlag out of every model.
# ---------------------------------------------------------------
if os.path.exists(FEATURES_FILE):
    engineered = pd.read_csv(FEATURES_FILE)

    # ServiceCount already exists on df and is identical by construction, so
    # it is excluded from the merge to avoid a duplicated column.
    merge_columns = ["customerID"] + [
        column for column in ENGINEERED_FEATURE_COLUMNS
        if column != "ServiceCount"
    ]

    before_columns = df.shape[1]
    df = df.merge(engineered[merge_columns], on="customerID", how="left")

    print("\n7. ENGINEERED FEATURE MERGE")
    print(f"Merged from: {FEATURES_FILE}")
    print(f"Columns added: {df.shape[1] - before_columns}")
    print(f"Features now available to phases 6-8: {merge_columns[1:]}")

    unmatched = df[merge_columns[1]].isna().sum()
    print(f"Rows without a matching engineered record: {unmatched}")

    if unmatched > 0:
        raise SystemExit(
            "Engineered features are missing for some customers. "
            "Re-run phase 4 before phase 5."
        )
else:
    raise SystemExit(
        f"Required input not found: {FEATURES_FILE}\n"
        "Run phase4_feature_engineering.py before phase 5."
    )

# Save segmented dataset
df.to_csv(OUTPUT_FILE, index=False)

print("\n8. OUTPUT")
print(f"Segmented dataset saved successfully: {OUTPUT_FILE}")
print(f"Final shape: {df.shape}")

print("\n9. PHASE 5 SUMMARY")
print("- Used tenure as the longevity proxy.")
print("- Used ServiceCount as the engagement proxy.")
print("- Used MonthlyCharges as the monetary-value proxy.")
print("- Standardized all clustering dimensions before K-Means.")
print(f"- Used {OPTIMAL_K} clusters for business-actionable customer segmentation.")
print("- Profiled clusters by tenure, charges, service adoption, revenue, and churn.")
print("- Named segments from centroid geometry only, so no target information")
print("  reaches the models through the segment label.")
print("- Persisted the scaler and K-Means model for consistent scoring of new data.")
print("- Merged the phase 4 engineered features so they reach phases 6-8.")
print("- Added CustomerSegment for later modeling and dashboard analysis.")