import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

INPUT_FILE = "data/Customer-Churn-Cleaned.csv"
OUTPUT_FILE = "data/Customer-Churn-Segmented.csv"
CHART_FOLDER = "charts"

RANDOM_STATE = 42
OPTIMAL_K = 4

sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams["figure.dpi"] = 120

os.makedirs(CHART_FOLDER, exist_ok=True)

print("=" * 70)
print("PHASE 5: CUSTOMER SEGMENTATION")
print("=" * 70)

# Load cleaned dataset
df = pd.read_csv(INPUT_FILE)

# Create engagement feature: number of active add-on services
service_columns = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

df["ServiceCount"] = df[service_columns].eq("Yes").sum(axis=1)

# Create churn flag only for cluster profiling
df["ChurnFlag"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# RFM-style input features for clustering
segmentation_columns = [
    "tenure",
    "ServiceCount",
    "MonthlyCharges"
]

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

# Create initial business-readable segment names
profile = cluster_profile.set_index("Cluster")

tenure_median = cluster_profile["MeanTenure"].median()
charge_median = cluster_profile["MeanMonthlyCharges"].median()
churn_median = cluster_profile["ChurnRate"].median()

segment_names = {}

for cluster_id, values in profile.iterrows():
    tenure = values["MeanTenure"]
    charges = values["MeanMonthlyCharges"]
    churn = values["ChurnRate"]

    if tenure >= tenure_median and charges >= charge_median and churn < churn_median:
        segment_names[cluster_id] = "High-Value Loyal"

    elif tenure >= tenure_median and charges >= charge_median and churn >= churn_median:
        segment_names[cluster_id] = "High-Value At-Risk"

    elif tenure < tenure_median and churn >= churn_median:
        segment_names[cluster_id] = "New & Fragile"

    else:
        segment_names[cluster_id] = "Low-Value Price-Sensitive"

# Ensure each segment has a unique name
name_counts = {}

for cluster_id, segment_name in segment_names.items():
    if segment_name in name_counts:
        name_counts[segment_name] += 1
        segment_names[cluster_id] = (
            f"{segment_name} {name_counts[segment_name]}"
        )
    else:
        name_counts[segment_name] = 1

# Append customer segment labels
df["CustomerSegment"] = df["Cluster"].map(segment_names)
cluster_profile["CustomerSegment"] = cluster_profile["Cluster"].map(
    segment_names
)

print("\n4. BUSINESS-READABLE SEGMENTS")
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

# Save segmented dataset
df.to_csv(OUTPUT_FILE, index=False)

print("\n5. OUTPUT")
print(f"Segmented dataset saved successfully: {OUTPUT_FILE}")

print("\n6. PHASE 5 SUMMARY")
print("- Used tenure as the longevity proxy.")
print("- Used ServiceCount as the engagement proxy.")
print("- Used MonthlyCharges as the monetary-value proxy.")
print("- Standardized all clustering dimensions before K-Means.")
print(f"- Used {OPTIMAL_K} clusters for business-actionable customer segmentation.")
print("- Profiled clusters by tenure, charges, service adoption, revenue, and churn.")
print("- Added CustomerSegment for later modeling and dashboard analysis.")