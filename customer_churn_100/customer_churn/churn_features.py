"""
Shared feature-engineering and segmentation helpers.

Every derived feature that reaches a model is built here, so the training
pipeline (phases 4 to 8) and the serving path (the Streamlit dashboard) construct
features in exactly the same way.

Two rules are enforced by this module:

1. No function reads the churn target. Every feature and every segment name is
   derived from customer attributes only, so nothing that reaches a model can
   carry information about the outcome being predicted.

2. Anything learned from the training data (ARPU quartile edges, the clustering
   scaler, the K-Means centroids) is returned as an artefact dictionary and
   persisted to disk, so unseen customers are transformed with the boundaries
   fitted during training rather than boundaries recomputed from themselves.
"""

import numpy as np
import pandas as pd

SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

AUTOMATIC_PAYMENT_METHODS = [
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]

TENURE_BINS = [-1, 6, 12, 24, 48, 72]
TENURE_LABELS = [
    "0-6 months",
    "7-12 months",
    "13-24 months",
    "25-48 months",
    "49-72 months",
]

VALUE_TIER_LABELS = ["Low", "Medium", "High", "Premium"]

# The engineered columns that phases 6 to 8 consume as model inputs.
ENGINEERED_FEATURE_COLUMNS = [
    "TenureBand",
    "ARPU",
    "ChargeRatio",
    "ServiceCount",
    "ContractRiskFlag",
    "ValueTier",
    "AutoPayFlag",
]

SEGMENTATION_COLUMNS = ["tenure", "ServiceCount", "MonthlyCharges"]


# ---------------------------------------------------------------------------
# Engineered features
# ---------------------------------------------------------------------------
def add_service_count(df):
    """Count add-on services. Requires the phase 2 label standardisation."""
    available = [c for c in SERVICE_COLUMNS if c in df.columns]

    if not available:
        df["ServiceCount"] = 0
    else:
        df["ServiceCount"] = df[available].astype(str).eq("Yes").sum(axis=1)

    return df


def add_tenure_band(df):
    df["TenureBand"] = pd.cut(
        df["tenure"],
        bins=TENURE_BINS,
        labels=TENURE_LABELS,
    )
    return df


def add_arpu(df):
    """Average revenue per active month.

    Initialise to a safe default and assign only where the division is defined.
    Computing first and repairing afterwards is fragile: an inf that survives
    into a scaler poisons the mean and standard deviation of the whole column.
    """
    df["ARPU"] = 0.0
    non_zero_tenure = df["tenure"] > 0

    df.loc[non_zero_tenure, "ARPU"] = (
        df.loc[non_zero_tenure, "TotalCharges"] / df.loc[non_zero_tenure, "tenure"]
    )
    return df


def add_charge_ratio(df):
    """Current monthly charge against the historical average monthly charge.

    A value above 1 means the customer now pays more than they have averaged,
    which is a plausible recent-price-increase signal.
    """
    df["ChargeRatio"] = 0.0
    valid = (df["tenure"] > 0) & (df["TotalCharges"] > 0)

    df.loc[valid, "ChargeRatio"] = df.loc[valid, "MonthlyCharges"] / (
        df.loc[valid, "TotalCharges"] / df.loc[valid, "tenure"]
    )
    return df


def add_contract_risk_flag(df):
    df["ContractRiskFlag"] = (df["Contract"] == "Month-to-month").astype(int)
    return df


def add_auto_pay_flag(df):
    df["AutoPayFlag"] = df["PaymentMethod"].isin(AUTOMATIC_PAYMENT_METHODS).astype(int)
    return df


def fit_value_tier_edges(df):
    """Learn ARPU quartile edges on the training data and return them.

    The edges are persisted so that an unseen customer is placed into the tier
    the training distribution defines, rather than into a tier recomputed from
    whatever batch they happen to arrive in.
    """
    _, edges = pd.qcut(df["ARPU"], q=4, retbins=True, duplicates="drop")

    edges = list(edges)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def apply_value_tier(df, edges):
    labels = VALUE_TIER_LABELS[: len(edges) - 1]

    df["ValueTier"] = pd.cut(
        df["ARPU"],
        bins=edges,
        labels=labels,
        include_lowest=True,
    )

    # An ARPU outside the fitted range still needs a tier rather than a null.
    if df["ValueTier"].isna().any():
        df["ValueTier"] = df["ValueTier"].cat.add_categories(["Unknown"]).fillna("Unknown")

    return df


def add_engineered_features(df, artifacts=None):
    """Build every engineered feature.

    Pass artifacts=None to fit (training). Pass the dictionary returned by the
    fitting call to transform unseen data with the training boundaries.

    Returns (dataframe, artifacts).
    """
    df = df.copy()

    df = add_service_count(df)
    df = add_tenure_band(df)
    df = add_arpu(df)
    df = add_charge_ratio(df)
    df = add_contract_risk_flag(df)
    df = add_auto_pay_flag(df)

    if artifacts is None:
        artifacts = {"value_tier_edges": fit_value_tier_edges(df)}

    df = apply_value_tier(df, artifacts["value_tier_edges"])

    return df, artifacts


# ---------------------------------------------------------------------------
# Segment naming, derived from cluster geometry only
# ---------------------------------------------------------------------------
def name_segments_from_centroids(centroids):
    """Name each cluster from where its centroid sits, never from its churn rate.

    centroids: DataFrame indexed by cluster id with columns tenure,
    MonthlyCharges and ServiceCount, in original (unscaled) units.

    Naming a cluster after its observed churn rate and then feeding that name to
    a classifier leaks the target: the label would summarise the very outcome
    the model is asked to predict. Positioning against the median centroid uses
    customer attributes only, so the resulting label is reproducible for a new
    customer whose outcome is unknown.
    """
    tenure_median = centroids["tenure"].median()
    charge_median = centroids["MonthlyCharges"].median()

    names = {}

    for cluster_id, row in centroids.iterrows():
        established = row["tenure"] >= tenure_median
        high_spend = row["MonthlyCharges"] >= charge_median

        if established and high_spend:
            names[cluster_id] = "Established High-Value"
        elif established and not high_spend:
            names[cluster_id] = "Established Budget"
        elif not established and high_spend:
            names[cluster_id] = "New High-Spend"
        else:
            names[cluster_id] = "New Budget"

    # Guarantee uniqueness deterministically if a quadrant holds more than one
    # cluster, which can happen when OPTIMAL_K is changed away from 4.
    duplicated = [n for n in set(names.values()) if list(names.values()).count(n) > 1]

    for name in duplicated:
        tied = [cid for cid, n in names.items() if n == name]
        # Rank the tied clusters by service adoption, then by spend.
        ordered = centroids.loc[tied].sort_values(
            ["ServiceCount", "MonthlyCharges"], ascending=False
        ).index.tolist()

        qualifiers = ["High-Engagement", "Mid-Engagement", "Low-Engagement"]

        for position, cluster_id in enumerate(ordered):
            qualifier = (
                qualifiers[position]
                if position < len(qualifiers)
                else f"Group {position + 1}"
            )
            names[cluster_id] = f"{name} ({qualifier})"

    return names


def assign_clusters(df, segmentation_artifacts):
    """Assign Cluster and CustomerSegment to new data using the fitted model."""
    df = df.copy()

    if "ServiceCount" not in df.columns:
        df = add_service_count(df)

    scaler = segmentation_artifacts["scaler"]
    kmeans = segmentation_artifacts["kmeans"]
    segment_names = segmentation_artifacts["segment_names"]

    scaled = scaler.transform(df[SEGMENTATION_COLUMNS])

    df["Cluster"] = kmeans.predict(scaled)
    df["CustomerSegment"] = df["Cluster"].map(segment_names)

    return df


# ---------------------------------------------------------------------------
# Model matrix
# ---------------------------------------------------------------------------
# Columns removed before modelling.
#
# CustomerSegment is a presentation label only: Cluster carries the identical
# information as a nominal code, so passing both would duplicate the signal and
# re-expose the naming rule to the model.
#
# ContractRiskFlag equals (Contract == "Month-to-month") for every row, and the
# one-hot encoding of Contract already produces that exact indicator. Keeping
# both gives the model the same column twice; a tree ensemble then splits the
# importance between them arbitrarily, which distorts every importance chart
# and SHAP explanation built on top. Removing it cost nothing: hold-out
# ROC-AUC moved from 0.8470 to 0.8472. The flag is still computed and kept in
# the datasets as a readable business attribute.
MODEL_EXCLUDED_COLUMNS = [
    "customerID",
    "Churn",
    "ChurnFlag",
    "CustomerSegment",
    "ContractRiskFlag",
]


def to_nominal_cluster(df):
    """Cast Cluster to a string label.

    Left as an integer the column is scaled as though cluster 3 were three
    times cluster 1, which is meaningless for an arbitrary K-Means label. Both
    the training matrix and the dashboard's scoring path call this, so the
    encoder sees the same representation in each.
    """
    df = df.copy()

    if "Cluster" in df.columns and not df["Cluster"].astype(str).str.startswith(
        "Cluster_"
    ).all():
        df["Cluster"] = "Cluster_" + df["Cluster"].astype(int).astype(str)

    return df


def build_model_matrix(df):
    """Return the feature matrix phases 6 to 8 train on."""
    X = df.drop(
        columns=[c for c in MODEL_EXCLUDED_COLUMNS if c in df.columns]
    )

    return to_nominal_cluster(X)


# ---------------------------------------------------------------------------
# Per-customer churn drivers from SHAP
# ---------------------------------------------------------------------------
# Every high-risk customer is on a month-to-month contract, so contract term is
# the shared condition of the whole cohort rather than something that tells
# one customer apart from another. A driver rule that tests contract first
# therefore puts every customer in the same category. Instead, contract is
# treated as the cohort's shared base driver, and each customer's personal
# driver is the actionable lever with the largest SHAP contribution to their
# own predicted risk.
#
# Tenure, the segment cluster and demographics are deliberately not levers: an
# offer cannot change how long someone has been a customer or who they are.
# They describe when and to whom to reach out, not what to offer.
SHARED_DRIVER_FEATURES = ["Contract"]

LEVER_GROUPS = {
    "Service quality": ["InternetService"],
    "Billing and payment": ["PaymentMethod", "PaperlessBilling", "AutoPayFlag"],
    "Protection and support": [
        "TechSupport", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    ],
    "Engagement and bundling": [
        "StreamingTV", "StreamingMovies", "ServiceCount", "MultipleLines",
        "PhoneService",
    ],
    "Price and plan fit": [
        "MonthlyCharges", "ChargeRatio", "ARPU", "ValueTier", "TotalCharges",
    ],
}

LEVER_ACTIONS = {
    "Service quality":
        "Service-quality audit with proactive fix and loyalty credit",
    "Billing and payment":
        "Move to automatic payment with a billing reward",
    "Protection and support":
        "Complimentary tech-support and online-security trial",
    "Engagement and bundling":
        "Bundled add-on offer to deepen product engagement",
    "Price and plan fit":
        "Plan right-sizing review and personalised tariff",
}

SHARED_ACTION = "Discounted 12- or 24-month contract with loyalty bonus"


def aggregate_shap_by_feature(shap_values, transformed_names, original_columns):
    """Sum one-hot SHAP columns back to the original feature they came from.

    The encoder turns Contract into Contract_Month-to-month, Contract_One year
    and Contract_Two year. Their SHAP values are additive, so summing them gives
    the total contribution of Contract to that customer's prediction.
    """
    ordered = sorted(original_columns, key=len, reverse=True)

    def original(name):
        base = name.split("__", 1)[1] if "__" in name else name
        for column in ordered:
            if base == column or base.startswith(column + "_"):
                return column
        return base

    groups = [original(n) for n in transformed_names]

    frame = pd.DataFrame(shap_values, columns=list(transformed_names))
    return frame.T.groupby(groups, sort=False).sum().T


def assign_shap_drivers(feature_shap):
    """Return per-customer lever contributions and primary/secondary levers."""
    levers = pd.DataFrame({
        lever: feature_shap[[c for c in columns if c in feature_shap.columns]].sum(axis=1)
        for lever, columns in LEVER_GROUPS.items()
    })

    ranked = np.argsort(-levers.to_numpy(), axis=1)
    names = np.array(levers.columns)

    result = pd.DataFrame(index=feature_shap.index)
    result["SharedDriverContribution"] = feature_shap[
        [c for c in SHARED_DRIVER_FEATURES if c in feature_shap.columns]
    ].sum(axis=1)
    result["PrimaryLever"] = names[ranked[:, 0]]
    result["SecondaryLever"] = names[ranked[:, 1]]
    result["PrimaryLeverContribution"] = levers.to_numpy()[
        np.arange(len(levers)), ranked[:, 0]
    ]

    # A lever that lowers risk is not a reason to act on it.
    no_positive_lever = result["PrimaryLeverContribution"] <= 0
    result.loc[no_positive_lever, "PrimaryLever"] = "No actionable lever"

    for lever in levers.columns:
        result[f"SHAP_{lever.replace(' ', '_')}"] = levers[lever]

    return result
