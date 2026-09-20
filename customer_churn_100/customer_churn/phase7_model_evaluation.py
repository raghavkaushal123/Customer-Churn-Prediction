import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from churn_features import MODEL_EXCLUDED_COLUMNS, build_model_matrix

warnings.filterwarnings("ignore")

INPUT_FILE = "data/Customer-Churn-Segmented.csv"
PERFORMANCE_FILE = "data/phase6_model_performance.csv"
CHART_FOLDER = "charts"
OUTPUT_FILE = "data/phase7_model_evaluation_results.csv"

RANDOM_STATE = 42
# A 5% hold-out left 353 records with only 94 churners, an interval on
# ROC-AUC of roughly +/-0.045 -- wider than the gap between the top models,
# so they could not be ranked. 20% gives 1409 records and ~374 churners.
TEST_SIZE = 0.20
DEFAULT_THRESHOLD = 0.50

os.makedirs(CHART_FOLDER, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

print("=" * 75)
print("PHASE 7: MODEL EVALUATION AND COMPARISON")
print("=" * 75)

# ---------------------------------------------------------------
# 1. Load dataset and recreate Phase 6 split
# ---------------------------------------------------------------
df = pd.read_csv(INPUT_FILE)

df["ChurnFlag"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

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

print("\n1. EVALUATION DATASET")
print(f"Split ratio: {1 - TEST_SIZE:.0%} train / {TEST_SIZE:.0%} test")
print(f"Training records: {len(X_train)}")
print(f"Test records: {len(X_test)}")
print(f"Churners in test set: {int(y_test.sum())}")
print(f"Test churn rate: {y_test.mean() * 100:.2f}%")
print(f"Decision threshold: {DEFAULT_THRESHOLD}")

# ---------------------------------------------------------------
# 2. Build preprocessing pipeline
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
# 3. Define models
# Match the models from the approved Phase 6 run
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        penalty="l2",
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=2000,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Support Vector Machine": SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        C=1.0,
        gamma="scale",
        random_state=RANDOM_STATE
    ),

    "Tuned XGBoost": XGBClassifier(
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
}

print("\n2. MODELS FOR COMPARISON")
for model_name in models:
    print(f"- {model_name}")

# ---------------------------------------------------------------
# 4. Evaluate models
# ---------------------------------------------------------------
print("\n3. CONSOLIDATED MODEL EVALUATION")

results = []
model_outputs = {}

for model_name, model in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= DEFAULT_THRESHOLD).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )
    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)

    matrix = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = matrix.ravel()

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "True_Negative": tn,
        "False_Positive": fp,
        "False_Negative": fn,
        "True_Positive": tp
    })

    model_outputs[model_name] = {
        "pipeline": pipeline,
        "probabilities": probabilities,
        "predictions": predictions,
        "confusion_matrix": matrix
    }

results_df = pd.DataFrame(results).sort_values(
    by="ROC_AUC",
    ascending=False
)

print(results_df.round(4).to_string(index=False))

results_df.to_csv(OUTPUT_FILE, index=False)

# ---------------------------------------------------------------
# 5. Select production model
# ROC-AUC is primary, then accuracy / recall are checked
# ---------------------------------------------------------------
production_model_name = results_df.iloc[0]["Model"]

production_model_metrics = results_df[
    results_df["Model"] == production_model_name
].iloc[0]

print("\n4. PRODUCTION MODEL SELECTION")
print(f"Selected model: {production_model_name}")
print(f"ROC-AUC: {production_model_metrics['ROC_AUC']:.4f}")
print(f"Accuracy: {production_model_metrics['Accuracy']:.4f}")
print(f"Recall: {production_model_metrics['Recall']:.4f}")
print(f"PR-AUC: {production_model_metrics['PR_AUC']:.4f}")

if production_model_metrics["Accuracy"] >= 0.85:
    print("Accuracy is above the stated 85% target.")
else:
    print(
        "Accuracy is below 85%; selection is based on highest ROC-AUC "
        "and acceptable churn recall, as agreed in model development."
    )

# ---------------------------------------------------------------
# 6. Classification report for production model
# ---------------------------------------------------------------
production_output = model_outputs[production_model_name]

print("\n5. PRODUCTION MODEL CLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        production_output["predictions"],
        target_names=["No Churn", "Churn"],
        zero_division=0
    )
)

# ---------------------------------------------------------------
# 7. Consolidated comparison chart
# ---------------------------------------------------------------
metrics_for_chart = results_df.melt(
    id_vars="Model",
    value_vars=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score",
        "ROC_AUC",
        "PR_AUC"
    ],
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(13, 7))

sns.barplot(
    data=metrics_for_chart,
    x="Model",
    y="Score",
    hue="Metric"
)

plt.title("Model Performance Comparison")
plt.xlabel("Model")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=15, ha="right")
plt.legend(title="Metric", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_FOLDER,
        "phase7_model_comparison.png"
    ),
    bbox_inches="tight"
)

plt.close()

print("Saved: charts/phase7_model_comparison.png")

# ---------------------------------------------------------------
# 8. Overlay ROC curves
# ---------------------------------------------------------------
plt.figure(figsize=(9, 7))

for model_name, output in model_outputs.items():
    fpr, tpr, _ = roc_curve(
        y_test,
        output["probabilities"]
    )

    auc_score = results_df.loc[
        results_df["Model"] == model_name,
        "ROC_AUC"
    ].iloc[0]

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{model_name} (AUC = {auc_score:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="black",
    label="Random Classifier"
)

plt.title("Overlay ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_FOLDER,
        "phase7_overlay_roc_curves.png"
    ),
    bbox_inches="tight"
)

plt.close()

print("Saved: charts/phase7_overlay_roc_curves.png")

# ---------------------------------------------------------------
# 9. Overlay precision-recall curves
# ---------------------------------------------------------------
plt.figure(figsize=(9, 7))

for model_name, output in model_outputs.items():
    precision_values, recall_values, _ = precision_recall_curve(
        y_test,
        output["probabilities"]
    )

    pr_auc = results_df.loc[
        results_df["Model"] == model_name,
        "PR_AUC"
    ].iloc[0]

    plt.plot(
        recall_values,
        precision_values,
        linewidth=2,
        label=f"{model_name} (PR-AUC = {pr_auc:.3f})"
    )

baseline_precision = y_test.mean()

plt.axhline(
    y=baseline_precision,
    linestyle="--",
    color="black",
    label=f"Baseline Churn Rate = {baseline_precision:.3f}"
)

plt.title("Overlay Precision-Recall Curves")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left")
plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_FOLDER,
        "phase7_precision_recall_curves.png"
    ),
    bbox_inches="tight"
)

plt.close()

print("Saved: charts/phase7_precision_recall_curves.png")

# ---------------------------------------------------------------
# 10. Confusion matrices for all models
# ---------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for axis, (model_name, output) in zip(
    axes,
    model_outputs.items()
):
    sns.heatmap(
        output["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=axis
    )

    axis.set_title(model_name)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")

plt.suptitle(
    "Confusion Matrices at Threshold 0.50",
    fontsize=14,
    y=1.02
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_FOLDER,
        "phase7_all_confusion_matrices.png"
    ),
    bbox_inches="tight"
)

plt.close()

print("Saved: charts/phase7_all_confusion_matrices.png")

# ---------------------------------------------------------------
# 11. Threshold analysis for production model
# Business interpretation only; production threshold stays 0.50
# ---------------------------------------------------------------
print("\n6. PRODUCTION MODEL THRESHOLD ANALYSIS")

thresholds = np.arange(0.10, 0.91, 0.05)
threshold_results = []

for threshold in thresholds:
    threshold_predictions = (
        production_output["probabilities"] >= threshold
    ).astype(int)

    threshold_results.append({
        "Threshold": round(float(threshold), 2),
        "Accuracy": accuracy_score(y_test, threshold_predictions),
        "Precision": precision_score(
            y_test,
            threshold_predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            threshold_predictions,
            zero_division=0
        ),
        "F1_Score": f1_score(
            y_test,
            threshold_predictions,
            zero_division=0
        )
    })

threshold_df = pd.DataFrame(threshold_results)

print(threshold_df.round(4).to_string(index=False))

threshold_df.to_csv(
    "data/phase7_threshold_analysis.csv",
    index=False
)

plt.figure(figsize=(10, 6))

for metric in ["Accuracy", "Precision", "Recall", "F1_Score"]:
    plt.plot(
        threshold_df["Threshold"],
        threshold_df[metric],
        marker="o",
        linewidth=2,
        label=metric
    )

plt.axvline(
    DEFAULT_THRESHOLD,
    color="black",
    linestyle="--",
    label="Current Production Threshold = 0.50"
)

plt.title(
    f"Threshold Trade-Off Analysis: {production_model_name}"
)

plt.xlabel("Churn Probability Threshold")
plt.ylabel("Metric Score")
plt.ylim(0, 1.05)
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_FOLDER,
        "phase7_threshold_tradeoff.png"
    ),
    bbox_inches="tight"
)

plt.close()

print("Saved: charts/phase7_threshold_tradeoff.png")

# ---------------------------------------------------------------
# 12. Final summary
# ---------------------------------------------------------------
print("\n7. OUTPUT FILES")
print(f"Saved: {OUTPUT_FILE}")
print("Saved: data/phase7_threshold_analysis.csv")
print("Saved: charts/phase7_model_comparison.png")
print("Saved: charts/phase7_overlay_roc_curves.png")
print("Saved: charts/phase7_precision_recall_curves.png")
print("Saved: charts/phase7_all_confusion_matrices.png")
print("Saved: charts/phase7_threshold_tradeoff.png")

print("\n8. PHASE 7 SUMMARY")
print("- Compared all required models on one consolidated evaluation table.")
print("- Evaluated Accuracy, Precision, Recall, F1-score, ROC-AUC, and PR-AUC.")
print("- Created overlay ROC and precision-recall curves.")
print("- Created confusion matrices for every model.")
print("- Selected the production model primarily using ROC-AUC.")
print("- Documented the decision-threshold trade-off for churn operations.")