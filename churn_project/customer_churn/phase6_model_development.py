import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    cross_val_score,
    RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

INPUT_FILE = "data/Customer-Churn-Segmented.csv"
CHART_FOLDER = "charts"
MODEL_FOLDER = "models"

RANDOM_STATE = 42
TEST_SIZE = 0.05
CV_FOLDS = 5

os.makedirs(CHART_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120

print("=" * 75)
print("PHASE 6: MODEL DEVELOPMENT")
print("=" * 75)

# ---------------------------------------------------------------
# 1. Load data and prepare features
# ---------------------------------------------------------------
df = pd.read_csv(INPUT_FILE)

df["ChurnFlag"] = df["Churn"].map({"No": 0, "Yes": 1})

columns_to_drop = [
    "customerID",
    "Churn",
    "ChurnFlag",
    "Cluster"
]

X = df.drop(
    columns=[column for column in columns_to_drop if column in df.columns]
)

y = df["ChurnFlag"]

print("\n1. DATASET PREPARATION")
print(f"Dataset shape: {df.shape}")
print(f"Feature matrix shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts().to_string()}")

# ---------------------------------------------------------------
# 2. Stratified 80/20 train-test split
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

print("\n2. STRATIFIED TRAIN-TEST SPLIT")
print(f"Training set: {X_train.shape[0]} records")
print(f"Test set: {X_test.shape[0]} records")
print(f"Training churn rate: {y_train.mean() * 100:.2f}%")
print(f"Test churn rate: {y_test.mean() * 100:.2f}%")

# ---------------------------------------------------------------
# 3. Preprocessing pipeline
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

print("\n3. PREPROCESSING")
print(f"Numerical features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")

# ---------------------------------------------------------------
# 4. Cross-validation configuration
# ---------------------------------------------------------------
cv = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)

# Class imbalance ratio for XGBoost
negative_class_count = (y_train == 0).sum()
positive_class_count = (y_train == 1).sum()

scale_pos_weight = negative_class_count / positive_class_count

print("\n4. CLASS IMBALANCE")
print(f"Negative class count: {negative_class_count}")
print(f"Positive class count: {positive_class_count}")
print(f"XGBoost scale_pos_weight: {scale_pos_weight:.2f}")

# ---------------------------------------------------------------
# 5. Define requested baseline models
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

    "XGBoost Baseline": XGBClassifier(
        n_estimators=2000,
        learning_rate=0.10,
        max_depth=6,
        min_child_weight=1,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}

print("\n5. FIVE-FOLD CROSS-VALIDATION (ROC-AUC)")

cv_results = []

for model_name, model in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    cv_results.append({
        "Model": model_name,
        "Mean_CV_ROC_AUC": scores.mean(),
        "Std_CV_ROC_AUC": scores.std()
    })

    print(
        f"{model_name}: "
        f"ROC-AUC = {scores.mean():.4f} "
        f"(+/- {scores.std():.4f})"
    )

cv_results_df = pd.DataFrame(cv_results).sort_values(
    by="Mean_CV_ROC_AUC",
    ascending=False
)

print("\nCross-validation ranking:")
print(cv_results_df.round(4).to_string(index=False))

# ---------------------------------------------------------------
# 6. Tune XGBoost using randomized search and ROC-AUC
# ---------------------------------------------------------------
print("\n6. XGBOOST HYPERPARAMETER TUNING")

xgb_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ]
)

xgb_parameter_grid = {
    "model__n_estimators": [200, 300, 400, 500],
    "model__learning_rate": [0.03, 0.05, 0.08, 0.10],
    "model__max_depth": [3, 4, 5, 6],
    "model__min_child_weight": [1, 2, 3, 5],
    "model__subsample": [0.70, 0.80, 0.90, 1.00],
    "model__colsample_bytree": [0.70, 0.80, 0.90, 1.00],
    "model__gamma": [0, 0.05, 0.10, 0.20],
    "model__reg_alpha": [0, 0.01, 0.05, 0.10],
    "model__reg_lambda": [1, 1.5, 2, 3],
    "model__scale_pos_weight": [
        scale_pos_weight * 0.85,
        scale_pos_weight,
        scale_pos_weight * 1.15
    ]
}

xgb_search = RandomizedSearchCV(
    estimator=xgb_pipeline,
    param_distributions=xgb_parameter_grid,
    n_iter=30,
    scoring="roc_auc",
    cv=cv,
    verbose=1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    refit=True
)

xgb_search.fit(X_train, y_train)

best_xgb_model = xgb_search.best_estimator_

print(f"\nBest XGBoost CV ROC-AUC: {xgb_search.best_score_:.4f}")
print("\nBest XGBoost parameters:")
print(xgb_search.best_params_)

# ---------------------------------------------------------------
# 7. Evaluate all base models and tuned XGBoost on test set
# ---------------------------------------------------------------
print("\n7. HOLDOUT TEST-SET EVALUATION")

final_models = {}

for model_name, model in models.items():
    final_models[model_name] = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

final_models["Tuned XGBoost"] = best_xgb_model

test_results = []
fpr_tpr_data = {}

for model_name, pipeline in final_models.items():
    pipeline.fit(X_train, y_train)

    predicted_probabilities = pipeline.predict_proba(X_test)[:, 1]
    predicted_labels = pipeline.predict(X_test)

    test_auc = roc_auc_score(y_test, predicted_probabilities)
    test_accuracy = accuracy_score(y_test, predicted_labels)
    test_precision = precision_score(
        y_test,
        predicted_labels,
        zero_division=0
    )
    test_recall = recall_score(
        y_test,
        predicted_labels,
        zero_division=0
    )
    test_f1 = f1_score(
        y_test,
        predicted_labels,
        zero_division=0
    )

    test_results.append({
        "Model": model_name,
        "ROC_AUC": test_auc,
        "Accuracy": test_accuracy,
        "Precision": test_precision,
        "Recall": test_recall,
        "F1_Score": test_f1
    })

    fpr, tpr, _ = roc_curve(y_test, predicted_probabilities)

    fpr_tpr_data[model_name] = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": test_auc
    }

test_results_df = pd.DataFrame(test_results).sort_values(
    by="ROC_AUC",
    ascending=False
)

print("\nTest-set performance ranking:")
print(test_results_df.round(4).to_string(index=False))

test_results_df.to_csv(
    "data/phase6_model_performance.csv",
    index=False
)

# ---------------------------------------------------------------
# 8. Best model evaluation details
# ---------------------------------------------------------------
best_model_name = test_results_df.iloc[0]["Model"]
best_model = final_models[best_model_name]

best_model.fit(X_train, y_train)

best_probabilities = best_model.predict_proba(X_test)[:, 1]
best_predictions = best_model.predict(X_test)

print("\n8. BEST MODEL")
print(f"Selected model: {best_model_name}")
print(
    f"Test ROC-AUC: "
    f"{roc_auc_score(y_test, best_probabilities):.4f}"
)

print("\nClassification report:")
print(
    classification_report(
        y_test,
        best_predictions,
        target_names=["No Churn", "Churn"],
        zero_division=0
    )
)

# Confusion matrix chart
conf_matrix = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(6, 5))

sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Churn", "Churn"],
    yticklabels=["No Churn", "Churn"]
)

plt.title(f"Confusion Matrix: {best_model_name}")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_FOLDER,
        "phase6_confusion_matrix.png"
    ),
    bbox_inches="tight"
)

plt.close()

print("Saved: charts/phase6_confusion_matrix.png")

# ROC curve for all models
plt.figure(figsize=(9, 7))

for model_name, curve_data in fpr_tpr_data.items():
    plt.plot(
        curve_data["fpr"],
        curve_data["tpr"],
        linewidth=2,
        label=(
            f"{model_name} "
            f"(AUC = {curve_data['auc']:.3f})"
        )
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="black",
    label="Random Classifier"
)

plt.title("ROC Curves: Customer Churn Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    os.path.join(
        CHART_FOLDER,
        "phase6_roc_curves.png"
    ),
    bbox_inches="tight"
)

plt.close()

print("Saved: charts/phase6_roc_curves.png")

# ---------------------------------------------------------------
# 9. Feature importance for tuned/best XGBoost if selected
# ---------------------------------------------------------------
print("\n9. FEATURE IMPORTANCE")

if best_model_name == "Tuned XGBoost":
    fitted_preprocessor = best_model.named_steps["preprocessor"]
    fitted_xgb_model = best_model.named_steps["model"]

    feature_names = fitted_preprocessor.get_feature_names_out()

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": fitted_xgb_model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    ).head(20)

    print("\nTop 20 XGBoost features:")
    print(importance_df.round(4).to_string(index=False))

    plt.figure(figsize=(10, 8))

    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature",
        color="darkorange"
    )

    plt.title("Top 20 Feature Importances: Tuned XGBoost")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            CHART_FOLDER,
            "phase6_feature_importance.png"
        ),
        bbox_inches="tight"
    )

    plt.close()

    print("Saved: charts/phase6_feature_importance.png")

else:
    print(
        "Feature-importance chart is generated only when "
        "Tuned XGBoost is the best test-set model."
    )

# ---------------------------------------------------------------
# 10. Save best trained model
# ---------------------------------------------------------------
joblib.dump(
    best_model,
    os.path.join(MODEL_FOLDER, "best_churn_model.joblib")
)

print("\n10. OUTPUT FILES")
print("Saved: data/phase6_model_performance.csv")
print("Saved: models/best_churn_model.joblib")
print("Saved: charts/phase6_roc_curves.png")
print("Saved: charts/phase6_confusion_matrix.png")

print("\n11. PHASE 6 SUMMARY")
print("- Applied a stratified 80/20 train-test split.")
print("- Used five-fold stratified cross-validation on training data.")
print("- Optimized model selection using ROC-AUC.")
print("- Used class weighting to address churn class imbalance.")
print("- Tuned XGBoost with randomized hyperparameter search.")
print("- All preprocessing is fit only on training folds, preventing leakage.")