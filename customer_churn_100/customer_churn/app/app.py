import os
import sys
import sqlite3
import hashlib
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(PROJECT_ROOT)

from churn_features import (  # noqa: E402  (import needs PROJECT_ROOT on the path)
    add_engineered_features,
    add_service_count,
    assign_clusters,
    to_nominal_cluster,
)

DATA_FOLDER = os.path.join(PROJECT_ROOT, "data")
CHART_FOLDER = os.path.join(PROJECT_ROOT, "charts")
MODEL_FOLDER = os.path.join(PROJECT_ROOT, "models")
DATABASE_FILE = os.path.join(DATA_FOLDER, "users.db")

st.set_page_config(
    page_title="Customer Churn Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)

sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------
# Login and account creation functions
# ---------------------------------------------------------------
def get_database_connection():
    return sqlite3.connect(DATABASE_FILE)


def initialize_database():
    connection = get_database_connection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()


def hash_password(password):
    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


def create_account(username, password):
    username = username.strip()

    if len(username) < 3:
        return False, "Username must be at least 3 characters."

    if len(password) < 6:
        return False, "Password must be at least 6 characters."

    connection = get_database_connection()
    cursor = connection.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO users (username, password_hash)
            VALUES (?, ?)
            """,
            (
                username,
                hash_password(password)
            )
        )

        connection.commit()
        return True, "Account created successfully. Please log in."

    except sqlite3.IntegrityError:
        return False, "Username already exists. Please choose another."

    finally:
        connection.close()


def check_login(username, password):
    username = username.strip()

    connection = get_database_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        SELECT password_hash
        FROM users
        WHERE username = ?
        """,
        (username,)
    )

    result = cursor.fetchone()
    connection.close()

    if result is None:
        return False

    return result[0] == hash_password(password)


def show_login_screen():
    st.title("🔐 Customer Churn Intelligence")
    st.subheader("Login or Create an Account")
    st.write(
        "Sign in to access the customer churn dashboard, "
        "risk predictions, retention recommendations, and "
        "batch-scoring features."
    )

    login_tab, signup_tab = st.tabs([
        "Login",
        "Create Account"
    ])

    with login_tab:
        st.subheader("Login")

        login_username = st.text_input(
            "Username",
            key="login_username"
        )

        login_password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button("Login", key="login_button"):
            if check_login(login_username, login_password):
                st.session_state.authenticated = True
                st.session_state.current_user = (
                    login_username.strip()
                )
                st.success("Login successful.")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with signup_tab:
        st.subheader("Create Account")

        new_username = st.text_input(
            "Choose a Username",
            key="new_username"
        )

        new_password = st.text_input(
            "Choose a Password",
            type="password",
            key="new_password"
        )

        confirm_password = st.text_input(
            "Confirm Password",
            type="password",
            key="confirm_password"
        )

        if st.button(
            "Create Account",
            key="create_account_button"
        ):
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                success, message = create_account(
                    new_username,
                    new_password
                )

                if success:
                    st.success(message)
                else:
                    st.error(message)


# ---------------------------------------------------------------
# Session initialization and access protection
# ---------------------------------------------------------------
initialize_database()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "current_user" not in st.session_state:
    st.session_state.current_user = ""

if not st.session_state.authenticated:
    show_login_screen()
    st.stop()


# ---------------------------------------------------------------
# Dashboard helper functions
# ---------------------------------------------------------------
@st.cache_data
def load_csv(filename):
    return pd.read_csv(
        os.path.join(DATA_FOLDER, filename)
    )


@st.cache_resource
def load_model(filename):
    return joblib.load(
        os.path.join(MODEL_FOLDER, filename)
    )


def show_chart(filename, caption):
    chart_path = os.path.join(CHART_FOLDER, filename)

    if os.path.exists(chart_path):
        st.image(
            chart_path,
            caption=caption,
            use_container_width=True
        )
    else:
        st.warning(f"Chart not found: {filename}")


def currency(value):
    return f"${value:,.2f}"


def create_service_count(dataframe):
    """Retained for backward compatibility; delegates to the shared module."""
    return add_service_count(dataframe)


@st.cache_resource
def load_feature_artifacts():
    """ARPU quartile edges fitted by phase 4.

    Reusing the trained edges means an uploaded customer is placed in the tier
    the training distribution defines, instead of a tier recomputed from
    whatever batch they happen to arrive in.
    """
    path = os.path.join(MODEL_FOLDER, "feature_engineering_artifacts.joblib")

    if not os.path.exists(path):
        return None

    return joblib.load(path)


@st.cache_resource
def load_segmentation_model():
    """Scaler, K-Means model and segment names fitted by phase 5.

    The dashboard previously guessed a segment from a hand-written rule on
    tenure and monthly charges. That rule did not reproduce the clusters the
    model was trained on, so an uploaded customer could be described by a
    segment the classifier had never associated with their behaviour. Scoring
    now runs the trained centroids.
    """
    path = os.path.join(MODEL_FOLDER, "customer_segmentation_model.joblib")

    if not os.path.exists(path):
        return None

    return joblib.load(path)


def prepare_scoring_dataframe(uploaded_dataframe):
    """Rebuild every model input for an uploaded file.

    The features are constructed by the same shared functions the training
    phases use, so a customer is described identically at training time and at
    prediction time.
    """
    scoring_dataframe = uploaded_dataframe.copy()

    feature_artifacts = load_feature_artifacts()
    segmentation_model = load_segmentation_model()

    if segmentation_model is None:
        st.error(
            "Segmentation model not found. Run phase5_customer_segmentation.py "
            "to generate models/customer_segmentation_model.joblib before "
            "scoring uploaded files."
        )
        st.stop()

    if feature_artifacts is None:
        st.error(
            "Feature-engineering artefacts not found. Run "
            "phase4_feature_engineering.py to generate "
            "models/feature_engineering_artifacts.joblib before scoring "
            "uploaded files."
        )
        st.stop()

    # Engineered features, using the quartile edges fitted during training.
    scoring_dataframe, _ = add_engineered_features(
        scoring_dataframe,
        artifacts=feature_artifacts
    )

    # Cluster and segment, assigned by the trained centroids.
    scoring_dataframe = assign_clusters(
        scoring_dataframe,
        segmentation_model
    )

    # The model was trained on the nominal form, so the encoder expects it.
    scoring_dataframe = to_nominal_cluster(scoring_dataframe)

    return scoring_dataframe


# ---------------------------------------------------------------
# Load project data and models
# ---------------------------------------------------------------
risk_predictions = load_csv(
    "phase8_customer_churn_risk_predictions.csv"
)

feature_importance = load_csv(
    "phase8_global_feature_importance.csv"
)

high_risk_reason_codes = load_csv(
    "phase8_high_risk_customer_reason_codes.csv"
)

financial_impact = load_csv(
    "phase9_financial_impact_summary.csv"
)

high_risk_targets = load_csv(
    "phase9_high_risk_retention_targets.csv"
)

recommendation_mapping = load_csv(
    "phase9_recommendation_mapping.csv"
)

model_results = load_csv(
    "phase7_model_evaluation_results.csv"
)

segmented_data = load_csv(
    "Customer-Churn-Segmented.csv"
)

best_model = load_model(
    "best_churn_model.joblib"
)

calibrated_model = load_model(
    "calibrated_churn_probability_model.joblib"
)

model_feature_names = (
    calibrated_model.feature_names_in_.tolist()
)

# ---------------------------------------------------------------
# Dashboard header and sidebar
# ---------------------------------------------------------------
st.title("📊 Customer Churn Intelligence Dashboard")

st.caption(
    "Churn prediction, customer segmentation, explainability, "
    "retention actions, and revenue impact analysis."
)

st.sidebar.success(
    f"Logged in as: {st.session_state.current_user}"
)

if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.session_state.current_user = ""
    st.rerun()

st.sidebar.markdown("---")

pages = [
    "Overview KPIs",
    "Exploratory Analysis",
    "Segmentation",
    "Model Comparison",
    "High-Risk Customers",
    "Explainability",
    "Business Impact",
    "Batch Scoring",
    "Single Customer Prediction"
]

selected_page = st.sidebar.radio(
    "Navigate Dashboard",
    pages
)

st.sidebar.markdown("---")

st.sidebar.info(
    "Final selected model: Tuned XGBoost\n\n"
    "Probability model: Calibrated XGBoost"
)

# ---------------------------------------------------------------
# Page 1: Overview KPIs
# ---------------------------------------------------------------
if selected_page == "Overview KPIs":
    st.header("Overview KPIs")

    total_customers = len(risk_predictions)

    high_risk_customers = (
        risk_predictions["RiskTier"] == "High Risk"
    ).sum()

    medium_risk_customers = (
        risk_predictions["RiskTier"] == "Medium Risk"
    ).sum()

    baseline_churn_rate = (
        risk_predictions["ObservedChurn"] == "Yes"
    ).mean() * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Baseline Churn Rate",
        f"{baseline_churn_rate:.2f}%"
    )

    col3.metric(
        "High-Risk Customers",
        f"{high_risk_customers:,}"
    )

    col4.metric(
        "Medium-Risk Customers",
        f"{medium_risk_customers:,}"
    )

    st.subheader("Churn Risk Tier Distribution")

    risk_counts = (
        risk_predictions["RiskTier"]
        .value_counts()
        .reindex([
            "Low Risk",
            "Medium Risk",
            "High Risk"
        ])
        .reset_index()
    )

    risk_counts.columns = ["RiskTier", "Customers"]

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.barplot(
        data=risk_counts,
        x="RiskTier",
        y="Customers",
        hue="RiskTier",
        legend=False,
        palette={
            "Low Risk": "green",
            "Medium Risk": "orange",
            "High Risk": "crimson"
        },
        ax=ax
    )

    ax.set_title("Customer Distribution by Risk Tier")
    ax.set_xlabel("Risk Tier")
    ax.set_ylabel("Customers")

    st.pyplot(fig)

    st.subheader("Risk Prediction Sample")

    st.dataframe(
        risk_predictions[
            [
                "customerID",
                "ChurnProbability",
                "RiskTier",
                "CustomerSegment",
                "Contract",
                "MonthlyCharges"
            ]
        ].head(20),
        use_container_width=True
    )

# ---------------------------------------------------------------
# Page 2: Exploratory analysis
# ---------------------------------------------------------------
elif selected_page == "Exploratory Analysis":
    st.header("Exploratory Data Analysis")

    chart_options = {
        "Univariate Analysis":
            "phase3_univariate_analysis.png",
        "Churn by Contract":
            "phase3_churn_by_contract.png",
        "Churn by Payment Method":
            "phase3_churn_by_payment_method.png",
        "Churn by Internet Service":
            "phase3_churn_by_internet_service.png",
        "Churn by Senior Citizen":
            "phase3_churn_by_senior_citizen.png",
        "Churn by Tech Support":
            "phase3_churn_by_tech_support.png",
        "Correlation Heatmap":
            "phase3_correlation_heatmap.png",
        "Tenure versus Monthly Charges":
            "phase3_tenure_monthlycharges_churn.png",
        "Churn by Tenure Band":
            "phase3_churn_by_tenure_band.png"
    }

    selected_chart = st.selectbox(
        "Choose EDA Chart",
        list(chart_options.keys())
    )

    show_chart(
        chart_options[selected_chart],
        selected_chart
    )

# ---------------------------------------------------------------
# Page 3: Customer segmentation
# ---------------------------------------------------------------
elif selected_page == "Segmentation":
    st.header("Customer Segmentation")

    st.write(
        "K-Means customer segmentation uses tenure, service "
        "adoption, and monthly charges."
    )

    show_chart(
        "phase5_elbow_method.png",
        "Elbow Method for Cluster Selection"
    )

    show_chart(
        "phase5_silhouette_scores.png",
        "Silhouette Scores by Cluster Count"
    )

    show_chart(
        "phase5_cluster_profiles.png",
        "Customer Segment Profiles"
    )

    show_chart(
        "phase5_customer_segments.png",
        "Customer Segments by Tenure and Monthly Charges"
    )

    st.subheader("Segment Summary")

    segment_summary = (
        segmented_data.groupby("CustomerSegment")
        .agg(
            Customers=("customerID", "count"),
            MeanTenure=("tenure", "mean"),
            MeanMonthlyCharges=(
                "MonthlyCharges",
                "mean"
            ),
            ChurnRate=("ChurnFlag", "mean")
        )
        .reset_index()
    )

    segment_summary["ChurnRate"] = (
        segment_summary["ChurnRate"] * 100
    ).round(2)

    st.dataframe(
        segment_summary.round(2),
        use_container_width=True
    )

# ---------------------------------------------------------------
# Page 4: Model comparison
# ---------------------------------------------------------------
elif selected_page == "Model Comparison":
    st.header("Model Evaluation and Comparison")

    st.dataframe(
        model_results.round(4),
        use_container_width=True
    )

    show_chart(
        "phase7_model_comparison.png",
        "Consolidated Model Performance"
    )

    show_chart(
        "phase7_overlay_roc_curves.png",
        "ROC Curve Comparison"
    )

    show_chart(
        "phase7_precision_recall_curves.png",
        "Precision-Recall Curve Comparison"
    )

    show_chart(
        "phase7_all_confusion_matrices.png",
        "Confusion Matrices"
    )

    show_chart(
        "phase7_threshold_tradeoff.png",
        "Threshold Trade-Off Analysis"
    )

# ---------------------------------------------------------------
# Page 5: High-risk customers
# ---------------------------------------------------------------
elif selected_page == "High-Risk Customers":
    st.header("High-Risk Customer Target List")

    st.write(
        "Customers are prioritized by their predicted revenue at risk."
    )

    st.dataframe(
        high_risk_targets.head(50),
        use_container_width=True
    )

    high_risk_csv = high_risk_targets.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Prioritized High-Risk Target List",
        data=high_risk_csv,
        file_name="high_risk_retention_targets.csv",
        mime="text/csv"
    )

# ---------------------------------------------------------------
# Page 6: Explainability
# ---------------------------------------------------------------
elif selected_page == "Explainability":
    st.header("Explainability and Churn Drivers")

    st.subheader("Global Feature Importance")

    st.dataframe(
        feature_importance,
        use_container_width=True
    )

    show_chart(
        "phase8_global_feature_importance.png",
        "Global XGBoost Feature Importance"
    )

    show_chart(
        "phase8_shap_global_summary.png",
        "SHAP Global Summary"
    )

    st.subheader("High-Risk Customer Reason Codes")

    st.dataframe(
        high_risk_reason_codes,
        use_container_width=True
    )

    st.info(
        "High-risk customers commonly have short tenure, "
        "month-to-month contracts, and Fiber-optic service."
    )

# ---------------------------------------------------------------
# Page 7: Business impact
# ---------------------------------------------------------------
elif selected_page == "Business Impact":
    st.header("Revenue and Retention Impact")

    financial_values = {
        row["Metric"]: row["Value"]
        for _, row in financial_impact.iterrows()
    }

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Revenue at Risk",
        currency(
            financial_values.get(
                "Total Revenue at Risk",
                0
            )
        )
    )

    col2.metric(
        "Expected Saved Revenue",
        currency(
            financial_values.get(
                "Expected Saved Revenue",
                0
            )
        )
    )

    col3.metric(
        "Expected Net Benefit",
        currency(
            financial_values.get(
                "Net Benefit",
                0
            )
        )
    )

    col4.metric(
        "Retention ROI",
        (
            f"{financial_values.get('Retention ROI (%)', 0):,.2f}%"
        )
    )

    st.subheader("Financial Impact Summary")

    st.dataframe(
        financial_impact.round(2),
        use_container_width=True
    )

    show_chart(
        "phase9_revenue_at_risk_by_segment.png",
        "Revenue at Risk by Customer Segment"
    )

    show_chart(
        "phase9_revenue_at_risk_by_driver.png",
        "Revenue at Risk by Churn Driver"
    )

    show_chart(
        "phase9_retention_financial_impact.png",
        "Retention Campaign Financial Impact"
    )

    st.subheader("Recommended Retention Actions")

    st.dataframe(
        recommendation_mapping,
        use_container_width=True
    )

# ---------------------------------------------------------------
# Page 8: Batch CSV scoring
# ---------------------------------------------------------------
elif selected_page == "Batch Scoring":
    st.header("Batch Customer Churn Scoring")

    st.write(
        "Upload the original or cleaned customer churn CSV. The app rebuilds "
        "every engineered feature and assigns the customer segment using the "
        "trained K-Means centroids, so the inputs match those the model was "
        "trained on, then returns churn probability and risk tier."
    )

    uploaded_file = st.file_uploader(
        "Upload Customer CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded File Preview")

        st.dataframe(
            uploaded_df.head(),
            use_container_width=True
        )

        scoring_df = prepare_scoring_dataframe(uploaded_df)

        missing_columns = [
            column
            for column in model_feature_names
            if column not in scoring_df.columns
        ]

        if missing_columns:
            st.error(
                "The uploaded file is missing model input columns: "
                + ", ".join(missing_columns)
            )

        else:
            model_input_df = scoring_df[
                model_feature_names
            ].copy()

            probabilities = calibrated_model.predict_proba(
                model_input_df
            )[:, 1]

            scoring_df["ChurnProbability"] = (
                probabilities.round(4)
            )

            scoring_df["RiskTier"] = pd.cut(
                scoring_df["ChurnProbability"],
                bins=[-0.01, 0.35, 0.65, 1.00],
                labels=[
                    "Low Risk",
                    "Medium Risk",
                    "High Risk"
                ]
            )

            scoring_df["RecommendedAction"] = (
                "Standard engagement support"
            )

            scoring_df.loc[
                (
                    scoring_df["RiskTier"] == "High Risk"
                ) &
                (
                    scoring_df["Contract"] == "Month-to-month"
                ),
                "RecommendedAction"
            ] = (
                "Offer discounted annual contract and loyalty bonus"
            )

            scoring_df.loc[
                (
                    scoring_df["RiskTier"] == "High Risk"
                ) &
                (
                    scoring_df["TechSupport"] == "No"
                ),
                "RecommendedAction"
            ] = (
                "Offer complimentary technical-support trial"
            )

            scoring_df.loc[
                (
                    scoring_df["RiskTier"] == "High Risk"
                ) &
                (
                    scoring_df["PaymentMethod"] == "Electronic check"
                ),
                "RecommendedAction"
            ] = (
                "Offer automatic-payment incentive"
            )

            scoring_df = scoring_df.sort_values(
                by="ChurnProbability",
                ascending=False
            )

            st.success("Batch scoring completed successfully.")

            st.subheader("Scored Customer Results")

            st.dataframe(
                scoring_df.head(50),
                use_container_width=True
            )

            batch_csv = scoring_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Scored Customer File",
                data=batch_csv,
                file_name="scored_customer_churn_predictions.csv",
                mime="text/csv"
            )

# ---------------------------------------------------------------
# Page 9: Single customer what-if prediction
# ---------------------------------------------------------------
elif selected_page == "Single Customer Prediction":
    st.header("Single Customer What-If Prediction")

    st.write(
        "Enter customer details to calculate calibrated churn "
        "probability, risk tier, estimated segment, and "
        "recommended retention action."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1]
        )

        partner = st.selectbox(
            "Partner",
            ["No", "Yes"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["No", "Yes"]
        )

        tenure = st.slider(
            "Tenure in Months",
            min_value=0,
            max_value=72,
            value=12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["No", "Yes"]
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes"]
        )

    with col2:
        internet_service = st.selectbox(
            "Internet Service",
            ["No", "DSL", "Fiber optic"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["No", "Yes"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes"]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes"]
        )

    with col3:
        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["No", "Yes"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=1.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=10000.0,
            value=float(monthly_charges * tenure),
            step=10.0
        )

    if st.button("Predict Customer Churn Risk"):
        service_count = sum([
            online_security == "Yes",
            online_backup == "Yes",
            device_protection == "Yes",
            tech_support == "Yes",
            streaming_tv == "Yes",
            streaming_movies == "Yes"
        ])

        input_df = pd.DataFrame([{
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "ServiceCount": service_count
        }])

        # Same construction path as batch scoring, so a single customer and an
        # uploaded file are described identically.
        input_df = prepare_scoring_dataframe(input_df)

        model_input_df = input_df[
            model_feature_names
        ].copy()

        probability = calibrated_model.predict_proba(
            model_input_df
        )[0, 1]

        if probability < 0.35:
            risk_tier = "Low Risk"
            recommendation = (
                "Maintain engagement through standard customer support."
            )

        elif probability <= 0.65:
            risk_tier = "Medium Risk"
            recommendation = (
                "Offer targeted engagement or a value-added service."
            )

        else:
            risk_tier = "High Risk"

            if contract == "Month-to-month":
                recommendation = (
                    "Offer a discounted annual contract and loyalty bonus."
                )

            elif tech_support == "No":
                recommendation = (
                    "Offer a complimentary technical-support trial."
                )

            elif payment_method == "Electronic check":
                recommendation = (
                    "Offer an automatic-payment incentive."
                )

            else:
                recommendation = (
                    "Offer a personalized retention incentive."
                )

        metric_col1, metric_col2, metric_col3 = st.columns(3)

        metric_col1.metric(
            "Predicted Churn Probability",
            f"{probability:.2%}"
        )

        metric_col2.metric(
            "Risk Tier",
            risk_tier
        )

        metric_col3.metric(
            "Estimated Customer Segment",
            input_df["CustomerSegment"].iloc[0]
        )

        st.subheader("Recommended Retention Action")
        st.success(recommendation)