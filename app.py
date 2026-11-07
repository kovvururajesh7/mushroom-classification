from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)



# 1. PAGE CONFIGURATION


st.set_page_config(
    page_title="Mushroom Classification | ML Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
SCHEMA_PATH = MODEL_DIR / "feature_schema.json"
DEFAULT_TEST_PATH = BASE_DIR / "test.csv"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k_nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib"
}



# 2. PREMIUM DASHBOARD STYLING


st.markdown("""
<style>

/* ----------------------------------------------------------
   MAIN APPLICATION
---------------------------------------------------------- */

.stApp {
    background: #F3F6FB;
    color: #26364A;
}

.block-container {
    max-width: 1500px;
    padding: 2rem 2.5rem 2rem 2.5rem;
}

/* ----------------------------------------------------------
   TOP STREAMLIT HEADER AND TOOLBAR
---------------------------------------------------------- */

header[data-testid="stHeader"] {
    background-color: #F3F6FB !important;
}

[data-testid="stToolbar"] {
    background-color: transparent !important;
}

[data-testid="stDecoration"] {
    background: transparent !important;
}

/* ----------------------------------------------------------
   MAIN TEXT
---------------------------------------------------------- */

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4 {
    color: #142B49;
}

.stApp p,
.stApp label,
.stApp li {
    color: #46566D;
}

/* ----------------------------------------------------------
   SIDEBAR
---------------------------------------------------------- */

[data-testid="stSidebar"] {
    background: #14243A;
    border-right: 1px solid #263A53;
}

[data-testid="stSidebar"] > div {
    background: #14243A;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #E8EEF7 !important;
}

[data-testid="stSidebar"] small {
    color: #A9B8CC !important;
}

[data-testid="stSidebar"] hr {
    border-color: #35465D;
}

/* Sidebar file uploader */

[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    background: #20344D;
    border: 1px dashed #7186A1;
    border-radius: 10px;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] * {
    color: #E8EEF7 !important;
}

/* Sidebar selectbox */

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #20344D !important;
    border: 1px solid #52677F !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #FFFFFF !important;
}

/* Brand */

.brand-title {
    color: #FFFFFF;
    font-size: 23px;
    font-weight: 800;
    letter-spacing: -0.5px;
}

.brand-subtitle {
    color: #A9B8CC;
    font-size: 12px;
    margin-top: 5px;
}

/* ----------------------------------------------------------
   INFORMATION BANNER
---------------------------------------------------------- */

[data-testid="stAlert"] {
    background-color: #EAF2FF !important;
    border: 1px solid #C8D9F2 !important;
    border-radius: 10px !important;
}

[data-testid="stAlert"] p {
    color: #244B78 !important;
}

/* ----------------------------------------------------------
   HERO HEADER
---------------------------------------------------------- */

.hero {
    background: linear-gradient(
        120deg,
        #142B49 0%,
        #244B78 100%
    );
    border-radius: 16px;
    padding: 27px 30px;
    margin-bottom: 25px;
    box-shadow: 0 7px 20px rgba(20, 43, 73, 0.12);
}

.hero-title {
    color: #FFFFFF !important;
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -0.7px;
    margin: 0;
    line-height: 1.4;
}

.hero-subtitle {
    color: #D1DEEF !important;
    font-size: 14px;
    margin-top: 9px;
    line-height: 1.6;
}

.hero-tag {
    display: inline-block;
    margin-top: 14px;
    padding: 5px 11px;
    background: rgba(255, 255, 255, 0.13);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 20px;
    color: #FFFFFF !important;
    font-size: 11px;
    font-weight: 600;
}

/* ----------------------------------------------------------
   SECTION HEADINGS
---------------------------------------------------------- */

.section-title {
    color: #142B49 !important;
    font-size: 20px;
    font-weight: 750;
    margin-top: 12px;
    margin-bottom: 15px;
}

/* ----------------------------------------------------------
   DATASET SUMMARY CARDS
---------------------------------------------------------- */

.summary-card {
    background: #FFFFFF;
    border: 1px solid #E0E7F0;
    border-radius: 13px;
    padding: 18px 20px;
    box-shadow: 0 3px 12px rgba(20, 43, 73, 0.045);
    min-height: 105px;
}

.summary-label {
    font-size: 12px;
    color: #6B7C93 !important;
    font-weight: 600;
}

.summary-value {
    color: #17365D !important;
    font-size: 27px;
    font-weight: 800;
    margin-top: 8px;
}

/* ----------------------------------------------------------
   PERFORMANCE METRIC CARDS
---------------------------------------------------------- */

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #DFE7F1;
    border-radius: 12px;
    padding: 16px 17px;
    box-shadow: 0 3px 12px rgba(20, 43, 73, 0.045);
    min-height: 100px;
}

[data-testid="stMetricLabel"] {
    color: #687A91 !important;
    font-size: 12px;
    font-weight: 650;
}

[data-testid="stMetricValue"] {
    color: #17365D !important;
    font-size: 25px;
    font-weight: 800;
}

/* ----------------------------------------------------------
   TABS
---------------------------------------------------------- */

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #E8EEF6;
    padding: 6px;
    border-radius: 10px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 7px;
    padding: 10px 18px;
    color: #52657D !important;
    font-weight: 650;
}

.stTabs [data-baseweb="tab"] p {
    color: #52657D !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: #FFFFFF;
    color: #1D4ED8 !important;
    box-shadow: 0 2px 6px rgba(20, 43, 73, 0.08);
}

.stTabs [data-baseweb="tab"][aria-selected="true"] p {
    color: #1D4ED8 !important;
}

/* ----------------------------------------------------------
   SELECT BOXES
---------------------------------------------------------- */

[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px;
}

[data-baseweb="select"] * {
    color: #26364A !important;
}

/* ----------------------------------------------------------
   BUTTONS
---------------------------------------------------------- */

/* Download Prediction Results Button */

.stDownloadButton button {
    background-color: #2459A6 !important;
    color: #FFFFFF !important;
    border: 1px solid #2459A6 !important;
    border-radius: 8px !important;
    font-weight: 650 !important;
}

.stDownloadButton button p,
.stDownloadButton button span,
.stDownloadButton button div {
    color: #FFFFFF !important;
}

.stDownloadButton button:hover {
    background-color: #173F7C !important;
    color: #FFFFFF !important;
}

/* ----------------------------------------------------------
   DATAFRAMES
---------------------------------------------------------- */

[data-testid="stDataFrame"] {
    border: 1px solid #DFE7F1;
    border-radius: 10px;
    background: #FFFFFF;
}

/* ----------------------------------------------------------
   EXPANDERS
---------------------------------------------------------- */

[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #DFE7F1;
    border-radius: 10px;
}

/* ----------------------------------------------------------
   DIVIDERS
---------------------------------------------------------- */

hr {
    border-color: #DFE7F1 !important;
}

/* ----------------------------------------------------------
   FOOTER
---------------------------------------------------------- */

.footer {
    border-top: 1px solid #DCE5EF;
    margin-top: 35px;
    padding-top: 18px;
    text-align: center;
    color: #7B899B !important;
    font-size: 12px;
}
            

/* Information and warning message text */
[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div {
    color: #FFFFFF !important;
}

/* Information banner */
[data-testid="stAlert"][data-baseweb="notification"] {
    background-color: #142B49 !important;
    border: 1px solid #244B78 !important;
}

</style>
            


""", unsafe_allow_html=True)



# 3. LOAD MODELS AND FEATURE SCHEMA


@st.cache_resource
def load_assets():

    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Feature schema not found: {SCHEMA_PATH}"
        )

    with open(SCHEMA_PATH, "r", encoding="utf-8") as file:
        schema = json.load(file)

    loaded_models = {}

    for name, filename in MODEL_FILES.items():

        model_path = MODEL_DIR / filename

        if not model_path.exists():
            raise FileNotFoundError(
                f"Saved model missing: {model_path}"
            )

        loaded_models[name] = joblib.load(model_path)

    return schema, loaded_models


try:
    schema, models = load_assets()

except Exception as exc:
    st.error(f"Unable to load model artifacts: {exc}")
    st.stop()


feature_columns = schema["feature_columns"]
target_column = schema["target_column"]



# 4. SIDEBAR


with st.sidebar:

    st.markdown("""
    <div style="padding: 12px 0 22px 0;">
        <div class="brand-title">MUSHROOM ML</div>
        <div class="brand-subtitle">
            MACHINE LEARNING ANALYTICS
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### Dataset")

    uploaded_file = st.file_uploader(
        "Upload test dataset",
        type=["csv"],
        help="Upload a CSV containing all required features and the class column."
    )

    st.caption(
        f"{len(feature_columns)} required input features"
    )

    st.divider()

    st.markdown("### Model Selection")

    selected_model = st.selectbox(
        "Choose classifier",
        list(MODEL_FILES.keys())
    )

    st.divider()

    st.markdown("### Classification Details")

    st.caption("Problem Type: Binary Classification")
    st.caption("Class 0: Edible")
    st.caption("Class 1: Poisonous")
    st.caption("Positive Class: Poisonous")

    st.divider()

    st.markdown("### Project Information")

    st.caption("Assignment: Machine Learning")
    st.caption("Models: Five classifiers")
    st.caption("Evaluation: Six metrics")



# 5. LOAD TEST DATA


if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

    except Exception as exc:
        st.error(f"Unable to read uploaded CSV: {exc}")
        st.stop()

elif DEFAULT_TEST_PATH.exists():

    data = pd.read_csv(DEFAULT_TEST_PATH)

    st.info(
        "Using the default held-out test dataset. "
        "Upload another CSV from the sidebar to evaluate a different dataset."
    )

else:

    st.warning(
        "Upload a test dataset from the sidebar to begin evaluation."
    )

    st.stop()


if data.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()



# 6. DATA VALIDATION


required_columns = feature_columns + [target_column]

missing_columns = [
    col for col in required_columns
    if col not in data.columns
]

if missing_columns:

    st.error(
        f"Missing required columns: {missing_columns}"
    )

    st.stop()


X_eval = data[feature_columns].copy()

y_raw = (
    data[target_column]
    .astype(str)
    .str.strip()
    .str.lower()
)

y_eval = y_raw.map(schema["label_mapping"])

if y_eval.isna().any():

    invalid_labels = sorted(
        y_raw[y_eval.isna()].unique().tolist()
    )

    st.error(
        f"Invalid target labels: {invalid_labels}. "
        "Expected values are 'e' and 'p'."
    )

    st.stop()


y_eval = y_eval.astype(int)



# 7. MODEL EVALUATION FUNCTION


def evaluate_model(model, X, y):

    predictions = model.predict(X)

    probabilities = (
        model.predict_proba(X)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    auc = None

    if probabilities is not None and pd.Series(y).nunique() == 2:

        auc = roc_auc_score(
            y,
            probabilities
        )

    return {

        "predictions": predictions,

        "probabilities": probabilities,

        "Accuracy": accuracy_score(y, predictions),

        "AUC": auc,

        "Precision": precision_score(
            y,
            predictions,
            pos_label=1,
            zero_division=0
        ),

        "Recall": recall_score(
            y,
            predictions,
            pos_label=1,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y,
            predictions,
            pos_label=1,
            zero_division=0
        ),

        "MCC": matthews_corrcoef(y, predictions)

    }



# 8. EVALUATE ALL FIVE MODELS


all_results = {}

for name, model in models.items():

    try:

        all_results[name] = evaluate_model(
            model,
            X_eval,
            y_eval
        )

    except Exception as exc:

        st.error(
            f"Evaluation failed for {name}: {exc}"
        )

        st.stop()



# 9. DASHBOARD HEADER



st.markdown("""
<div class="hero">
<div class="hero-title">Mushroom Classification</div>
<div class="hero-subtitle">
Intelligent classification, model evaluation and performance analytics
</div>
<div class="hero-tag">MACHINE LEARNING PROJECT</div>
</div>
""", unsafe_allow_html=True)


# 10. DATASET OVERVIEW


st.markdown(
    '<div class="section-title">Dataset Overview</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

overview = [
    ("Total Records", f"{len(data):,}"),
    ("Input Features", f"{len(feature_columns)}"),
    ("Edible Samples", f"{int((y_eval == 0).sum()):,}"),
    ("Poisonous Samples", f"{int((y_eval == 1).sum()):,}")
]

for col, (label, value) in zip([c1, c2, c3, c4], overview):

    with col:

        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-label">{label}</div>
            <div class="summary-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)


st.write("")



# 11. DASHBOARD TABS


tab1, tab2, tab3 = st.tabs([
    "Model Evaluation",
    "Model Comparison",
    "Dataset Analysis"
])



# TAB 1 - MODEL EVALUATION


with tab1:

    st.markdown(
        '<div class="section-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    result = all_results[selected_model]

    st.markdown(
        f"**Selected Classifier:** {selected_model}"
    )

    metric_names = [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1 Score",
        "MCC"
    ]

    metric_cols = st.columns(6)

    for col, metric in zip(metric_cols, metric_names):

        value = result[metric]

        col.metric(
            metric,
            "N/A" if value is None else f"{value:.4f}"
        )

    st.write("")

    left, right = st.columns(
        [1, 1],
        gap="large"
    )

    # Confusion Matrix

    with left:

        st.markdown("#### Confusion Matrix")

        cm = confusion_matrix(
            y_eval,
            result["predictions"],
            labels=[0, 1]
        )

        fig, ax = plt.subplots(figsize=(6, 4.5))

        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")

        ax.imshow(
            cm,
            interpolation="nearest",
            cmap="Blues"
        )

        ax.set_title(
            selected_model,
            fontsize=13,
            fontweight="bold",
            color="#142B49",
            pad=15
        )

        ax.set_xlabel(
            "Predicted Class",
            color="#334155",
            fontsize=10
        )

        ax.set_ylabel(
            "Actual Class",
            color="#334155",
            fontsize=10
        )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])

        ax.set_xticklabels(
            ["Edible", "Poisonous"],
            color="#334155"
        )

        ax.set_yticklabels(
            ["Edible", "Poisonous"],
            color="#334155"
        )

        threshold = cm.max() / 2 if cm.max() else 0

        for i in range(2):
            for j in range(2):

                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:,}",
                    ha="center",
                    va="center",
                    fontsize=16,
                    fontweight="bold",
                    color=(
                        "white"
                        if cm[i, j] > threshold
                        else "#142B49"
                    )
                )

        for spine in ax.spines.values():
            spine.set_visible(False)

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    # Classification Report

    with right:

        st.markdown("#### Classification Report")

        report = classification_report(
            y_eval,
            result["predictions"],
            labels=[0, 1],
            target_names=["Edible", "Poisonous"],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).T

        st.dataframe(
            report_df.round(4),
            use_container_width=True,
            height=310
        )

    st.divider()

    st.markdown("#### Prediction Results")

    download_data = data.copy()

    download_data[
        f"Predicted_{target_column}"
    ] = pd.Series(
        result["predictions"],
        index=data.index
    ).map({
        0: "e",
        1: "p"
    })

    if result["probabilities"] is not None:

        download_data[
            "Poisonous_probability"
        ] = result["probabilities"]

    st.download_button(
        label="Download Prediction Results",
        data=download_data.to_csv(
            index=False
        ).encode("utf-8"),
        file_name="mushroom_classification_predictions.csv",
        mime="text/csv"
    )



# TAB 2 - MODEL COMPARISON


with tab2:

    st.markdown(
        '<div class="section-title">Model Comparison</div>',
        unsafe_allow_html=True
    )

    comparison_rows = []

    for name, values in all_results.items():

        comparison_rows.append({

            "ML Model Name": name,

            "Accuracy": values["Accuracy"],

            "AUC": values["AUC"],

            "Precision": values["Precision"],

            "Recall": values["Recall"],

            "F1 Score": values["F1 Score"],

            "MCC": values["MCC"]

        })

    comparison = pd.DataFrame(
        comparison_rows
    ).set_index("ML Model Name")

    st.markdown("#### Evaluation Metrics")

    st.dataframe(
        comparison.style.format("{:.4f}"),
        use_container_width=True
    )

    st.write("")

    st.markdown("#### Performance Visualization")

    selected_metric = st.selectbox(
        "Choose metric to compare",
        [
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1 Score",
            "MCC"
        ]
    )

    chart_data = comparison[
        selected_metric
    ].sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    bars = ax.barh(
        chart_data.index,
        chart_data.values,
        color="#3478C8",
        height=0.55
    )

    ax.set_xlim(0, 1.08)

    ax.set_xlabel(
        selected_metric,
        color="#46566D"
    )

    ax.set_title(
        f"{selected_metric} Across Five Classifiers",
        color="#142B49",
        fontsize=14,
        fontweight="bold",
        pad=15
    )

    ax.grid(
        axis="x",
        alpha=0.18,
        linestyle="--"
    )

    ax.set_axisbelow(True)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    for bar, value in zip(bars, chart_data.values):

        ax.text(
            value + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            color="#243247",
            fontsize=10,
            fontweight="bold"
        )

    plt.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)

    st.divider()

    st.markdown("#### Top Performing Models")

    best_score = comparison["F1 Score"].max()

    best_models = comparison.index[
        comparison["F1 Score"] == best_score
    ].tolist()

    if len(best_models) > 1:

        st.success(
            f"Joint top-performing models: "
            f"{', '.join(best_models)} "
            f"| F1 Score: {best_score:.4f}"
        )

    else:

        st.success(
            f"{best_models[0]} achieved the highest F1 Score "
            f"of {best_score:.4f}."
        )

    st.caption(
        "Models with identical F1 scores are considered tied. "
        "The comparison is based on the currently loaded test dataset."
    )



# TAB 3 - DATASET ANALYSIS


with tab3:

    st.markdown(
        '<div class="section-title">Dataset Analysis</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    # Class Distribution

    with left:

        st.markdown("#### Class Distribution")

        class_counts = (
            y_eval.map({
                0: "Edible",
                1: "Poisonous"
            }).value_counts()
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        fig.patch.set_facecolor("#FFFFFF")
        ax.set_facecolor("#FFFFFF")

        ax.bar(
            class_counts.index,
            class_counts.values,
            color=["#3478C8", "#159D91"],
            width=0.55
        )

        ax.set_ylabel(
            "Number of Samples",
            color="#46566D"
        )

        ax.set_title(
            "Distribution of Mushroom Classes",
            color="#142B49",
            fontweight="bold"
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.grid(
            axis="y",
            alpha=0.18
        )

        ax.set_axisbelow(True)

        plt.tight_layout()

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    # Missing Values

    with right:

        st.markdown("#### Missing Values Analysis")

        missing_df = (
            X_eval.isnull()
            .sum()
            .sort_values(ascending=False)
        )

        missing_df = missing_df[
            missing_df > 0
        ]

        if missing_df.empty:

            st.success(
                "No missing values found in the uploaded feature data."
            )

        else:

            st.dataframe(
                missing_df.rename(
                    "Missing Count"
                ).to_frame(),
                use_container_width=True
            )

    st.divider()

    st.markdown("#### Dataset Preview")

    st.dataframe(
        data.head(20),
        use_container_width=True
    )

    st.markdown("#### Feature Summary")

    st.dataframe(
        X_eval.describe(include="all").T,
        use_container_width=True
    )



# 12. FOOTER


st.markdown("""
<div class="footer">
    MUSHROOM CLASSIFICATION | MACHINE LEARNING ASSIGNMENT 2
    <br><br>
    Model evaluation is based on the uploaded test dataset.
</div>
""", unsafe_allow_html=True)