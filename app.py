from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import clone
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "heart.csv"
MODEL_FILES = {
    "Logistic Regression": "LogisticR.pkl",
    "Decision Tree": "tree.pkl",
    "Support Vector Machine": "SVM.pkl",
}
FEATURES = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
]
CATEGORICAL_COLUMNS = {
    "Sex": ["M", "F"],
    "ChestPainType": ["ATA", "NAP", "ASY", "TA"],
    "RestingECG": ["Normal", "ST", "LVH"],
    "ExerciseAngina": ["N", "Y"],
    "ST_Slope": ["Up", "Flat", "Down"],
}
DISPLAY_LABELS = {
    "M": "Male",
    "F": "Female",
    "ATA": "Atypical Angina",
    "NAP": "Non-Anginal Pain",
    "ASY": "Asymptomatic",
    "TA": "Typical Angina",
    "Normal": "Normal",
    "ST": "ST-T Wave Abnormality",
    "LVH": "Left Ventricular Hypertrophy",
    "N": "No",
    "Y": "Yes",
    "Up": "Upsloping",
    "Flat": "Flat",
    "Down": "Downsloping",
}


st.set_page_config(
    page_title="HeartCare | Heart Disease Analytics",
    page_icon="♥",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_models():
    models = {}
    for name, filename in MODEL_FILES.items():
        with open(ROOT / filename, "rb") as model_file:
            models[name] = pickle.load(model_file)
    return models


def encode_features(data):
    encoded = data[FEATURES].copy()
    for column, categories in CATEGORICAL_COLUMNS.items():
        encoded[column] = pd.Categorical(encoded[column], categories=categories).codes
    return encoded


def preprocess_for_evaluation(data):
    encoded = encode_features(data)
    encoded["Cholesterol"] = encoded["Cholesterol"].replace(0, np.nan)
    encoded["RestingBP"] = encoded["RestingBP"].replace(0, np.nan)
    encoded["FastingBS"] = data["FastingBS"].astype(float)
    encoded["Age"] = data["Age"].astype(float)
    encoded["MaxHR"] = data["MaxHR"].astype(float)
    encoded["Oldpeak"] = data["Oldpeak"].astype(float)
    return encoded


def format_input(values):
    raw = pd.DataFrame([values])
    return encode_features(raw)


def metric_value(value):
    return f"{value:.1%}"


def apply_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
        :root { --orange:#F97316; --orange-soft:#FFF7ED; --ink:#171717; --muted:#737373; --border:#E5E7EB; }
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: #FFFFFF !important; color: #000000 !important; }
        [data-testid="stHeader"] { background: #FFFFFF !important; box-shadow: none !important; }
        [data-testid="stToolbar"] { background: #FFFFFF !important; }
        [data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid var(--border); }
        [data-testid="stSidebar"] > div:first-child { padding: 1.5rem 1rem; }
        .brand { display:flex; align-items:center; gap:10px; padding: 0.4rem 0.6rem 2rem; }
        .brand-mark { width:34px; height:34px; display:grid; place-items:center; border-radius:10px; background:var(--orange-soft); color:var(--orange); font-size:22px; }
        .brand-name { color:#000000 !important; font-size:20px; font-weight:700; letter-spacing:-.03em; }
        .sidebar-caption { color:#000000 !important; font-size:11px; text-transform:uppercase; letter-spacing:.09em; margin:1.6rem .7rem .45rem; }
        .sidebar-footer { border-top:1px solid var(--border); margin-top:2rem; padding:1rem .7rem; color:#000000 !important; font-size:12px; line-height:1.7; }
        .sidebar-footer strong { color:#000000 !important; }
        h1, h2, h3 { color:#000000 !important; }
        h1 { font-size:32px !important; letter-spacing:-.04em; margin-bottom:.2rem !important; }
        h2 { font-size:21px !important; letter-spacing:-.025em; }
        h3 { font-size:16px !important; }
        .subtitle { color:#000000 !important; font-size:14px; margin-bottom:1.4rem; }
        .topbar { display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--border); padding-bottom:1.2rem; margin-bottom:1.8rem; }
        .breadcrumb { color:#000000 !important; font-size:12px; margin-bottom:.35rem; }
        .status { display:inline-flex; align-items:center; gap:7px; color:#000000 !important; background:#F0FDF4; border:1px solid #DCFCE7; border-radius:999px; padding:7px 11px; font-size:12px; font-weight:600; }
        .status-dot { width:7px; height:7px; border-radius:50%; background:#22C55E; }
        .kpi { background:#fff; border:1px solid var(--border); border-radius:12px; padding:1.1rem 1.2rem; min-height:108px; }
        .kpi-label { color:#000000 !important; font-size:12px; margin-bottom:.75rem; }
        .kpi-value { color:#000000 !important; font-size:26px; font-weight:700; letter-spacing:-.04em; }
        .kpi-note { color:#000000 !important; font-size:11px; margin-top:.3rem; }
        .section-card { background:#fff; border:1px solid var(--border); border-radius:12px; padding:1.2rem; margin-bottom:1.1rem; }
        .section-label { color:#000000 !important; font-size:11px; font-weight:700; letter-spacing:.1em; text-transform:uppercase; margin-bottom:.7rem; }
        .result-card { border:1px solid #FED7AA; background:var(--orange-soft); border-radius:12px; padding:1.2rem; }
        .result-card.healthy { border-color:#BBF7D0; background:#F0FDF4; }
        .result-title { font-size:17px; font-weight:700; margin-bottom:.25rem; }
        .result-copy { color:#000000 !important; font-size:13px; }
        .disclaimer { color:#000000 !important; font-size:12px; line-height:1.55; border-left:3px solid var(--orange); padding:.25rem 0 .25rem .8rem; }
        .upload-note { background:var(--orange-soft); border:1px solid #FED7AA; border-radius:10px; padding:.9rem 1rem; color:#000000 !important; font-size:13px; line-height:1.55; }
        .stApp [data-testid="stMarkdownContainer"] p,
        .stApp [data-testid="stMarkdownContainer"] li,
        .stApp [data-testid="stMarkdownContainer"] span,
        .stApp [data-testid="stWidgetLabel"] p,
        .stApp [data-testid="stRadio"] label p,
        .stApp [data-testid="stMetricLabel"],
        .stApp [data-testid="stMetricValue"],
        .stApp [data-testid="stMetricDelta"] { color:#000000 !important; }
        .stApp [data-testid="stWidgetLabel"] p,
        .stApp [data-testid="stRadio"] label p { color:var(--ink) !important; }
        .stApp input, .stApp textarea, .stApp [data-baseweb="select"] *, .stApp [data-testid="stDataFrame"] * { color:#000000 !important; }
        .stApp [data-testid="stFileUploader"] * { color:#000000 !important; }
        .stApp button { color:#000000; }
        .stApp .stButton > button { color:#FFFFFF !important; }
        .stApp .stDownloadButton > button { color:#000000 !important; }
        .stApp [data-baseweb="select"] > div { background:#FFFFFF !important; border-color:var(--border) !important; }
        .stApp [data-testid="stDataFrame"] { color:#000000 !important; }
        .stApp [data-testid="stDataFrame"] div { color:#000000 !important; }
        div[data-testid="stMetric"] { background:#fff; border:1px solid var(--border); border-radius:12px; padding:1rem; }
        div[data-testid="stMetric"] label { color:var(--muted); }
        .stButton > button { border-radius:8px; border:1px solid var(--orange); background:var(--orange); color:#fff; font-weight:600; padding:.55rem 1.2rem; }
        .stButton > button:hover { border-color:#EA580C; background:#EA580C; color:#fff; }
        .stDownloadButton > button { border-radius:8px; }
        [data-testid="stFileUploader"] { border:1px dashed #FDBA74; border-radius:10px; background:var(--orange-soft); padding:.4rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def chart_layout(fig, height=300):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="DM Sans", color="#000000", size=12),
        legend=dict(orientation="h", y=-0.16, font=dict(color="#000000")),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#F1F1F1", zeroline=False, title_font=dict(color="#000000"), tickfont=dict(color="#000000"))
    fig.update_yaxes(showgrid=True, gridcolor="#F1F1F1", zeroline=False, title_font=dict(color="#000000"), tickfont=dict(color="#000000"))
    return fig


def render_topbar(title, subtitle, breadcrumb):
    left, right = st.columns([4, 1])
    with left:
        st.markdown(f'<div class="breadcrumb">{breadcrumb}</div>', unsafe_allow_html=True)
        st.title(title)
        st.markdown(f'<div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div style="text-align:right;padding-top:1.2rem"><span class="status"><span class="status-dot"></span>Models Loaded</span></div>', unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="brand"><div class="brand-mark">♥</div><div class="brand-name">HeartCare</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-caption">Workspace</div>', unsafe_allow_html=True)
        pages = {
            "⌂  Overview": "Overview",
            "♥  Heart Disease Prediction": "Heart Disease Prediction",
            "⇩  Bulk Prediction": "Bulk Prediction",
            "▦  Data Explorer": "Data Explorer",
            "◒  Model Performance": "Model Performance",
            "ⓘ  About Project": "About Project",
        }
        selected = st.radio("Navigate", list(pages), label_visibility="collapsed")
        st.markdown(
            '<div class="sidebar-footer"><strong>Project information</strong><br>Version 1.0<br>Academic FDS Project</div>',
            unsafe_allow_html=True,
        )
    return pages[selected]


def render_kpi(label, value, note):
    st.markdown(
        f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def overview_page(data, models):
    render_topbar(
        "Heart Disease Analytics",
        "Explore health indicators, understand risk patterns and evaluate machine learning predictions.",
        "Workspace / Overview",
    )
    cases = int(data["HeartDisease"].sum())
    healthy = len(data) - cases
    metrics = evaluate_models(data, models)
    best_accuracy = metrics["Accuracy"].max()
    cards = st.columns(4)
    with cards[0]:
        render_kpi("Total Records", f"{len(data):,}", "Patient records in dataset")
    with cards[1]:
        render_kpi("Heart Disease Cases", f"{cases:,}", f"{cases / len(data):.1%} of records")
    with cards[2]:
        render_kpi("Healthy Cases", f"{healthy:,}", f"{healthy / len(data):.1%} of records")
    with cards[3]:
        render_kpi("Best Model Accuracy", metric_value(best_accuracy), "Stratified 20% holdout")

    st.markdown("### Dataset signals")
    c1, c2 = st.columns(2)
    with c1:
        distribution = data["HeartDisease"].map({0: "Healthy", 1: "Heart disease"}).value_counts().reset_index()
        distribution.columns = ["Outcome", "Count"]
        fig = px.pie(distribution, names="Outcome", values="Count", hole=.68, title="Heart Disease Distribution",
                     color="Outcome", color_discrete_map={"Healthy": "#E5E7EB", "Heart disease": "#F97316"})
        st.plotly_chart(chart_layout(fig), use_container_width=True)
    with c2:
        fig = px.histogram(data, x="Age", color=data["HeartDisease"].map({0: "Healthy", 1: "Heart disease"}),
                           nbins=18, title="Age Distribution", color_discrete_map={"Healthy": "#D4D4D4", "Heart disease": "#F97316"})
        st.plotly_chart(chart_layout(fig), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        gender = data.assign(Sex=data["Sex"].map({"M": "Male", "F": "Female"})).groupby(["Sex", "HeartDisease"]).size().reset_index(name="Count")
        gender["Outcome"] = gender["HeartDisease"].map({0: "Healthy", 1: "Heart disease"})
        fig = px.bar(gender, x="Sex", y="Count", color="Outcome", barmode="group", title="Heart Disease by Gender",
                     color_discrete_map={"Healthy": "#D4D4D4", "Heart disease": "#F97316"})
        st.plotly_chart(chart_layout(fig), use_container_width=True)
    with c2:
        corr = encode_features(data.assign(
            Sex=data["Sex"], ChestPainType=data["ChestPainType"], RestingECG=data["RestingECG"],
            ExerciseAngina=data["ExerciseAngina"], ST_Slope=data["ST_Slope"]
        )).assign(HeartDisease=data["HeartDisease"]).corr()
        fig = px.imshow(corr, text_auto=".1f", aspect="auto", color_continuous_scale=["#FFF7ED", "#F97316", "#171717"], title="Correlation Heatmap")
        st.plotly_chart(chart_layout(fig, 360), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        chest = data.assign(ChestPainType=data["ChestPainType"].map(DISPLAY_LABELS)).groupby("ChestPainType").size().reset_index(name="Count")
        fig = px.bar(chest, x="ChestPainType", y="Count", title="Chest Pain Type Distribution", color="Count", color_continuous_scale=["#FED7AA", "#F97316"])
        st.plotly_chart(chart_layout(fig), use_container_width=True)
    with c2:
        scatter = data.assign(Outcome=data["HeartDisease"].map({0: "Healthy", 1: "Heart disease"}))
        fig = px.scatter(scatter, x="Cholesterol", y="MaxHR", color="Outcome", opacity=.7, title="Cholesterol vs Maximum Heart Rate",
                         color_discrete_map={"Healthy": "#A3A3A3", "Heart disease": "#F97316"})
        st.plotly_chart(chart_layout(fig), use_container_width=True)


def prediction_form():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Personal information</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Age", min_value=1, max_value=120, value=53, help="Patient age in years.")
    with c2:
        sex = st.selectbox("Sex", ["Male", "Female"])
    st.markdown('<div class="section-label" style="margin-top:1rem">Cardiac symptoms</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        chest_pain = st.selectbox("Chest Pain Type", ["Atypical Angina", "Non-Anginal Pain", "Asymptomatic", "Typical Angina"])
    with c2:
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["No", "Yes"])
    st.markdown('<div class="section-label" style="margin-top:1rem">Clinical measurements</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        resting_bp = st.number_input("Resting Blood Pressure", min_value=1, max_value=300, value=130, help="mm Hg")
        fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    with c2:
        cholesterol = st.number_input("Serum Cholesterol", min_value=0, max_value=700, value=200, help="mg/dl")
        max_hr = st.number_input("Maximum Heart Rate", min_value=60, max_value=202, value=140)
    st.markdown('<div class="section-label" style="margin-top:1rem">ECG and stress indicators</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    with c2:
        oldpeak = st.number_input("Oldpeak", min_value=-3.0, max_value=10.0, value=0.0, step=0.1, help="ST depression induced by exercise.")
    with c3:
        st_slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
    st.markdown("</div>", unsafe_allow_html=True)
    return {
        "Age": age,
        "Sex": "M" if sex == "Male" else "F",
        "ChestPainType": {"Atypical Angina": "ATA", "Non-Anginal Pain": "NAP", "Asymptomatic": "ASY", "Typical Angina": "TA"}[chest_pain],
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": 1 if fasting_bs == "> 120 mg/dl" else 0,
        "RestingECG": {"Normal": "Normal", "ST-T Wave Abnormality": "ST", "Left Ventricular Hypertrophy": "LVH"}[resting_ecg],
        "MaxHR": max_hr,
        "ExerciseAngina": "Y" if exercise_angina == "Yes" else "N",
        "Oldpeak": oldpeak,
        "ST_Slope": {"Upsloping": "Up", "Flat": "Flat", "Downsloping": "Down"}[st_slope],
    }


def prediction_page(models):
    render_topbar("Heart Disease Prediction", "Enter patient health parameters to generate a machine learning prediction.", "Workspace / Prediction")
    left, right = st.columns([1.7, 1])
    with left:
        values = prediction_form()
        if st.button("Analyze Heart Disease Risk", use_container_width=True):
            st.session_state["prediction_values"] = values
    with right:
        st.markdown('<div class="section-card"><div class="section-label">Clinical note</div><h3>How to read the result</h3><p style="color:#737373;font-size:13px;line-height:1.6">Each model assesses the same encoded health profile independently. Compare the three outputs to identify agreement or variation.</p><div class="disclaimer">This tool is for educational and research use only. It is not a diagnosis and should not replace advice from a qualified healthcare professional.</div></div>', unsafe_allow_html=True)
    if "prediction_values" not in st.session_state:
        return
    encoded = format_input(st.session_state["prediction_values"])
    st.markdown("### Model results")
    result_columns = st.columns(3)
    for column, (name, model) in zip(result_columns, models.items()):
        prediction = int(model.predict(encoded)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(encoded)[0][list(model.classes_).index(1)])
        with column:
            status = "Heart disease detected" if prediction else "No heart disease detected"
            note = f"Estimated probability: {probability:.1%}" if probability is not None else "Probability output not available for this model"
            css_class = "" if prediction else " healthy"
            st.markdown(f'<div class="result-card{css_class}"><div class="section-label">{name}</div><div class="result-title">{status}</div><div class="result-copy">{note}</div></div>', unsafe_allow_html=True)


def bulk_page(models):
    render_topbar("Bulk Prediction", "Upload a CSV to evaluate multiple patient profiles at once.", "Workspace / Bulk Prediction")
    st.markdown('<div class="upload-note"><strong>Required format</strong><br>CSV files must contain exactly these feature columns: ' + ", ".join(FEATURES) + '. Categorical values must use the training codes: M/F, ATA/NAP/ASY/TA, Normal/ST/LVH, N/Y and Up/Flat/Down.</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop a CSV file here", type=["csv"], label_visibility="visible")
    if uploaded_file is None:
        st.info("Upload a CSV file to preview and predict records.")
        return
    try:
        input_data = pd.read_csv(uploaded_file)
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        st.error(f"Could not read this CSV file: {error}")
        return
    missing = [column for column in FEATURES if column not in input_data.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        return
    if input_data[FEATURES].isnull().any().any():
        st.error("The uploaded data contains missing values. Fill all required fields before predicting.")
        return
    try:
        encoded = encode_features(input_data)
        if (encoded == -1).any().any():
            raise ValueError("One or more categorical values are not supported by the training format.")
        encoded = encoded.astype(float)
    except (TypeError, ValueError) as error:
        st.error(f"Invalid input data: {error}")
        return
    st.markdown("### Preview")
    st.dataframe(input_data[FEATURES].head(10), use_container_width=True, hide_index=True)
    if st.button("Predict uploaded records", use_container_width=False):
        results = input_data[FEATURES].copy()
        for name, model in models.items():
            results[name] = model.predict(encoded)
        st.markdown("### Prediction results")
        st.dataframe(results, use_container_width=True, hide_index=True)
        st.download_button("Download results as CSV", results.to_csv(index=False), "heartcare_predictions.csv", "text/csv")


def explorer_page(data):
    render_topbar("Data Explorer", "Inspect the source dataset, distributions and data quality.", "Workspace / Data Explorer")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{data.shape[0]:,}")
    c2.metric("Columns", data.shape[1])
    c3.metric("Missing values", int(data.isna().sum().sum()))
    c4.metric("Duplicate records", int(data.duplicated().sum()))
    st.markdown("### Dataset preview")
    st.dataframe(data, use_container_width=True, hide_index=True)
    st.markdown("### Column information")
    info = pd.DataFrame({"Column": data.columns, "Data type": data.dtypes.astype(str), "Missing": data.isna().sum().values, "Unique values": data.nunique().values})
    st.dataframe(info, use_container_width=True, hide_index=True)
    with st.expander("Summary statistics"):
        st.dataframe(data.describe(include="all").T, use_container_width=True)
    st.markdown("### Feature distributions")
    feature = st.selectbox("Select a feature", [column for column in data.columns if column != "HeartDisease"])
    if data[feature].dtype == "object":
        counts = data[feature].value_counts().reset_index()
        counts.columns = [feature, "Count"]
        counts[feature] = counts[feature].map(lambda value: DISPLAY_LABELS.get(value, value))
        fig = px.bar(counts, x=feature, y="Count", color="Count", color_continuous_scale=["#FED7AA", "#F97316"])
    else:
        fig = px.histogram(data, x=feature, color=data["HeartDisease"].map({0: "Healthy", 1: "Heart disease"}), color_discrete_map={"Healthy": "#D4D4D4", "Heart disease": "#F97316"})
    st.plotly_chart(chart_layout(fig), use_container_width=True)


def evaluate_models(data, models):
    x = preprocess_for_evaluation(data)
    y = data["HeartDisease"].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42, stratify=y)
    imputer = KNNImputer(n_neighbors=3)
    x_train = pd.DataFrame(imputer.fit_transform(x_train), columns=FEATURES, index=x_train.index)
    x_test = pd.DataFrame(imputer.transform(x_test), columns=FEATURES, index=x_test.index)
    rows = []
    for name, original in models.items():
        model = clone(original)
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1-score": f1_score(y_test, predictions, zero_division=0),
        })
    return pd.DataFrame(rows)


def performance_page(data, models):
    render_topbar("Model Performance", "Compare model quality on a consistent stratified holdout evaluation.", "Workspace / Model Performance")
    st.markdown('<div class="upload-note">Metrics below are calculated by cloning the existing model configurations and evaluating them on a fixed, stratified 20% holdout. No training-set accuracy is shown.</div>', unsafe_allow_html=True)
    metrics = evaluate_models(data, models)
    best = metrics.loc[metrics["Accuracy"].idxmax(), "Model"]
    st.markdown(f"### Model comparison <span style='color:#F97316;font-size:13px;font-weight:500'>Best accuracy: {best}</span>", unsafe_allow_html=True)
    display_metrics = metrics.copy()
    for column in ["Accuracy", "Precision", "Recall", "F1-score"]:
        display_metrics[column] = display_metrics[column].map(metric_value)
    st.dataframe(display_metrics, use_container_width=True, hide_index=True)
    st.markdown("### Diagnostic views")
    x = preprocess_for_evaluation(data)
    y = data["HeartDisease"].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42, stratify=y)
    imputer = KNNImputer(n_neighbors=3)
    x_train = pd.DataFrame(imputer.fit_transform(x_train), columns=FEATURES)
    x_test = pd.DataFrame(imputer.transform(x_test), columns=FEATURES)
    tabs = st.tabs(list(models))
    for tab, (name, original) in zip(tabs, models.items()):
        with tab:
            model = clone(original).fit(x_train, y_train)
            predictions = model.predict(x_test)
            c1, c2 = st.columns(2)
            with c1:
                matrix = confusion_matrix(y_test, predictions)
                fig = px.imshow(matrix, text_auto=True, x=["Healthy", "Heart disease"], y=["Healthy", "Heart disease"], color_continuous_scale=["#FFF7ED", "#F97316"], title=f"{name} confusion matrix")
                st.plotly_chart(chart_layout(fig, 340), use_container_width=True)
            with c2:
                if hasattr(model, "predict_proba"):
                    scores = model.predict_proba(x_test)[:, 1]
                    auc = roc_auc_score(y_test, scores)
                    fpr, tpr, _ = roc_curve(y_test, scores)
                    fig = go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color="#F97316", width=3), name=f"AUC {auc:.2f}"))
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#D4D4D4", dash="dash"), showlegend=False))
                    fig.update_layout(title=f"ROC curve · AUC {auc:.2f}", xaxis_title="False positive rate", yaxis_title="True positive rate")
                    st.plotly_chart(chart_layout(fig, 340), use_container_width=True)
                else:
                    st.info("This SVM was trained without probability estimation, so an ROC/AUC curve is not available.")


def about_page():
    render_topbar("About Project", "A transparent look at the HeartCare academic analytics project.", "Workspace / About Project")
    st.markdown(
        """
        <div class="section-card">
        <div class="section-label">HeartCare</div>
        <h2>Heart disease analytics for informed exploration</h2>
        <p style="color:#737373;line-height:1.7">HeartCare is a Streamlit application built around the UCI heart disease dataset. It provides an approachable interface for exploring health indicators and comparing three trained classification models: Logistic Regression, Decision Tree and Support Vector Machine.</p>
        <p style="color:#737373;line-height:1.7">The application preserves the original feature encoding and serialized model files. Predictions are intended for academic and educational use only, not for clinical decision-making.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    apply_styles()
    data = load_dataset()
    models = load_models()
    page = render_sidebar()
    if page == "Overview":
        overview_page(data, models)
    elif page == "Heart Disease Prediction":
        prediction_page(models)
    elif page == "Bulk Prediction":
        bulk_page(models)
    elif page == "Data Explorer":
        explorer_page(data)
    elif page == "Model Performance":
        performance_page(data, models)
    else:
        about_page()


if __name__ == "__main__":
    main()
