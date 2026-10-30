# HeartCare: Heart Disease Analytics

HeartCare is a Streamlit-based healthcare analytics dashboard for exploring heart
disease data and comparing machine-learning predictions. It supports interactive
dataset exploration, individual patient prediction, bulk CSV prediction, and
holdout-based model evaluation.

> **Medical disclaimer:** HeartCare is an academic and educational project. It
> is not a medical device, diagnostic tool, or substitute for advice from a
> qualified healthcare professional. Never use an application prediction as the
> sole basis for a medical decision.

## Contents

- [Project overview](#project-overview)
- [Features](#features)
- [Repository structure](#repository-structure)
- [Dataset](#dataset)
- [Preprocessing and feature encoding](#preprocessing-and-feature-encoding)
- [Prediction workflow](#prediction-workflow)
- [Machine-learning models](#machine-learning-models)
- [Evaluation metrics and formulas](#evaluation-metrics-and-formulas)
- [Dashboard pages](#dashboard-pages)
- [CSV bulk prediction](#csv-bulk-prediction)
- [Installation and local execution](#installation-and-local-execution)
- [Docker execution](#docker-execution)
- [Configuration](#configuration)
- [Validation and error handling](#validation-and-error-handling)
- [Known limitations](#known-limitations)
- [Development notes](#development-notes)

## Project overview

HeartCare loads:

1. The source dataset from `heart.csv`.
2. Three serialized scikit-learn models:
   - `LogisticR.pkl`
   - `tree.pkl`
   - `SVM.pkl`
3. A Streamlit interface defined in `app.py`.

The application preserves the original model input contract: exactly 11
numeric features in a fixed order. The dashboard adds presentation, analysis,
validation, and comparison features without replacing the trained model files.

## Features

- Premium light healthcare dashboard with a white background and black text.
- Orange accent styling and responsive Streamlit layout.
- Overview KPIs calculated from the actual dataset:
  - Total records
  - Heart disease cases
  - Healthy cases
  - Best holdout accuracy
- Interactive Plotly visualizations:
  - Heart disease distribution
  - Age distribution
  - Heart disease by gender
  - Correlation heatmap
  - Chest pain distribution
  - Cholesterol versus maximum heart rate
- Individual prediction using all three existing models.
- Probability confidence when the model exposes `predict_proba`.
- Explicitly reports when probability output is unavailable.
- Bulk CSV upload, preview, validation, prediction, and download.
- Dataset preview, data types, missing values, duplicates, statistics, and
  feature distributions.
- Consistent stratified holdout model evaluation:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion matrix
  - ROC curve and AUC where probability scores are available.

## Repository structure

```text
Heart-Disease-Analytics/
├── .gitignore                 # Local files excluded from Git
├── .streamlit/
│   └── config.toml            # Light theme and black text configuration
├── Dockerfile                 # Container image definition
├── LogisticR.pkl              # Serialized Logistic Regression model
├── README.md                  # Project documentation
├── SVM.pkl                    # Serialized linear SVM model
├── app.py                     # Streamlit application
├── datapreprocessing.ipynb    # Original exploration and preprocessing notebook
├── heart.csv                 # Source dataset
├── requirements.txt           # Python dependencies
└── tree.pkl                  # Serialized Decision Tree model
```

The local `venv/` directory and Python cache files are intentionally excluded
from version control.

## Dataset

The project dataset contains **918 records** and **12 columns**:

| Column | Type | Meaning |
|---|---:|---|
| `Age` | Integer | Patient age in years |
| `Sex` | Categorical | `M` for male, `F` for female |
| `ChestPainType` | Categorical | `ATA`, `NAP`, `ASY`, or `TA` |
| `RestingBP` | Integer | Resting blood pressure in mm Hg |
| `Cholesterol` | Integer | Serum cholesterol in mg/dl |
| `FastingBS` | Integer | `1` when fasting blood sugar is above 120 mg/dl, otherwise `0` |
| `RestingECG` | Categorical | `Normal`, `ST`, or `LVH` |
| `MaxHR` | Integer | Maximum heart rate achieved |
| `ExerciseAngina` | Categorical | `Y` for yes, `N` for no |
| `Oldpeak` | Float | ST depression induced by exercise |
| `ST_Slope` | Categorical | `Up`, `Flat`, or `Down` |
| `HeartDisease` | Integer | Target: `0` healthy, `1` heart disease |

The target column `HeartDisease` is used for analysis and evaluation. It is
not included in prediction input.

### Dataset observations

The current dataset contains:

- 918 rows
- 11 input features
- 1 target column
- 0 missing values in the raw CSV
- 0 duplicate records
- 508 heart disease cases
- 410 healthy cases

These values are calculated at runtime by the application and are not used as
hardcoded prediction statistics.

## Preprocessing and feature encoding

The original notebook investigates categorical values, replaces categories
with integer codes, replaces invalid zero measurements, and applies
`KNNImputer(n_neighbors=3)` to the affected numeric data.

### Categorical encoding used by the models

The application uses the same category order as the original implementation:

| Feature | Original value | Encoded value |
|---|---|---:|
| `Sex` | `M` | `0` |
| `Sex` | `F` | `1` |
| `ChestPainType` | `ATA` | `0` |
| `ChestPainType` | `NAP` | `1` |
| `ChestPainType` | `ASY` | `2` |
| `ChestPainType` | `TA` | `3` |
| `RestingECG` | `Normal` | `0` |
| `RestingECG` | `ST` | `1` |
| `RestingECG` | `LVH` | `2` |
| `ExerciseAngina` | `N` | `0` |
| `ExerciseAngina` | `Y` | `1` |
| `ST_Slope` | `Up` | `0` |
| `ST_Slope` | `Flat` | `1` |
| `ST_Slope` | `Down` | `2` |

The final model input order is always:

```text
Age,
Sex,
ChestPainType,
RestingBP,
Cholesterol,
FastingBS,
RestingECG,
MaxHR,
ExerciseAngina,
Oldpeak,
ST_Slope
```

### K-nearest-neighbor imputation

The notebook treats zero `Cholesterol` and zero `RestingBP` values as missing
measurements, then estimates replacements with the three nearest rows.

For a missing value in row `i`, KNN imputation can be expressed as:

```text
x̂ᵢ = (1 / k) × Σⱼ∈Nₖ(i) xⱼ
```

where:

- `x̂ᵢ` is the estimated value,
- `Nₖ(i)` is the set of `k = 3` nearest valid rows,
- `xⱼ` is the corresponding observed value for neighbor `j`.

Direct individual and bulk prediction use the original serialized models and
their established numeric input contract. Evaluation creates a fresh
stratified split and fits its imputer on the training partition only.

## Prediction workflow

### Individual prediction

1. The user enters the 11 health parameters.
2. Human-readable selections are converted to the encoded values above.
3. A one-row pandas DataFrame is created in the fixed feature order.
4. Each serialized model receives the same encoded row.
5. Each model returns class `0` or `1`.
6. Logistic Regression and Decision Tree also expose class probabilities.
7. SVM is reported without a probability because the saved SVM was trained
   with probability estimation disabled.

### Prediction interpretation

```text
0 → No heart disease detected
1 → Heart disease detected
```

The result is a model output, not a clinical diagnosis.

## Machine-learning models

### Logistic Regression

The saved model is a scikit-learn `LogisticRegression` classifier.

For input vector `x`, the linear score is:

```text
z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

The probability of class `1` is calculated using the logistic sigmoid:

```text
P(y = 1 | x) = σ(z) = 1 / (1 + e⁻ᶻ)
```

The class decision is:

```text
ŷ = 1  if P(y = 1 | x) ≥ 0.5
ŷ = 0  otherwise
```

The application displays this probability when `predict_proba` is available.

### Decision Tree

The saved model is a scikit-learn `DecisionTreeClassifier` with the
serialized configuration used by the project.

For a node containing class proportions `p₀, p₁, ..., p₍c₋₁₎`, the Gini
impurity is:

```text
Gini = 1 - Σᶜₖ₌₁ pₖ²
```

For a binary target:

```text
Gini = 1 - p₀² - p₁²
```

For a candidate split, the weighted impurity is:

```text
G_split =
    (n_left / n_total)  × G_left
  + (n_right / n_total) × G_right
```

The tree selects splits that reduce impurity. Its probability output is the
class distribution of the terminal leaf reached by the input.

### Support Vector Machine

The saved model is a scikit-learn `SVC` using a linear kernel.

The separating hyperplane is:

```text
f(x) = wᵀx + b
```

The predicted class is determined by the sign:

```text
ŷ = 1  if f(x) ≥ 0
ŷ = 0  if f(x) < 0
```

The saved SVM has `probability=False`, so the app does not invent or display a
probability estimate for it.

## Evaluation metrics and formulas

The Model Performance page evaluates cloned versions of the saved model
configurations on a fixed, stratified 20% holdout:

```text
train_test_split(
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

The models are fitted only on the training partition. Metrics are calculated
on the untouched test partition.

For binary classification, define:

- `TP`: true positives
- `TN`: true negatives
- `FP`: false positives
- `FN`: false negatives

### Accuracy

The proportion of all predictions that are correct:

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Precision

Of the cases predicted positive, the proportion that is actually positive:

```text
Precision = TP / (TP + FP)
```

### Recall

Of the actual positive cases, the proportion correctly identified:

```text
Recall = TP / (TP + FN)
```

### F1-score

The harmonic mean of precision and recall:

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### Confusion matrix

The application displays:

```text
                 Predicted 0   Predicted 1
Actual 0             TN            FP
Actual 1             FN            TP
```

### ROC curve

The ROC curve plots true-positive rate against false-positive rate as the
classification threshold changes:

```text
TPR = TP / (TP + FN)
FPR = FP / (FP + TN)
```

### AUC

The area under the ROC curve summarizes ranking performance across thresholds.
It is shown only for models that provide usable probability scores. The saved
linear SVM does not provide `predict_proba`, so its ROC/AUC view is omitted.

### Current holdout results

With the current dataset and `random_state=42`, the application calculates:

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 86.4% | 86.0% | 90.2% | 88.0% |
| Support Vector Machine | 84.2% | 85.4% | 86.3% | 85.9% |
| Decision Tree | 81.0% | 83.2% | 82.4% | 82.8% |

These metrics are reproducible for the current files and evaluation seed, but
they should not be treated as clinical validation.

## Dashboard pages

### Overview

The landing page calculates actual dataset KPIs and displays:

- Heart disease versus healthy distribution
- Age histogram
- Gender comparison
- Encoded-feature correlation heatmap
- Chest pain counts
- Cholesterol versus maximum heart rate scatter plot

### Heart Disease Prediction

The form groups inputs into:

1. Personal information
   - Age
   - Sex
2. Cardiac symptoms
   - Chest pain type
   - Exercise-induced angina
3. Clinical measurements
   - Resting blood pressure
   - Serum cholesterol
   - Fasting blood sugar
   - Maximum heart rate
4. ECG and stress indicators
   - Resting ECG
   - Oldpeak
   - ST slope

The page returns a separate result card for Logistic Regression, Decision Tree,
and SVM.

### Bulk Prediction

The user uploads a CSV, reviews a preview, clicks the prediction button, and
downloads a results file containing the original features plus one prediction
column per model.

### Data Explorer

The page provides:

- Full dataset preview
- Row and column count
- Missing-value count
- Duplicate count
- Column data types
- Unique-value counts
- Summary statistics
- Selectable feature distributions
- Actual-data Plotly charts

### Model Performance

The page displays:

- Holdout comparison table
- Best model based on accuracy
- Confusion matrix per model
- ROC/AUC for probability-capable models
- A clear statement that training-set accuracy is not reported

### About Project

The page documents the academic purpose, model set, and non-clinical scope of
the application.

## CSV bulk prediction

### Required columns

```csv
Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope
```

### Example row

```csv
53,M,ATA,130,200,0,Normal,140,N,0.0,Up
```

### Accepted categorical values

```text
Sex:             M, F
ChestPainType:   ATA, NAP, ASY, TA
RestingECG:      Normal, ST, LVH
ExerciseAngina:  N, Y
ST_Slope:        Up, Flat, Down
```

The target column `HeartDisease` is optional in an upload and is not required
for prediction. If present, it is ignored as an input feature.

### Validation behavior

The upload workflow rejects:

- Files that cannot be parsed as CSV.
- Missing required columns.
- Rows containing missing values.
- Unsupported categorical values.

The output contains the original required columns followed by:

```text
Logistic Regression
Decision Tree
Support Vector Machine
```

## Installation and local execution

### Requirements

- Python 3.9 or newer
- pip
- Windows, macOS, or Linux

### Clone the repository

```bash
git clone https://github.com/dhruv-gupta-28/Heart-Disease-Analytics.git
cd Heart-Disease-Analytics
```

### Create a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Start Streamlit

```bash
python -m streamlit run app.py
```

Open:

```text
http://127.0.0.1:8501
```

Windows users with the included environment can run:

```powershell
Set-Location "C:\MINI PROJECT FDS"
& ".\venv\Scripts\python.exe" -m streamlit run app.py
```

## Docker execution

Build the image:

```bash
docker build -t heartcare .
```

Run the container:

```bash
docker run --rm -p 8501:8501 heartcare
```

Open:

```text
http://127.0.0.1:8501
```

The Dockerfile copies the project into `/app`, installs
`requirements.txt`, exposes port `8501`, and starts Streamlit.

## Configuration

`.streamlit/config.toml` sets:

```toml
[theme]
base = "light"
primaryColor = "#F97316"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#FFFFFF"
textColor = "#000000"
font = "sans serif"
```

The application also applies page-level CSS to keep text, tables, controls,
cards, charts, and navigation readable across Streamlit theme updates.

## Dependencies

```text
pandas
scikit-learn
numpy
plotly
streamlit
```

The serialized models were created with scikit-learn `1.7.2`. Loading them
with a different scikit-learn version may produce compatibility warnings.
For reproducible deployments, pin the model-training and serving versions
together and regenerate the pickle files when upgrading scikit-learn.

## Validation and error handling

The project has been validated for:

- Python syntax compilation.
- Streamlit startup and health endpoint response.
- Loading all three serialized models.
- Individual predictions through all models.
- Bulk prediction on dataset rows.
- Correct 11-column feature order.
- Probability display only for models supporting `predict_proba`.
- Holdout metric calculation.
- CSV parsing and required-column validation.
- Visible rendering of every dashboard page.

The application intentionally uses explicit error messages for invalid uploads
instead of silently substituting data or returning success-shaped fallbacks.

## Known limitations

- The models are loaded from pickle files. Pickle files should only be loaded
  from trusted sources because deserialization can execute arbitrary code.
- The source notebook does not persist a formal train/test split, preprocessing
  pipeline, or model-training metadata. The performance page therefore uses a
  documented reproducible stratified holdout for comparison.
- The current serialized SVM does not support probability estimation.
- The dataset may contain measurement values that require domain-specific
  interpretation.
- Holdout metrics are not a substitute for external validation, calibration,
  fairness analysis, or clinical trials.
- The application has no authentication, user database, patient record storage,
  or production-grade audit logging.
- Uploaded data is processed in the active Streamlit session and should not
  contain unnecessary personal or identifying information.

## Development notes

- Keep all model inputs in the `FEATURES` list in `app.py`.
- If category encoding changes, update the category order and all CSV
  documentation together.
- Do not report training accuracy as generalization performance.
- Re-run the holdout evaluation after changing models or preprocessing.
- Keep generated environments, secrets, and caches out of Git.
- Use a proper model format and pinned dependencies for production systems.

## Author and repository

- Repository: https://github.com/dhruv-gupta-28/Heart-Disease-Analytics
- Project: Academic FDS Heart Disease Analytics
