# ❤️ Heart Disease Prediction App

A **Machine Learning-powered web application** built using **Streamlit** that predicts the likelihood of heart disease based on patient health parameters. 
The app supports both **single prediction** and **bulk prediction via CSV upload**.

---

## 🚀 Features

- 🔍 **Single Prediction**
  - Enter patient details manually
  - Get predictions from multiple ML models:
    - Decision Tree  
    - Logistic Regression  
    - Support Vector Machine  

- 📊 **Bulk Prediction**
  - Upload CSV file with multiple patient records
  - Download predictions as a CSV file

- 📈 Clean and interactive UI using Streamlit

---

## 🧠 Machine Learning Models Used

- Decision Tree Classifier  
- Logistic Regression  
- Support Vector Machine (SVM)  

All models are loaded using `.pkl` files and used for prediction.

---

## 📂 Project Structure

```
├── app.py                  # Main Streamlit app
├── datapreprocessing.ipynb # Data preprocessing & model training
├── heart.csv              # Dataset
├── tree.pkl               # Decision Tree model
├── LogisticR.pkl          # Logistic Regression model
├── SVM.pkl                # SVM model
├── requirements.txt       # Dependencies
├── Dockerfile             # Docker file
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/sakshi-0101/HeartDiseasePrediction.git
cd HeartDiseasePredictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

---

## 📥 Input Features

The model uses the following 11 features:

- Age  
- Sex (0 = Male, 1 = Female)  
- Chest Pain Type  
- Resting Blood Pressure  
- Cholesterol  
- Fasting Blood Sugar  
- Resting ECG  
- Max Heart Rate  
- Exercise Induced Angina  
- Oldpeak  
- ST Slope  

---

## 📊 CSV Upload Format

Ensure your CSV file:

- Has **no missing values**
- Contains exactly these columns:

```
Age, Sex, ChestPainType, RestingBP, Cholesterol,
FastingBS, RestingECG, MaxHR, ExerciseAngina,
Oldpeak, ST_Slope
```

---

## 📤 Output

- Displays prediction results in the app
- Allows downloading results as a CSV file

---

## 📦 Requirements

```
pandas
scikit-learn
plotly
streamlit
numpy
```

---



## 👩‍💻 Author

**Sakshi Grawal**
