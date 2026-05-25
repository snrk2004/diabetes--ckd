# 🩺 Frictionless Early Screening for CKD & Diabetes
**An Ultra-Minimalist Machine Learning Bridge Architecture**

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4-orange?logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey?logo=flask)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## 📌 Project Overview
Chronic diseases like Type 2 Diabetes and Chronic Kidney Disease (CKD) are "silent killers." Current machine learning models for early detection often suffer from fatal flaws: they require invasive diagnostic lab tests (causing target leakage), they evaluate based on default 0.5 thresholds (resulting in dangerous false negatives), and they apply data-balancing techniques globally (causing data leakage).

This project resolves these issues by introducing a **"Frictionless ML Bridge Architecture."** It accurately screens for both diseases using only **8 routine, non-invasive patient vitals**.

### ✨ Core Innovations
1. **Zero-Leakage Cross-Validation:** SMOTE (Synthetic Minority Oversampling Technique) is strictly isolated *inside* Stratified 10-Fold CV loops using `imblearn.pipeline`, guaranteeing realistic test evaluations.
2. **Dynamic Threshold Optimization:** Standard "Accuracy" is discarded. The model parses ROC curves to compute exact clinical probability thresholds (e.g., 0.148 and 0.644) that mathematically guarantee **>90% Sensitivity (Recall)**, prioritizing patient safety over raw precision.
3. **The "Minimalist Bridge":** The pipeline first assesses Diabetes risk, then dynamically bridges that boolean output into a restricted CKD model alongside basic vitals (Age, BP, Glucose)—mimicking a doctor's sequential diagnostic logic.
4. **Explainable AI (XAI):** Integrated SHAP (SHapley Additive exPlanations) to crack the ML "black box," proving that the models base decisions on verified clinical biomarkers.

## 🏗️ System Architecture
The system accepts 8 basic inputs: *Age, Blood Pressure, Fasting Glucose, BMI, Insulin, Skin Thickness, Pregnancies, and Diabetes Pedigree Function*.

1. **Preprocessing Layer:** Safe Median Imputation & Standard Scaling.
2. **Layer 1 (Diabetes Model):** Gradient Boosting evaluates all 8 vitals. 
3. **The Bridge:** High-risk patients trigger a `dm_flag` (1 or 0), which is passed forward.
4. **Layer 2 (CKD Model):** A highly restricted model evaluates *only* Age, BP, Glucose, and the `dm_flag` to predict kidney failure risk.
5. **Clinical Routing:** The Flask Web App synthesizes the results into actionable clinical states: `Clear (Green)`, `Warning (Orange)`, or `Urgent Escalation (Red)`.

## 📊 Final Performance Metrics
Models were rigorously evaluated using Stratified 10-Fold Cross-Validation. **Gradient Boosting** was selected as the optimal algorithm.

| Pipeline | Model | 10-Fold AUC | Computed Threshold | Clinical Recall | Precision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Diabetes** | Gradient Boosting | 0.838 | 0.148 | **90.7%** | 53.8% |
| **CKD (Minimalist)** | Gradient Boosting | 0.941 | 0.644 | **90.0%** | 100.0% |

> *Note: Diabetes precision is intentionally lowered. In clinical screening, a False Positive results in a harmless follow-up blood test, whereas a False Negative is fatal. The threshold was artificially lowered to 0.148 to cast a wider net and maximize Recall.*

## 📂 Project Structure
```text
FINAL_INTERNSHIP/
│
├── production_pipeline.py    # Main ML training, CV, thresholding, and XAI script
├── app.py                    # Flask REST API and clinical routing logic
├── datasets/                 # Raw CSV files (Diabetes & CKD)
├── clinical_plots/           # Auto-generated ROC, PR, Confusion Matrix & SHAP plots
├── saved_models/             # Serialized .pkl files (Imputers, Scalers, Models)
├── templates/                # HTML/JS for the Flask Web UI
└── README.md                 # Project documentation
🚀 How to Run the Project
1. Prerequisites
Ensure you have Python 3.10+ installed. Install the required dependencies in your terminal:

Bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib shap flask openpyxl
2. Train the Models (Optional)
If you want to retrain the models and generate the SHAP/ROC plots from scratch, run the production pipeline:

Bash
python production_pipeline.py
(Check the clinical_plots/ folder to view the generated evaluation metrics and architecture diagrams).

3. Launch the Clinical Web App
To start the clinical routing interface, run the Flask application:

Bash
python app.py
Once the terminal displays that the server is running (usually Running on http://127.0.0.1:5000/), open your web browser and navigate to that local URL.

🧪 Sample Inputs for Testing
To test the dynamic routing logic of the web application, use the following sample patient profiles. These will trigger the different risk thresholds and demonstrate the UI's color-coded response system.

Case 1: CLEAR (Green State)
Patient is currently low risk for both conditions.

Age: 25

Blood Pressure: 70

Fasting Glucose: 90

BMI: 22

Insulin: 80

Skin Thickness: 15

Pregnancies: 0

Diabetes Pedigree Function: 0.25

Case 2: WARNING (Orange State)
High risk for early-stage issues. Triggers a warning based on elevated age and glucose.

Age: 65

Blood Pressure: 90

Fasting Glucose: 110

BMI: 28

Insulin: 100

Skin Thickness: 22

Pregnancies: 1

Diabetes Pedigree Function: 0.45

Case 3: URGENT ESCALATION (Red State)
High risk for both Diabetes and Early-Stage CKD. Both thresholds breached.

Age: 55

Blood Pressure: 85

Fasting Glucose: 150

BMI: 32

Insulin: 180

Skin Thickness: 28

Pregnancies: 2

Diabetes Pedigree Function: 0.65

👨‍💻 Author
Shankara Narayana Rao K SRN: PES1UG22AM149 | PES University B.Tech in Computer Science and Engineering (AI & ML) - 8th Semester Internship Project

Guide: Prof. Agha Alfi Mirza