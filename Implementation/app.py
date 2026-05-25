import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. LOAD TRAINED ARTIFACTS
# ==========================================
diab_model = joblib.load(os.path.join(BASE_DIR, 'saved_models', 'Diabetes_model.pkl'))
diab_imputer = joblib.load(os.path.join(BASE_DIR, 'saved_models', 'Diabetes_imputer.pkl'))
diab_scaler = joblib.load(os.path.join(BASE_DIR, 'saved_models', 'Diabetes_scaler.pkl'))

ckd_model = joblib.load(os.path.join(BASE_DIR, 'saved_models', 'CKD_model.pkl'))
ckd_imputer = joblib.load(os.path.join(BASE_DIR, 'saved_models', 'CKD_imputer.pkl'))
ckd_scaler = joblib.load(os.path.join(BASE_DIR, 'saved_models', 'CKD_scaler.pkl'))

# Saftey thresholds based on your model's 90% Sensitivity computation
DIAB_THRESHOLD = 0.148
CKD_THRESHOLD = 0.644

# ==========================================
# 2. INFERENCE LOGIC (DUAL DISEASE PIPELINE)
# ==========================================
def predict_clinical_risk(patient_data):
    # --- MODEL 1: Diabetes Check ---
    diab_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    df_diab = pd.DataFrame([patient_data], columns=diab_features)
    
    df_diab_imp = pd.DataFrame(diab_imputer.transform(df_diab), columns=diab_features)
    df_diab_scaled = pd.DataFrame(diab_scaler.transform(df_diab_imp), columns=diab_features)
    
    diab_prob = diab_model.predict_proba(df_diab_scaled)[0][1]
    diab_risk = bool(diab_prob >= DIAB_THRESHOLD)

    # --- MODEL 2: Ultra-Minimalist CKD Bridge ---
    # It takes ONLY Age, BP, Glucose, and the 'diab_risk' calculated from Model 1
    ckd_features = ['age', 'bp', 'bgr', 'dm']
    ckd_mapping = {
        'age': patient_data.get('Age'),
        'bp': patient_data.get('BloodPressure'),
        'bgr': patient_data.get('Glucose'),
        'dm': 1 if diab_risk else 0  # <--- THE BRIDGE HAPPENS HERE
    }
    df_ckd = pd.DataFrame([ckd_mapping], columns=ckd_features)
    
    df_ckd_imp = pd.DataFrame(ckd_imputer.transform(df_ckd), columns=ckd_features)
    df_ckd_scaled = pd.DataFrame(ckd_scaler.transform(df_ckd_imp), columns=ckd_features)
    
    ckd_prob = ckd_model.predict_proba(df_ckd_scaled)[0][1]
    ckd_risk = bool(ckd_prob >= CKD_THRESHOLD)

    # --- CLINICAL ROUTING ---
    if ckd_risk and diab_risk:
        status = "URGENT ESCALATION: High risk for both Diabetes and Early-Stage CKD."
        color = "#b42318" # Dark Red
    elif ckd_risk:
        status = "WARNING: High risk for Early-Stage CKD."
        color = "#b54708" # Orange
    elif diab_risk:
        status = "WARNING: High risk for Diabetes. Monitor renal function."
        color = "#b54708" # Orange
    else:
        status = "CLEAR: Patient is currently low risk for both conditions."
        color = "#127c48" # Green

    return {
        "Diabetes_Probability": f"{diab_prob:.2f}",
        "CKD_Probability": f"{ckd_prob:.2f}",
        "Clinical_Recommendation": status,
        "UI_Color": color
    }

# ==========================================
# 3. FLASK WEB ROUTES
# ==========================================
@app.route('/')
def home():
    html = """
    <html>
        <head>
            <title>Minimalist Clinical Screening</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 40px; max-width: 600px; margin: auto; background-color: #f5f7fb; color: #333; }
                .card { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                input, button { padding: 12px; margin: 8px 0; width: 100%; box-sizing: border-box; border-radius: 6px; border: 1px solid #ccc; font-size: 14px; }
                button { background-color: #155eef; color: white; font-weight: bold; cursor: pointer; border: none; transition: 0.3s; }
                button:hover { background-color: #104ab5; }
                #result { margin-top: 20px; padding: 20px; font-weight: bold; border-radius: 6px; color: white; line-height: 1.5; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
                h2 { color: #155eef; margin-top: 0; }
            </style>
        </head>
        <body>
            <div class="card">
                <h2>Frictionless Early Screening</h2>
                <p>Enter 8 baseline vitals to generate a dual-disease risk forecast.</p>
                <form id="patientForm">
                    <label>Age:</label> <input type="number" id="Age" value="55"><br>
                    <label>Blood Pressure (mmHg):</label> <input type="number" id="BloodPressure" value="90"><br>
                    <label>Glucose (Fasting):</label> <input type="number" id="Glucose" value="140"><br>
                    <label>BMI:</label> <input type="number" id="BMI" value="32.5"><br>
                    <label>Insulin:</label> <input type="number" id="Insulin" value="0"><br>
                    <label>Skin Thickness:</label> <input type="number" id="SkinThickness" value="35"><br>
                    <label>Pregnancies:</label> <input type="number" id="Pregnancies" value="2"><br>
                    <label>Diabetes Pedigree Function:</label> <input type="number" step="0.01" id="DiabetesPedigreeFunction" value="0.6"><br>
                    <button type="button" onclick="runInference()">Run AI Clinical Inference</button>
                </form>
            </div>
            <div id="result" style="display:none;"></div>

            <script>
                async function runInference() {
                    const data = {
                        Age: document.getElementById('Age').value,
                        Glucose: document.getElementById('Glucose').value,
                        BloodPressure: document.getElementById('BloodPressure').value,
                        BMI: document.getElementById('BMI').value,
                        Insulin: document.getElementById('Insulin').value,
                        SkinThickness: document.getElementById('SkinThickness').value,
                        Pregnancies: document.getElementById('Pregnancies').value,
                        DiabetesPedigreeFunction: document.getElementById('DiabetesPedigreeFunction').value
                    };
                    
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(data)
                    });
                    
                    const result = await response.json();
                    const resultDiv = document.getElementById('result');
                    resultDiv.style.display = 'block';
                    resultDiv.style.backgroundColor = result.UI_Color;
                    
                    resultDiv.innerHTML = `
                        <h3 style="margin-top: 0;">${result.Clinical_Recommendation}</h3>
                        <hr style="border-color: rgba(255,255,255,0.3);">
                        <b>Diabetes Risk Score:</b> ${result.Diabetes_Probability} (Trigger Threshold: 0.14)<br>
                        <b>CKD Risk Score:</b> ${result.CKD_Probability} (Trigger Threshold: 0.64)
                    `;
                }
            </script>
        </body>
    </html>
    """
    return render_template_string(html)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    patient_data = {k: float(v) for k, v in data.items()}
    return jsonify(predict_clinical_risk(patient_data))

if __name__ == '__main__':
    print("Starting ML Backend Server...")
    app.run(debug=True, port=5000)