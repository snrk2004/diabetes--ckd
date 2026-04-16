"""
================================================================================
  CKD Early Detection — Web Inference Application
  Flask REST API + Single-page HTML frontend

  Run:
    !pip install flask flask-cors -q
    !python ckd_web_inference.py &
    # Then open the printed URL in a browser or use the embedded HTML below.

  Endpoints:
    GET  /              → serves the frontend HTML page
    POST /predict/diabetes  → diabetes risk assessment
    POST /predict/ckd       → direct CKD risk assessment
    POST /predict/bridge    → DM→CKD escalation risk (bridge model)
    POST /predict/full      → full pipeline: diabetes → bridge → CKD
    GET  /thresholds        → returns all auto-computed thresholds
    GET  /features          → returns bridge feature names
    GET  /health            → health check

  Input format (JSON):
    /predict/diabetes  → all Diabetes dataset features
    /predict/ckd       → all CKD dataset features
    /predict/bridge    → bridge features only (auto-discovered)
    /predict/full      → pass both sets; pipeline runs in order
================================================================================
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# ── Load all saved artefacts ─────────────────────────────────────────────────
MODEL_DIR = r"saved_models"

# ── Version-safe imputer wrapper ────────────────────────────────────────────
# Root cause: SimpleImputer pickled with sklearn < 1.4 is missing the internal
# attribute _fill_dtype which was introduced in sklearn 1.4. When sklearn 1.4+
# tries to call .transform(), it accesses _fill_dtype and raises AttributeError.
#
# Fix: after loading any imputer, call _refit_imputer() which rebuilds the
# internal state from the always-present statistics_ array by re-fitting the
# imputer on a dummy 2-row array constructed from those stored medians.
# This is safe because statistics_ (the per-column medians) is always saved
# regardless of sklearn version, so the imputation values are unchanged.
def _refit_imputer(imp):
    """Restore sklearn-version-specific internal attributes from statistics_."""
    if not hasattr(imp, '_fill_dtype'):
        try:
            stats = imp.statistics_
            dummy = np.tile(stats, (2, 1)).astype(float)
            imp.fit(dummy)
        except Exception as e:
            print(f"  WARNING: imputer refit failed ({e}) — using fallback")
    return imp

def load_artefacts():
    art = {}
    art["rf_diab"]     = joblib.load(f"{MODEL_DIR}/rf_model_diabetes.pkl")
    art["rf_ckd"]      = joblib.load(f"{MODEL_DIR}/rf_model_ckd.pkl")
    art["rf_bridge"]   = joblib.load(f"{MODEL_DIR}/rf_model_bridge.pkl")
    # Load and immediately re-fit imputers to restore version-specific internals
    art["imp_diab"]    = _refit_imputer(joblib.load(f"{MODEL_DIR}/imputer_diabetes.pkl"))
    art["imp_ckd"]     = _refit_imputer(joblib.load(f"{MODEL_DIR}/imputer_ckd.pkl"))
    art["imp_bridge"]  = _refit_imputer(joblib.load(f"{MODEL_DIR}/imputer_bridge.pkl"))
    art["scl_diab"]    = joblib.load(f"{MODEL_DIR}/scaler_diabetes.pkl")
    art["scl_ckd"]     = joblib.load(f"{MODEL_DIR}/scaler_ckd.pkl")
    art["scl_bridge"]  = joblib.load(f"{MODEL_DIR}/scaler_bridge.pkl")
    art["cols_diab"]   = joblib.load(f"{MODEL_DIR}/diab_columns.pkl")
    art["cols_ckd"]    = joblib.load(f"{MODEL_DIR}/ckd_columns.pkl")
    art["cols_bridge"] = joblib.load(f"{MODEL_DIR}/bridge_features_diab.pkl")
    art["cols_bridge_ckd"] = joblib.load(f"{MODEL_DIR}/bridge_features_ckd.pkl")
    art["binary_map"]  = joblib.load(f"{MODEL_DIR}/ckd_binary_map.pkl")
    art["thresholds"]  = joblib.load(f"{MODEL_DIR}/optimal_thresholds.pkl")
    art["lit_thrs"]    = joblib.load(f"{MODEL_DIR}/literature_thresholds.pkl")
    art["dt_thrs"]     = joblib.load(f"{MODEL_DIR}/dt_thresholds.pkl")
    print("✅ All artefacts loaded.")
    print(f"   Bridge features : {art['cols_bridge']} ↔ {art['cols_bridge_ckd']}")
    print(f"   Thresholds      : {art['thresholds']}")
    return art

A = load_artefacts()

# ── Inference helpers ─────────────────────────────────────────────────────────
def encode_ckd(data_dict):
    enc = {}
    for k, v in data_dict.items():
        if k in A["binary_map"]:
            enc[k] = A["binary_map"][k].get(str(v).strip().lower(), np.nan)
        else:
            enc[k] = v
    return enc

def prepare(data_dict, cols, imp, scl):
    df = pd.DataFrame([{c: data_dict.get(c, np.nan) for c in cols}],
                       columns=cols)
    df = df.apply(pd.to_numeric, errors="coerce")
    try:
        imputed = imp.transform(df)
    except AttributeError:
        # Fallback if _fill_dtype still missing after refit attempt:
        # manually fill NaNs using imp.statistics_ (always present)
        df_filled = df.copy()
        for i, stat in enumerate(imp.statistics_):
            df_filled.iloc[:, i] = df_filled.iloc[:, i].fillna(
                float(stat) if not np.isnan(float(stat)) else 0.0)
        imputed = df_filled.values
    return scl.transform(imputed)

def risk_label(prob):
    if prob >= 0.75: return "HIGH",   "#FF6B6B"
    if prob >= 0.50: return "MEDIUM", "#FFD93D"
    if prob >= 0.30: return "LOW",    "#FF9A3C"
    return "MINIMAL", "#6BCB77"

def predict_diabetes(data):
    X = prepare(data, A["cols_diab"], A["imp_diab"], A["scl_diab"])
    prob = float(A["rf_diab"].predict_proba(X)[0][1])
    thr  = A["thresholds"].get("youden", 0.5)
    label, color = risk_label(prob)
    return {
        "probability": round(prob * 100, 2),
        "verdict": "POSITIVE" if prob >= thr else "NEGATIVE",
        "risk_level": label,
        "risk_color": color,
        "threshold_used": round(thr, 3),
        "threshold_method": "Youden's J"
    }

def predict_ckd(data):
    encoded = encode_ckd(data)
    X = prepare(encoded, A["cols_ckd"], A["imp_ckd"], A["scl_ckd"])
    prob = float(A["rf_ckd"].predict_proba(X)[0][1])
    thr  = A["thresholds"].get("youden", 0.5)
    label, color = risk_label(prob)
    return {
        "probability": round(prob * 100, 2),
        "verdict": "POSITIVE" if prob >= thr else "NEGATIVE",
        "risk_level": label,
        "risk_color": color,
        "threshold_used": round(thr, 3),
        "threshold_method": "Youden's J"
    }

def predict_bridge(data, threshold_mode="youden"):
    # Build bridge input from either Diabetes or CKD feature names
    bridge_data = {}
    for bf, cf in zip(A["cols_bridge"], A["cols_bridge_ckd"]):
        if bf in data:
            bridge_data[bf] = data[bf]
        elif cf in data:
            bridge_data[bf] = data[cf]
        else:
            bridge_data[bf] = np.nan
    X = prepare(bridge_data, A["cols_bridge"], A["imp_bridge"], A["scl_bridge"])
    prob = float(A["rf_bridge"].predict_proba(X)[0][1])
    thr  = A["thresholds"].get(threshold_mode, 0.5)
    label, color = risk_label(prob)
    return {
        "probability": round(prob * 100, 2),
        "verdict": "ESCALATE TO CKD SCREENING" if prob >= thr else "MONITOR",
        "risk_level": label,
        "risk_color": color,
        "threshold_used": round(thr, 3),
        "threshold_method": threshold_mode,
        "bridge_features_used": A["cols_bridge"]
    }

# ── Routes ───────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({"status": "ok", "models_loaded": True,
                    "bridge_features": A["cols_bridge"]})

@app.route("/thresholds")
def get_thresholds():
    return jsonify({
        "optimal_thresholds": A["thresholds"],
        "literature_thresholds": [
            {k: v for k, v in lit.items() if k != "color"}
            for lit in A["lit_thrs"]
        ],
        "dt_thresholds": {k: v for k, v in A["dt_thrs"].items()}
    })

@app.route("/features")
def get_features():
    return jsonify({
        "diabetes_features": A["cols_diab"],
        "ckd_features": A["cols_ckd"],
        "bridge_features_diabetes": A["cols_bridge"],
        "bridge_features_ckd": A["cols_bridge_ckd"],
        "ckd_binary_features": list(A["binary_map"].keys()),
        "ckd_binary_values": A["binary_map"]
    })

@app.route("/predict/diabetes", methods=["POST"])
def api_diabetes():
    try:
        data = request.get_json(force=True)
        return jsonify({"status": "ok", "result": predict_diabetes(data)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/predict/ckd", methods=["POST"])
def api_ckd():
    try:
        data = request.get_json(force=True)
        return jsonify({"status": "ok", "result": predict_ckd(data)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/predict/bridge", methods=["POST"])
def api_bridge():
    try:
        data = request.get_json(force=True)
        thr_mode = data.pop("threshold_mode", "youden")
        return jsonify({"status": "ok",
                        "result": predict_bridge(data, thr_mode)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/predict/full", methods=["POST"])
def api_full():
    """
    Full pipeline:
      1. Assess diabetes risk from diabetes features
      2. If diabetes probability is meaningful, run bridge escalation
      3. Assess CKD risk from CKD features
      4. Return composite clinical decision
    """
    try:
        data = request.get_json(force=True)
        diab_result   = predict_diabetes(data)
        bridge_result = predict_bridge(data)
        ckd_result    = predict_ckd(data)

        # Clinical decision logic
        diab_prob   = diab_result["probability"]
        bridge_prob = bridge_result["probability"]
        ckd_prob    = ckd_result["probability"]

        if ckd_prob >= 50:
            decision = "URGENT: CKD screening recommended immediately"
            urgency  = "URGENT"
        elif bridge_prob >= A["thresholds"].get("youden", 50) * 100:
            decision = "ESCALATE: Diabetes pattern warrants CKD investigation"
            urgency  = "ESCALATE"
        elif diab_prob >= 50:
            decision = "MONITOR: Diabetic patient — periodic CKD monitoring advised"
            urgency  = "MONITOR"
        else:
            decision = "ROUTINE: No immediate CKD escalation required"
            urgency  = "ROUTINE"

        urgency_colors = {
            "URGENT":   "#FF6B6B",
            "ESCALATE": "#FF9A3C",
            "MONITOR":  "#FFD93D",
            "ROUTINE":  "#6BCB77",
        }

        return jsonify({
            "status": "ok",
            "pipeline": {
                "step1_diabetes":      diab_result,
                "step2_bridge_dm_ckd": bridge_result,
                "step3_ckd":           ckd_result,
            },
            "clinical_decision": {
                "verdict":  decision,
                "urgency":  urgency,
                "color":    urgency_colors[urgency],
                "diabetes_probability":  diab_prob,
                "bridge_probability":    bridge_prob,
                "ckd_probability":       ckd_prob,
            }
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# ── Frontend HTML ─────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CKD Early Detection System</title>
<style>
  :root {
    --bg:      #0F1117; --panel:  #1A1D27; --border: #2D3142;
    --c1: #00D4FF; --c2: #FF6B6B; --c3: #FFD93D;
    --c4: #6BCB77; --c5: #C77DFF; --c6: #FF9A3C;
    --text: #E8E8E8; --sub: #9CA3AF;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', sans-serif; min-height: 100vh; }

  header { background: var(--panel); border-bottom: 1px solid var(--border);
           padding: 18px 32px; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 22px; font-weight: 600; color: var(--c1); }
  header span { font-size: 13px; color: var(--sub); }
  .badge { background: var(--c4)22; border: 1px solid var(--c4);
           color: var(--c4); padding: 3px 10px; border-radius: 20px; font-size: 12px; }

  .container { max-width: 1200px; margin: 0 auto; padding: 28px 24px; }

  .tabs { display: flex; gap: 4px; border-bottom: 1px solid var(--border);
          margin-bottom: 28px; overflow-x: auto; }
  .tab { padding: 10px 22px; cursor: pointer; border-radius: 8px 8px 0 0;
         font-size: 14px; color: var(--sub); transition: all .2s;
         white-space: nowrap; }
  .tab:hover { color: var(--text); background: var(--panel); }
  .tab.active { color: var(--c1); border-bottom: 2px solid var(--c1);
                background: var(--panel); }

  .panel { display: none; }
  .panel.active { display: block; }

  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  .grid3 { display: grid; grid-template-columns: repeat(3,1fr); gap: 18px; }
  @media(max-width:800px) { .grid2,.grid3 { grid-template-columns:1fr; } }

  .card { background: var(--panel); border: 1px solid var(--border);
          border-radius: 12px; padding: 22px; }
  .card h3 { font-size: 14px; color: var(--c3); margin-bottom: 16px;
             display: flex; align-items: center; gap: 8px; }

  .field { margin-bottom: 14px; }
  .field label { display: block; font-size: 12px; color: var(--sub);
                 margin-bottom: 5px; }
  .field input, .field select {
    width: 100%; padding: 9px 12px; background: var(--bg);
    border: 1px solid var(--border); border-radius: 8px;
    color: var(--text); font-size: 13px; transition: border-color .2s;
  }
  .field input:focus, .field select:focus {
    outline: none; border-color: var(--c1);
  }
  .hint { font-size: 10px; color: var(--sub); margin-top: 3px; }

  .btn { padding: 11px 24px; border-radius: 8px; border: none;
         font-size: 14px; font-weight: 600; cursor: pointer; transition: all .2s; }
  .btn-primary { background: var(--c1); color: var(--bg); }
  .btn-primary:hover { background: #33dcff; }
  .btn-full { background: var(--c5); color: var(--bg); }
  .btn-full:hover { background: #d49aff; }
  .btn-row { display: flex; gap: 10px; margin-top: 18px; flex-wrap: wrap; }

  .result-box { border-radius: 10px; padding: 18px; margin-top: 20px;
                border: 1px solid var(--border); display: none; }
  .result-box.show { display: block; }
  .result-title { font-size: 13px; color: var(--sub); margin-bottom: 10px; }
  .prob-bar-wrap { background: var(--bg); border-radius: 8px;
                   height: 12px; overflow: hidden; margin: 8px 0; }
  .prob-bar { height: 100%; border-radius: 8px; transition: width .6s ease; }
  .verdict { font-size: 20px; font-weight: 700; margin: 8px 0; }
  .risk-pill { display: inline-block; padding: 4px 14px; border-radius: 20px;
               font-size: 12px; font-weight: 600; }
  .thr-note { font-size: 11px; color: var(--sub); margin-top: 6px; }
  .stats-row { display: flex; gap: 16px; flex-wrap: wrap; margin-top: 12px; }
  .stat { background: var(--bg); border-radius: 8px; padding: 10px 14px;
          flex: 1; min-width: 100px; }
  .stat-val { font-size: 22px; font-weight: 700; }
  .stat-lbl { font-size: 11px; color: var(--sub); margin-top: 2px; }

  .pipeline-steps { display: flex; flex-direction: column; gap: 12px; margin-top: 16px; }
  .pipe-step { display: flex; gap: 14px; align-items: flex-start; }
  .pipe-num { background: var(--c1); color: var(--bg); border-radius: 50%;
              width: 26px; height: 26px; flex-shrink: 0; display: flex;
              align-items: center; justify-content: center;
              font-size: 12px; font-weight: 700; }
  .pipe-body { background: var(--bg); border-radius: 8px; padding: 12px 14px; flex: 1; }
  .pipe-body h4 { font-size: 13px; font-weight: 600; margin-bottom: 4px; }
  .pipe-body p  { font-size: 12px; color: var(--sub); }

  .decision-banner { border-radius: 12px; padding: 18px 22px; margin-top: 20px;
                     border: 2px solid; display: none; }
  .decision-banner.show { display: block; }
  .decision-banner h2 { font-size: 18px; font-weight: 700; margin-bottom: 6px; }
  .decision-banner p  { font-size: 13px; opacity: 0.85; }

  .thr-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; }
  .thr-card { background: var(--bg); border-radius: 8px; padding: 12px; }
  .thr-card .key { font-size: 11px; color: var(--sub); text-transform: uppercase; }
  .thr-card .val { font-size: 22px; font-weight: 700; color: var(--c1); margin: 4px 0; }
  .thr-card .desc { font-size: 10px; color: var(--sub); }

  .loading { display: none; color: var(--sub); font-size: 13px;
             align-items: center; gap: 8px; }
  .loading.show { display: flex; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .spinner { width: 16px; height: 16px; border: 2px solid var(--border);
             border-top-color: var(--c1); border-radius: 50%;
             animation: spin .7s linear infinite; }

  pre { background: var(--bg); padding: 14px; border-radius: 8px;
        font-size: 11px; overflow-x: auto; color: var(--c4); line-height: 1.6; }

  .lit-item { background: var(--bg); border-radius: 8px; padding: 12px 14px;
              margin-bottom: 10px; border-left: 3px solid var(--c3); }
  .lit-item .val { font-size: 18px; font-weight: 700; color: var(--c3); }
  .lit-item .desc { font-size: 12px; color: var(--sub); margin-top: 4px; }
  .lit-item .doi  { font-size: 11px; color: var(--c1); }
</style>
</head>
<body>
<header>
  <div>
    <h1>🩺 CKD Early Detection System</h1>
    <span>Diabetes → CKD Bridge Analysis · Zero-Hardcoding ML Pipeline</span>
  </div>
  <span class="badge" id="status-badge">Loading...</span>
</header>

<div class="container">
  <div class="tabs">
    <div class="tab active" onclick="switchTab('full')">Full Pipeline</div>
    <div class="tab" onclick="switchTab('diabetes')">Diabetes Check</div>
    <div class="tab" onclick="switchTab('ckd')">CKD Assessment</div>
    <div class="tab" onclick="switchTab('bridge')">DM→CKD Bridge</div>
    <div class="tab" onclick="switchTab('thresholds')">Thresholds</div>
    <div class="tab" onclick="switchTab('api')">API Docs</div>
  </div>

  <!-- ══ FULL PIPELINE ══════════════════════════════════════════════════════ -->
  <div class="panel active" id="panel-full">
    <div class="grid2">
      <div>
        <div class="card">
          <h3>🩸 Diabetes Features</h3>
          <div class="grid2" id="diab-fields"></div>
        </div>
        <div class="card" style="margin-top:16px;">
          <h3>💊 CKD Features (key)</h3>
          <div class="grid2" id="ckd-fields-full"></div>
        </div>
        <div class="btn-row">
          <button class="btn btn-full" onclick="runFull()">
            ▶ Run Full Pipeline
          </button>
        </div>
        <div class="loading" id="loading-full">
          <div class="spinner"></div> Running pipeline...
        </div>
      </div>

      <div>
        <div class="card">
          <h3>⚡ Clinical Decision</h3>
          <div class="decision-banner" id="decision-banner">
            <h2 id="decision-text"></h2>
            <p id="decision-sub"></p>
          </div>
          <div class="pipeline-steps" id="pipeline-steps"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- ══ DIABETES ══════════════════════════════════════════════════════════ -->
  <div class="panel" id="panel-diabetes">
    <div class="grid2">
      <div class="card">
        <h3>🩸 Patient Diabetes Features</h3>
        <div id="diab-form-fields"></div>
        <div class="btn-row">
          <button class="btn btn-primary" onclick="runDiabetes()">Predict Diabetes</button>
        </div>
        <div class="loading" id="loading-diab"><div class="spinner"></div> Running...</div>
      </div>
      <div class="card">
        <h3>📊 Result</h3>
        <div class="result-box" id="result-diab"></div>
      </div>
    </div>
  </div>

  <!-- ══ CKD ═══════════════════════════════════════════════════════════════ -->
  <div class="panel" id="panel-ckd">
    <div class="grid2">
      <div class="card">
        <h3>🫘 Patient CKD Features</h3>
        <div id="ckd-form-fields"></div>
        <div class="btn-row">
          <button class="btn btn-primary" onclick="runCKD()">Predict CKD</button>
        </div>
        <div class="loading" id="loading-ckd"><div class="spinner"></div> Running...</div>
      </div>
      <div class="card">
        <h3>📊 Result</h3>
        <div class="result-box" id="result-ckd"></div>
      </div>
    </div>
  </div>

  <!-- ══ BRIDGE ════════════════════════════════════════════════════════════ -->
  <div class="panel" id="panel-bridge">
    <div class="grid2">
      <div class="card">
        <h3>🌉 Bridge Features (auto-discovered)</h3>
        <p style="font-size:12px;color:var(--sub);margin-bottom:14px;">
          These features were discovered automatically by the pipeline as
          shared between the Diabetes and CKD datasets.
        </p>
        <div id="bridge-form-fields"></div>
        <div class="field">
          <label>Threshold Method</label>
          <select id="bridge-thr-mode">
            <option value="youden">Youden's J (balanced)</option>
            <option value="f1">F1-maximising</option>
            <option value="se90">Sensitivity ≥ 0.90</option>
            <option value="fbeta2">Fβ=2 (recall-weighted)</option>
            <option value="gmean">G-mean</option>
            <option value="pr_f1">PR-curve F1</option>
          </select>
        </div>
        <div class="btn-row">
          <button class="btn btn-primary" onclick="runBridge()">
            Assess CKD Escalation Risk
          </button>
        </div>
        <div class="loading" id="loading-bridge"><div class="spinner"></div> Running...</div>
      </div>
      <div class="card">
        <h3>📊 Escalation Result</h3>
        <div class="result-box" id="result-bridge"></div>
      </div>
    </div>
  </div>

  <!-- ══ THRESHOLDS ════════════════════════════════════════════════════════ -->
  <div class="panel" id="panel-thresholds">
    <div class="card" style="margin-bottom:20px;">
      <h3>🎯 Auto-Computed Optimal Thresholds</h3>
      <p style="font-size:12px;color:var(--sub);margin-bottom:18px;">
        All threshold values were computed from the model's ROC/PR curves.
        No values were hardcoded.
      </p>
      <div class="thr-grid" id="thr-grid"></div>
    </div>
    <div class="card" style="margin-bottom:20px;">
      <h3>📚 Literature Reference Thresholds</h3>
      <div id="lit-thrs"></div>
    </div>
    <div class="card">
      <h3>🌳 Decision Tree — Auto-Discovered CKD Thresholds</h3>
      <p style="font-size:12px;color:var(--sub);margin-bottom:14px;">
        Thresholds extracted from the surrogate decision tree trained on CKD data.
      </p>
      <div id="dt-thrs"></div>
    </div>
  </div>

  <!-- ══ API DOCS ══════════════════════════════════════════════════════════ -->
  <div class="panel" id="panel-api">
    <div class="grid2">
      <div class="card">
        <h3>📡 REST API Endpoints</h3>
        <div style="display:flex;flex-direction:column;gap:10px;margin-top:4px;">
          <div class="lit-item">
            <strong>GET /health</strong>
            <div class="desc">Health check — confirms models are loaded</div>
          </div>
          <div class="lit-item">
            <strong>GET /thresholds</strong>
            <div class="desc">Returns all auto-computed and literature thresholds</div>
          </div>
          <div class="lit-item">
            <strong>GET /features</strong>
            <div class="desc">Returns feature names and binary field mappings</div>
          </div>
          <div class="lit-item">
            <strong>POST /predict/diabetes</strong>
            <div class="desc">Diabetes risk from Pregnancies, Glucose, BMI, etc.</div>
          </div>
          <div class="lit-item">
            <strong>POST /predict/ckd</strong>
            <div class="desc">CKD risk from haemoglobin, serum creatinine, etc.</div>
          </div>
          <div class="lit-item">
            <strong>POST /predict/bridge</strong>
            <div class="desc">DM→CKD escalation risk from auto-discovered bridge features</div>
          </div>
          <div class="lit-item">
            <strong>POST /predict/full</strong>
            <div class="desc">Complete pipeline: diabetes → bridge escalation → CKD</div>
          </div>
        </div>
      </div>
      <div class="card">
        <h3>💻 Example cURL</h3>
        <pre id="curl-example">Loading feature names...</pre>
        <h3 style="margin-top:20px;">Example Python</h3>
        <pre id="python-example">Loading...</pre>
      </div>
    </div>
  </div>
</div>

<script>
const BASE = "";
let FEATURES = null;

// ── field definitions ────────────────────────────────────────────────────
const DIAB_FIELDS = [
  {id:"Pregnancies",    label:"Pregnancies",            hint:"Count (0–17)",       type:"number", min:0,  max:20},
  {id:"Glucose",        label:"Glucose (mg/dL)",         hint:"Fasting plasma glucose", type:"number", min:50, max:300},
  {id:"BloodPressure",  label:"Blood Pressure (mmHg)",  hint:"Diastolic BP",       type:"number", min:40, max:150},
  {id:"SkinThickness",  label:"Skin Thickness (mm)",    hint:"Triceps skinfold",   type:"number", min:5,  max:100},
  {id:"Insulin",        label:"Insulin (μU/mL)",         hint:"2-hour serum",       type:"number", min:10, max:900},
  {id:"BMI",            label:"BMI (kg/m²)",             hint:"Body mass index",    type:"number", min:15, max:70},
  {id:"DiabetesPedigreeFunction", label:"Pedigree Function", hint:"Genetic score",  type:"number", min:0,  max:3, step:0.001},
  {id:"Age",            label:"Age (years)",             hint:"Patient age",        type:"number", min:18, max:100},
];

const CKD_FIELDS_KEY = [
  {id:"age",  label:"Age",               hint:"years",     type:"number", min:2,  max:100},
  {id:"bp",   label:"Blood Pressure",    hint:"mmHg",      type:"number", min:50, max:180},
  {id:"bgr",  label:"Blood Glucose",     hint:"mg/dL",     type:"number", min:50, max:500},
  {id:"bu",   label:"Blood Urea",        hint:"mg/dL",     type:"number", min:5,  max:200},
  {id:"sc",   label:"Serum Creatinine",  hint:"mg/dL",     type:"number", min:0.5,max:15,  step:0.1},
  {id:"hemo", label:"Haemoglobin",       hint:"g/dL",      type:"number", min:5,  max:18,  step:0.1},
  {id:"pcv",  label:"Packed Cell Volume",hint:"%",         type:"number", min:15, max:55},
  {id:"sod",  label:"Sodium",            hint:"mEq/L",     type:"number", min:110,max:160},
  {id:"htn",  label:"Hypertension",      hint:"yes/no",    type:"select", options:["yes","no"]},
  {id:"dm",   label:"Diabetes Mellitus", hint:"yes/no",    type:"select", options:["yes","no"]},
];

function makeField(f) {
  const d = document.createElement("div");
  d.className = "field";
  const lbl = `<label for="${f.id}">${f.label}</label>`;
  let input;
  if (f.type === "select") {
    input = `<select id="${f.id}">` +
             f.options.map(o => `<option value="${o}">${o}</option>`).join("") +
             `</select>`;
  } else {
    input = `<input type="number" id="${f.id}" 
             min="${f.min||0}" max="${f.max||9999}" 
             step="${f.step||1}" placeholder="${f.hint||''}">`;
  }
  d.innerHTML = lbl + input + `<div class="hint">${f.hint||''}</div>`;
  return d;
}

function populateFields(containerId, fields) {
  const c = document.getElementById(containerId);
  if (!c) return;
  c.innerHTML = "";
  fields.forEach(f => c.appendChild(makeField(f)));
}

function getFieldValues(fields) {
  const d = {};
  fields.forEach(f => {
    const el = document.getElementById(f.id);
    if (!el) return;
    const v = el.value.trim();
    if (v === "") return;
    d[f.id] = f.type === "select" ? v : parseFloat(v);
  });
  return d;
}

// ── result rendering ─────────────────────────────────────────────────────
function renderResult(containerId, res) {
  const box = document.getElementById(containerId);
  if (!box) return;
  box.className = "result-box show";
  const p = res.probability;
  box.innerHTML = `
    <div class="result-title">Risk Probability</div>
    <div class="prob-bar-wrap">
      <div class="prob-bar" style="width:${p}%;background:${res.risk_color}"></div>
    </div>
    <div class="verdict" style="color:${res.risk_color}">${res.verdict}</div>
    <span class="risk-pill" style="background:${res.risk_color}22;color:${res.risk_color}">
      ${res.risk_level} RISK
    </span>
    <div class="stats-row">
      <div class="stat">
        <div class="stat-val" style="color:${res.risk_color}">${p.toFixed(1)}%</div>
        <div class="stat-lbl">Probability</div>
      </div>
      <div class="stat">
        <div class="stat-val">${(res.threshold_used*100).toFixed(1)}%</div>
        <div class="stat-lbl">Decision threshold</div>
      </div>
    </div>
    <div class="thr-note">Threshold: ${res.threshold_method}</div>
    ${res.bridge_features_used ? 
      `<div class="thr-note">Bridge features: ${res.bridge_features_used.join(", ")}</div>` : ""}
  `;
}

function showLoading(id, show) {
  const el = document.getElementById(id);
  if (el) el.className = show ? "loading show" : "loading";
}

// ── API calls ─────────────────────────────────────────────────────────────
async function post(endpoint, data) {
  const r = await fetch(endpoint, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(data)
  });
  return r.json();
}

async function runDiabetes() {
  showLoading("loading-diab", true);
  try {
    const data = getFieldValues(DIAB_FIELDS);
    const res  = await post("/predict/diabetes", data);
    if (res.status === "ok") renderResult("result-diab", res.result);
    else alert("Error: " + res.message);
  } finally { showLoading("loading-diab", false); }
}

async function runCKD() {
  showLoading("loading-ckd", true);
  try {
    const data = getFieldValues(CKD_FIELDS_KEY);
    const res  = await post("/predict/ckd", data);
    if (res.status === "ok") renderResult("result-ckd", res.result);
    else alert("Error: " + res.message);
  } finally { showLoading("loading-ckd", false); }
}

async function runBridge() {
  showLoading("loading-bridge", true);
  try {
    const data = getFieldValues(DIAB_FIELDS.concat(CKD_FIELDS_KEY));
    data.threshold_mode = document.getElementById("bridge-thr-mode").value;
    const res = await post("/predict/bridge", data);
    if (res.status === "ok") renderResult("result-bridge", res.result);
    else alert("Error: " + res.message);
  } finally { showLoading("loading-bridge", false); }
}

async function runFull() {
  showLoading("loading-full", true);
  const banner  = document.getElementById("decision-banner");
  const stepsEl = document.getElementById("pipeline-steps");
  banner.className  = "decision-banner";
  stepsEl.innerHTML = "";
  try {
    const data = getFieldValues(DIAB_FIELDS.concat(CKD_FIELDS_KEY));
    const res  = await post("/predict/full", data);
    if (res.status !== "ok") { alert("Error: " + res.message); return; }

    const cd = res.clinical_decision;
    banner.className = "decision-banner show";
    banner.style.borderColor  = cd.color;
    banner.style.background   = cd.color + "18";
    document.getElementById("decision-text").style.color = cd.color;
    document.getElementById("decision-text").textContent = cd.verdict;
    document.getElementById("decision-sub").textContent  =
      `Diabetes: ${cd.diabetes_probability.toFixed(1)}%  ·  ` +
      `Bridge: ${cd.bridge_probability.toFixed(1)}%  ·  ` +
      `CKD: ${cd.ckd_probability.toFixed(1)}%`;

    const steps = [
      {num:1, title:"Diabetes Assessment",
       prob: cd.diabetes_probability, res: res.pipeline.step1_diabetes},
      {num:2, title:"DM→CKD Bridge Escalation",
       prob: cd.bridge_probability,   res: res.pipeline.step2_bridge_dm_ckd},
      {num:3, title:"CKD Direct Assessment",
       prob: cd.ckd_probability,      res: res.pipeline.step3_ckd},
    ];
    steps.forEach(s => {
      const div = document.createElement("div");
      div.className = "pipe-step";
      div.innerHTML = `
        <div class="pipe-num">${s.num}</div>
        <div class="pipe-body">
          <h4>${s.title}</h4>
          <div class="prob-bar-wrap" style="height:8px;margin:6px 0;">
            <div class="prob-bar"
                 style="width:${s.prob}%;background:${s.res.risk_color}"></div>
          </div>
          <p>${s.res.verdict}  —  ${s.prob.toFixed(1)}%  
             (${s.res.risk_level} RISK)</p>
        </div>`;
      stepsEl.appendChild(div);
    });
  } finally { showLoading("loading-full", false); }
}

// ── thresholds tab ────────────────────────────────────────────────────────
async function loadThresholds() {
  try {
    const res = await fetch("/thresholds").then(r => r.json());

    const grid = document.getElementById("thr-grid");
    grid.innerHTML = "";
    const methodNames = {
      youden:"Youden's J", f1:"F1-max", se90:"Se ≥ 0.90",
      fbeta2:"Fβ=2", gmean:"G-mean", pr_f1:"PR-F1"
    };
    Object.entries(res.optimal_thresholds).forEach(([k,v]) => {
      grid.innerHTML += `
        <div class="thr-card">
          <div class="key">${methodNames[k]||k}</div>
          <div class="val">${(v*100).toFixed(1)}%</div>
          <div class="desc">threshold = ${v.toFixed(3)}</div>
        </div>`;
    });

    const lit = document.getElementById("lit-thrs");
    lit.innerHTML = "";
    (res.literature_thresholds || []).forEach(l => {
      lit.innerHTML += `
        <div class="lit-item">
          <div class="val">${l.value} ${l.unit}</div>
          <strong>${l.label}</strong>
          <div class="desc">${l.description}</div>
          <div class="doi">DOI: ${l.doi}</div>
        </div>`;
    });

    const dt = document.getElementById("dt-thrs");
    dt.innerHTML = "";
    Object.entries(res.dt_thresholds || {}).forEach(([feat, info]) => {
      dt.innerHTML += `
        <div class="thr-card" style="margin-bottom:8px;">
          <div class="key">${feat}</div>
          <div class="val" style="font-size:18px;">${info.threshold}</div>
          <div class="desc">n=${info.n} · gini=${info.gini}</div>
        </div>`;
    });
  } catch(e) { console.error("Threshold load error:", e); }
}

// ── API docs examples ─────────────────────────────────────────────────────
async function loadApiDocs() {
  try {
    const f = await fetch("/features").then(r => r.json());
    const sample = {};
    f.diabetes_features.slice(0,3).forEach(k => sample[k] = 0);
    document.getElementById("curl-example").textContent =
`curl -X POST http://localhost:5000/predict/full \\
  -H "Content-Type: application/json" \\
  -d '${JSON.stringify({...sample, hemo:12.5, sc:1.2, htn:"yes", dm:"yes"}, null, 2)}'`;

    document.getElementById("python-example").textContent =
`import requests, json

payload = {
    # Diabetes features
    "Glucose": 148, "BloodPressure": 72, "BMI": 33.6,
    "Age": 50, "Pregnancies": 6,
    # CKD features
    "hemo": 11.2, "sc": 1.8, "bp": 90,
    "htn": "yes", "dm": "yes",
}
r = requests.post("http://localhost:5000/predict/full", json=payload)
print(json.dumps(r.json(), indent=2))`;
  } catch(e) {}
}

// ── bridge fields (from /features) ───────────────────────────────────────
async function loadBridgeFields() {
  try {
    const f = await fetch("/features").then(r => r.json());
    FEATURES = f;
    const c = document.getElementById("bridge-form-fields");
    if (!c) return;
    c.innerHTML = "";
    f.bridge_features_diabetes.forEach(bf => {
      const match = DIAB_FIELDS.find(d => d.id === bf);
      const def = match || {id:bf, label:bf, hint:"Bridge feature", type:"number"};
      c.appendChild(makeField(def));
    });
  } catch(e) {}
}

// ── health check ──────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const h = await fetch("/health").then(r => r.json());
    const b = document.getElementById("status-badge");
    b.textContent = "✅ Online";
    b.style.background  = "#6BCB7722";
    b.style.borderColor = "#6BCB77";
    b.style.color       = "#6BCB77";
  } catch(e) {
    document.getElementById("status-badge").textContent = "⚠ Offline";
  }
}

// ── tab switching ─────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
  event.target.classList.add("active");
  document.getElementById("panel-"+name).classList.add("active");
  if (name === "thresholds") loadThresholds();
  if (name === "api")        loadApiDocs();
}

// ── init ─────────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  populateFields("diab-fields",       DIAB_FIELDS);
  populateFields("diab-form-fields",  DIAB_FIELDS);
  populateFields("ckd-fields-full",   CKD_FIELDS_KEY);
  populateFields("ckd-form-fields",   CKD_FIELDS_KEY);
  checkHealth();
  loadBridgeFields();
});
</script>
</body>
</html>"""

@app.route("/")
def index():
    return render_template_string(HTML)

if __name__ == "__main__":
    import socket
    host = "0.0.0.0"
    port = 5000
    # Auto-detect Colab public URL
    try:
        from google.colab import output
        print("Running in Google Colab")
        print(f"Starting server on port {port} ...")
        output.serve_kernel_port_as_window(port, anchor_text="Open CKD App")
    except ImportError:
        pass
    print(f"App running at: http://localhost:{port}")
    print("Endpoints:")
    print("  GET  /           → Web UI")
    print("  POST /predict/full    → Full pipeline")
    print("  POST /predict/diabetes")
    print("  POST /predict/ckd")
    print("  POST /predict/bridge")
    print("  GET  /thresholds → All auto-thresholds")
    print("  GET  /features   → Feature names")
    app.run(host=host, port=port, debug=False, use_reloader=False)
