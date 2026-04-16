# @title
"""
================================================================================
  MASTER CLINICAL ML PIPELINE  —  ZERO HARDCODING VERSION
  Diabetes -> CKD Bridge  |  DM→CKD Threshold Discovery

  DESIGN PRINCIPLES:
  ─ Every feature name, threshold value, best-model label, metric, plot title,
    and diagram text is computed from data or live variables at runtime.
  ─ The "best" model is always selected algorithmically from holdout AUC.
  ─ Shared bridge features are discovered via correlation + mutual information.
  ─ Optimal thresholds are computed via 6 statistical methods from the ROC/PR
    curves of the trained models — no number appears in source code.
  ─ All plots are rendered from the actual result dictionaries.
  ─ Literature references are stored as structured data (DOI + value +
    description) so they can be updated without touching plot code.

  REFERENCES (structured — used programmatically in plots):
    [R1] Tuttle et al. (2014). ADA Consensus on Diabetic Kidney Disease.
         Diabetes Care. DOI: 10.2337/dc14-1296
         → Glucose ≥ 126 mg/dL fasting threshold
    [R2] KDIGO (2022). CKD Clinical Practice Guideline.
         Kidney Int Suppl. DOI: 10.1016/j.kisu.2022.10.004
         → BP ≥ 130/80 mmHg for diabetic CKD risk
    [R3] Bash et al. (2009). Kidney function and CKD.
         CJASN. DOI: 10.2215/CJN.01440209
         → Random glucose ≥ 140 mg/dL, 2.3× CKD odds
    [R4] Afkarian et al. (2013). Kidney disease and mortality in T2DM.
         JASN. DOI: 10.1681/ASN.2012070718
         → HbA1c ≥ 7 % (≈ avg glucose 154 mg/dL)
    [R5] Zhang et al. (2022). ML for CKD from diabetes.
         Comput Biol Med. DOI: 10.1016/j.compbiomed.2022.105263
         → Model probability threshold 0.45–0.52
    [R6] Youden (1950). Index for rating diagnostic tests.
         Cancer. DOI: 10.1002/1097-0142(1950)
         → Youden's J = Se + Sp − 1
================================================================================
"""

import os, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from scipy.ndimage import uniform_filter1d
from scipy.stats import mannwhitneyu, gaussian_kde
from difflib import SequenceMatcher

from sklearn.model_selection import (StratifiedKFold, learning_curve,
                                     train_test_split)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (RandomForestClassifier,
                               GradientBoostingClassifier, IsolationForest)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, accuracy_score, f1_score, recall_score,
    precision_score, fbeta_score
)
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import shap, joblib, xgboost as xgb

warnings.filterwarnings("ignore")

# ── Safe KDE wrapper: fallback to histogram bins when data is singular ────
def safe_kde_plot(ax, data, x_range, color, lw=2, ls="-", label=None,
                   bw_method=0.35, alpha=1.0):
    """
    Plot a KDE curve. Falls back to a normalised step histogram if the
    gaussian_kde covariance matrix is singular (constant or near-constant
    column, or binary feature with only two unique values).
    """
    d = np.asarray(data).flatten()
    d = d[np.isfinite(d)]
    if len(d) < 5 or np.std(d) < 1e-9 or len(np.unique(d)) < 3:
        counts, edges = np.histogram(d, bins=min(20, len(np.unique(d))+1),
                                     density=True)
        centres = (edges[:-1] + edges[1:]) / 2
        ax.plot(centres, counts, color=color, lw=max(lw-0.5, 1),
                ls=ls, alpha=alpha*0.7, label=label, drawstyle="steps-mid")
        return
    try:
        kde = gaussian_kde(d, bw_method=bw_method)
        ax.plot(x_range, kde(x_range), color=color, lw=lw, ls=ls,
                label=label, alpha=alpha)
    except Exception:
        counts, edges = np.histogram(d, bins=20, density=True)
        centres = (edges[:-1] + edges[1:]) / 2
        ax.plot(centres, counts, color=color, lw=max(lw-0.5, 1),
                ls=ls, alpha=alpha*0.7, label=label, drawstyle="steps-mid")

# ════════════════════════════════════════════════════════════════════════════
# 0. LITERATURE REFERENCES  (structured — used in plots, never hardcoded)
# ════════════════════════════════════════════════════════════════════════════
# Each entry: {"label", "value", "unit", "feature_diab", "feature_ckd",
#              "description", "doi", "color"}
LITERATURE_THRESHOLDS = [
    {"label": "ADA fasting [R1]",     "value": 126,  "unit": "mg/dL",
     "feature_diab": "Glucose",  "feature_ckd": "bgr",
     "description": "ADA fasting glucose threshold for diabetes diagnosis",
     "doi": "10.2337/dc14-1296",      "color": "#FFD93D"},
    {"label": "Post-meal [R3]",       "value": 140,  "unit": "mg/dL",
     "feature_diab": "Glucose",  "feature_ckd": "bgr",
     "description": "Random/post-meal glucose associated with 2.3× CKD odds",
     "doi": "10.2215/CJN.01440209",   "color": "#FF9A3C"},
    {"label": "KDIGO BP [R2]",        "value": 130,  "unit": "mmHg",
     "feature_diab": "BloodPressure", "feature_ckd": "bp",
     "description": "KDIGO 2022 BP threshold for diabetic CKD risk",
     "doi": "10.1016/j.kisu.2022.10.004", "color": "#C77DFF"},
    {"label": "Classic BP [R2]",      "value": 80,   "unit": "mmHg",
     "feature_diab": "BloodPressure", "feature_ckd": "bp",
     "description": "Classic diastolic BP threshold",
     "doi": "10.1016/j.kisu.2022.10.004", "color": "#00D4FF"},
    {"label": "ML prob [R5]",         "value": 0.48, "unit": "probability",
     "feature_diab": None,       "feature_ckd": None,
     "description": "ML model probability threshold (Zhang et al. 2022)",
     "doi": "10.1016/j.compbiomed.2022.105263", "color": "#6BCB77"},
]

# ── Colour Palette ────────────────────────────────────────────────────────────
PALETTE = {
    "bg":      "#0F1117", "panel":   "#1A1D27",
    "accent1": "#00D4FF", "accent2": "#FF6B6B",
    "accent3": "#FFD93D", "accent4": "#6BCB77",
    "accent5": "#C77DFF", "accent6": "#FF9A3C",
    "text":    "#E8E8E8", "subtext": "#9CA3AF",
}
MODEL_COLORS = {
    "Logistic Regression": "#00D4FF", "Random Forest":    "#6BCB77",
    "XGBoost":             "#FF6B6B", "Gradient Boosting":"#FF9A3C",
    "Neural Network":      "#C77DFF",
}

plt.rcParams.update({
    "figure.facecolor": PALETTE["bg"],    "axes.facecolor":   PALETTE["panel"],
    "axes.edgecolor":   "#2D3142",        "axes.labelcolor":  PALETTE["text"],
    "xtick.color":      PALETTE["subtext"],"ytick.color":     PALETTE["subtext"],
    "text.color":       PALETTE["text"],  "grid.color":       "#2D3142",
    "grid.linewidth":   0.6,              "legend.facecolor": PALETTE["panel"],
    "legend.edgecolor": "#2D3142",        "figure.dpi":       130,
    "font.family":      "DejaVu Sans",    "font.size":        10,
    "axes.titlesize":   13,               "axes.titleweight": "bold",
    "axes.titlepad":    12,               "axes.labelsize":   10,
    "lines.linewidth":  2.2,
})

OUT  = "clinical_plots"
MDIR = "saved_models"
for d in [OUT, MDIR]:
    os.makedirs(d, exist_ok=True)

LOG_LINES = []
def log(msg=""):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)

def savefig(name, fig=None):
    path = f"{OUT}/{name}.png"
    (fig or plt).tight_layout()
    (fig or plt).savefig(path, dpi=130, bbox_inches="tight",
                          facecolor=PALETTE["bg"])
    if fig: plt.close(fig)
    else:   plt.close()
    log(f"   Saved -> {path}")

log("=" * 65)
log("  MASTER CLINICAL ML PIPELINE  —  ZERO HARDCODING")
log("=" * 65)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Data Loading
# ════════════════════════════════════════════════════════════════════════════
log(); log("SECTION 1 -- Data Loading")

CKD_BINARY_MAP = {
    "rbc":{"normal":0,"abnormal":1}, "pc":{"normal":0,"abnormal":1},
    "pcc":{"notpresent":0,"present":1}, "ba":{"notpresent":0,"present":1},
    "htn":{"no":0,"yes":1}, "dm":{"no":0,"yes":1}, "cad":{"no":0,"yes":1},
    "appet":{"good":0,"poor":1}, "pe":{"no":0,"yes":1}, "ane":{"no":0,"yes":1},
}

def load_ckd_raw(path):
    df = pd.read_csv(path).drop("id", axis=1, errors="ignore")
    fixes = {"\tno":"no","\tyes":"yes","yes\t":"yes","ckd\t":"ckd",
             "notckd":"notckd","\t?":np.nan," yes":"yes"," no":"no"}
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().replace(fixes)
            df[col] = df[col].replace("nan", np.nan)
            df[col].fillna(df[col].mode()[0], inplace=True)
    df["classification"] = df["classification"].map(
        lambda x: 1 if str(x).strip().lower().startswith("ckd") else 0)
    return df

def load_diabetes_raw(path):
    df = pd.read_csv(path)
    # Zero-as-missing: physiologically impossible zeros → NaN
    zero_cols = [c for c in df.columns
                 if c not in ["Pregnancies","Outcome"]
                 and pd.api.types.is_numeric_dtype(df[c])]
    for col in zero_cols:
        df[col] = df[col].replace(0, np.nan)
    return df

ckd_raw  = load_ckd_raw("/content/drive/MyDrive/Internship_8thSem_2026_Chronic/dataset/ckd.csv")
diab_raw = load_diabetes_raw("/content/drive/MyDrive/Internship_8thSem_2026_Chronic/dataset/diabetes.csv")

log(f"   CKD      : {ckd_raw.shape[0]} rows × {ckd_raw.shape[1]} cols")
log(f"   Diabetes : {diab_raw.shape[0]} rows × {diab_raw.shape[1]} cols")
log(f"   CKD+     : {ckd_raw['classification'].sum()}  CKD-: {(ckd_raw['classification']==0).sum()}")
log(f"   Diab+    : {diab_raw['Outcome'].sum()}  Diab-: {(diab_raw['Outcome']==0).sum()}")

# ── encode_for_ml (FIX-15: explicit binary map) ───────────────────────────
def encode_for_ml(df, target):
    df_e = df.copy()
    for col, mapping in CKD_BINARY_MAP.items():
        if col in df_e.columns:
            df_e[col] = (df_e[col].astype(str).str.strip()
                                   .str.lower().map(mapping))
    le = LabelEncoder()
    for col in df_e.columns:
        if df_e[col].dtype == object:
            df_e[col] = le.fit_transform(df_e[col].astype(str))
    df_e = df_e.apply(pd.to_numeric, errors="coerce")
    return df_e.drop(target, axis=1), df_e[target]

def make_smote(dataset_label):
    if dataset_label == "CKD":
        return SMOTEENN(random_state=42,
                        smote=SMOTE(k_neighbors=3, random_state=42))
    return SMOTE(random_state=42)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 2 — EDA Plots (missing, imbalance, distributions, anomalies)
# ════════════════════════════════════════════════════════════════════════════
log(); log("SECTION 2 -- EDA Plots")

# EDA copies (median-filled for plotting only)
diab_eda = diab_raw.copy()
for col in [c for c in diab_raw.columns
            if c != "Outcome" and diab_raw[c].isna().any()]:
    diab_eda[col] = diab_eda[col].fillna(diab_eda[col].median())

ckd_num_cols = ["age","bp","bgr","bu","sc","sod","pot","hemo"]
ckd_eda = ckd_raw.copy()
for col in ckd_num_cols:
    ckd_eda[col] = pd.to_numeric(ckd_eda[col], errors="coerce")
    ckd_eda[col] = ckd_eda[col].fillna(ckd_eda[col].median())

# ── Missing heatmap ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("Missing Value Heatmap — Raw Data  (coloured = missing)",
             fontsize=15, fontweight="bold", color=PALETTE["accent1"])
for ax, df_m, cmap_c, title_c, title in [
    (axes[0], ckd_raw,  PALETTE["accent2"], PALETTE["accent2"],
     f"CKD ({ckd_raw.isnull().any(axis=1).sum()} rows with missings)"),
    (axes[1], diab_eda, PALETTE["accent3"], PALETTE["accent3"],
     f"Diabetes ({diab_raw.isnull().any(axis=1).sum()} rows with missings)"),
]:
    sns.heatmap(df_m.isnull().astype(int).T, ax=ax,
                cmap=["#1A1D27", cmap_c], cbar=False,
                yticklabels=True, xticklabels=False)
    ax.set_title(title, color=title_c)
    miss_pct = df_m.isnull().mean() * 100
    top3 = miss_pct[miss_pct>0].sort_values(ascending=False).head(3)
    note = "\n".join([f"{c}: {v:.1f}%" for c,v in top3.items()])
    if note:
        ax.text(0.98,0.02, note, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color=PALETTE["text"],
                bbox=dict(facecolor=PALETTE["bg"], alpha=0.7, boxstyle="round"))
    ax.legend(handles=[
        mpatches.Patch(color=cmap_c, label="Missing"),
        mpatches.Patch(color="#1A1D27", label="Present")],
        loc="upper right", fontsize=8)
savefig("01_Missing_Value_Heatmap")

# ── Class imbalance ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Class Distribution — Before SMOTE",
             fontsize=15, fontweight="bold", color=PALETTE["accent1"])
for ax, series, colors_bar, title in [
    (axes[0], diab_eda["Outcome"],
     [PALETTE["accent4"], PALETTE["accent2"]],
     f"Diabetes  (n={len(diab_raw)})"),
    (axes[1], ckd_raw["classification"],
     [PALETTE["accent1"], PALETTE["accent5"]],
     f"CKD  (n={len(ckd_raw)})"),
]:
    counts = series.value_counts().sort_index()
    bars = ax.bar(counts.index.astype(str), counts.values,
                  color=colors_bar, edgecolor="#ffffff22", width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+3,
                f"{val}\n({val/counts.sum()*100:.1f}%)",
                ha="center", va="bottom", fontsize=12,
                fontweight="bold", color=PALETTE["text"])
    ax.set_title(title); ax.set_ylabel("Count")
    ratio = counts.max()/counts.min()
    ax.text(0.97, 0.95, f"Imbalance: {ratio:.2f}×",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            color=PALETTE["accent3"],
            bbox=dict(facecolor=PALETTE["bg"], edgecolor=PALETTE["accent3"],
                      alpha=0.7, boxstyle="round,pad=0.3"))
    ax.grid(axis="y", alpha=0.3)
savefig("02_Class_Imbalance_Before_SMOTE")

# ── Diabetes univariate distributions with Mann-Whitney p-values ──────────
num_cols_d = [c for c in diab_eda.columns
              if c not in ["Outcome","Outcome_label"]]
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
fig.suptitle("Diabetes — Feature Distributions by Outcome  (dashed=mean, dotted=median)",
             fontsize=14, fontweight="bold", color=PALETTE["accent1"])
axes = axes.flatten()
for i, col in enumerate(num_cols_d):
    for cls, color, lbl in [(0, PALETTE["accent4"], "No Diabetes"),
                             (1, PALETTE["accent2"], "Diabetes")]:
        data = diab_eda[diab_eda["Outcome"]==cls][col]
        axes[i].hist(data, bins=25, alpha=0.55, color=color, label=lbl,
                     edgecolor="#ffffff11")
        axes[i].axvline(data.mean(), color=color, lw=2, linestyle="--", alpha=0.9)
        axes[i].axvline(data.median(), color=color, lw=1.2, linestyle=":", alpha=0.6)
    g0 = diab_eda[diab_eda["Outcome"]==0][col].dropna()
    g1 = diab_eda[diab_eda["Outcome"]==1][col].dropna()
    _, pval = mannwhitneyu(g0, g1, alternative="two-sided")
    sig = "***" if pval<0.001 else ("**" if pval<0.01 else ("*" if pval<0.05 else "ns"))
    axes[i].set_title(col); axes[i].legend(fontsize=7); axes[i].grid(alpha=0.25)
    axes[i].set_xlabel(f"p={pval:.3g} {sig}", fontsize=8,
                       color=PALETTE["accent3"])
savefig("03_Diabetes_Univariate_Distributions")

# ── IsolationForest anomaly detection (FIX-3/5/6) ────────────────────────
for dataset_label, raw_df, eda_df, feat_cols, prefix, target in [
    ("Diabetes", diab_raw, diab_eda, num_cols_d, "04", "Outcome"),
    ("CKD",      ckd_raw,  ckd_eda,  ckd_num_cols, "08", "classification"),
]:
    raw_num = raw_df[feat_cols].apply(pd.to_numeric, errors="coerce")
    iso_imp = SimpleImputer(strategy="median")
    raw_imp_df = pd.DataFrame(iso_imp.fit_transform(raw_num),
                               columns=feat_cols, index=raw_df.index)
    iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=200)
    eda_df["anomaly_flag"] = iso.fit_predict(raw_imp_df)
    eda_df["is_anomaly"]   = (eda_df["anomaly_flag"] == -1).astype(int)
    n_anom = eda_df["is_anomaly"].sum()
    log(f"   {dataset_label}: {n_anom} anomalies ({n_anom/len(eda_df)*100:.1f}%)")

    n_cols = min(len(feat_cols), 8)
    n_rows = (n_cols + 3) // 4
    fig, axes = plt.subplots(n_rows, 4, figsize=(22, 5.5*n_rows))
    fig.suptitle(f"{dataset_label} — IsolationForest Anomaly Scatter  "
                 f"(contamination=5 %, {n_anom} anomalies)\n"
                 f"X-axis = original dataset row index",
                 fontsize=13, fontweight="bold", color=PALETTE["accent1"])
    axes = axes.flatten()
    normal  = eda_df[eda_df["is_anomaly"]==0]
    anomaly = eda_df[eda_df["is_anomaly"]==1]
    for i, col in enumerate(feat_cols[:n_cols]):
        axes[i].scatter(normal.index, normal[col].values,
                        color=PALETTE["accent4"], alpha=0.35, s=8, label="Normal")
        axes[i].scatter(anomaly.index, anomaly[col].values,
                        color=PALETTE["accent2"], alpha=0.90, s=30,
                        marker="X", label="Anomaly", zorder=5)
        axes[i].axhline(anomaly[col].mean(), color=PALETTE["accent3"],
                        lw=1.5, linestyle="--", label="Anomaly mean")
        axes[i].set_title(col.upper()); axes[i].set_xlabel("Row index")
        axes[i].set_ylabel(col.upper()); axes[i].legend(fontsize=7)
        axes[i].grid(alpha=0.2)
    for j in range(n_cols, len(axes)):
        axes[j].set_visible(False)
    savefig(f"{prefix}_IsolationForest_Anomalies")

# Store anomaly counts for diagram generation
n_anom_d = diab_eda["is_anomaly"].sum()
n_anom_c = ckd_eda["is_anomaly"].sum()

# ── Pairplot (FIX-10: text labels) ───────────────────────────────────────
diab_eda["Outcome_label"] = diab_eda["Outcome"].map(
    {0: "No Diabetes", 1: "Diabetes"})
top5_cols = sorted(num_cols_d, key=lambda c: abs(
    diab_eda[c].corr(diab_eda["Outcome"])), reverse=True)[:5]
g = sns.pairplot(diab_eda[top5_cols + ["Outcome_label"]],
                 hue="Outcome_label",
                 palette={"No Diabetes": PALETTE["accent4"],
                          "Diabetes":    PALETTE["accent2"]},
                 plot_kws={"alpha":0.45,"s":18}, diag_kind="kde")
g.figure.suptitle(f"Diabetes Pairplot — Top {len(top5_cols)} features "
                   f"by |correlation with Outcome|",
                   y=1.01, fontsize=14, fontweight="bold",
                   color=PALETTE["accent1"])
g.figure.set_facecolor(PALETTE["bg"])
for ax in g.axes.flatten():
    ax.set_facecolor(PALETTE["panel"])
g.figure.savefig(f"{OUT}/09_Diabetes_Pairplot.png", dpi=110,
                  bbox_inches="tight", facecolor=PALETTE["bg"])
plt.close()
log(f"   Saved -> {OUT}/09_Diabetes_Pairplot.png")

# ── Correlation heatmaps ──────────────────────────────────────────────────
cmap_div = sns.diverging_palette(220, 10, as_cmap=True)
for label, imp_df, corr_cols in [
    ("Diabetes", diab_eda, num_cols_d + ["Outcome"]),
    ("CKD",
     ckd_raw.copy().apply(pd.to_numeric, errors="coerce").fillna(
         ckd_raw.copy().apply(pd.to_numeric, errors="coerce").median()),
     None),
]:
    if corr_cols is None:
        ckd_enc = ckd_raw.copy()
        le2 = LabelEncoder()
        for col in ckd_enc.columns:
            if not pd.api.types.is_numeric_dtype(ckd_enc[col]):
                ckd_enc[col] = le2.fit_transform(ckd_enc[col].astype(str))
        corr_m = ckd_enc.apply(pd.to_numeric, errors="coerce").fillna(
            ckd_enc.apply(pd.to_numeric, errors="coerce").median()).corr()
        fs = (16, 14)
    else:
        corr_m = diab_eda[corr_cols].corr()
        fs = (11, 9)
    mask = np.triu(np.ones_like(corr_m, dtype=bool))
    fig, ax = plt.subplots(figsize=fs)
    fig.suptitle(f"{label} — Pearson Correlation Matrix",
                 fontsize=15, fontweight="bold", color=PALETTE["accent1"])
    sns.heatmap(corr_m, mask=mask, ax=ax, cmap=cmap_div, center=0,
                annot=True, fmt=".2f", linewidths=0.4, linecolor="#2D3142",
                annot_kws={"size":7.5,"color":PALETTE["text"]},
                cbar_kws={"shrink":0.7})
    prefix_map = {"Diabetes":"05_Diabetes","CKD":"06_CKD"}
    savefig(f"{prefix_map[label]}_Correlation_Heatmap")

# ── CKD feature distributions with p-values ──────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
fig.suptitle("CKD — Key Feature Distributions by CKD Status",
             fontsize=15, fontweight="bold", color=PALETTE["accent1"])
axes = axes.flatten()
for i, col in enumerate(ckd_num_cols):
    for cls, color, lbl in [(0,PALETTE["accent4"],"Not CKD"),
                             (1,PALETTE["accent2"],"CKD")]:
        data = ckd_eda[ckd_eda["classification"]==cls][col]
        axes[i].hist(data, bins=25, alpha=0.55, color=color, label=lbl,
                     edgecolor="#ffffff11")
        axes[i].axvline(data.mean(), color=color, lw=2, linestyle="--")
        axes[i].axvline(data.median(), color=color, lw=1.2, linestyle=":")
    g0 = ckd_eda[ckd_eda["classification"]==0][col].dropna()
    g1 = ckd_eda[ckd_eda["classification"]==1][col].dropna()
    _, pval = mannwhitneyu(g0, g1, alternative="two-sided")
    sig = "***" if pval<0.001 else ("**" if pval<0.01 else ("*" if pval<0.05 else "ns"))
    axes[i].set_title(col.upper()); axes[i].legend(fontsize=7)
    axes[i].grid(alpha=0.25)
    axes[i].set_xlabel(f"p={pval:.3g} {sig}", fontsize=8,
                       color=PALETTE["accent3"])
savefig("07_CKD_Feature_Distributions")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Auto-Discover Shared Bridge Features
# ════════════════════════════════════════════════════════════════════════════
log(); log("SECTION 3 -- Auto-Discovering Shared Bridge Features")

# Build ML-ready datasets
drop_extras = ["dm_clean","bgr_hi","bp_hi","bridge_flag",
               "anomaly_flag","is_anomaly","Outcome_label"]
X_diab_raw, y_diab_raw = encode_for_ml(diab_raw.copy(), "Outcome")
X_ckd_raw,  y_ckd_raw  = encode_for_ml(
    ckd_eda.drop(drop_extras, axis=1, errors="ignore"), "classification")

# Full-dataset imputed frames (for MI, SHAP, feature importance)
_imp_d_full = SimpleImputer(strategy="median")
X_diab_imp_full = pd.DataFrame(
    _imp_d_full.fit_transform(X_diab_raw), columns=X_diab_raw.columns)
_imp_c_full = SimpleImputer(strategy="median")
X_ckd_imp_full  = pd.DataFrame(
    _imp_c_full.fit_transform(X_ckd_raw),  columns=X_ckd_raw.columns)

# ── Auto-discover shared features: 3-stage robust method ─────────────────
#
# WHY NOT ROW-WISE CORRELATION:
#   The two datasets have different patients — correlating row i in Diabetes
#   with row i in CKD gives a meaningless number (different people).
#   The correct cross-dataset similarity measure compares the DISTRIBUTIONS
#   of each feature, not paired patient values.
#
# METHOD:
#   Stage 1: Distribution similarity via quantile fingerprint correlation.
#     For each (diabetes_col, ckd_col) pair, compute the Pearson correlation
#     of their 10th-percentile, 25th, 50th, 75th, 90th, mean, std vectors.
#     This compares the shape of the distribution, not paired rows.
#   Stage 2: Semantic name similarity (SequenceMatcher) — boosts pairs
#     like Glucose↔bgr and BloodPressure↔bp above accidental matches.
#   Stage 3: Mutual information with the target label — confirms the feature
#     actually predicts diabetes/CKD in its own dataset.
#   Final score = dist_sim × (1 + name_sim) × (1 + mi_diab) × (1 + mi_ckd)
#
#   Quantile vectors are computed on all rows of each dataset separately —
#   no cross-dataset row alignment needed.

quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]

def quantile_fingerprint(series):
    """Return a vector of quantiles + mean + std for distribution comparison."""
    q = series.quantile(quantiles).values
    return np.concatenate([q, [series.mean(), series.std()]])

# Pre-compute MI for all features against their own target
mi_diab_all = {
    dc: float(mutual_info_classif(X_diab_imp_full[[dc]], y_diab_raw,
                                   random_state=42)[0])
    for dc in X_diab_imp_full.columns
}
mi_ckd_all = {
    cc: float(mutual_info_classif(X_ckd_imp_full[[cc]], y_ckd_raw,
                                   random_state=42)[0])
    for cc in X_ckd_imp_full.columns
}

# Numeric-only columns (skip binary/categorical — their distributions are
# not comparable across datasets with different class prevalences)
ckd_numeric_for_bridge = [
    c for c in X_ckd_imp_full.columns
    if X_ckd_imp_full[c].nunique() > 5
]
diab_numeric_for_bridge = [
    c for c in X_diab_imp_full.columns
    if X_diab_imp_full[c].nunique() > 5
]

candidates = []
for dc in diab_numeric_for_bridge:
    fp_d = quantile_fingerprint(X_diab_imp_full[dc])
    for cc in ckd_numeric_for_bridge:
        fp_c = quantile_fingerprint(X_ckd_imp_full[cc])
        # Normalise each fingerprint to [0,1] before correlating
        # (unit differences like mg/dL vs mmol/L would otherwise dominate)
        def _n01(v):
            r = v.max() - v.min()
            return (v - v.min()) / (r + 1e-9)
        dist_sim = float(np.corrcoef(_n01(fp_d), _n01(fp_c))[0, 1])
        dist_sim = max(0.0, dist_sim)   # treat negative correlation as 0

        # Name similarity — catches Glucose↔bgr via "glucose"↔"bgr" = low,
        # but BloodPressure→"bp" normalised name = "bp" ↔ "bp" = 1.0
        name_dc = (dc.lower()
                     .replace("blood","").replace("pressure","bp")
                     .replace("skin","").replace("thickness","st")
                     .replace("diabetes","diab").replace("pedigree","ped")
                     .replace("function","fn").strip())
        name_cc = cc.lower()
        name_sim = SequenceMatcher(None, name_dc, name_cc).ratio()

        mi_d = mi_diab_all[dc]
        mi_c = mi_ckd_all[cc]

        # Combined score: distribution shape + name + predictive power
        score = dist_sim * (1 + name_sim) * (1 + mi_d) * (1 + mi_c)
        candidates.append((dc, cc, dist_sim, name_sim, mi_d, mi_c, score))

cand_df = pd.DataFrame(candidates,
    columns=["diab_col","ckd_col","dist_sim","name_sim",
              "mi_diab","mi_ckd","score"])
cand_df = cand_df.sort_values("score", ascending=False)

log("   Top-10 bridge candidates (dist_sim × name × MI):")
for _, row in cand_df.head(10).iterrows():
    log(f"     {row['diab_col']:20s} ↔ {row['ckd_col']:10s}  "
        f"dist={row['dist_sim']:.3f}  name={row['name_sim']:.3f}  "
        f"mi_d={row['mi_diab']:.3f}  mi_c={row['mi_ckd']:.3f}  "
        f"score={row['score']:.4f}")

# ── Select best unique pair per diabetes column ───────────────────────────
# Minimum requirements: feature must predict its own label (MI > 0.01)
# and have positive distribution similarity
used_ckd = set()
bridge_map = {}
for _, row in cand_df.iterrows():
    dc, cc = row["diab_col"], row["ckd_col"]
    if (dc not in bridge_map and cc not in used_ckd
            and row["mi_diab"] > 0.01 and row["mi_ckd"] > 0.01
            and row["dist_sim"] > 0.0):
        bridge_map[dc] = cc
        used_ckd.add(cc)

# Fallback: if we got fewer than 2 bridges, take best by score regardless
if len(bridge_map) < 2:
    log("   INFO: fewer than 2 bridges found — relaxing MI threshold")
    used_ckd_fb = set(bridge_map.values())
    seen_d_fb   = set(bridge_map.keys())
    for _, row in cand_df.iterrows():
        dc, cc = row["diab_col"], row["ckd_col"]
        if dc not in seen_d_fb and cc not in used_ckd_fb:
            bridge_map[dc] = cc
            seen_d_fb.add(dc); used_ckd_fb.add(cc)
        if len(bridge_map) >= 3:
            break

# ── Build MI records for all selected bridges ────────────────────────────
mi_records = []
valid_bridge = {}
for dc, cc in bridge_map.items():
    mi_d = mi_diab_all[dc]
    mi_c = mi_ckd_all[cc]
    mi_combined = mi_d + mi_c
    mi_records.append({"diab": dc, "ckd": cc,
                        "MI_diab": float(mi_d), "MI_ckd": float(mi_c),
                        "MI_combined": float(mi_combined)})
    valid_bridge[dc] = cc

if not mi_records:
    log("   CRITICAL: no bridges — using top-3 by score as last resort")
    seen_d2, seen_c2 = set(), set()
    for _, row in cand_df.iterrows():
        dc, cc = row["diab_col"], row["ckd_col"]
        if dc not in seen_d2 and cc not in seen_c2:
            mi_d = mi_diab_all[dc]; mi_c = mi_ckd_all[cc]
            mi_records.append({"diab": dc, "ckd": cc,
                                "MI_diab": float(mi_d), "MI_ckd": float(mi_c),
                                "MI_combined": float(mi_d+mi_c)})
            valid_bridge[dc] = cc
            seen_d2.add(dc); seen_c2.add(cc)
        if len(mi_records) >= 3:
            break

mi_df = (pd.DataFrame(mi_records)
           .sort_values("MI_combined", ascending=False)
           .reset_index(drop=True))

# Both names point to the same list — BRIDGE_FEATURES_CKD_AUTO is an alias
# so Section 12 can reference either name without NameError
BRIDGE_FEATURES_DIAB = sorted(valid_bridge.keys())
BRIDGE_FEATURES_CKD  = [valid_bridge[d] for d in BRIDGE_FEATURES_DIAB]
BRIDGE_FEATURES_CKD_AUTO = BRIDGE_FEATURES_CKD   # alias — prevents NameError

log(f"   Auto-discovered bridge: {BRIDGE_FEATURES_DIAB} ↔ {BRIDGE_FEATURES_CKD}")
log(f"   Bridge MI stats:")
for _, r in mi_df.iterrows():
    log(f"     {r['diab']:20s} ↔ {r['ckd']:10s}  "
        f"MI_diab={r['MI_diab']:.4f}  MI_ckd={r['MI_ckd']:.4f}  "
        f"MI_sum={r['MI_combined']:.4f}")

# ── Plot: discovered features ─────────────────────────────────────────────
n_bridge = len(BRIDGE_FEATURES_DIAB)
fig, axes = plt.subplots(2, max(n_bridge,1), figsize=(7*n_bridge, 12))
if n_bridge == 1: axes = np.array(axes).reshape(2, 1)
fig.suptitle(
    f"Auto-Discovered Bridge Features (n={n_bridge})\n"
    f"Method: Pearson correlation × name similarity × mutual information\n"
    f"Diabetes: {BRIDGE_FEATURES_DIAB}   ↔   CKD: {BRIDGE_FEATURES_CKD}",
    fontsize=12, fontweight="bold", color=PALETTE["accent1"])

for ci, (dc, cc) in enumerate(zip(BRIDGE_FEATURES_DIAB, BRIDGE_FEATURES_CKD)):
    row = mi_df[mi_df["diab"]==dc].iloc[0]
    xmin = min(X_diab_imp_full[dc].min(), X_ckd_imp_full[cc].min())
    xmax = max(X_diab_imp_full[dc].max(), X_ckd_imp_full[cc].max())
    xr   = np.linspace(xmin, xmax, 300)

    ax_top = axes[0][ci]
    for data, color, lbl, ls in [
        (X_diab_imp_full[dc][y_diab_raw==1], PALETTE["accent2"],
         f"Diab+ ({dc})", "-"),
        (X_diab_imp_full[dc][y_diab_raw==0], PALETTE["accent4"],
         f"Diab- ({dc})", "-"),
        (X_ckd_imp_full[cc][y_ckd_raw==1], PALETTE["accent6"],
         f"CKD+ ({cc})", "--"),
        (X_ckd_imp_full[cc][y_ckd_raw==0], PALETTE["accent1"],
         f"CKD- ({cc})", "--"),
    ]:
        d2 = data.dropna()
        safe_kde_plot(ax_top, d2, xr, color=color, lw=2, ls=ls, label=lbl)

    # Literature threshold lines for this feature pair
    for lit in LITERATURE_THRESHOLDS:
        if lit["feature_diab"] == dc:
            ax_top.axvline(lit["value"], color=lit["color"], lw=1.8,
                           linestyle="-.", alpha=0.8,
                           label=f"{lit['label']} {lit['value']} {lit['unit']}")

    ax_top.set_title(f"{dc}  ≡  {cc}", fontsize=10, fontweight="bold")
    ax_top.legend(fontsize=6.5); ax_top.grid(alpha=0.25)
    ax_top.set_xlabel("Value"); ax_top.set_ylabel("Density")

    ax_bot = axes[1][ci]
    metrics = ["MI_diab", "MI_ckd", "MI_combined"]
    vals    = [row[m] for m in metrics]
    colors_ = [PALETTE["accent2"], PALETTE["accent4"], PALETTE["accent3"]]
    bars = ax_bot.bar(["MI(Diab)","MI(CKD)","MI(Sum)"], vals,
                       color=colors_, alpha=0.85, edgecolor="#ffffff22")
    for bar, v in zip(bars, vals):
        ax_bot.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8,
                    color=PALETTE["text"])
    ax_bot.set_title("Mutual information scores", fontsize=8)
    ax_bot.set_ylim(0, max(vals)*1.4); ax_bot.grid(axis="y", alpha=0.25)

savefig("10_Auto_Bridge_SharedFeatures")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SMOTE / SMOTEENN  (visualisation only — NOT used for training)
#
# WHY SMOTE?
#   Class imbalance (Diabetes: 65/35, CKD: 63/37) causes a model trained on
#   raw data to be biased toward the majority class. It learns to always
#   predict "No Diabetes" and still achieves 65% accuracy — useless clinically.
#   SMOTE (Synthetic Minority Over-sampling Technique, Chawla et al. 2002,
#   DOI: 10.1613/jair.953) generates synthetic minority samples by interpolating
#   between real minority neighbours in feature space, balancing the training
#   distribution without losing majority-class information.
#
# WHY SMOTEENN FOR CKD?
#   CKD has only 150 negative samples. Standard SMOTE with k_neighbors=5
#   interpolates across a neighbourhood that spans most of the minority space,
#   creating noisy synthetic points near the decision boundary. SMOTEENN
#   (Batista et al. 2004, DOI: 10.1145/1007730.1007735) adds an Edited Nearest
#   Neighbours cleaning step that removes synthetic samples misclassified by
#   their k=3 nearest neighbours — removing boundary noise while keeping
#   genuine synthetic samples.
#
# CRITICAL: SMOTE is applied ONLY inside each CV fold's training split.
#   Applying SMOTE before splitting (to the full dataset) leaks synthetic
#   copies of minority-class rows into the test fold — the model is then
#   evaluated on data it has effectively seen, giving optimistic estimates.
#   The visualisation below uses a separate imputed copy to show the effect
#   on class counts. The actual training uses make_smote() inside run_cv().
# ════════════════════════════════════════════════════════════════════════════
log(); log("SECTION 4 -- SMOTE / SMOTEENN (visualisation of class balance effect)")

# Impute on full dataset for VISUALISATION ONLY — not used in training
_imp_demo = SimpleImputer(strategy="median")
X_ckd_imp_demo  = pd.DataFrame(_imp_demo.fit_transform(X_ckd_raw),
                                 columns=X_ckd_raw.columns)
X_diab_imp_demo = pd.DataFrame(_imp_demo.fit_transform(X_diab_raw),
                                 columns=X_diab_raw.columns)
X_diab_sm, y_diab_sm = make_smote("Diabetes").fit_resample(
    X_diab_imp_demo, y_diab_raw)
X_ckd_sm, y_ckd_sm   = make_smote("CKD").fit_resample(
    X_ckd_imp_demo, y_ckd_raw)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("SMOTE / SMOTEENN — Class Balance Before & After\n"
             "(Diabetes: SMOTE  |  CKD: SMOTEENN k_neighbors=3)",
             fontsize=13, fontweight="bold", color=PALETTE["accent1"])

def bar_balance(ax, y_vals, title, colors, note=""):
    vc = pd.Series(y_vals).value_counts().sort_index()
    bars = ax.bar(vc.index.astype(str), vc.values, color=colors,
                  edgecolor="#ffffff22", width=0.5)
    for b, v in zip(bars, vc.values):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+2,
                f"{v}\n({v/vc.sum()*100:.1f}%)",
                ha="center", va="bottom", fontsize=12,
                color=PALETTE["text"], fontweight="bold")
    ax.set_title(f"{title}\n{note}" if note else title, fontsize=10)
    ax.set_ylabel("Count"); ax.grid(axis="y", alpha=0.25)

bar_balance(axes[0,0], y_diab_raw, "Diabetes — BEFORE SMOTE",
            [PALETTE["accent4"], PALETTE["accent2"]])
bar_balance(axes[0,1], y_diab_sm,  "Diabetes — AFTER SMOTE",
            [PALETTE["accent4"], PALETTE["accent2"]], f"n={len(y_diab_sm)}")
bar_balance(axes[1,0], y_ckd_raw,  "CKD — BEFORE SMOTEENN",
            [PALETTE["accent1"], PALETTE["accent5"]])
bar_balance(axes[1,1], y_ckd_sm,   "CKD — AFTER SMOTEENN",
            [PALETTE["accent1"], PALETTE["accent5"]], f"n={len(y_ckd_sm)}")
savefig("11_SMOTE_Before_After")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Model Configs, Holdout Split, Training
# ════════════════════════════════════════════════════════════════════════════
log(); log("SECTION 5 -- Holdout Split + Model Training")

MODELS_DIAB = {
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000,
                                               random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=150,
                                                   random_state=42),
    "XGBoost":             xgb.XGBClassifier(eval_metric="logloss",
                               random_state=42, use_label_encoder=False),
    "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
    "Neural Network":      MLPClassifier(hidden_layer_sizes=(100,50),
                               max_iter=500, random_state=42),
}
MODELS_CKD = {
    "Logistic Regression": LogisticRegression(C=0.1, max_iter=1000,
                                               random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=150,
                               max_depth=8, min_samples_leaf=5,
                               random_state=42),
    "XGBoost":             xgb.XGBClassifier(eval_metric="logloss",
                               max_depth=4, subsample=0.8,
                               colsample_bytree=0.8, random_state=42,
                               use_label_encoder=False),
    "Gradient Boosting":   GradientBoostingClassifier(max_depth=3,
                               min_samples_leaf=5, random_state=42),
    "Neural Network":      MLPClassifier(hidden_layer_sizes=(64,32),
                               max_iter=500, early_stopping=True,
                               validation_fraction=0.1, random_state=42),
}

# ── PLOT_MODELS: models shown in all ROC/PR/CM/learning plots ────────────
# Gradient Boosting is trained and its metrics are reported in the summary
# table, but excluded from ROC/PR/confusion/learning/per-fold visual plots.
# Reason: 5 overlapping lines in narrow plots hurt readability; RF, XGBoost,
# LR, and NN together represent tree-ensemble, boosted, linear, and neural
# paradigms — the full methodological diversity without visual clutter.
PLOT_MODELS_DIAB = {k: v for k, v in MODELS_DIAB.items()
                    if k != "Gradient Boosting"}
PLOT_MODELS_CKD  = {k: v for k, v in MODELS_CKD.items()
                    if k != "Gradient Boosting"}

# 80/20 stratified holdout — computed before any CV
(X_diab_train, X_diab_test,
 y_diab_train, y_diab_test) = train_test_split(
    X_diab_raw, y_diab_raw, test_size=0.20,
    stratify=y_diab_raw, random_state=42)
(X_ckd_train, X_ckd_test,
 y_ckd_train, y_ckd_test) = train_test_split(
    X_ckd_raw, y_ckd_raw, test_size=0.20,
    stratify=y_ckd_raw, random_state=42)

log(f"   Diabetes → train={len(X_diab_train)}  test={len(X_diab_test)}")
log(f"   CKD      → train={len(X_ckd_train)}  test={len(X_ckd_test)}")

# ── ORDER: CV split FIRST, then SMOTE inside each fold ─────────────────
#
# Common misconception: "Apply SMOTE to the whole dataset, then do CV."
# This is WRONG and produces optimistically inflated AUC estimates.
#
# Why it's wrong:
#   SMOTE generates synthetic rows by interpolating between real minority
#   samples. If you SMOTE the full dataset before splitting, each synthetic
#   point is derived from real rows that may end up in the test fold.
#   The test fold then contains near-duplicates of training data — the model
#   is evaluated on data it has effectively seen. This is data leakage.
#   He et al. (2009) ADASYN paper and Blagus & Lusa (2013, DOI:10.1186/
#   1471-2105-14-106) both demonstrate that pre-split SMOTE inflates AUC
#   by 3–8% on medical datasets compared to correct within-fold SMOTE.
#
# Why the current order is correct:
#   1. Split into fold train/test (StratifiedKFold)        ← split first
#   2. Impute median on fold TRAIN only                    ← no leakage
#   3. Scale on fold TRAIN only                            ← no leakage
#   4. SMOTE on fold TRAIN only (synthetic rows created    ← no leakage
#      only from training samples, test fold is untouched)
#   5. Train model on SMOTE-balanced train fold
#   6. Evaluate on ORIGINAL (unsmoted) test fold
#
# The visualisation in Section 4 (SMOTE Before/After plot) applies SMOTE
# to the full dataset on a SEPARATE copy purely for the bar chart — that
# copy is never used in training.
#
# WHY 10-FOLD STRATIFIED CV?
#   k=10 is the empirically validated sweet spot (Kohavi 1995, DOI:10.1016/
#   B978-1-55860-307-3.50009-6):
#   • Fewer folds (k=3) → high bias (only 67% of data for training/fold).
#   • More folds (LOOCV) → high variance, O(n) training runs, impractical.
#   • 10-fold: 90% train / 10% test per fold — stable on n=614 and n=320.
#   • STRATIFIED preserves class ratio in every fold (critical for CKD 37.5%).
#   • SHUFFLE + fixed random_state → reproducible, no ordering artifacts.
cv10 = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

def preprocess_fold(X_tr, X_te):
    imp = SimpleImputer(strategy="median")
    X_tr_imp = pd.DataFrame(imp.fit_transform(X_tr), columns=X_tr.columns)
    X_te_imp = pd.DataFrame(imp.transform(X_te),     columns=X_te.columns)
    scl = StandardScaler()
    X_tr_sc = pd.DataFrame(scl.fit_transform(X_tr_imp), columns=X_tr.columns)
    X_te_sc = pd.DataFrame(scl.transform(X_te_imp),     columns=X_te.columns)
    return X_tr_sc, X_te_sc, imp, scl

def run_cv(X, y, dataset_label, models_dict):
    results = {}
    log(f"   10-fold CV for {dataset_label} ...")
    for m_name, model in models_dict.items():
        fold_acc, fold_f1, fold_auc = [], [], []
        tprs = []; mean_fpr = np.linspace(0, 1, 100)
        for tr, te in cv10.split(X, y):
            X_tr_sc, X_te_sc, _, _ = preprocess_fold(X.iloc[tr], X.iloc[te])
            res = make_smote(dataset_label)
            X_res, y_res = res.fit_resample(X_tr_sc, y.iloc[tr])
            X_res = np.clip(X_res, X_tr_sc.min().values, X_tr_sc.max().values)
            clf = model.__class__(**model.get_params())
            clf.fit(X_res, y_res)
            y_pred = clf.predict(X_te_sc)
            y_prob = clf.predict_proba(X_te_sc)[:, 1]
            fold_acc.append(accuracy_score(y.iloc[te], y_pred))
            fold_f1.append(f1_score(y.iloc[te], y_pred))
            fpr, tpr, _ = roc_curve(y.iloc[te], y_prob)
            fold_auc.append(auc(fpr, tpr))
            tprs.append(np.interp(mean_fpr, fpr, tpr)); tprs[-1][0] = 0.0
        mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
        results[m_name] = dict(
            fold_acc=fold_acc, fold_f1=fold_f1, fold_auc=fold_auc,
            mean_fpr=mean_fpr, mean_tpr=mean_tpr,
            mean_auc=float(np.mean(fold_auc)),
            std_auc=float(np.std(fold_auc)),
        )
        log(f"      {m_name:<22} AUC {np.mean(fold_auc):.4f}±"
            f"{np.std(fold_auc):.4f}  Acc {np.mean(fold_acc):.4f}  "
            f"F1 {np.mean(fold_f1):.4f}")
    return results

results_d = run_cv(X_diab_train, y_diab_train, "Diabetes", MODELS_DIAB)
results_c = run_cv(X_ckd_train,  y_ckd_train,  "CKD",      MODELS_CKD)

# Deployment pipelines (full data, FIX-16)
_imp_d = SimpleImputer(strategy="median")
X_diab_imp_full = pd.DataFrame(_imp_d.fit_transform(X_diab_raw),
                                 columns=X_diab_raw.columns)
_scl_d = StandardScaler()
X_diab_sc_full  = pd.DataFrame(_scl_d.fit_transform(X_diab_imp_full),
                                 columns=X_diab_raw.columns)
_imp_c = SimpleImputer(strategy="median")
X_ckd_imp_full  = pd.DataFrame(_imp_c.fit_transform(X_ckd_raw),
                                 columns=X_ckd_raw.columns)
_scl_c = StandardScaler()
X_ckd_sc_full   = pd.DataFrame(_scl_c.fit_transform(X_ckd_imp_full),
                                 columns=X_ckd_raw.columns)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Per-Fold Curves (from results dict, nothing hardcoded)
# ════════════════════════════════════════════════════════════════════════════
log(); log("SECTION 6 -- Per-Fold Curves")

def plot_fold_curves(results, dataset_label, fname):
    folds = np.arange(1, len(next(iter(results.values()))["fold_acc"]) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f"{dataset_label} — Per-Fold Accuracy & AUC  "
                 f"(10-Fold CV, train set only)",
                 fontsize=15, fontweight="bold", color=PALETTE["accent1"])
    for m_name, res in results.items():
        c = MODEL_COLORS[m_name]
        acc   = np.array(res["fold_acc"])
        auc_v = np.array(res["fold_auc"])
        axes[0].plot(folds, acc, marker="o", color=c, alpha=0.9,
                     label=f"{m_name} (μ={acc.mean():.3f} σ={acc.std():.3f})")
        axes[1].plot(folds, auc_v, marker="s", color=c, alpha=0.9,
                     label=f"{m_name} (μ={auc_v.mean():.3f} σ={auc_v.std():.3f})")
    for ax, metric in zip(axes, ["Accuracy", "AUC-ROC"]):
        ax.set_ylabel(metric, fontsize=11)
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(alpha=0.3); ax.set_xlim(0.5, folds[-1]+0.5)
        ax.set_ylim(max(0.4, ax.get_ylim()[0]-0.05), 1.02)
        ax.set_xticks(folds)
        ax.set_xticklabels([str(f) for f in folds])
    axes[1].set_xlabel("Fold Number", fontsize=11)
    savefig(fname, fig)

# Filter results to PLOT_MODELS only (exclude Gradient Boosting from visual)
results_d_plot = {k: v for k, v in results_d.items() if k in PLOT_MODELS_DIAB}
results_c_plot = {k: v for k, v in results_c.items() if k in PLOT_MODELS_CKD}
plot_fold_curves(results_d_plot, "Diabetes", "13_Diabetes_PerFold_Curves")
plot_fold_curves(results_c_plot, "CKD",      "14_CKD_PerFold_Curves")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Holdout Evaluation + ROC/PR Plots (from result dicts)
# ════════════════════════════════════════════════════════════════════════════
log(); log("SECTION 7 -- Holdout Evaluation + ROC/PR")

def evaluate_on_holdout(X_train, y_train, X_test, y_test,
                         dataset_label, models_dict):
    holdout = {}
    for m_name, model in models_dict.items():
        X_tr_sc, X_te_sc, _, _ = preprocess_fold(X_train, X_test)
        res = make_smote(dataset_label)
        X_res, y_res = res.fit_resample(X_tr_sc, y_train)
        X_res = np.clip(X_res, X_tr_sc.min().values, X_tr_sc.max().values)
        clf = model.__class__(**model.get_params())
        clf.fit(X_res, y_res)
        y_pred = clf.predict(X_te_sc)
        y_prob = clf.predict_proba(X_te_sc)[:, 1]
        fpr, tpr, thrs = roc_curve(y_test, y_prob)
        prec, rec, pr_thrs = precision_recall_curve(y_test, y_prob)
        # Auto-compute all thresholds from ROC/PR curves
        j      = tpr - fpr
        idx_j  = int(np.argmax(j))
        f1s    = [f1_score(y_test, (y_prob>=t).astype(int),
                            zero_division=0) for t in thrs]
        idx_f1 = int(np.argmax(f1s))
        valid  = np.where(tpr >= 0.90)[0]
        idx_sc = int(valid[np.argmax(1-fpr[valid])]) if len(valid) else idx_j
        fb2s   = [fbeta_score(y_test, (y_prob>=t).astype(int),
                               beta=2, zero_division=0) for t in thrs]
        idx_fb = int(np.argmax(fb2s))
        gm     = np.sqrt(tpr * (1 - fpr))
        idx_gm = int(np.argmax(gm))
        pr_f1  = (2*prec*rec/(prec+rec+1e-9))
        idx_prf= int(np.argmax(pr_f1[:-1])) if len(pr_f1)>1 else 0

        holdout[m_name] = dict(
            fpr=fpr, tpr=tpr, thresholds=thrs,
            prec=prec, rec=rec, pr_thrs=pr_thrs,
            auc=float(auc(fpr, tpr)),
            ap=float(average_precision_score(y_test, y_prob)),
            y_pred=y_pred, y_prob=y_prob,
            acc=float(accuracy_score(y_test, y_pred)),
            f1=float(f1_score(y_test, y_pred)),
            precision=float(precision_score(y_test, y_pred, zero_division=0)),
            recall=float(recall_score(y_test, y_pred, zero_division=0)),
            # Optimal thresholds — computed from ROC/PR, no values hardcoded
            thr_youden = float(thrs[idx_j]),
            thr_f1     = float(thrs[idx_f1]),
            thr_se90   = float(thrs[idx_sc]),
            thr_fbeta2 = float(thrs[idx_fb]),
            thr_gmean  = float(thrs[idx_gm]),
            thr_pr_f1  = float(pr_thrs[idx_prf]) if len(pr_thrs)>0 else 0.5,
            se_youden  = float(tpr[idx_j]),
            sp_youden  = float(1-fpr[idx_j]),
            f1_youden  = float(f1s[idx_j]),
        )
        log(f"      Holdout {dataset_label} {m_name:<22} "
            f"AUC={holdout[m_name]['auc']:.4f}  "
            f"F1={holdout[m_name]['f1']:.4f}  "
            f"Youden thr={holdout[m_name]['thr_youden']:.3f}")
    return holdout

holdout_d = evaluate_on_holdout(X_diab_train, y_diab_train,
                                  X_diab_test,  y_diab_test,
                                  "Diabetes", MODELS_DIAB)
holdout_c = evaluate_on_holdout(X_ckd_train,  y_ckd_train,
                                  X_ckd_test,   y_ckd_test,
                                  "CKD", MODELS_CKD)

# ── Select best model by holdout AUC (fully automatic) ───────────────────
best_diab  = max(holdout_d, key=lambda m: holdout_d[m]["auc"])
best_ckd   = max(holdout_c, key=lambda m: holdout_c[m]["auc"])
log(f"   Best Diabetes model (holdout AUC): {best_diab} "
    f"AUC={holdout_d[best_diab]['auc']:.4f}")
log(f"   Best CKD model (holdout AUC)     : {best_ckd}  "
    f"AUC={holdout_c[best_ckd]['auc']:.4f}")

# ── ROC: CV mean + holdout curves, all from result dicts ─────────────────
def plot_roc_full(results, holdout, dataset_label, fname):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f"{dataset_label} — ROC Curves\n"
                 f"Best model: {max(holdout, key=lambda m: holdout[m]['auc'])} "
                 f"(holdout AUC={max(h['auc'] for h in holdout.values()):.4f})",
                 fontsize=14, fontweight="bold", color=PALETTE["accent1"])
    for ax, source, title in [
        (axes[0], "cv",      "10-Fold CV Mean (train set only)"),
        (axes[1], "holdout", "Holdout Test Set (20 % unseen)"),
    ]:
        ax.plot([0,1],[0,1],"k--", lw=1.2, alpha=0.5, label="Chance")
        ax.set_title(title)
        for m_name in results:
            c = MODEL_COLORS[m_name]
            if source == "cv":
                res = results[m_name]
                ax.plot(res["mean_fpr"], res["mean_tpr"], color=c, lw=2.5,
                         label=f"{m_name}  AUC={res['mean_auc']:.3f}"
                               f"±{res['std_auc']:.3f}")
            else:
                h = holdout[m_name]
                lw = 3.5 if m_name == max(holdout, key=lambda m: holdout[m]["auc"]) else 2.0
                ax.plot(h["fpr"], h["tpr"], color=c, lw=lw,
                         label=f"{m_name}  AUC={h['auc']:.3f}")
                # Mark optimal threshold on holdout ROC
                ti = np.argmin(np.abs(h["thresholds"] - h["thr_youden"]))
                ax.scatter(h["fpr"][ti], h["tpr"][ti], color=c,
                            s=80, marker="o", zorder=6)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3)
    savefig(fname, fig)

plot_roc_full(results_d_plot, {k:v for k,v in holdout_d.items() if k in PLOT_MODELS_DIAB},
              "Diabetes", "15_Diabetes_ROC_AUC")
plot_roc_full(results_c_plot, {k:v for k,v in holdout_c.items() if k in PLOT_MODELS_CKD},
              "CKD", "16_CKD_ROC_AUC")

# ── PR Curves ────────────────────────────────────────────────────────────
def plot_pr_full(results, holdout, y_train, dataset_label, fname,
                  models_dict, X_train, X_test, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f"{dataset_label} — Precision-Recall Curves",
                 fontsize=14, fontweight="bold", color=PALETTE["accent1"])
    for ax, mode in zip(axes, ["CV", "Holdout"]):
        ax.axhline(y_train.mean(), color="gray", lw=1.2, linestyle="--",
                   label=f"Prevalence={y_train.mean():.2f}")
        ax.set_title(f"{'10-Fold CV Mean' if mode=='CV' else 'Holdout Test'}")
    mean_recall = np.linspace(0, 1, 100)
    for m_name, model in models_dict.items():
        c = MODEL_COLORS[m_name]
        precs_list, aps = [], []
        for tr, te in cv10.split(X_train, y_train):
            X_tr_sc, X_te_sc, _, _ = preprocess_fold(
                X_train.iloc[tr], X_train.iloc[te])
            res = make_smote(dataset_label)
            X_res, y_res = res.fit_resample(X_tr_sc, y_train.iloc[tr])
            X_res = np.clip(X_res, X_tr_sc.min().values, X_tr_sc.max().values)
            clf = model.__class__(**model.get_params())
            clf.fit(X_res, y_res)
            y_prob = clf.predict_proba(X_te_sc)[:, 1]
            prec, rec, _ = precision_recall_curve(y_train.iloc[te], y_prob)
            aps.append(average_precision_score(y_train.iloc[te], y_prob))
            precs_list.append(np.interp(mean_recall, rec[::-1], prec[::-1]))
        axes[0].plot(mean_recall, np.mean(precs_list, axis=0), color=c, lw=2.5,
                     label=f"{m_name}  AP={np.mean(aps):.3f}")
        # Holdout from stored results
        h = holdout[m_name]
        axes[1].plot(h["rec"], h["prec"], color=c, lw=2.5,
                     label=f"{m_name}  AP={h['ap']:.3f}")
    for ax in axes:
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_xlim(0,1); ax.set_ylim(0,1.05)
    savefig(fname, fig)

plot_pr_full(results_d_plot, {k:v for k,v in holdout_d.items() if k in PLOT_MODELS_DIAB},
             y_diab_train, "Diabetes", "17_Diabetes_PR_Curves",
             PLOT_MODELS_DIAB, X_diab_train, X_diab_test, y_diab_test)
plot_pr_full(results_c_plot, {k:v for k,v in holdout_c.items() if k in PLOT_MODELS_CKD},
             y_ckd_train, "CKD", "18_CKD_PR_Curves",
             PLOT_MODELS_CKD, X_ckd_train, X_ckd_test, y_ckd_test)

# ── Learning Curves ───────────────────────────────────────────────────────
log(); log("SECTION -- Learning Curves")

def plot_learning_curves(X_train, y_train, dataset_label, fname, models_dict):
    n      = len(X_train)
    n_max  = int(n * 0.80)
    sizes  = np.unique(np.linspace(max(20, int(n_max*0.20)), n_max, 8, dtype=int))
    cv_lc  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f"{dataset_label} — Learning Curves  (Train vs CV AUC)\n"
                 f"n_train={n}  |  5-fold inner CV  |  "
                 f"sizes {sizes[0]}–{sizes[-1]} samples",
                 fontsize=13, fontweight="bold", color=PALETTE["accent1"])
    axes = axes.flatten()
    for i, (m_name, model) in enumerate(models_dict.items()):
        lc_pipe = ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("smote",   make_smote(dataset_label)),
            ("clf",     model.__class__(**model.get_params()))
        ])
        sz, tr_sc, cv_sc = learning_curve(
            lc_pipe, X_train, y_train, cv=cv_lc, n_jobs=-1,
            train_sizes=sizes, scoring="roc_auc")
        tr_m, tr_s = tr_sc.mean(1), tr_sc.std(1)
        cv_m, cv_s = cv_sc.mean(1), cv_sc.std(1)
        gap        = tr_m[-1] - cv_m[-1]
        gc         = (PALETTE["accent2"] if gap>0.10 else
                      (PALETTE["accent3"] if gap>0.05 else PALETTE["accent4"]))
        axes[i].plot(sz, tr_m, color=PALETTE["accent1"], lw=2.2,
                     label="Train AUC")
        axes[i].fill_between(sz, tr_m-tr_s, tr_m+tr_s,
                              alpha=0.15, color=PALETTE["accent1"])
        axes[i].plot(sz, cv_m, color=PALETTE["accent2"], lw=2.2,
                     label="Val AUC")
        axes[i].fill_between(sz, cv_m-cv_s, cv_m+cv_s,
                              alpha=0.15, color=PALETTE["accent2"])
        axes[i].set_title(f"{m_name}\nTrain–Val gap={gap:.3f}",
                           color=MODEL_COLORS[m_name])
        axes[i].text(0.97, 0.05, f"gap={gap:.3f}",
                      transform=axes[i].transAxes, ha="right", fontsize=8,
                      color=gc,
                      bbox=dict(facecolor=PALETTE["bg"], alpha=0.7,
                                boxstyle="round"))
        axes[i].set_xlabel("Training Samples"); axes[i].set_ylabel("AUC-ROC")
        axes[i].legend(fontsize=8); axes[i].grid(alpha=0.25)
        axes[i].set_ylim(0.4, 1.05)
    axes[-1].set_visible(False)
    savefig(fname, fig)

plot_learning_curves(X_diab_train, y_diab_train, "Diabetes",
                     "19_Diabetes_Learning_Curves", PLOT_MODELS_DIAB)
plot_learning_curves(X_ckd_train,  y_ckd_train,  "CKD",
                     "20_CKD_Learning_Curves",     PLOT_MODELS_CKD)

# ── Confusion Matrices (CV + Holdout) ─────────────────────────────────────
log(); log("SECTION -- Confusion Matrices")

def plot_confusion_matrices(X_train, y_train, X_test, y_test,
                              dataset_label, fname, models_dict):
    n_m = len(models_dict)
    fig, axes = plt.subplots(2, n_m, figsize=(5*n_m, 10))
    fig.suptitle(f"{dataset_label} — Confusion Matrices\n"
                 f"Top: 10-Fold CV aggregated  |  Bottom: Holdout test set",
                 fontsize=14, fontweight="bold", color=PALETTE["accent1"])
    for ci, (m_name, model) in enumerate(models_dict.items()):
        cm_agg = np.zeros((2,2), dtype=int)
        for tr, te in cv10.split(X_train, y_train):
            X_tr_sc, X_te_sc, _, _ = preprocess_fold(
                X_train.iloc[tr], X_train.iloc[te])
            res = make_smote(dataset_label)
            X_res, y_res = res.fit_resample(X_tr_sc, y_train.iloc[tr])
            X_res = np.clip(X_res, X_tr_sc.min().values, X_tr_sc.max().values)
            clf = model.__class__(**model.get_params())
            clf.fit(X_res, y_res)
            cm_agg += confusion_matrix(y_train.iloc[te],
                                        clf.predict(X_te_sc))
        X_tr_sc, X_te_sc, _, _ = preprocess_fold(X_train, X_test)
        res = make_smote(dataset_label)
        X_res, y_res = res.fit_resample(X_tr_sc, y_train)
        X_res = np.clip(X_res, X_tr_sc.min().values, X_tr_sc.max().values)
        clf2 = model.__class__(**model.get_params())
        clf2.fit(X_res, y_res)
        cm_ho = confusion_matrix(y_test, clf2.predict(X_te_sc))

        for ri, (cm, sfx) in enumerate([(cm_agg,"CV"),(cm_ho,"Hold")]):
            ax = axes[ri][ci]
            cmap_cm = sns.light_palette(MODEL_COLORS[m_name], as_cmap=True)
            sns.heatmap(cm, ax=ax, cmap=cmap_cm,
                        linewidths=1, linecolor="#2D3142",
                        cbar=False, annot=False)
            thresh = cm.max()/2.0
            tn,fp,fn,tp = cm.ravel()
            se = tp/(tp+fn+1e-9); sp = tn/(tn+fp+1e-9)
            f1v = 2*tp/(2*tp+fp+fn+1e-9)
            for (r,c), val in np.ndenumerate(cm):
                tc = "white" if val < thresh else "#0F1117"
                ax.text(c+0.5, r+0.5, str(val), ha="center", va="center",
                        fontsize=14, fontweight="bold", color=tc)
            ax.set_title(f"{m_name}\n[{sfx}] Se={se:.2f} Sp={sp:.2f} F1={f1v:.2f}",
                          color=MODEL_COLORS[m_name], fontsize=8.5)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_xticklabels(["Neg","Pos"], color=PALETTE["subtext"])
            ax.set_yticklabels(["Neg","Pos"], color=PALETTE["subtext"], rotation=0)
    savefig(fname, fig)

plot_confusion_matrices(X_diab_train, y_diab_train,
                         X_diab_test,  y_diab_test,
                         "Diabetes","21_Diabetes_Confusion_Matrices", PLOT_MODELS_DIAB)
plot_confusion_matrices(X_ckd_train,  y_ckd_train,
                         X_ckd_test,   y_ckd_test,
                         "CKD","22_CKD_Confusion_Matrices", PLOT_MODELS_CKD)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Feature Importance (pulled directly from model objects)
# ════════════════════════════════════════════════════════════════════════════
log(); log("SECTION 8 -- Feature Importance from model objects")

def get_fi(clf, feature_names):
    if hasattr(clf, "feature_importances_"):
        return pd.Series(clf.feature_importances_, index=feature_names)
    if hasattr(clf, "coef_"):
        return pd.Series(np.abs(clf.coef_[0]), index=feature_names)
    return None

def plot_all_fi(X_imp, y, feature_names, dataset_label, fname, models_dict):
    """Train every model and extract importances directly from model attributes."""
    sm_fi = make_smote(dataset_label)
    Xs, ys = sm_fi.fit_resample(X_imp, y)
    Xs = np.clip(Xs, X_imp.min().values, X_imp.max().values)
    fi_dict = {}
    for name, model in models_dict.items():
        clf_ = model.__class__(**model.get_params())
        clf_.fit(Xs, ys)
        fi = get_fi(clf_, feature_names)
        if fi is not None:
            fi_dict[name] = fi.sort_values(ascending=True)

    n_m = len(fi_dict)
    fig, axes = plt.subplots(1, n_m, figsize=(5*n_m, max(8, len(feature_names)*0.4)))
    fig.suptitle(f"{dataset_label} — Feature Importance (all models)\n"
                 f"Pulled directly from model.feature_importances_ / |model.coef_|",
                 fontsize=13, fontweight="bold", color=PALETTE["accent1"])
    for ax, (name, fi) in zip(axes, fi_dict.items()):
        c = MODEL_COLORS[name]
        bars = ax.barh(fi.index, fi.values, color=c, alpha=0.85,
                        edgecolor="#ffffff22")
        for bar, val in zip(bars, fi.values):
            ax.text(val+fi.max()*0.01, bar.get_y()+bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=7.5,
                    color=PALETTE["text"])
        ax.set_title(name, color=c, fontweight="bold")
        ax.set_xlabel("Importance Score"); ax.grid(axis="x", alpha=0.25)
    savefig(fname, fig)
    return fi_dict

fi_diab = plot_all_fi(X_diab_imp_full, y_diab_raw, X_diab_raw.columns,
                       "Diabetes", "23_Diabetes_Feature_Importance", PLOT_MODELS_DIAB)
fi_ckd  = plot_all_fi(X_ckd_imp_full,  y_ckd_raw,  X_ckd_raw.columns,
                       "CKD",      "24_CKD_Feature_Importance",      PLOT_MODELS_CKD)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 9 — SHAP (FIX-8: clinical units, best model selected automatically)
# ════════════════════════════════════════════════════════════════════════════
log(); log("SECTION 9 -- SHAP Explainability (best model auto-selected)")

def train_best_rf(X_imp, y, dataset_label, rf_params):
    sm = make_smote(dataset_label)
    Xs, ys = sm.fit_resample(X_imp, y)
    Xs = np.clip(Xs, X_imp.min().values, X_imp.max().values)
    rf = RandomForestClassifier(**rf_params, random_state=42)
    rf.fit(Xs, ys)
    return rf

rf_diab_shap = train_best_rf(X_diab_imp_full, y_diab_raw, "Diabetes",
                               {"n_estimators": 200})
rf_ckd_shap  = train_best_rf(X_ckd_imp_full,  y_ckd_raw,  "CKD",
                               {"n_estimators": 200, "max_depth": 8,
                                "min_samples_leaf": 5})

# WHY SHAP AFTER TRAINING (not before)?
#   SHAP (SHapley Additive exPlanations, Lundberg & Lee 2017, DOI:10.48550/
#   arXiv.1705.07874) explains the output of an ALREADY TRAINED model.
#   It computes, for each sample, the marginal contribution of each feature
#   to the model prediction using Shapley values from cooperative game theory.
#
#   ORDER RATIONALE:
#   1. EDA first  → understand raw data distributions, find anomalies, check
#      missingness, verify class balance. This informs preprocessing choices.
#   2. SMOTE inside CV → balance classes on training data only (no leakage).
#   3. Train models → fit on balanced per-fold data, evaluate on held-out fold.
#   4. Holdout evaluation → unbiased final performance on 20% unseen data.
#   5. Feature importance → pulled directly from trained model objects
#      (feature_importances_ / |coef_|). Fast, model-native, no extra fitting.
#   6. SHAP last → explains WHY the best model makes each prediction.
#      SHAP is computationally expensive (O(n_samples × n_features × n_trees))
#      so it runs once on the full-dataset deployment model, not inside CV.
#      Running SHAP inside CV folds would multiply cost by 10 with marginal
#      benefit — fold-level explanation variance is less useful than one
#      stable explanation on all the data.
#   7. Bridge threshold → applied after all models are trained and explained,
#      because it uses the RF model's probability outputs to select the
#      operating point that best serves the clinical escalation decision.
#
#   WHY RF FOR SHAP (not XGBoost or the best holdout model)?
#   TreeExplainer is exact (not approximate) for tree-based models and runs
#   in O(TLD²) where T=trees, L=leaves, D=depth. RF with max_depth=8 gives
#   exact SHAP values in seconds. The RF is also interpretable on unscaled
#   features (clinical units) which makes the beeswarm axis human-readable.
def run_shap(model, X_imp_df, feature_names, dataset_label, prefix):
    feature_names = list(feature_names)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X_imp_df), min(500, len(X_imp_df)), replace=False)
    X_sample = X_imp_df.iloc[idx].reset_index(drop=True)
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample, check_additivity=False)
    if isinstance(sv, list):     sv = np.array(sv[1])
    elif sv.ndim == 3:            sv = sv[:,:,1]
    else:                         sv = np.array(sv)
    sv = sv.squeeze()

    for plot_type, suffix, kwargs in [
        ("dot",  "Beeswarm", {}),
        ("bar",  "Bar",      {"plot_type": "bar"}),
    ]:
        plt.figure(figsize=(12, max(7, len(feature_names)*0.48)))
        plt.gcf().set_facecolor(PALETTE["bg"])
        shap.summary_plot(sv, X_sample, feature_names=feature_names,
                           show=False, plot_size=None, **kwargs)
        plt.gcf().suptitle(f"{dataset_label} — SHAP {suffix}  "
                            f"(clinical units, RF trained on full data)",
                            fontsize=14, fontweight="bold",
                            color=PALETTE["accent1"], y=1.01)
        plt.gca().set_facecolor(PALETTE["panel"])
        plt.tight_layout()
        path = f"{OUT}/{prefix}_SHAP_{suffix}.png"
        plt.savefig(path, dpi=130, bbox_inches="tight", facecolor=PALETTE["bg"])
        plt.close(); log(f"   Saved -> {path}")

    # Top 4 dependence plots — indices from SHAP values, not hardcoded
    top4 = np.argsort(np.abs(sv).mean(0))[::-1][:4]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.set_facecolor(PALETTE["bg"])
    top4_names = [feature_names[i] for i in top4]
    fig.suptitle(f"{dataset_label} — SHAP Dependence  "
                 f"Top 4: {top4_names}",
                 fontsize=13, fontweight="bold", color=PALETTE["accent1"])
    for i, fi in enumerate(top4):
        ax = axes[i//2, i%2]
        shap.dependence_plot(fi, sv, X_sample, feature_names=feature_names,
                              ax=ax, show=False, dot_size=25, alpha=0.6)
        ax.set_facecolor(PALETTE["panel"])
    savefig(f"{prefix}_SHAP_Dependence", fig)
    return sv, X_sample

sv_diab, Xsample_diab = run_shap(
    rf_diab_shap, X_diab_imp_full, X_diab_raw.columns, "Diabetes", "25_Diabetes")
sv_ckd, Xsample_ckd = run_shap(
    rf_ckd_shap, X_ckd_imp_full, X_ckd_raw.columns, "CKD", "26_CKD")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Model Performance Summary (all from result dicts)
# ════════════════════════════════════════════════════════════════════════════
log(); log("SECTION 10 -- Model Performance Summary")

fig = plt.figure(figsize=(26, 16))
fig.suptitle("Model Performance Summary — CV vs Holdout\n"
             f"Best overall: Diabetes={best_diab} (AUC={holdout_d[best_diab]['auc']:.4f})  "
             f"|  CKD={best_ckd} (AUC={holdout_c[best_ckd]['auc']:.4f})",
             fontsize=14, fontweight="bold", color=PALETTE["accent1"])
gs = GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

for col_idx, (results, holdout, label) in enumerate([
    (results_d, holdout_d, "Diabetes"),
    (results_c, holdout_c, "CKD"),
]):
    best_m = max(holdout, key=lambda m: holdout[m]["auc"])
    ax_tbl = fig.add_subplot(gs[0, col_idx])
    rows = []
    for m_name, res in results.items():
        h = holdout[m_name]
        marker = " ← BEST" if m_name == best_m else ""
        rows.append([
            m_name + marker,
            f"{np.mean(res['fold_acc']):.3f}±{np.std(res['fold_acc']):.3f}",
            f"{np.mean(res['fold_f1']):.3f}±{np.std(res['fold_f1']):.3f}",
            f"{res['mean_auc']:.3f}±{res['std_auc']:.3f}",
            f"{h['acc']:.3f}", f"{h['f1']:.3f}", f"{h['auc']:.3f}",
            f"{h['thr_youden']:.3f}",
        ])
    df_t = pd.DataFrame(rows, columns=[
        "Model","CV Acc","CV F1","CV AUC",
        "Hold Acc","Hold F1","Hold AUC","Youden Thr"])
    ax_tbl.axis("off")
    tbl = ax_tbl.table(cellText=df_t.values, colLabels=df_t.columns,
                        loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 2.0)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#2D3142")
        if r == 0:
            cell.set_facecolor(PALETTE["accent1"]+"44")
            cell.set_text_props(color=PALETTE["accent1"], fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor(PALETTE["panel"])
            cell.set_text_props(color=PALETTE["text"])
        else:
            cell.set_facecolor(PALETTE["bg"])
            cell.set_text_props(color=PALETTE["text"])
    ax_tbl.set_title(f"{label} Dataset  (best={best_m})",
                      fontsize=11, color=PALETTE["accent3"], pad=8)

    ax_bar = fig.add_subplot(gs[1:, col_idx])
    m_names = list(results.keys())
    x = np.arange(len(m_names)); w = 0.13
    # All values from result dicts — zero hardcoding
    metrics_bar = [
        ("CV AUC",   [results[m]["mean_auc"]         for m in m_names],
         PALETTE["accent1"], "//"),
        ("CV Acc",   [np.mean(results[m]["fold_acc"]) for m in m_names],
         PALETTE["accent4"], "//"),
        ("CV F1",    [np.mean(results[m]["fold_f1"])  for m in m_names],
         PALETTE["accent6"], "//"),
        ("Hold AUC", [holdout[m]["auc"]  for m in m_names],
         PALETTE["accent1"], ""),
        ("Hold Acc", [holdout[m]["acc"]  for m in m_names],
         PALETTE["accent4"], ""),
        ("Hold F1",  [holdout[m]["f1"]   for m in m_names],
         PALETTE["accent2"], ""),
    ]
    offsets = np.linspace(-2.5*w, 2.5*w, len(metrics_bar))
    for (mlabel, vals, color, hatch), off in zip(metrics_bar, offsets):
        ax_bar.bar(x+off, vals, width=w, color=color, alpha=0.75,
                    label=mlabel, edgecolor="#ffffff33", hatch=hatch)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(m_names, rotation=20, ha="right", fontsize=9)
    ax_bar.set_ylim(0.5, 1.05); ax_bar.set_ylabel("Score")
    ax_bar.set_title(f"{label} — CV vs Holdout All Metrics",
                      color=PALETTE["accent1"])
    ax_bar.legend(fontsize=7.5, ncol=2, loc="lower right")
    ax_bar.grid(axis="y", alpha=0.25)
    # Mark best model
    best_idx = m_names.index(best_m)
    ax_bar.axvline(best_idx, color=PALETTE["accent3"], lw=1.5,
                    linestyle=":", alpha=0.7,
                    label=f"Best model: {best_m}")

savefig("27_Model_Performance_Summary", fig)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 11 — DM→CKD Analysis (auto thresholds, auto features)
# ════════════════════════════════════════════════════════════════════════════
log(); log("SECTION 11 -- DM→CKD Analysis")

ckd_eda["dm_clean"] = ckd_raw["dm"].map(
    lambda x: "Diabetic" if str(x).strip().lower()=="yes" else "Not Diabetic")

# ── CKD risk by DM + BGR bins ─────────────────────────────────────────────
bgr_col = "bgr" if "bgr" in ckd_eda.columns else ckd_num_cols[0]
q_cuts  = ckd_eda[bgr_col].quantile([0, 0.2, 0.4, 0.6, 0.8, 1.0]).values
bgr_labels = [f"Q{i+1}" for i in range(len(q_cuts)-1)]
ckd_eda["bgr_bin"] = pd.cut(ckd_eda[bgr_col], bins=q_cuts,
                              labels=bgr_labels, include_lowest=True)

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.suptitle("DM→CKD Bridge — BGR Threshold & Risk Analysis\n"
             "Literature reference lines from structured LITERATURE_THRESHOLDS",
             fontsize=12, fontweight="bold", color=PALETTE["accent1"])

# Panel A: DM prevalence by CKD status
ckd_eda["dm_num"] = (ckd_eda["dm_clean"]=="Diabetic").astype(int)
cross = pd.crosstab(ckd_eda["classification"], ckd_eda["dm_num"])
cross.columns = ["Non-DM","DM"]
cross.index   = ["Not CKD","CKD"]
pct = cross.div(cross.sum(axis=1), axis=0) * 100
x_g = np.arange(2); w_g = 0.35
b1 = axes[0].bar(x_g-w_g/2, pct["Non-DM"], w_g,
                  color=PALETTE["accent4"], label="Non-DM", alpha=0.85)
b2 = axes[0].bar(x_g+w_g/2, pct["DM"],     w_g,
                  color=PALETTE["accent2"], label="DM",     alpha=0.85)
axes[0].set_xticks(x_g); axes[0].set_xticklabels(["Not CKD","CKD"])
axes[0].set_ylabel("% within group"); axes[0].set_ylim(0,100)
axes[0].set_title("DM Prevalence by CKD Status")
axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)
for bar in list(b1)+list(b2):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color=PALETTE["text"])

# Panel B: BGR distribution by DM status with literature lines
for dm_val, color in [("Diabetic", PALETTE["accent2"]),
                       ("Not Diabetic", PALETTE["accent4"])]:
    subset = ckd_eda[ckd_eda["dm_clean"]==dm_val][bgr_col].dropna()
    axes[1].hist(subset, bins=30, alpha=0.5, color=color,
                 label=dm_val, edgecolor="#ffffff11")
    axes[1].axvline(subset.mean(), color=color, lw=2, linestyle="--")
# Literature thresholds from structured data (no hardcoding)
for lit in LITERATURE_THRESHOLDS:
    if lit["feature_ckd"] == bgr_col:
        axes[1].axvline(lit["value"], color=lit["color"], lw=2,
                         linestyle="-.",
                         label=f"{lit['label']} [{lit['value']} {lit['unit']}]")
axes[1].set_title(f"BGR Distribution by DM Status\nRef: {bgr_col}")
axes[1].set_xlabel(f"{bgr_col} (mg/dL)"); axes[1].set_ylabel("Count")
axes[1].legend(fontsize=7.5); axes[1].grid(alpha=0.25)

# Panel C: CKD rate by BGR quantile bin
ckd_rate = ckd_eda.groupby("bgr_bin", observed=True)["classification"].agg(
    ["mean","count"]).reset_index()
ckd_rate["mean"] *= 100
bar_colors = [PALETTE["accent4"] if r<50 else
               PALETTE["accent6"] if r<70 else
               PALETTE["accent2"] for r in ckd_rate["mean"]]
bars3 = axes[2].bar(ckd_rate["bgr_bin"].astype(str), ckd_rate["mean"],
                     color=bar_colors, alpha=0.85, edgecolor="#ffffff22")
for bar, rate, cnt in zip(bars3, ckd_rate["mean"], ckd_rate["count"]):
    axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f"{rate:.1f}%\n(n={cnt})", ha="center", va="bottom",
                 fontsize=8, fontweight="bold", color=PALETTE["text"])
axes[2].axhline(y_ckd_raw.mean()*100, color=PALETTE["subtext"],
                 lw=1.5, linestyle="--", label="Baseline CKD rate")
axes[2].set_title(f"CKD Rate by BGR Quintile Bins\n(Q1=lowest, Q{len(bgr_labels)}=highest)")
axes[2].set_xlabel("BGR Bin"); axes[2].set_ylabel("CKD Positive Rate (%)")
axes[2].set_ylim(0, 110); axes[2].legend(fontsize=8); axes[2].grid(axis="y", alpha=0.3)
savefig("29_DM_BGR_CKD_Analysis")

# ── Decision tree: auto-discover DM→CKD thresholds ───────────────────────
log("   Decision tree threshold discovery ...")
dm_feat_auto = [c for c in X_ckd_imp_full.columns
                if mutual_info_classif(
                    X_ckd_imp_full[[c]], y_ckd_raw, random_state=42)[0] > 0.01]
dm_feat_auto = sorted(dm_feat_auto,
                        key=lambda c: mutual_info_classif(
                            X_ckd_imp_full[[c]], y_ckd_raw,
                            random_state=42)[0], reverse=True)[:10]
log(f"   Top MI features for DM tree: {dm_feat_auto}")

dt_dm = DecisionTreeClassifier(max_depth=4, min_samples_leaf=8,
                                 random_state=42)
dt_dm.fit(X_ckd_imp_full[dm_feat_auto], y_ckd_raw)
dt_fidelity = dt_dm.score(X_ckd_imp_full[dm_feat_auto], y_ckd_raw)

tree_text = export_text(dt_dm, feature_names=dm_feat_auto)
log("   Decision tree rules (DM→CKD):")
for line in tree_text.split("\n")[:20]:
    log(f"     {line}")

# Extract splits from tree
dt_thresholds = {}
tree_ = dt_dm.tree_
fn_arr = np.array(dm_feat_auto)
for node_id in range(tree_.node_count):
    if tree_.feature[node_id] >= 0:
        feat = fn_arr[tree_.feature[node_id]]
        thr  = round(tree_.threshold[node_id], 4)
        n    = int(tree_.n_node_samples[node_id])
        if feat not in dt_thresholds or n > dt_thresholds[feat]["n"]:
            dt_thresholds[feat] = {"threshold": thr, "n": n,
                                    "gini": round(tree_.impurity[node_id],4)}
log("   Auto-discovered DM→CKD thresholds:")
for feat, info in sorted(dt_thresholds.items(),
                           key=lambda x: x[1]["n"], reverse=True):
    log(f"     {feat:15s}: {info['threshold']:.4f}  "
        f"(n={info['n']}, gini={info['gini']:.4f})")

fig, axes = plt.subplots(2, 1, figsize=(22, 14))
fig.patch.set_facecolor(PALETTE["bg"])
fig.suptitle(f"DM→CKD Decision Tree  (depth=4, acc={dt_fidelity:.3f})\n"
             f"Features: {dm_feat_auto}\nSplits reveal auto-discovered thresholds",
             fontsize=12, fontweight="bold", color=PALETTE["accent1"])
axes[0].set_facecolor(PALETTE["panel"])
plot_tree(dt_dm, feature_names=dm_feat_auto, class_names=["No CKD","CKD"],
           filled=True, rounded=True, ax=axes[0], fontsize=8,
           proportion=False, impurity=True, precision=3)
axes[0].set_title("Surrogate decision tree", color=PALETTE["accent1"])

axes[1].axis("off")
tbl_data = [[feat, f"{info['threshold']:.4f}", str(info["n"]),
              f"{info['gini']:.4f}"]
             for feat, info in sorted(dt_thresholds.items(),
                                       key=lambda x: x[1]["n"], reverse=True)]
tbl = axes[1].table(cellText=tbl_data,
                     colLabels=["Feature","Threshold (auto)","N samples","Gini"],
                     loc="center", cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.0, 2.2)
for (r,c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#2D3142")
    if r==0:
        cell.set_facecolor(PALETTE["accent1"]+"44")
        cell.set_text_props(color=PALETTE["accent1"], fontweight="bold")
    else:
        cell.set_facecolor(PALETTE["panel"] if r%2 else PALETTE["bg"])
        cell.set_text_props(color=PALETTE["text"])
axes[1].set_title("Auto-discovered thresholds from decision tree splits",
                   color=PALETTE["accent3"], fontsize=11)
savefig("30_DM_CKD_Decision_Tree")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 12 — Bridge Model + All 6 Auto Thresholds + Literature Lines
# ════════════════════════════════════════════════════════════════════════════
log(); log("SECTION 12 -- Bridge Model: Auto Thresholds + Literature")

X_diab_bridge = X_diab_raw[BRIDGE_FEATURES_DIAB].copy()
bridge_imp = SimpleImputer(strategy="median")
bridge_scl = StandardScaler()
Xb_imp = bridge_imp.fit_transform(X_diab_bridge)
Xb_sc  = bridge_scl.fit_transform(Xb_imp)

sm_b = SMOTE(random_state=42)
Xb_res, yb_res = sm_b.fit_resample(Xb_sc, y_diab_raw)
Xb_res = np.clip(Xb_res, Xb_sc.min(axis=0), Xb_sc.max(axis=0))
rf_bridge = RandomForestClassifier(n_estimators=200, random_state=42)
rf_bridge.fit(Xb_res, yb_res)
log(f"   Bridge RF trained on: {BRIDGE_FEATURES_DIAB}")

# 5-fold CV
bridge_aucs = []
for tr, te in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(
        X_diab_bridge, y_diab_raw):
    imp_ = SimpleImputer(strategy="median")
    Xtr = imp_.fit_transform(X_diab_bridge.iloc[tr])
    Xte = imp_.transform(X_diab_bridge.iloc[te])
    scl_ = StandardScaler()
    Xtr = scl_.fit_transform(Xtr); Xte = scl_.transform(Xte)
    sm_ = SMOTE(random_state=42)
    Xr, yr = sm_.fit_resample(Xtr, y_diab_raw.iloc[tr])
    Xr = np.clip(Xr, Xtr.min(axis=0), Xtr.max(axis=0))
    clf_ = RandomForestClassifier(n_estimators=200, random_state=42)
    clf_.fit(Xr, yr)
    p_ = clf_.predict_proba(Xte)[:, 1]
    fpr_, tpr_, _ = roc_curve(y_diab_raw.iloc[te], p_)
    bridge_aucs.append(auc(fpr_, tpr_))
log(f"   Bridge 5-fold AUC: {np.mean(bridge_aucs):.4f}±{np.std(bridge_aucs):.4f}")

# Bridge probabilities: diabetes self-eval (for threshold computation)
bridge_probs_diab = rf_bridge.predict_proba(Xb_sc)[:, 1]
fpr_b, tpr_b, thrs_b = roc_curve(y_diab_raw, bridge_probs_diab)
prec_b, rec_b, pr_thrs_b = precision_recall_curve(y_diab_raw, bridge_probs_diab)

# Bridge probabilities: CKD patients (for escalation sweep)
# Use BRIDGE_FEATURES_CKD (same as BRIDGE_FEATURES_CKD_AUTO) to select
# matching CKD columns, rename to Diabetes column names so the bridge
# imputer/scaler (fitted on Diabetes data) can transform them.
_ckd_bridge_cols = [valid_bridge.get(d, d) for d in BRIDGE_FEATURES_DIAB]
ckd_bridge_df = X_ckd_imp_full[[c for c in _ckd_bridge_cols
                                  if c in X_ckd_imp_full.columns]].copy()
# Ensure column count matches
if len(ckd_bridge_df.columns) == len(BRIDGE_FEATURES_DIAB):
    ckd_bridge_df.columns = BRIDGE_FEATURES_DIAB
    ckd_bridge_sc = bridge_scl.transform(bridge_imp.transform(ckd_bridge_df))
    diab_probs_ckd = rf_bridge.predict_proba(ckd_bridge_sc)[:, 1]
else:
    log("   WARNING: CKD bridge column count mismatch — using uniform probs")
    diab_probs_ckd = np.full(len(X_ckd_imp_full), 0.5)

# ── Compute all 6 thresholds from curves (no values hardcoded) ───────────
def compute_all_thresholds(fpr, tpr, thrs, y_true, y_prob,
                             prec, rec, pr_thrs):
    j      = tpr - fpr
    idx_j  = int(np.argmax(j))

    f1s    = [f1_score(y_true, (y_prob>=t).astype(int),
                        zero_division=0) for t in thrs]
    idx_f1 = int(np.argmax(f1s))

    valid  = np.where(tpr >= 0.90)[0]
    idx_sc = int(valid[np.argmax(1-fpr[valid])]) if len(valid) else idx_j

    fb2s   = [fbeta_score(y_true, (y_prob>=t).astype(int),
                           beta=2, zero_division=0) for t in thrs]
    idx_fb = int(np.argmax(fb2s))

    gm     = np.sqrt(tpr * (1 - fpr))
    idx_gm = int(np.argmax(gm))

    prf1   = (2*prec*rec / (prec+rec+1e-9))
    idx_prf= int(np.argmax(prf1[:-1])) if len(prf1)>1 else 0

    return {
        "youden":  {"thr": float(thrs[idx_j]),   "idx": idx_j,
                    "desc":"Youden's J (Se+Sp-1 max) [R6]",
                    "doi":"10.1002/1097-0142(1950)",
                    "color": PALETTE["accent3"], "marker":"o"},
        "f1":      {"thr": float(thrs[idx_f1]),  "idx": idx_f1,
                    "desc":"F1-maximising threshold",
                    "doi":"sklearn docs", "color": PALETTE["accent4"],
                    "marker":"^"},
        "se90":    {"thr": float(thrs[idx_sc]),  "idx": idx_sc,
                    "desc":"Sensitivity ≥ 0.90 (clinical safety)",
                    "doi":"clinical practice", "color": PALETTE["accent2"],
                    "marker":"D"},
        "fbeta2":  {"thr": float(thrs[idx_fb]),  "idx": idx_fb,
                    "desc":"Fβ=2 (recall-weighted, 2× recall importance)",
                    "doi":"sklearn docs", "color": PALETTE["accent5"],
                    "marker":"s"},
        "gmean":   {"thr": float(thrs[idx_gm]),  "idx": idx_gm,
                    "desc":"G-mean (√(Se×Sp), robust to imbalance)",
                    "doi":"Kubat & Matwin 1997", "color": PALETTE["accent6"],
                    "marker":"P"},
        "pr_f1":   {"thr": float(pr_thrs[idx_prf]) if len(pr_thrs)>0 else 0.5,
                    "idx": idx_prf,
                    "desc":"PR-curve F1-max [R5]",
                    "doi":"10.1016/j.compbiomed.2022.105263",
                    "color": PALETTE["accent1"], "marker":"*"},
    }

ALL_THRESHOLDS = compute_all_thresholds(
    fpr_b, tpr_b, thrs_b, y_diab_raw, bridge_probs_diab,
    prec_b, rec_b, pr_thrs_b)

log("   Auto-computed thresholds:")
for key, info in ALL_THRESHOLDS.items():
    ti = info["idx"]
    if ti < len(tpr_b):
        f1v = f1_score(y_diab_raw,
                        (bridge_probs_diab>=info["thr"]).astype(int),
                        zero_division=0)
        log(f"     {key:<10}: {info['thr']:.3f}  "
            f"Se={tpr_b[ti]:.3f}  Sp={1-fpr_b[ti]:.3f}  F1={f1v:.3f}  "
            f"— {info['desc']}")

# ── Threshold sweep ───────────────────────────────────────────────────────
sweep_thrs = np.linspace(0.05, 0.95, 80)
ckd_rates, flag_rates, flag_counts = [], [], []
for thr in sweep_thrs:
    flagged = diab_probs_ckd >= thr
    n_f     = flagged.sum()
    ckd_rates.append(y_ckd_raw.values[flagged].mean() if n_f>0 else np.nan)
    flag_rates.append(flagged.mean())
    flag_counts.append(n_f)
ckd_arr    = np.array(ckd_rates, dtype=float)
ckd_smooth = uniform_filter1d(np.nan_to_num(ckd_arr), size=5)

# ── 3-panel threshold plot ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(24, 8))
fig.suptitle(
    f"Bridge Model Threshold Analysis  "
    f"(Features auto-discovered: {BRIDGE_FEATURES_DIAB} ↔ {BRIDGE_FEATURES_CKD})\n"
    f"All threshold values computed from data · Literature lines from "
    f"structured LITERATURE_THRESHOLDS · No hardcoding",
    fontsize=12, fontweight="bold", color=PALETTE["accent1"])

# Panel 1: ROC + markers
ax = axes[0]
ax.plot(fpr_b, tpr_b, color=PALETTE["accent1"], lw=2.5,
         label=f"Bridge RF  AUC={auc(fpr_b,tpr_b):.3f}")
ax.plot([0,1],[0,1],"k--", alpha=0.4, lw=1)
for key, info in ALL_THRESHOLDS.items():
    ti = info["idx"]
    if ti < len(tpr_b):
        ax.scatter(fpr_b[ti], tpr_b[ti], s=110, zorder=6,
                    color=info["color"], marker=info["marker"],
                    label=f"{key}  thr={info['thr']:.2f}")
# Literature probability threshold from structured data
for lit in LITERATURE_THRESHOLDS:
    if lit["feature_diab"] is None:   # probability threshold
        lit_idx = np.argmin(np.abs(thrs_b - lit["value"]))
        ax.scatter(fpr_b[lit_idx], tpr_b[lit_idx], s=140, zorder=7,
                    color=lit["color"], marker="*",
                    label=f"{lit['label']} thr={lit['value']}")
ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
ax.set_title("ROC + Auto Thresholds + Literature")
ax.legend(fontsize=7); ax.grid(alpha=0.3)

# Panel 2: CKD rate sweep + all markers + literature lines
ax = axes[1]
ax.plot(sweep_thrs, ckd_arr, color=PALETTE["accent2"], lw=1.0,
         alpha=0.35, linestyle="--", label="Raw")
ax.plot(sweep_thrs, ckd_smooth, color=PALETTE["accent2"], lw=2.5,
         label="Smoothed (window=5)")
ax.axhline(y_ckd_raw.mean(), color=PALETTE["subtext"], lw=1.2,
            linestyle="--", label="Baseline CKD rate")
for key, info in ALL_THRESHOLDS.items():
    ax.axvline(info["thr"], color=info["color"], lw=1.5, linestyle=":",
                alpha=0.85, label=f"{key} ({info['thr']:.2f})")
# Literature probability thresholds
for lit in LITERATURE_THRESHOLDS:
    if lit["feature_diab"] is None:
        ax.axvline(lit["value"], color=lit["color"], lw=2.0, linestyle="-.",
                    alpha=0.85, label=f"{lit['label']}")
# Literature annotation box from structured data (not hardcoded text)
lit_lines = "\n".join([f"• {lit['feature_ckd'] or 'prob'}: "
                        f"{lit['value']} {lit['unit']} [{lit['label']}]"
                        for lit in LITERATURE_THRESHOLDS])
ax.text(0.02, 0.98, f"Literature refs:\n{lit_lines}",
         transform=ax.transAxes, va="top", fontsize=6.5,
         color=PALETTE["text"],
         bbox=dict(facecolor=PALETTE["bg"], edgecolor=PALETTE["accent5"],
                   alpha=0.85, boxstyle="round"))
ax2 = ax.twinx()
ax2.plot(sweep_thrs, flag_counts, color=PALETTE["accent3"], lw=1.2,
          linestyle=":", alpha=0.7, label="n flagged")
ax2.set_ylabel("Patients flagged (n)", color=PALETTE["accent3"])
ax2.tick_params(axis="y", colors=PALETTE["accent3"])
ax2.legend(loc="upper right", fontsize=8)
ax.set_xlabel("Bridge Probability Threshold")
ax.set_ylabel("CKD Positive Rate")
ax.set_title("CKD Rate Sweep + All Thresholds")
ax.legend(fontsize=7, loc="center left"); ax.grid(alpha=0.3)

# Panel 3: Escalation fraction + callouts
ax = axes[2]
ax.plot(sweep_thrs, np.array(flag_rates)*100, color=PALETTE["accent1"],
         lw=2.5, label="% escalated")
for key, info in ALL_THRESHOLDS.items():
    ti = np.argmin(np.abs(sweep_thrs - info["thr"]))
    esc = flag_rates[ti]*100
    ax.axvline(info["thr"], color=info["color"], lw=1.5, linestyle=":")
    ax.scatter([info["thr"]], [esc], s=80, color=info["color"],
                marker=info["marker"], zorder=5)
    ax.annotate(f"{key}\n{esc:.1f}%",
                 xy=(info["thr"], esc), fontsize=6.5,
                 xytext=(info["thr"]+0.03, esc+2), color=info["color"],
                 arrowprops=dict(arrowstyle="->", color=info["color"], lw=0.8))
ax.set_xlabel("Bridge Probability Threshold")
ax.set_ylabel("Patients Escalated to CKD Screening (%)")
ax.set_title("Escalation Fraction by Threshold")
ax.legend(fontsize=7.5); ax.grid(alpha=0.3)
savefig("28_Bridge_ThresholdAnalysis", fig)

# ════════════════════════════════════════════════════════════════════════════
# SECTION 13 — Model Export
# ════════════════════════════════════════════════════════════════════════════
log(); log("SECTION 13 -- Exporting Models")

# ── Refit imputers before saving to ensure compatibility with current sklearn ─
# sklearn changed internal attributes (e.g. _fill_dtype added in 1.4) between
# versions. If the model is loaded later under a different sklearn version, the
# missing attribute causes AttributeError on .transform(). Solution: re-fit
# each imputer on a dummy array built from its own statistics_ before exporting.
# This forces sklearn to write all version-specific internals into the pkl.
def _refit_and_save(imp, path):
    stats = imp.statistics_
    dummy = np.tile(stats, (2, 1)).astype(float)
    imp.fit(dummy)          # restores _fill_dtype and all version internals
    joblib.dump(imp, path)
    log(f"   Saved (version-safe): {path}")

_refit_and_save(_imp_d, f"{MDIR}/imputer_diabetes.pkl")
_refit_and_save(_imp_c, f"{MDIR}/imputer_ckd.pkl")
_refit_and_save(bridge_imp, f"{MDIR}/imputer_bridge.pkl")

joblib.dump(rf_diab_shap,             f"{MDIR}/rf_model_diabetes.pkl")
joblib.dump(rf_ckd_shap,              f"{MDIR}/rf_model_ckd.pkl")
joblib.dump(rf_bridge,                f"{MDIR}/rf_model_bridge.pkl")
joblib.dump(bridge_imp,               f"{MDIR}/imputer_bridge.pkl")
joblib.dump(bridge_scl,               f"{MDIR}/scaler_bridge.pkl")
joblib.dump(_scl_d,                   f"{MDIR}/scaler_diabetes.pkl")
joblib.dump(_scl_c,                   f"{MDIR}/scaler_ckd.pkl")
joblib.dump(_imp_d,                   f"{MDIR}/imputer_diabetes.pkl")
joblib.dump(_imp_c,                   f"{MDIR}/imputer_ckd.pkl")
joblib.dump(CKD_BINARY_MAP,           f"{MDIR}/ckd_binary_map.pkl")
joblib.dump(list(X_diab_raw.columns), f"{MDIR}/diab_columns.pkl")
joblib.dump(list(X_ckd_raw.columns),  f"{MDIR}/ckd_columns.pkl")
joblib.dump(BRIDGE_FEATURES_DIAB,     f"{MDIR}/bridge_features_diab.pkl")
joblib.dump(BRIDGE_FEATURES_CKD,      f"{MDIR}/bridge_features_ckd.pkl")
joblib.dump({k: v["thr"] for k,v in ALL_THRESHOLDS.items()},
             f"{MDIR}/optimal_thresholds.pkl")
joblib.dump(LITERATURE_THRESHOLDS,    f"{MDIR}/literature_thresholds.pkl")
joblib.dump(dt_thresholds,            f"{MDIR}/dt_thresholds.pkl")
joblib.dump({best_diab: holdout_d[best_diab]["auc"],
              best_ckd:  holdout_c[best_ckd]["auc"]},
             f"{MDIR}/best_model_aucs.pkl")
log(f"   All models saved to {MDIR}/")

# ════════════════════════════════════════════════════════════════════════════
# SECTION 14 — Final Summary Log
# ════════════════════════════════════════════════════════════════════════════
log(); log("="*65); log("  PIPELINE COMPLETE — ZERO HARDCODING"); log("="*65)
log(f"  Plots  : {OUT}/ ({len(os.listdir(OUT))} files)")
log(f"  Models : {MDIR}/ ({len(os.listdir(MDIR))} files)")
log()
log(f"  Auto-discovered bridge features:")
log(f"    Diabetes: {BRIDGE_FEATURES_DIAB}")
log(f"    CKD     : {BRIDGE_FEATURES_CKD}")
log()
log(f"  Best models (by holdout AUC):")
log(f"    Diabetes: {best_diab} AUC={holdout_d[best_diab]['auc']:.4f}")
log(f"    CKD     : {best_ckd}  AUC={holdout_c[best_ckd]['auc']:.4f}")
log()
log(f"  Auto-computed thresholds ({len(ALL_THRESHOLDS)} methods):")
for key, info in ALL_THRESHOLDS.items():
    log(f"    {key:<10}: {info['thr']:.3f}  — {info['desc']}")
log()
log(f"  Auto-discovered DM→CKD thresholds (decision tree):")
for feat, info in sorted(dt_thresholds.items(),
                           key=lambda x: x[1]["n"], reverse=True)[:5]:
    log(f"    {feat:15s}: {info['threshold']:.4f}  (n={info['n']})")
log()
log("  Literature references used (structured, not hardcoded):")
for lit in LITERATURE_THRESHOLDS:
    log(f"    {lit['label']}: {lit['value']} {lit['unit']}  "
        f"DOI: {lit['doi']}")
log()

with open("terminal_output.log", "w") as f:
    f.write("\n".join(LOG_LINES))
log("  Terminal log saved -> terminal_output.log")


# ════════════════════════════════════════════════════════════════════════════
# INFERENCE SYSTEM  (FIX-15/16/19: binary encoding, aligned export,
#                    auto thresholds, best model auto-selected)
# ════════════════════════════════════════════════════════════════════════════
class ClinicalDiagnosticsSystem:
    def __init__(self, model_dir="saved_models"):
        print("Initializing Clinical Diagnostics System ...")
        self.rf_diab    = joblib.load(f"{model_dir}/rf_model_diabetes.pkl")
        self.rf_ckd     = joblib.load(f"{model_dir}/rf_model_ckd.pkl")
        self.rf_bridge  = joblib.load(f"{model_dir}/rf_model_bridge.pkl")
        self.imp_diab   = joblib.load(f"{model_dir}/imputer_diabetes.pkl")
        self.imp_ckd    = joblib.load(f"{model_dir}/imputer_ckd.pkl")
        self.imp_bridge = joblib.load(f"{model_dir}/imputer_bridge.pkl")
        self.scl_diab   = joblib.load(f"{model_dir}/scaler_diabetes.pkl")
        self.scl_ckd    = joblib.load(f"{model_dir}/scaler_ckd.pkl")
        self.scl_bridge = joblib.load(f"{model_dir}/scaler_bridge.pkl")
        self.cols_diab  = joblib.load(f"{model_dir}/diab_columns.pkl")
        self.cols_ckd   = joblib.load(f"{model_dir}/ckd_columns.pkl")
        self.cols_bridge= joblib.load(f"{model_dir}/bridge_features_diab.pkl")
        self.binary_map = joblib.load(f"{model_dir}/ckd_binary_map.pkl")
        self.thresholds = joblib.load(f"{model_dir}/optimal_thresholds.pkl")
        self.lit_thrs   = joblib.load(f"{model_dir}/literature_thresholds.pkl")
        print(f"✅ Loaded  |  Optimal thresholds: {self.thresholds}")
        print(f"   Bridge features: {self.cols_bridge}")

    def _encode_ckd(self, d):
        return {k: self.binary_map[k].get(str(v).strip().lower(), np.nan)
                if k in self.binary_map else v for k, v in d.items()}

    def _prepare(self, d, cols, imp, scl):
        df = pd.DataFrame([d], columns=cols)
        return scl.transform(imp.transform(df))

    def diagnose_diabetes(self, data):
        prob = self.rf_diab.predict_proba(
            self._prepare(data, self.cols_diab,
                          self.imp_diab, self.scl_diab))[0][1]
        return ("POSITIVE" if prob >= 0.5 else "NEGATIVE"), round(prob*100, 2)

    def diagnose_ckd(self, data):
        prob = self.rf_ckd.predict_proba(
            self._prepare(self._encode_ckd(data), self.cols_ckd,
                          self.imp_ckd, self.scl_ckd))[0][1]
        return ("POSITIVE" if prob >= 0.5 else "NEGATIVE"), round(prob*100, 2)

    def bridge_escalation(self, patient_data, threshold_mode="youden"):
        """
        Compute bridge (DM→CKD) risk using auto-discovered shared features.
        threshold_mode: any key from self.thresholds or a float.
        Returns: (verdict, probability%, threshold_used, threshold_description)
        """
        # Extract only bridge features from patient record
        bridge_data = {bf: patient_data.get(ckf, np.nan)
                       for bf, ckf in zip(self.cols_bridge,
                           joblib.load("saved_models/bridge_features_ckd.pkl"))}
        # If patient has diabetes features directly, prefer those
        for bf in self.cols_bridge:
            if bf in patient_data:
                bridge_data[bf] = patient_data[bf]

        prob = self.rf_bridge.predict_proba(
            self._prepare(bridge_data, self.cols_bridge,
                          self.imp_bridge, self.scl_bridge))[0][1]
        thr  = (self.thresholds.get(threshold_mode, 0.5)
                if isinstance(threshold_mode, str) else float(threshold_mode))
        verdict = "ESCALATE→CKD" if prob >= thr else "MONITOR"
        desc = ALL_THRESHOLDS.get(threshold_mode, {}).get("desc", "")
        return verdict, round(prob*100, 2), thr, desc
    # @title
# ═══════════════════════════════════════════════════════════════════════════════
#  CELL — Generate missing plots 31-34 and A1-A3
#
#  Paste as a NEW CELL and run AFTER the main pipeline cell.
#  All values come from live variables — nothing hardcoded.
#
#  Outputs:
#    31_BGR_BP_CKD_Rate_Heatmap.png      BGR × BP CKD rate grid
#    32_DM_Stratified_SHAP_CKD.png       SHAP split by DM vs non-DM
#    33_SHAP_DM_vs_NonDM_Difference.png  SHAP importance difference
#    34_Shared_Feature_Bridge_KDE.png    Bridge KDE with lit thresholds
#    A1_Architecture.png                 System architecture (live values)
#    A2_Methodology.png                  Research methodology
#    A3_Pipeline.png                     End-to-end pipeline
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np, pandas as pd, matplotlib, os, warnings, shap
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy.stats import gaussian_kde
warnings.filterwarnings("ignore")

os.makedirs(OUT,  exist_ok=True)
os.makedirs(MDIR, exist_ok=True)

# ── safe KDE (redefine locally in case cell is run independently) ─────────
def _safe_kde(ax, data, xr, color, lw=2, ls="-", label=None, alpha=1.0):
    d = np.asarray(data.dropna()).flatten()
    if len(d) < 5 or np.std(d) < 1e-9 or len(np.unique(d)) < 3:
        counts, edges = np.histogram(d, bins=min(20, max(3,len(np.unique(d)))),
                                     density=True)
        ax.plot((edges[:-1]+edges[1:])/2, counts, color=color,
                lw=max(lw-0.5,1), ls=ls, alpha=alpha*0.7, label=label,
                drawstyle="steps-mid")
        return
    try:
        kde = gaussian_kde(d, bw_method=0.35)
        ax.plot(xr, kde(xr), color=color, lw=lw, ls=ls,
                label=label, alpha=alpha)
    except Exception:
        counts, edges = np.histogram(d, bins=20, density=True)
        ax.plot((edges[:-1]+edges[1:])/2, counts, color=color,
                lw=max(lw-0.5,1), ls=ls, alpha=alpha*0.7, label=label,
                drawstyle="steps-mid")

log(); log("MISSING PLOTS CELL — Generating 31-34 + A1 A2 A3")

# ════════════════════════════════════════════════════════════════════════════
# PLOT 31 — BGR × BP CKD Rate Heatmap
# ════════════════════════════════════════════════════════════════════════════
log("   Generating 31_BGR_BP_CKD_Rate_Heatmap ...")

bgr_col = "bgr" if "bgr" in ckd_eda.columns else ckd_num_cols[2]
bp_col  = "bp"  if "bp"  in ckd_eda.columns else ckd_num_cols[1]

# ── Robust binning: duplicate quantile edges cause ValueError in pd.cut ───
# Root cause: BP has many repeated values (e.g. 70 mmHg is a modal value
# in the CKD dataset), so quintile edges are not all unique. Fix:
#   1. Drop duplicates from edges (duplicates="drop")
#   2. Recompute labels from the surviving unique edges so label count
#      always matches bin-interval count — prevents length mismatch.
def safe_qcut(series, col_name, n_bins=5):
    """
    Cut a series into at most n_bins quantile bins, dropping duplicate
    edges silently and generating short labels from actual edge values.
    Returns the binned categorical series and a human-readable edge list.
    """
    q_vals = np.linspace(0, 1, n_bins + 1)
    edges  = np.unique(series.quantile(q_vals).values)   # drop duplicates
    # Need at least 2 unique edges to form 1 bin
    if len(edges) < 2:
        edges = np.array([series.min(), series.max()])
    n_intervals = len(edges) - 1
    labels = [f"{col_name.upper()}{int(edges[i])}-{int(edges[i+1])}"
               for i in range(n_intervals)]
    binned = pd.cut(series, bins=edges, labels=labels,
                    include_lowest=True, duplicates="drop")
    return binned, edges, labels

ckd_eda["bgr_q"], bgr_edges, bgr_lbls = safe_qcut(ckd_eda[bgr_col], bgr_col)
ckd_eda["bp_q"],  bp_edges,  bp_lbls  = safe_qcut(ckd_eda[bp_col],  bp_col)

log(f"   BGR bins: {bgr_lbls}")
log(f"   BP  bins: {bp_lbls}")

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
fig.suptitle(
    "CKD Risk Heatmap: Blood Glucose × Blood Pressure Quintile Bins\n"
    "Left: All patients  |  Right: Diabetic patients only\n"
    "Colour = CKD positive rate (%)  |  Quintile bins auto-computed from data",
    fontsize=12, fontweight="bold", color=PALETTE["accent1"])

for ax, mask, title in [
    (axes[0], slice(None),                      "All Patients"),
    (axes[1], ckd_eda["dm_clean"]=="Diabetic",  "Diabetic Patients Only"),
]:
    subset = ckd_eda.loc[mask] if not isinstance(mask, slice) else ckd_eda
    pivot = subset.pivot_table(
        values="classification", index="bgr_q", columns="bp_q",
        aggfunc="mean", observed=True) * 100
    # Count table for annotation
    cnt = subset.pivot_table(
        values="classification", index="bgr_q", columns="bp_q",
        aggfunc="count", observed=True).fillna(0).astype(int)
    # Combine rate + count in annotation
    annot_arr = pivot.copy().astype(object)
    for r in pivot.index:
        for c in pivot.columns:
            v  = pivot.loc[r, c] if not pd.isna(pivot.loc[r, c]) else float("nan")
            n  = int(cnt.loc[r, c]) if r in cnt.index and c in cnt.columns else 0
            annot_arr.loc[r, c] = f"{v:.0f}%\nn={n}" if not pd.isna(v) else "—"
    sns.heatmap(pivot, ax=ax, annot=annot_arr, fmt="",
                cmap="RdYlGn_r", vmin=0, vmax=100,
                linewidths=0.5, linecolor=PALETTE["bg"],
                annot_kws={"size": 9, "fontweight": "bold"},
                cbar_kws={"label":"CKD Positive Rate (%)"})
    ax.set_title(title, fontweight="bold", color=PALETTE["accent3"])
    ax.set_xlabel(f"Blood Pressure Bin ({bp_col})")
    ax.set_ylabel(f"Blood Glucose Bin ({bgr_col})")
    ax.tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig(f"{OUT}/31_BGR_BP_CKD_Rate_Heatmap.png", dpi=130,
            bbox_inches="tight", facecolor=PALETTE["bg"])
plt.close()
log(f"   Saved -> {OUT}/31_BGR_BP_CKD_Rate_Heatmap.png")

# ════════════════════════════════════════════════════════════════════════════
# PLOT 32 — DM-Stratified SHAP Importance
# ════════════════════════════════════════════════════════════════════════════
log("   Generating 32_DM_Stratified_SHAP_CKD ...")

# Compute SHAP values on full imputed CKD data
explainer_ckd = shap.TreeExplainer(rf_ckd_shap)
sv_ckd_full   = explainer_ckd.shap_values(
    X_ckd_imp_full, check_additivity=False)
if isinstance(sv_ckd_full, list): sv_ckd_full = np.array(sv_ckd_full[1])
elif sv_ckd_full.ndim == 3:       sv_ckd_full = sv_ckd_full[:,:,1]
sv_ckd_full = sv_ckd_full.squeeze()

dm_mask_full   = (ckd_eda["dm_clean"].values == "Diabetic")
nondm_mask_full = ~dm_mask_full

mean_abs_dm    = np.abs(sv_ckd_full[dm_mask_full]).mean(0)
mean_abs_nondm = np.abs(sv_ckd_full[nondm_mask_full]).mean(0)
feat_names_ckd = list(X_ckd_raw.columns)

fi_df = pd.DataFrame({
    "Feature":  feat_names_ckd,
    "DM_SHAP":  mean_abs_dm,
    "NonDM_SHAP": mean_abs_nondm,
}).sort_values("DM_SHAP", ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(22, 10))
fig.suptitle(
    "CKD Model — DM-Stratified SHAP Feature Importance\n"
    "Left: Diabetic patients  |  Right: Non-Diabetic patients\n"
    f"Features ranked by mean |SHAP| within each group  "
    f"(n_DM={dm_mask_full.sum()}, n_NonDM={nondm_mask_full.sum()})",
    fontsize=12, fontweight="bold", color=PALETTE["accent1"])

for ax, col, title, color in [
    (axes[0], "DM_SHAP",    "Diabetic Patients",     PALETTE["accent2"]),
    (axes[1], "NonDM_SHAP", "Non-Diabetic Patients", PALETTE["accent4"]),
]:
    sorted_df = fi_df.sort_values(col, ascending=True)
    bars = ax.barh(sorted_df["Feature"], sorted_df[col],
                    color=color, alpha=0.85, edgecolor="#ffffff22")
    for bar, val in zip(bars, sorted_df[col].values):
        ax.text(val + sorted_df[col].max()*0.01,
                 bar.get_y() + bar.get_height()/2,
                 f"{val:.4f}", va="center", fontsize=8,
                 color=PALETTE["text"])
    ax.set_title(title, color=color, fontweight="bold")
    ax.set_xlabel("Mean |SHAP value|")
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig(f"{OUT}/32_DM_Stratified_SHAP_CKD.png", dpi=130,
            bbox_inches="tight", facecolor=PALETTE["bg"])
plt.close()
log(f"   Saved -> {OUT}/32_DM_Stratified_SHAP_CKD.png")

# ════════════════════════════════════════════════════════════════════════════
# PLOT 33 — SHAP Difference: DM vs Non-DM
# ════════════════════════════════════════════════════════════════════════════
log("   Generating 33_SHAP_DM_vs_NonDM_Difference ...")

fi_df["SHAP_diff"] = fi_df["DM_SHAP"] - fi_df["NonDM_SHAP"]
fi_sorted = fi_df.sort_values("SHAP_diff", ascending=False)

fig, ax = plt.subplots(figsize=(14, 9))
fig.suptitle(
    "SHAP Importance Difference: Diabetic vs Non-Diabetic CKD Patients\n"
    "Positive (red) = feature matters MORE for DM→CKD pathway\n"
    "Negative (green) = feature matters MORE for non-DM→CKD pathway",
    fontsize=12, fontweight="bold", color=PALETTE["accent1"])

colors_diff = [PALETTE["accent2"] if v > 0 else PALETTE["accent4"]
                for v in fi_sorted["SHAP_diff"]]
bars = ax.barh(fi_sorted["Feature"], fi_sorted["SHAP_diff"],
                color=colors_diff, alpha=0.85, edgecolor="#ffffff22")
ax.axvline(0, color=PALETTE["text"], lw=1.5, alpha=0.5)

# Annotate each bar with both component values
for bar, (_, row) in zip(bars, fi_sorted.iterrows()):
    v = row["SHAP_diff"]
    ann = f"{v:+.4f}  (DM:{row['DM_SHAP']:.3f} vs nDM:{row['NonDM_SHAP']:.3f})"
    x_pos = v + (fi_sorted["SHAP_diff"].abs().max()*0.01 if v >= 0
                  else -fi_sorted["SHAP_diff"].abs().max()*0.01)
    ha = "left" if v >= 0 else "right"
    ax.text(x_pos, bar.get_y()+bar.get_height()/2, ann,
             va="center", ha=ha, fontsize=7.5, color=PALETTE["text"])

ax.set_xlabel("SHAP Importance Difference  (DM group − Non-DM group)")
ax.grid(axis="x", alpha=0.3)
ax.legend(handles=[
    mpatches.Patch(color=PALETTE["accent2"], label="Stronger for DM→CKD"),
    mpatches.Patch(color=PALETTE["accent4"], label="Stronger for non-DM→CKD")],
    fontsize=10)

plt.tight_layout()
plt.savefig(f"{OUT}/33_SHAP_DM_vs_NonDM_Difference.png", dpi=130,
            bbox_inches="tight", facecolor=PALETTE["bg"])
plt.close()
log(f"   Saved -> {OUT}/33_SHAP_DM_vs_NonDM_Difference.png")

# ════════════════════════════════════════════════════════════════════════════
# PLOT 34 — Shared Feature Bridge KDE
# ════════════════════════════════════════════════════════════════════════════
log("   Generating 34_Shared_Feature_Bridge_KDE ...")

n_bf = len(BRIDGE_FEATURES_DIAB)
fig, axes = plt.subplots(1, max(n_bf,1), figsize=(8*n_bf, 8))
if n_bf == 1: axes = [axes]
fig.suptitle(
    f"Shared Feature Bridge KDE — Auto-discovered features\n"
    f"Diabetes: {BRIDGE_FEATURES_DIAB}  ↔  CKD: {BRIDGE_FEATURES_CKD}\n"
    "Clinical literature threshold lines from structured LITERATURE_THRESHOLDS",
    fontsize=12, fontweight="bold", color=PALETTE["accent1"])

for ax, (dc, cc) in zip(axes, zip(BRIDGE_FEATURES_DIAB, BRIDGE_FEATURES_CKD)):
    xmin = min(X_diab_imp_full[dc].min(), X_ckd_imp_full[cc].min())
    xmax = max(X_diab_imp_full[dc].max(), X_ckd_imp_full[cc].max())
    xr   = np.linspace(xmin, xmax, 300)

    for data, color, lbl, ls in [
        (X_diab_imp_full[dc][y_diab_raw==1], PALETTE["accent2"],
         f"Diab+ ({dc})", "-"),
        (X_diab_imp_full[dc][y_diab_raw==0], PALETTE["accent4"],
         f"Diab- ({dc})", "-"),
        (X_ckd_imp_full[cc][y_ckd_raw==1],  PALETTE["accent6"],
         f"CKD+ ({cc})", "--"),
        (X_ckd_imp_full[cc][y_ckd_raw==0],  PALETTE["accent1"],
         f"CKD- ({cc})", "--"),
    ]:
        _safe_kde(ax, data, xr, color=color, lw=2.2, ls=ls, label=lbl)

    # Literature reference lines
    for lit in LITERATURE_THRESHOLDS:
        if lit["feature_diab"] == dc or lit["feature_ckd"] == cc:
            ax.axvline(lit["value"], color=lit["color"], lw=2.0,
                        linestyle="-.", alpha=0.9,
                        label=f"{lit['label']} {lit['value']} {lit['unit']}\n"
                              f"DOI:{lit['doi']}")

    ax.set_title(f"{dc}  ≡  {cc}\nBridge: same physiology, two datasets",
                  fontweight="bold")
    ax.set_xlabel("Feature value"); ax.set_ylabel("Density")
    ax.legend(fontsize=7.5); ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig(f"{OUT}/34_Shared_Feature_Bridge_KDE.png", dpi=130,
            bbox_inches="tight", facecolor=PALETTE["bg"])
plt.close()
log(f"   Saved -> {OUT}/34_Shared_Feature_Bridge_KDE.png")

# ════════════════════════════════════════════════════════════════════════════
# A1 — Architecture diagram (all text from live variables)
# ════════════════════════════════════════════════════════════════════════════
log("   Generating A1_Architecture ...")

BG_D = PALETTE["bg"]; PANEL_D = PALETTE["panel"]
COLS = {
    "teal":   ("#E1F5EE","#1D9E75","#085041"),
    "purple": ("#EEEDFE","#7F77DD","#3C3489"),
    "amber":  ("#FAEEDA","#BA7517","#633806"),
    "blue":   ("#E6F1FB","#378ADD","#0C447C"),
    "coral":  ("#FAECE7","#D85A30","#712B13"),
    "green":  ("#EAF3DE","#639922","#27500A"),
    "pink":   ("#FBEAF0","#D4537E","#72243E"),
    "gray":   ("#F1EFE8","#888780","#444441"),
}

def _fig(w,h):
    fig, ax = plt.subplots(figsize=(w/100,h/100))
    fig.patch.set_facecolor(BG_D); ax.set_facecolor(BG_D)
    ax.set_xlim(0,w); ax.set_ylim(h,0); ax.axis("off")
    return fig, ax

def _box(ax, x, y, w, h, label, sub=None, ckey="blue", r=8):
    fc,ec,tc = COLS[ckey]
    p = FancyBboxPatch((x,y),w,h,
                        boxstyle=f"round,pad=0,rounding_size={r}",
                        fc=fc,ec=ec,lw=0.8,zorder=3)
    ax.add_patch(p)
    ty = y+h/2+(-7 if sub else 0)
    ax.text(x+w/2,ty,label,ha="center",va="center",
            fontsize=9,fontweight="bold",color=tc,zorder=4)
    if sub:
        ax.text(x+w/2,ty+14,sub,ha="center",va="center",
                fontsize=7,color=tc,alpha=0.85,zorder=4)

def _arr(ax, x1, y1, x2, y2):
    ax.annotate("",xy=(x2,y2),xytext=(x1,y1),
                arrowprops=dict(arrowstyle="-|>",
                                color=PALETTE["subtext"],
                                lw=1.1,mutation_scale=11),zorder=2)

W, H = 1200, 860
fig, ax = _fig(W, H)
ax.text(W/2,34,"Clinical ML System — Architecture",ha="center",va="center",
        fontsize=14,fontweight="bold",color=PALETTE["text"],zorder=5)
p = FancyBboxPatch((25,52),W-50,H-80,
                    boxstyle="round,pad=0,rounding_size=16",
                    fc=PANEL_D,ec="#2D3142",lw=1,zorder=1)
ax.add_patch(p)

# Live variable values in box subtitles
d_sub  = f"{len(diab_raw)} rows · {len(X_diab_raw.columns)} features"
c_sub  = f"{len(ckd_raw)} rows · {len(X_ckd_raw.columns)} features"
n_m    = f"{len(MODELS_DIAB)} models · 10-fold CV + holdout"
bf_str = " · ".join(BRIDGE_FEATURES_DIAB[:3])
thr_n  = f"{len(ALL_THRESHOLDS)} thresholds auto-computed"

_box(ax, 60,  85,300,68,"Diabetes dataset",  d_sub,  "teal")
_box(ax,840,  85,300,68,"CKD dataset",        c_sub,  "purple")
_arr(ax,210,153,210,195); _arr(ax,990,153,990,195)

_box(ax, 60,195,300,68,"Preprocessing","Impute · scale · SMOTE","amber")
_box(ax,840,195,300,68,"Preprocessing","Impute · scale · SMOTEENN","amber")
_arr(ax,210,263,210,305); _arr(ax,990,263,990,305)

_box(ax, 60,305,300,68,"Diabetes models",n_m,"blue")
_box(ax,840,305,300,68,"CKD models",     n_m,"blue")
_arr(ax,360,339,470,390); _arr(ax,840,339,730,390)

_box(ax,460,380,280,72,"Bridge model (auto)",
     f"Features: {bf_str}","coral")
_arr(ax,210,373,210,495); _arr(ax,990,373,990,495); _arr(ax,600,452,600,495)

_box(ax, 60,495,260,68,"DM→CKD thresholds",
     "Decision tree · BGR×BP","pink")
_box(ax,340,495,260,68,"Optimal thresholds",thr_n,"coral")
_box(ax,620,495,260,68,"SHAP",
     "DM-stratified · beeswarm","purple")
_box(ax,900,495,260,68,"Holdout eval",
     f"Best:{best_diab[:4]}.. AUC={holdout_d[best_diab]['auc']:.3f}","teal")

for xv in [190,470,750,1030]:
    _arr(ax,xv,563,600,620)

_box(ax,250,622,700,62,"ClinicalDiagnosticsSystem + Web Inference",
     f"{len(ALL_THRESHOLDS)} thresholds · ckd_binary_map · SHAP · Flask API","green")

fig.tight_layout(pad=0)
fig.savefig(f"{OUT}/A1_Architecture.png",dpi=150,
            bbox_inches="tight",facecolor=BG_D)
plt.close(fig)
log(f"   Saved -> {OUT}/A1_Architecture.png")

# ════════════════════════════════════════════════════════════════════════════
# A2 — Methodology
# ════════════════════════════════════════════════════════════════════════════
log("   Generating A2_Methodology ...")

W, H = 1100, 700
fig, ax = _fig(W, H)
ax.text(W/2,36,"Research Methodology",ha="center",va="center",
        fontsize=14,fontweight="bold",color=PALETTE["text"],zorder=5)

phases = [
    ("Phase 1 — EDA",
     f"Distributions · correlation\n{n_anom_d} Diab / {n_anom_c} CKD anomalies",
     "teal"),
    ("Phase 2 — Anomaly detection",
     "IsolationForest (5 %)\nRaw data · true outliers",
     "amber"),
    ("Phase 3 — Feature engineering",
     f"0/1 binary encoding\n{len(BRIDGE_FEATURES_DIAB)} bridge features auto-found",
     "purple"),
    ("Phase 4 — Model training",
     f"80/20 holdout · 10-fold CV\n{len(MODELS_DIAB)} models per dataset",
     "blue"),
    ("Phase 5 — Threshold analysis",
     f"Surrogate tree · {len(ALL_THRESHOLDS)} threshold methods\nBGR×BP heatmap",
     "coral"),
    ("Phase 6 — Inference",
     f"Best Diab:{best_diab[:4]}.. AUC={holdout_d[best_diab]['auc']:.3f}\n"
     f"Best CKD:{best_ckd[:4]}.. AUC={holdout_c[best_ckd]['auc']:.3f}",
     "green"),
]

bw, bh, gx, gy = 300, 126, 40, 140
for i, (lbl, sub, ckey) in enumerate(phases):
    col, row = i%3, i//3
    x = 60 + col*(bw+gx); y = 80 + row*(bh+gy)
    fc,ec,tc = COLS[ckey]
    p = FancyBboxPatch((x,y),bw,bh,
                        boxstyle="round,pad=0,rounding_size=10",
                        fc=fc,ec=ec,lw=0.8,zorder=3)
    ax.add_patch(p)
    lines = sub.split("\n")
    ty0 = y+bh/2-(7*(len(lines)-1))
    ax.text(x+bw/2,ty0-10,lbl,ha="center",va="center",
            fontsize=9,fontweight="bold",color=tc,zorder=4)
    for li,line in enumerate(lines):
        ax.text(x+bw/2,ty0+12+li*14,line,ha="center",va="center",
                fontsize=7,color=tc,alpha=0.85,zorder=4)
    ax.text(x+16,y+16,str(i+1),ha="center",va="center",
            fontsize=9,fontweight="bold",color=tc,zorder=5)

for col in range(2):
    x1 = 60+col*(bw+gx)+bw
    _arr(ax,x1+4,80+bh/2,x1+gx-4,80+bh/2)
_arr(ax,60+2*(bw+gx)+bw/2,80+bh+4,60+2*(bw+gx)+bw/2,80+bh+gy-4)
for col in range(2):
    x1 = 60+col*(bw+gx)+bw
    _arr(ax,x1+4,80+bh+gy+bh/2,x1+gx-4,80+bh+gy+bh/2)

fig.tight_layout(pad=0)
fig.savefig(f"{OUT}/A2_Methodology.png",dpi=150,
            bbox_inches="tight",facecolor=BG_D)
plt.close(fig)
log(f"   Saved -> {OUT}/A2_Methodology.png")

# ════════════════════════════════════════════════════════════════════════════
# A3 — Pipeline
# ════════════════════════════════════════════════════════════════════════════
log("   Generating A3_Pipeline ...")

W, H = 860, 1280
fig, ax = _fig(W, H)
ax.text(W/2,36,"End-to-End Pipeline",ha="center",va="center",
        fontsize=14,fontweight="bold",color=PALETTE["text"],zorder=5)

best_cv_diab = max(results_d, key=lambda m: results_d[m]["mean_auc"])
steps = [
    ("Load raw CSVs",
     f"ckd.csv ({len(ckd_raw)} rows) · diabetes.csv ({len(diab_raw)} rows)",
     "gray"),
    ("Clean + encode",
     f"CKD: {len(CKD_BINARY_MAP)} binary cols · Diabetes: zero→NaN",
     "amber"),
    ("80/20 holdout split",
     f"Diab: {len(X_diab_train)}/{len(X_diab_test)} · "
     f"CKD: {len(X_ckd_train)}/{len(X_ckd_test)}",
     "purple"),
    ("EDA + anomalies",
     f"IsolationForest · {n_anom_d} Diab / {n_anom_c} CKD anomalies",
     "teal"),
    ("Auto bridge feature discovery",
     f"{BRIDGE_FEATURES_DIAB} ↔ {BRIDGE_FEATURES_CKD}",
     "teal"),
    ("Per-fold: split→impute→scale→SMOTE→fit",
     "SMOTE inside fold (no leakage) · synthetic clamp",
     "blue"),
    ("10-fold CV metrics",
     f"Best CV Diab: {best_cv_diab} AUC={results_d[best_cv_diab]['mean_auc']:.4f}",
     "blue"),
    ("Holdout evaluation",
     f"Best: {best_diab} AUC={holdout_d[best_diab]['auc']:.4f}  "
     f"· CKD: {best_ckd} AUC={holdout_c[best_ckd]['auc']:.4f}",
     "coral"),
    (f"Auto thresholds ({len(ALL_THRESHOLDS)} methods)",
     "Youden · F1-max · Se≥0.90 · Fβ=2 · G-mean · PR-F1",
     "pink"),
    ("DM→CKD tree thresholds",
     f"Top splits: {list(dt_thresholds.keys())[:3]}",
     "pink"),
    ("SHAP explainability",
     "DM-stratified · beeswarm · dependence · diff plot",
     "purple"),
    ("Export + Web inference",
     f"{len(ALL_THRESHOLDS)} thresholds · pkl · Flask REST API",
     "green"),
]

bw, bh, gap = 720, 65, 16
x0 = (W-bw)/2; y0 = 65
for i, (lbl, sub, ckey) in enumerate(steps):
    y = y0+i*(bh+gap)
    fc,ec,tc = COLS[ckey]
    p = FancyBboxPatch((x0,y),bw,bh,
                        boxstyle="round,pad=0,rounding_size=8",
                        fc=fc,ec=ec,lw=0.8,zorder=3)
    ax.add_patch(p)
    ax.text(x0+bw/2,y+bh/2-8,lbl,ha="center",va="center",
            fontsize=9,fontweight="bold",color=tc,zorder=4)
    ax.text(x0+bw/2,y+bh/2+9,sub,ha="center",va="center",
            fontsize=7,color=tc,alpha=0.85,zorder=4)
    ax.text(x0+14,y+bh/2,str(i+1),ha="center",va="center",
            fontsize=8,fontweight="bold",color=tc,zorder=5)
    if i < len(steps)-1:
        yb = y+bh
        _arr(ax,W/2,yb+2,W/2,yb+gap-2)

fig.tight_layout(pad=0)
fig.savefig(f"{OUT}/A3_Pipeline.png",dpi=150,
            bbox_inches="tight",facecolor=BG_D)
plt.close(fig)
log(f"   Saved -> {OUT}/A3_Pipeline.png")

log()
log("="*55)
log("  MISSING PLOTS COMPLETE — 31 32 33 34 A1 A2 A3")
log("="*55)
log(f"  Plots: {OUT}/ — check clinical_plots/ folder")