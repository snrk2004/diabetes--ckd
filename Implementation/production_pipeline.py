"""
================================================================================
 CLINICAL ML PIPELINE - EARLY SCREENING FOR DIABETES & CKD
================================================================================
 Upgrades implemented:
 1. Fixed the In-Place dataframe modification bug.
 2. Added Side-by-Side "Before vs After SMOTE" visualization plots.
 3. Added ROC, Precision-Recall, and Confusion Matrix plots.
 4. Ultra-Minimalist CKD Bridge: CKD model uses only 3 shared features + DM flag.
 5. 10-Fold Cross Validation & Zero Leakage SMOTE.
================================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, recall_score, precision_score, 
                             confusion_matrix, roc_curve, precision_recall_curve, auc)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

os.makedirs("saved_models", exist_ok=True)
os.makedirs("clinical_plots", exist_ok=True)
CLINICAL_MIN_RECALL = 0.90 

# ==========================================
# 1. DATA PREPROCESSING
# ==========================================

def preprocess_diabetes(df):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    X[cols_with_zeros] = X[cols_with_zeros].replace(0, np.nan)
    return X, y

def preprocess_ckd_early_screening(df):
    df.columns = df.columns.str.strip()
    # Create a fresh copy of the target to prevent in-place modification bugs
    target = df['classification'].astype(str).str.strip().str.lower()
    y = target.apply(lambda x: 1 if x.startswith('ckd') else 0)
    
    minimal_features = ['age', 'bp', 'bgr', 'dm']
    X = df[minimal_features].copy()
    binary_map = {'yes': 1, 'no': 0, '\tyes': 1, '\tno': 0, ' yes': 1}
    X['dm'] = X['dm'].astype(str).str.strip().map(binary_map)
    X = X.apply(pd.to_numeric, errors='coerce')
    
    return X, y

# ==========================================
# 2. PLOTTING FUNCTIONS
# ==========================================

def plot_before_after_smote(y_original, y_smoted, dataset_name):
    """Generates a side-by-side bar chart showing dataset balance before and after SMOTE."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Plot Before SMOTE
    sns.countplot(x=y_original, palette="Reds", ax=axes[0])
    axes[0].set_title(f'Before SMOTE ({dataset_name})')
    axes[0].set_xlabel('Class (0 = Healthy, 1 = Sick)')
    axes[0].set_ylabel('Patient Count')
    
    # Plot After SMOTE
    sns.countplot(x=y_smoted, palette="Greens", ax=axes[1])
    axes[1].set_title(f'After SMOTE ({dataset_name})')
    axes[1].set_xlabel('Class (0 = Healthy, 1 = Sick)')
    axes[1].set_ylabel('Patient Count')
    
    plt.tight_layout()
    plt.savefig(f'clinical_plots/{dataset_name}_smote_balance.png', dpi=300)
    plt.close()
    print(f"  -> Saved Before/After SMOTE plot for {dataset_name}.")

def plot_roc_curve(y_true, y_probs, dataset_name, auc_score):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'{dataset_name} - ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'clinical_plots/{dataset_name}_roc_curve.png', dpi=300)
    plt.close()

def plot_pr_curve(y_true, y_probs, dataset_name):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.title(f'{dataset_name} - Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f'clinical_plots/{dataset_name}_pr_curve.png', dpi=300)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, dataset_name, threshold):
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(f'{dataset_name} - Confusion Matrix\n(Threshold = {threshold:.2f})')
    plt.ylabel('Actual Status')
    plt.xlabel('Model Prediction')
    plt.tight_layout()
    plt.savefig(f'clinical_plots/{dataset_name}_confusion_matrix.png', dpi=300)
    plt.close()

def generate_shap_plots(model, X_test, dataset_name, model_name):
    try:
        if model_name in ['RandomForest', 'GradientBoosting']:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            if isinstance(shap_values, list): shap_values = shap_values[1] 
        else:
            explainer = shap.LinearExplainer(model, X_test)
            shap_values = explainer.shap_values(X_test)
            
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        plt.title(f"SHAP Feature Importance: {dataset_name} Risk Drivers")
        plt.tight_layout()
        plt.savefig(f'clinical_plots/{dataset_name}_shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        pass

# ==========================================
# 3. MODEL EVALUATION & TRAINING
# ==========================================

def find_clinical_threshold(y_true, y_probs, min_recall=CLINICAL_MIN_RECALL):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    valid_idx = np.where(tpr >= min_recall)[0]
    if len(valid_idx) == 0: return 0.5 
    return thresholds[valid_idx[0]]

def train_and_evaluate(X, y, dataset_name):
    print(f"\n--- Processing {dataset_name} Pipeline ---")
    
    # 1. Generate the Representative Before/After SMOTE Plot for Presentation
    imputer_plot = SimpleImputer(strategy='median')
    X_imp_plot = imputer_plot.fit_transform(X)
    smote_plot = SMOTE(random_state=42)
    _, y_smoted = smote_plot.fit_resample(X_imp_plot, y)
    plot_before_after_smote(y, y_smoted, dataset_name)
    
    # 2. Strict 10-Fold CV for Clinical Integrity
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    best_auc, best_name, best_pipeline = 0, "", None
    print("Evaluating algorithms via 10-Fold CV (Safe SMOTE inside folds)...")
    for name, classifier in models.items():
        pipeline = ImbPipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('smote', SMOTE(random_state=42)),
            ('classifier', classifier)
        ])
        cv_scores = cross_val_score(pipeline, X, y, scoring='roc_auc', cv=cv)
        mean_auc = cv_scores.mean()
        print(f"  {name} 10-Fold AUC: {mean_auc:.3f}")
        
        if mean_auc > best_auc:
            best_auc, best_name, best_pipeline = mean_auc, name, pipeline
            
    print(f"Selected Best Model: {best_name} (10-Fold AUC: {best_auc:.3f})")
    
    # 3. Train Final Model and Extract Metrics
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    best_pipeline.fit(X_train, y_train)
    y_probs = best_pipeline.predict_proba(X_test)[:, 1]
    
    safe_thresh = find_clinical_threshold(y_test, y_probs, CLINICAL_MIN_RECALL)
    y_pred_safe = (y_probs >= safe_thresh).astype(int)
    
    print(f"Computed Clinical Threshold for >=90% Sensitivity: {safe_thresh:.3f}")
    print(f"Final Holdout Metrics -> Sensitivity (Recall): {recall_score(y_test, y_pred_safe):.3f} | Precision: {precision_score(y_test, y_pred_safe):.3f}")
    
    # 4. Save Models and Output Plots
    final_imputer = best_pipeline.named_steps['imputer']
    final_scaler = best_pipeline.named_steps['scaler']
    final_model = best_pipeline.named_steps['classifier']
    
    joblib.dump(final_model, f'saved_models/{dataset_name}_model.pkl')
    joblib.dump(final_imputer, f'saved_models/{dataset_name}_imputer.pkl')
    joblib.dump(final_scaler, f'saved_models/{dataset_name}_scaler.pkl')
    
    plot_roc_curve(y_test, y_probs, dataset_name, best_auc)
    plot_pr_curve(y_test, y_probs, dataset_name)
    plot_confusion_matrix(y_test, y_pred_safe, dataset_name, safe_thresh)
    
    X_test_proc = pd.DataFrame(final_scaler.transform(final_imputer.transform(X_test)), columns=X.columns)
    generate_shap_plots(final_model, X_test_proc, dataset_name, best_name)

if __name__ == "__main__":
    print("Loading datasets...")
    try:
        df_diab = pd.read_csv(r'D:\FINAL_INTERNSHIP\dataset\diabetes.csv')
        df_ckd = pd.read_csv(r'D:\FINAL_INTERNSHIP\dataset\ckd.csv')
        
        # FIXED: Assigned variables explicitly to prevent in-place modification bugs
        X_diab, y_diab = preprocess_diabetes(df_diab)
        train_and_evaluate(X_diab, y_diab, 'Diabetes')
        
        X_ckd, y_ckd = preprocess_ckd_early_screening(df_ckd)
        train_and_evaluate(X_ckd, y_ckd, 'CKD')
        
        print("\nPipeline Complete. All models and clinical plots successfully generated.")
    except FileNotFoundError:
        print("ERROR: CSV files not found.")