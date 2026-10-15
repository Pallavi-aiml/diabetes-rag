"""
risk_prediction/predictor.py

Patent Modules 102, 103, 104:
  102 — Risk Prediction Module    : XGBoost diabetes risk scorer
  103 — Feature Explanation Module: SHAP-based feature contribution
  104 — Query Generation Module   : patient-specific retrieval query builder

Dataset: Pima Indians Diabetes Dataset (free, UCI ML Repository)
         768 female patients, 8 clinical features, binary outcome
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import requests
from pathlib import Path

warnings.filterwarnings("ignore")

MODEL_DIR   = Path(__file__).parent / "models"
MODEL_PATH  = MODEL_DIR / "xgb_diabetes_risk.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
DATA_PATH   = MODEL_DIR / "pima_diabetes.csv"
MODEL_DIR.mkdir(exist_ok=True)

# ── Feature definitions ───────────────────────────────────────────────────────
FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

FEATURE_DISPLAY = {
    "Pregnancies":               "Number of Pregnancies",
    "Glucose":                   "Fasting Plasma Glucose (mg/dL)",
    "BloodPressure":             "Diastolic Blood Pressure (mm Hg)",
    "SkinThickness":             "Skin Fold Thickness (mm)",
    "Insulin":                   "Serum Insulin (μU/mL)",
    "BMI":                       "Body Mass Index (kg/m²)",
    "DiabetesPedigreeFunction":  "Diabetes Pedigree (Family History)",
    "Age":                       "Age (years)",
}

RISK_THRESHOLDS = {
    (0.00, 0.30): ("Low Risk",       "low",       "#28a745"),
    (0.30, 0.55): ("Moderate Risk",  "moderate",  "#fd7e14"),
    (0.55, 0.75): ("High Risk",      "high",      "#dc3545"),
    (0.75, 1.01): ("Very High Risk", "very_high", "#7b0000"),
}


# ── Dataset download ──────────────────────────────────────────────────────────

def download_pima_dataset() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)

    url  = ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
             "pima-indians-diabetes.data.csv")
    cols = FEATURE_NAMES + ["Outcome"]
    try:
        print("Downloading Pima Indians Diabetes Dataset...")
        df = pd.read_csv(url, header=None, names=cols)
        df.to_csv(DATA_PATH, index=False)
        print(f"  Saved ({len(df)} rows)")
        return df
    except Exception:
        # Bundled fallback dataset
        data = [
            [6,148,72,35,0,33.6,0.627,50,1],[1,85,66,29,0,26.6,0.351,31,0],
            [8,183,64,0,0,23.3,0.672,32,1],[1,89,66,23,94,28.1,0.167,21,0],
            [0,137,40,35,168,43.1,2.288,33,1],[5,116,74,0,0,25.6,0.201,30,0],
            [3,78,50,32,88,31.0,0.248,26,1],[10,115,0,0,0,35.3,0.134,29,0],
            [2,197,70,45,543,30.5,0.158,53,1],[8,125,96,0,0,0.0,0.232,54,1],
            [4,110,92,0,0,37.6,0.191,30,0],[10,168,74,0,0,38.0,0.537,34,1],
            [10,139,80,0,0,27.1,1.441,57,0],[1,189,60,23,846,30.1,0.398,59,1],
            [5,166,72,19,175,25.8,0.587,51,1],[7,100,0,0,0,30.0,0.484,32,1],
            [0,118,84,47,230,45.8,0.551,31,1],[7,107,74,0,0,29.6,0.254,31,1],
            [1,103,30,38,83,43.3,0.183,33,0],[1,115,70,30,96,34.6,0.529,32,1],
        ]
        df = pd.DataFrame(data, columns=cols)
        df.to_csv(DATA_PATH, index=False)
        return df


# ── Model training ────────────────────────────────────────────────────────────

def train_model():
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score

    df = download_pima_dataset()

    # Replace impossible zeros with median
    zero_cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
    df[zero_cols] = df[zero_cols].replace(0, np.nan)
    df[zero_cols] = df[zero_cols].fillna(df[zero_cols].median())

    X = df[FEATURE_NAMES].values
    y = df["Outcome"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    model.fit(X_train_sc, y_train,
              eval_set=[(X_test_sc, y_test)], verbose=False)

    acc = round(accuracy_score(y_test, model.predict(X_test_sc)), 4)
    auc = round(roc_auc_score(y_test, model.predict_proba(X_test_sc)[:,1]), 4)

    with open(MODEL_PATH,  "wb") as f: pickle.dump(model,  f)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)

    print(f"Model trained — Accuracy: {acc}  ROC-AUC: {auc}")
    return model, scaler, {"accuracy": acc, "roc_auc": auc}


def load_or_train():
    if MODEL_PATH.exists() and SCALER_PATH.exists():
        with open(MODEL_PATH,  "rb") as f: model  = pickle.load(f)
        with open(SCALER_PATH, "rb") as f: scaler = pickle.load(f)
        return model, scaler
    print("No saved model — training now...")
    model, scaler, _ = train_model()
    return model, scaler


def _get_feature_vector(patient_params: dict) -> np.ndarray:
    df_ref    = download_pima_dataset()
    zero_cols = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
    df_ref[zero_cols] = df_ref[zero_cols].replace(0, np.nan)
    medians   = df_ref[FEATURE_NAMES].median()
    values = []
    for f in FEATURE_NAMES:
        try:
            val = float(str(patient_params.get(f, medians[f])).strip())
        except (ValueError, TypeError):
            val = float(medians[f])
        values.append(val)
    return np.array(values).reshape(1, -1)



# ── Module 102: Risk Prediction ───────────────────────────────────────────────

def predict_risk(patient_params: dict) -> dict:
    """
    Patent Module 102 — Risk Prediction Module.
    XGBoost diabetes risk scorer.

    Args:
        patient_params: dict with clinical parameter keys

    Returns:
        risk_score, risk_category, risk_level, risk_color, prediction
    """
    model, scaler = load_or_train()
    X      = _get_feature_vector(patient_params)
    X_sc   = scaler.transform(X)

    risk_score = float(model.predict_proba(X_sc)[0, 1])
    prediction = int(model.predict(X_sc)[0])

    risk_category = "Unknown"
    risk_level    = "unknown"
    risk_color    = "#6c757d"
    for (lo, hi), (cat, lvl, col) in RISK_THRESHOLDS.items():
        if lo <= risk_score < hi:
            risk_category = cat
            risk_level    = lvl
            risk_color    = col
            break

    return {
        "risk_score":     round(risk_score, 4),
        "risk_category":  risk_category,
        "risk_level":     risk_level,
        "risk_color":     risk_color,
        "prediction":     prediction,
        "feature_values": dict(zip(FEATURE_NAMES, _get_feature_vector(patient_params).flatten().tolist())),
    }


# ── Module 103: SHAP Feature Explanation ─────────────────────────────────────

def explain_features(patient_params: dict, top_n: int = 4) -> dict:
    """
    Patent Module 103 — Feature Explanation Module.
    SHAP TreeExplainer to identify feature contributions.

    Returns:
        shap_values, top_features, base_value, explanation string
    """
    import shap

    model, scaler = load_or_train()
    X    = _get_feature_vector(patient_params)
    X_sc = scaler.transform(X)

    explainer  = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_sc)[0]
    base_value = float(explainer.expected_value)

    shap_dict = {
        feat: round(float(shap_vals[i]), 4)
        for i, feat in enumerate(FEATURE_NAMES)
    }

    sorted_feats = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top_features = sorted_feats[:top_n]

    # Build explanation string
    lines = []
    for feat, val in top_features:
        direction = "increases" if val > 0 else "decreases"
        display   = FEATURE_DISPLAY.get(feat, feat)
        actual    = patient_params.get(feat, "N/A")
        lines.append(
            f"{display} = {actual} {direction} risk "
            f"(SHAP: {val:+.3f})"
        )
    explanation = "; ".join(lines)

    return {
        "shap_values":  shap_dict,
        "top_features": [
            {
                "feature":    f,
                "display":    FEATURE_DISPLAY.get(f, f),
                "shap_value": v,
                "direction":  "increases" if v > 0 else "decreases",
                "actual":     patient_params.get(f, "N/A"),
            }
            for f, v in top_features
        ],
        "base_value":   round(base_value, 4),
        "explanation":  explanation,
    }


# ── Module 104: Patient-Specific Query Generation ────────────────────────────

def generate_patient_query(
    user_query:     str,
    risk_result:    dict,
    shap_result:    dict,
    patient_params: dict,
) -> str:
    """
    Patent Module 104 — Query Generation Module.
    Combines user query + risk score + SHAP features
    into a personalised retrieval query.
    """
    risk_category = risk_result["risk_category"]
    risk_score    = risk_result["risk_score"]

    top_feat_strings = []
    for item in shap_result["top_features"]:
        feat = item["feature"]
        val  = patient_params.get(feat, "N/A")
        disp = item["display"]
        top_feat_strings.append(f"{disp}: {val}")

    clinical_context = ", ".join(top_feat_strings)
    enriched_query   = (
        f"{user_query} "
        f"[Patient: {risk_category} (score={risk_score:.2f}); "
        f"Key parameters: {clinical_context}]"
    )
    return enriched_query


# ── Full pipeline: 102 + 103 + 104 ───────────────────────────────────────────

def run_patient_pipeline(
    user_query:     str,
    patient_params: dict,
    top_shap:       int = 4,
) -> dict:
    """
    Run all three patent modules in sequence.
    Returns enriched query + risk result + SHAP result.
    """
    risk_result   = predict_risk(patient_params)
    shap_result   = explain_features(patient_params, top_n=top_shap)
    patient_query = generate_patient_query(
        user_query, risk_result, shap_result, patient_params
    )
    return {
        "original_query":         user_query,
        "patient_specific_query": patient_query,
        "risk_result":            risk_result,
        "shap_result":            shap_result,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Training XGBoost model...")
    train_model()

    patient = {
        "Pregnancies": 2, "Glucose": 148, "BloodPressure": 72,
        "SkinThickness": 35, "Insulin": 0, "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627, "Age": 50,
    }

    result = run_patient_pipeline(
        user_query="What medications are recommended for my diabetes?",
        patient_params=patient,
    )

    print(f"\nOriginal Query  : {result['original_query']}")
    print(f"Enriched Query  : {result['patient_specific_query']}")
    print(f"Risk Score      : {result['risk_result']['risk_score']}")
    print(f"Risk Category   : {result['risk_result']['risk_category']}")
    print(f"\nTop SHAP Features:")
    for item in result["shap_result"]["top_features"]:
        print(f"  {item['display']}: SHAP = {item['shap_value']:+.4f} "
              f"({item['direction']} risk)")
