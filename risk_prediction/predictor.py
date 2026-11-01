"""
risk_prediction/predictor.py

Diabetes risk model utilities.
IMPORTANT: This is an educational risk-estimation model, not a diagnostic tool.
The Pima Indians dataset is limited and does not establish clinical validity.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("default")

MODEL_DIR = Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "xgb_diabetes_risk.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
IMPUTER_PATH = MODEL_DIR / "imputer.pkl"
DATA_PATH = MODEL_DIR / "pima_diabetes.csv"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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
    "Pregnancies": "Number of Pregnancies",
    "Glucose": "2-hour Plasma Glucose in OGTT (mg/dL)",
    "BloodPressure": "Diastolic Blood Pressure (mm Hg)",
    "SkinThickness": "Skin Fold Thickness (mm)",
    "Insulin": "2-hour Serum Insulin in OGTT (uU/mL)",
    "BMI": "Body Mass Index (kg/m²)",
    "DiabetesPedigreeFunction": "Diabetes Pedigree (Family History)",
    "Age": "Age (years)",
}

# These are illustrative model-score bands, NOT clinically validated thresholds.
# Binary model prediction uses CLASSIFICATION_THRESHOLD.
CLASSIFICATION_THRESHOLD = 0.50
RISK_THRESHOLDS = {
    (0.00, 0.30): ("Low Risk", "low", "#28a745"),
    (0.30, 0.50): ("Moderate Risk", "moderate", "#fd7e14"),
    (0.50, 0.75): ("High Risk", "high", "#dc3545"),
    (0.75, 1.01): ("Very High Risk", "very_high", "#7b0000"),
}

ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
# Broad sanity bounds catch obvious data-entry mistakes; they are not diagnostic cutoffs.
FEATURE_BOUNDS = {
    "Pregnancies": (0, 30),
    "Glucose": (1, 500),
    "BloodPressure": (1, 300),
    "SkinThickness": (1, 150),
    "Insulin": (1, 3000),
    "BMI": (1, 150),
    "DiabetesPedigreeFunction": (0, 20),
    "Age": (1, 120),
}


def _load_dataset_from_path(path: Path) -> pd.DataFrame:
    """Load a CSV and validate its expected schema."""
    df = pd.read_csv(path)
    # Also support the original headerless Pima CSV format.
    if not set(FEATURE_NAMES + ["Outcome"]).issubset(df.columns):
        if df.shape[1] == 9:
            df.columns = FEATURE_NAMES + ["Outcome"]
        else:
            raise ValueError(
                f"Dataset at {path} must contain the eight features and Outcome."
            )
    return df[FEATURE_NAMES + ["Outcome"]]


def download_pima_dataset() -> pd.DataFrame:
    """Load a real local dataset or download it; never silently train on fake data."""
    candidates = [
        DATA_PATH,
        Path(__file__).resolve().parents[1] / "pima_diabetes.csv",
        Path.cwd() / "pima_diabetes.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            df = _load_dataset_from_path(candidate)
            if len(df) < 100:
                raise ValueError(
                    f"Dataset {candidate} contains only {len(df)} rows; refusing to train."
                )
            return df

    url = (
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
        "pima-indians-diabetes.data.csv"
    )
    try:
        import requests

        response = requests.get(url, timeout=20)
        response.raise_for_status()
        from io import StringIO

        df = pd.read_csv(StringIO(response.text), header=None, names=FEATURE_NAMES + ["Outcome"])
        if len(df) < 100:
            raise ValueError("Downloaded dataset is unexpectedly small.")
        df.to_csv(DATA_PATH, index=False)
        return df
    except Exception as exc:
        raise RuntimeError(
            "Could not load the Pima dataset. Place pima_diabetes.csv in the project "
            "root or risk_prediction/models/, or check the network connection. "
            "Training on a small hardcoded fallback dataset is disabled."
        ) from exc


def _clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Validate labels/features and convert clinically implausible zero placeholders to NaN."""
    missing = set(FEATURE_NAMES + ["Outcome"]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    clean = df[FEATURE_NAMES + ["Outcome"]].copy()
    for col in FEATURE_NAMES + ["Outcome"]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    clean = clean.dropna(subset=["Outcome"])
    clean = clean[clean["Outcome"].isin([0, 1])]
    if clean.empty:
        raise ValueError("Dataset has no valid binary Outcome labels.")

    clean[ZERO_AS_MISSING] = clean[ZERO_AS_MISSING].replace(0, np.nan)
    X = clean[FEATURE_NAMES].replace([np.inf, -np.inf], np.nan)
    y = clean["Outcome"].astype(int)
    return X, y


def _safe_float(value: Any, default: float | None = None) -> float:
    """Convert a value to float; reject invalid values unless a default is explicit."""
    if value is None or (isinstance(value, str) and not value.strip()):
        if default is not None:
            return float(default)
        raise ValueError("A numeric value is required.")
    try:
        result = float(str(value).strip())
    except (ValueError, TypeError) as exc:
        if default is not None:
            return float(default)
        raise ValueError(f"Expected a numeric value, got {value!r}.") from exc
    if not np.isfinite(result):
        raise ValueError("NaN and infinite values are not valid patient inputs.")
    return result


def _validate_patient_params(patient_params: dict) -> dict[str, float]:
    if not isinstance(patient_params, dict):
        raise TypeError("patient_params must be a dictionary.")

    unknown = set(patient_params) - set(FEATURE_NAMES)
    if unknown:
        raise ValueError(f"Unknown patient feature(s): {sorted(unknown)}")

    cleaned: dict[str, float] = {}
    for feature, raw in patient_params.items():
        # Missing features are left out and imputed by the training-fitted imputer.
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            continue
        value = _safe_float(raw)
        low, high = FEATURE_BOUNDS[feature]
        if feature in ZERO_AS_MISSING and value == 0:
            # Common Pima dataset convention: zero means unavailable for these measures.
            cleaned[feature] = np.nan
            continue
        if not low <= value <= high:
            raise ValueError(
                f"{feature} must be between {low} and {high}; received {value}."
            )
        cleaned[feature] = value
    return cleaned


def _get_feature_vector(patient_params: dict) -> np.ndarray:
    """Return a validated raw feature row; missing measurements remain NaN."""
    cleaned = _validate_patient_params(patient_params)
    values = [cleaned.get(feature, np.nan) for feature in FEATURE_NAMES]
    return np.asarray(values, dtype=float).reshape(1, -1)


def train_model():
    """Train and save the model. Imputer is fitted on training data only."""
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    df = download_pima_dataset()
    X, y = _clean_dataset(df)
    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("Training requires at least two examples of each outcome class.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # Kept for compatibility with the existing model/scaler interface.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train_scaled, y_train)

    probabilities = model.predict_proba(X_test_scaled)[:, 1]
    predictions = (probabilities >= CLASSIFICATION_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall_sensitivity": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "specificity": round(float(tn / (tn + fp)) if (tn + fp) else 0.0, 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(y_test, probabilities)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "test_size": int(len(y_test)),
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "note": "Single stratified holdout; not clinical validation.",
    }

    with MODEL_PATH.open("wb") as f:
        pickle.dump(model, f)
    with SCALER_PATH.open("wb") as f:
        pickle.dump(scaler, f)
    with IMPUTER_PATH.open("wb") as f:
        pickle.dump(imputer, f)

    print(f"Model trained. Holdout metrics: {metrics}")
    return model, scaler, metrics


_loaded_artifacts = None


def load_artifacts():
    """Load frozen model artifacts. Training is offline-only via train_model()."""
    global _loaded_artifacts
    if _loaded_artifacts is not None:
        return _loaded_artifacts
    missing = [str(path) for path in (MODEL_PATH, SCALER_PATH, IMPUTER_PATH) if not path.exists()]
    if missing:
        raise RuntimeError(
            "Risk model artifacts are missing. Train offline with "
            "`python -m risk_prediction.predictor` before serving queries. "
            f"Missing: {missing}"
        )
    try:
        with MODEL_PATH.open("rb") as f:
            model = pickle.load(f)
        with SCALER_PATH.open("rb") as f:
            scaler = pickle.load(f)
        with IMPUTER_PATH.open("rb") as f:
            imputer = pickle.load(f)
    except Exception as exc:
        raise RuntimeError(
            "Risk model artifacts could not be loaded. Retrain offline with "
            "`python -m risk_prediction.predictor`."
        ) from exc
    _loaded_artifacts = (model, scaler, imputer)
    return _loaded_artifacts


def load_or_train():
    """Load saved artifacts. Does not train during a live request."""
    model, scaler, _ = load_artifacts()
    return model, scaler


def _load_imputer():
    _, _, imputer = load_artifacts()
    return imputer


def _transform_patient(patient_params: dict, scaler=None) -> tuple[np.ndarray, np.ndarray]:
    
    raw_values = _get_feature_vector(patient_params)
    raw = pd.DataFrame(raw_values, columns=FEATURE_NAMES)
    imputer = _load_imputer()
    imputed = imputer.transform(raw)
    if scaler is None:
        _, scaler = load_or_train()
    return raw, scaler.transform(imputed)


def predict_risk(patient_params: dict) -> dict:
    """Return a model score and category; this is not a medical diagnosis."""
    model, scaler = load_or_train()
    raw, X_scaled = _transform_patient(patient_params, scaler)
    risk_score = float(model.predict_proba(X_scaled)[0, 1])
    prediction = int(risk_score >= CLASSIFICATION_THRESHOLD)

    risk_category, risk_level, risk_color = "Unknown", "unknown", "#6c757d"
    for (lo, hi), (category, level, color) in RISK_THRESHOLDS.items():
        if lo <= risk_score < hi:
            risk_category, risk_level, risk_color = category, level, color
            break

    # Return the actual imputed model inputs so the API/UI can show what was used.
    imputed = _load_imputer().transform(raw)
    feature_values = dict(zip(FEATURE_NAMES, imputed[0].astype(float).tolist()))
    return {
        "risk_score": round(risk_score, 4),
        "risk_category": risk_category,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "prediction": prediction,
        "classification_threshold": CLASSIFICATION_THRESHOLD,
        "feature_values": feature_values,
        "disclaimer": "Educational estimate only; not a diagnosis or a substitute for clinical assessment.",
    }



def explain_features(patient_params: dict, top_n: int = 4) -> dict:
    """Explain model predictions using XGBoost's native TreeSHAP contributions."""
    import xgboost as xgb

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    model, scaler = load_or_train()
    raw, X_scaled = _transform_patient(patient_params, scaler)

    # Native XGBoost TreeSHAP avoids SHAP-version parsing incompatibilities.
    booster = model.get_booster()
    dmatrix = xgb.DMatrix(X_scaled)
    contributions = booster.predict(dmatrix, pred_contribs=True)

    contributions = np.asarray(contributions)
    if contributions.ndim != 2 or contributions.shape[1] != len(FEATURE_NAMES) + 1:
        raise RuntimeError(
            f"Unexpected TreeSHAP output shape: {contributions.shape}"
        )

    # The last column is the base value; the others are feature contributions.
    shap_values = contributions[0, :-1]
    base_value = float(contributions[0, -1])

    imputed = _load_imputer().transform(raw)[0]
    actual_values = dict(zip(FEATURE_NAMES, imputed.astype(float).tolist()))

    shap_dict = {
        feature: round(float(shap_values[i]), 4)
        for i, feature in enumerate(FEATURE_NAMES)
    }

    sorted_features = sorted(
        shap_dict.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    top_features = sorted_features[:min(top_n, len(sorted_features))]

    feature_items = []
    lines = []

    for feature, contribution in top_features:
        if contribution > 0:
            direction = "increases"
        elif contribution < 0:
            direction = "decreases"
        else:
            direction = "neutral"
        display = FEATURE_DISPLAY.get(feature, feature)
        actual = actual_values[feature]

        feature_items.append({
            "feature": feature,
            "display": display,
            "shap_value": contribution,
            "direction": direction,
            "actual": actual,
        })
        lines.append(
            f"{display} = {actual:g} {direction} the model score "
            f"(TreeSHAP: {contribution:+.3f})"
        )

    return {
        "shap_values": shap_dict,
        "top_features": feature_items,
        "base_value": round(base_value, 4),
        "explanation": "; ".join(lines),
        "disclaimer": (
            "Feature contributions explain this model's score, not causal "
            "effects or a medical diagnosis."
        ),
    }



def generate_patient_query(
    user_query: str,
    risk_result: dict,
    shap_result: dict,
    patient_params: dict,
) -> str:
    """Build a patient-specific retrieval query without claiming a diagnosis."""
    risk_category = risk_result.get("risk_category", "Unknown")
    risk_score = risk_result.get("risk_score")
    top_feature_strings = []
    for item in shap_result.get("top_features", []):
        feature = item.get("feature")
        display = item.get("display", feature)
        value = patient_params.get(feature, item.get("actual", "not provided"))
        top_feature_strings.append(f"{display}: {value}")

    context = ", ".join(top_feature_strings) if top_feature_strings else "No key features available"
    score_text = f"{risk_score:.2f}" if isinstance(risk_score, (int, float)) else "unavailable"
    return (
        f"{user_query} [Model-estimated category: {risk_category} "
        f"(score={score_text}; not a diagnosis); "
        f"available key parameters: {context}]"
    )


def run_patient_pipeline(
    user_query: str,
    patient_params: dict,
    top_shap: int = 4,
) -> dict:
    """Run risk prediction, explanation, and retrieval-query generation."""
    if not isinstance(user_query, str) or not user_query.strip():
        raise ValueError("user_query must be a non-empty string.")
    # Validate once so invalid values fail clearly rather than becoming zero.
    clean_params = _validate_patient_params(patient_params)
    risk_result = predict_risk(clean_params)
    shap_result = explain_features(clean_params, top_n=top_shap)
    patient_query = generate_patient_query(
        user_query, risk_result, shap_result, clean_params
    )
    return {
        "original_query": user_query,
        "patient_specific_query": patient_query,
        "risk_result": risk_result,
        "shap_result": shap_result,
    }


if __name__ == "__main__":
    print("Training XGBoost model...")
    _, _, metrics = train_model()
    print("Holdout metrics:", metrics)
    sample_patient = {
        "Pregnancies": 2,
        "Glucose": 148,
        "BloodPressure": 72,
        "SkinThickness": 35,
        "Insulin": None,
        "BMI": 33.6,
        "DiabetesPedigreeFunction": 0.627,
        "Age": 50,
    }
    result = run_patient_pipeline(
        user_query="What lifestyle factors should I discuss with my clinician?",
        patient_params=sample_patient,
    )
    print("Risk result:", result["risk_result"])
    print("Top SHAP features:", result["shap_result"]["top_features"])
