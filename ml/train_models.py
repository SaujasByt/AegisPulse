"""
Machine Learning Training Pipeline for Hospital Patient Prioritization,
Queue Wait-Time Estimation, and Bed Allocation.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)

def train_all_models(data_path: str = "data/opd_patients.csv", output_dir: str = "ml/models"):
    """Train and persist all predictive models and evaluation metadata."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Feature definitions
    numeric_features = [
        "age", "comorbidity_count", "heart_rate_bpm", "systolic_bp",
        "diastolic_bp", "respiratory_rate", "spo2_percent", "temperature_c",
        "pain_score", "gcs", "mews_score"
    ]
    categorical_features = ["department", "gender", "mobility"]
    
    # Preprocessor pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )

    X = df[numeric_features + categorical_features]
    
    # -------------------------------------------------------------
    # 1. Triage Acuity Predictor (ESI Levels 1 to 5)
    # -------------------------------------------------------------
    print("\n--- Training Model 1: Triage Acuity Classifier (ESI 1-5) ---")
    y_triage = df["urgency_level"]
    X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
        X, y_triage, test_size=0.20, random_state=42, stratify=y_triage
    )

    triage_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.1, max_depth=4, random_state=42
        ))
    ])
    triage_model.fit(X_train_t, y_train_t)
    y_pred_t = triage_model.predict(X_test_t)

    triage_acc = accuracy_score(y_test_t, y_pred_t)
    triage_f1 = f1_score(y_test_t, y_pred_t, average="weighted")
    triage_report = classification_report(y_test_t, y_pred_t, output_dict=True)
    print(f"Triage Acuity Model Accuracy: {triage_acc * 100:.2f}% | Weighted F1: {triage_f1:.4f}")

    # -------------------------------------------------------------
    # 2. Deterioration Risk Score Regressor (0 - 100%)
    # -------------------------------------------------------------
    print("\n--- Training Model 2: Deterioration Risk Scorer ---")
    y_risk = df["deterioration_risk_score"]
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_risk, test_size=0.20, random_state=42
    )

    risk_model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1))
    ])
    risk_model.fit(X_train_r, y_train_r)
    y_pred_r = risk_model.predict(X_test_r)

    risk_mae = mean_absolute_error(y_test_r, y_pred_r)
    risk_r2 = r2_score(y_test_r, y_pred_r)
    print(f"Deterioration Risk MAE: {risk_mae:.2f}% | R2 Score: {risk_r2:.4f}")

    # -------------------------------------------------------------
    # 3. Service / Consultation Duration Regressor (Minutes)
    # -------------------------------------------------------------
    print("\n--- Training Model 3: Consultation Service Duration Regressor ---")
    # For duration, add urgency level as a feature
    X_duration = X.copy()
    X_duration["urgency_level"] = df["urgency_level"]
    num_feat_dur = numeric_features + ["urgency_level"]
    
    preprocessor_dur = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_feat_dur),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )

    y_dur = df["predicted_service_min"]
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
        X_duration, y_dur, test_size=0.20, random_state=42
    )

    duration_model = Pipeline([
        ("preprocessor", preprocessor_dur),
        ("regressor", RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1))
    ])
    duration_model.fit(X_train_d, y_train_d)
    y_pred_d = duration_model.predict(X_test_d)

    dur_mae = mean_absolute_error(y_test_d, y_pred_d)
    dur_r2 = r2_score(y_test_d, y_pred_d)
    print(f"Service Duration MAE: {dur_mae:.2f} mins | R2 Score: {dur_r2:.4f}")

    # -------------------------------------------------------------
    # 4. Admission Destination Ward Allocation Classifier
    # -------------------------------------------------------------
    print("\n--- Training Model 4: Bed / Ward Allocation Classifier ---")
    # Features include urgency and deterioration risk
    X_ward = X.copy()
    X_ward["urgency_level"] = df["urgency_level"]
    X_ward["deterioration_risk_score"] = df["deterioration_risk_score"]
    num_feat_ward = numeric_features + ["urgency_level", "deterioration_risk_score"]

    preprocessor_ward = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_feat_ward),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )

    y_ward = df["assigned_ward"]
    X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
        X_ward, y_ward, test_size=0.20, random_state=42, stratify=y_ward
    )

    ward_model = Pipeline([
        ("preprocessor", preprocessor_ward),
        ("classifier", RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42, n_jobs=-1))
    ])
    ward_model.fit(X_train_w, y_train_w)
    y_pred_w = ward_model.predict(X_test_w)

    ward_acc = accuracy_score(y_test_w, y_pred_w)
    ward_f1 = f1_score(y_test_w, y_pred_w, average="weighted")
    ward_report = classification_report(y_test_w, y_pred_w, output_dict=True)
    print(f"Bed Ward Allocation Accuracy: {ward_acc * 100:.2f}% | Weighted F1: {ward_f1:.4f}")

    # -------------------------------------------------------------
    # Feature Importances Extraction
    # -------------------------------------------------------------
    feature_names = []
    # numeric
    feature_names.extend(numeric_features)
    # categorical encoded
    cat_encoder = triage_model.named_steps["preprocessor"].named_transformers_["cat"]
    feature_names.extend(cat_encoder.get_feature_names_out(categorical_features).tolist())

    triage_gb = triage_model.named_steps["classifier"]
    raw_importances = triage_gb.feature_importances_
    sorted_idx = np.argsort(raw_importances)[::-1]

    feature_importance_list = [
        {"feature": feature_names[i], "importance": round(float(raw_importances[i]), 4)}
        for i in sorted_idx[:12]
    ]

    # -------------------------------------------------------------
    # Persist Models and Metadata
    # -------------------------------------------------------------
    joblib.dump(triage_model, os.path.join(output_dir, "triage_model.joblib"))
    joblib.dump(risk_model, os.path.join(output_dir, "risk_model.joblib"))
    joblib.dump(duration_model, os.path.join(output_dir, "duration_model.joblib"))
    joblib.dump(ward_model, os.path.join(output_dir, "ward_model.joblib"))

    metadata = {
        "dataset_records": len(df),
        "timestamp": pd.Timestamp.now().isoformat(),
        "metrics": {
            "triage_acuity": {
                "accuracy": round(float(triage_acc), 4),
                "f1_score": round(float(triage_f1), 4),
                "report": triage_report
            },
            "deterioration_risk": {
                "mae": round(float(risk_mae), 4),
                "r2": round(float(risk_r2), 4)
            },
            "service_duration": {
                "mae_minutes": round(float(dur_mae), 4),
                "r2": round(float(dur_r2), 4)
            },
            "ward_allocation": {
                "accuracy": round(float(ward_acc), 4),
                "f1_score": round(float(ward_f1), 4),
                "report": ward_report
            }
        },
        "top_features": feature_importance_list,
        "classes": {
            "triage_levels": [1, 2, 3, 4, 5],
            "wards": ward_model.named_steps["classifier"].classes_.tolist()
        }
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nAll models and metadata successfully saved to '{output_dir}'.")
    return metadata

if __name__ == "__main__":
    train_all_models()
