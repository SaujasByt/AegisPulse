"""
Inference Engine for Patient Triage, Deterioration Risk, Service Duration,
and Bed / Ward Allocation.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

class HospitalPredictor:
    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.triage_model = joblib.load(os.path.join(models_dir, "triage_model.joblib"))
        self.risk_model = joblib.load(os.path.join(models_dir, "risk_model.joblib"))
        self.duration_model = joblib.load(os.path.join(models_dir, "duration_model.joblib"))
        self.ward_model = joblib.load(os.path.join(models_dir, "ward_model.joblib"))
        
        with open(os.path.join(models_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)

    @staticmethod
    def calculate_mews(vitals: dict) -> int:
        """Calculate Modified Early Warning Score (MEWS) from physiological parameters."""
        sbp = vitals.get("systolic_bp", 120)
        hr = vitals.get("heart_rate_bpm", 75)
        rr = vitals.get("respiratory_rate", 16)
        temp = vitals.get("temperature_c", 36.8)
        gcs = vitals.get("gcs", 15)

        mews = 0
        if sbp <= 70: mews += 3
        elif sbp <= 80: mews += 2
        elif sbp <= 100: mews += 1
        elif sbp >= 200: mews += 2

        if hr <= 40: mews += 2
        elif hr <= 50: mews += 1
        elif hr >= 130: mews += 3
        elif hr >= 111: mews += 2
        elif hr >= 101: mews += 1

        if rr < 9: mews += 2
        elif rr >= 30: mews += 3
        elif rr >= 25: mews += 2
        elif rr >= 21: mews += 1

        if temp < 35.0: mews += 2
        elif temp >= 38.5: mews += 2

        if gcs <= 8: mews += 3
        elif gcs <= 13: mews += 2
        elif gcs == 14: mews += 1

        return mews

    def predict(self, patient: dict) -> dict:
        """
        Run end-to-end ML inference on a patient record.
        Returns triage level, deterioration risk, duration estimate,
        bed allocation recommendation, and clinical flags.
        """
        # Ensure MEWS is calculated
        mews = patient.get("mews_score")
        if mews is None:
            mews = self.calculate_mews(patient)
            patient["mews_score"] = mews

        df_input = pd.DataFrame([{
            "age": float(patient.get("age", 45)),
            "comorbidity_count": int(patient.get("comorbidity_count", 0)),
            "heart_rate_bpm": float(patient.get("heart_rate_bpm", 78)),
            "systolic_bp": float(patient.get("systolic_bp", 120)),
            "diastolic_bp": float(patient.get("diastolic_bp", 80)),
            "respiratory_rate": float(patient.get("respiratory_rate", 16)),
            "spo2_percent": float(patient.get("spo2_percent", 98)),
            "temperature_c": float(patient.get("temperature_c", 36.8)),
            "pain_score": float(patient.get("pain_score", 0)),
            "gcs": float(patient.get("gcs", 15)),
            "mews_score": float(mews),
            "department": str(patient.get("department", "General Medicine")),
            "gender": str(patient.get("gender", "Female")),
            "mobility": str(patient.get("mobility", "Walking Independent"))
        }])

        # 1. Triage Acuity
        pred_urgency = int(self.triage_model.predict(df_input)[0])
        triage_probs = self.triage_model.predict_proba(df_input)[0]
        urgency_classes = self.triage_model.named_steps["classifier"].classes_
        triage_prob_dict = {
            f"ESI_{cls}": round(float(prob), 4)
            for cls, prob in zip(urgency_classes, triage_probs)
        }

        # 2. Deterioration Risk Score
        pred_risk = float(self.risk_model.predict(df_input)[0])
        pred_risk = round(float(np.clip(pred_risk, 0.0, 100.0)), 1)

        # 3. Consultation Service Duration
        df_dur = df_input.copy()
        df_dur["urgency_level"] = pred_urgency
        pred_duration = float(self.duration_model.predict(df_dur)[0])
        pred_duration = round(max(3.0, pred_duration), 1)

        # 4. Bed / Ward Allocation
        df_ward = df_input.copy()
        df_ward["urgency_level"] = pred_urgency
        df_ward["deterioration_risk_score"] = pred_risk
        pred_ward = str(self.ward_model.predict(df_ward)[0])
        ward_probs = self.ward_model.predict_proba(df_ward)[0]
        ward_classes = self.ward_model.named_steps["classifier"].classes_
        ward_prob_dict = {
            cls: round(float(prob), 4)
            for cls, prob in zip(ward_classes, ward_probs)
        }

        # -------------------------------------------------------------
        # Clinical Guardrails & Safety Overrides (ESI Clinical Protocol)
        # -------------------------------------------------------------
        complaint_lower = str(patient.get("chief_complaint", "")).lower()
        spo2_val = float(df_input["spo2_percent"].iloc[0])
        pain_val = float(df_input["pain_score"].iloc[0])
        hr_val = float(df_input["heart_rate_bpm"].iloc[0])
        sbp_val = float(df_input["systolic_bp"].iloc[0])
        gcs_val = float(df_input["gcs"].iloc[0])

        is_high_risk_cardiac = any(term in complaint_lower for term in [
            "chest pain", "angina", "stemi", "myocardial", "crushing", "diaphoresis", "cardiac", "retrosternal"
        ])
        is_severe_respiratory = any(term in complaint_lower for term in [
            "respiratory distress", "stridor", "cyanosis", "apnea", "croup", "dyspnea", "asthma"
        ])

        # Rule 1: High-acuity cardiac events combined with hypoxia (SpO2 <= 92%)
        # or severe pain (>= 8) MUST strictly be classified as ESI 2 or ESI 1 (NEVER ESI 3+)
        if (is_high_risk_cardiac and (spo2_val <= 92 or pain_val >= 8)) or (is_severe_respiratory and spo2_val <= 92):
            if pred_urgency > 2:
                pred_urgency = 2

        # Rule 2: If an ICU bed is recommended, ESI can NEVER be ESI 3 or higher (strictly ESI 1 or ESI 2)
        if "ICU" in pred_ward and pred_urgency > 2:
            pred_urgency = 2

        # Rule 3: Extreme physiological decompensation (SpO2 <= 85% or GCS <= 8 or SBP <= 75) -> ESI 1
        if spo2_val <= 85 or gcs_val <= 8 or sbp_val <= 75:
            pred_urgency = 1
        elif (spo2_val <= 90 or gcs_val <= 12 or sbp_val >= 200 or hr_val >= 140) and pred_urgency > 2:
            pred_urgency = 2

        # If acuity was escalated to ESI 1 or 2, ensure admission is directed away from Outpatient Discharge
        if pred_urgency in [1, 2] and pred_ward == "Outpatient Discharge":
            pred_ward = "ICU (Intensive Care Unit)" if (pred_urgency == 1 or spo2_val <= 90) else "HDU / Telemetry Step-Down"

        # Determine if admission is recommended
        admission_recommended = pred_ward != "Outpatient Discharge"

        # Clinical Red Flags
        flags = []
        if df_input["spo2_percent"].iloc[0] < 90:
            flags.append({"level": "CRITICAL", "message": f"Critical Hypoxia (SpO2: {df_input['spo2_percent'].iloc[0]}%)"})
        elif df_input["spo2_percent"].iloc[0] < 94:
            flags.append({"level": "WARNING", "message": f"Mild Hypoxemia (SpO2: {df_input['spo2_percent'].iloc[0]}%)"})

        if df_input["systolic_bp"].iloc[0] >= 180 or df_input["diastolic_bp"].iloc[0] >= 110:
            flags.append({"level": "CRITICAL", "message": f"Hypertensive Crisis ({df_input['systolic_bp'].iloc[0]}/{df_input['diastolic_bp'].iloc[0]} mmHg)"})
        elif df_input["systolic_bp"].iloc[0] <= 85:
            flags.append({"level": "CRITICAL", "message": f"Hypotensive Shock (SBP: {df_input['systolic_bp'].iloc[0]} mmHg)"})

        if df_input["heart_rate_bpm"].iloc[0] >= 125:
            flags.append({"level": "WARNING", "message": f"Marked Tachycardia ({df_input['heart_rate_bpm'].iloc[0]} bpm)"})
        elif df_input["heart_rate_bpm"].iloc[0] <= 45:
            flags.append({"level": "WARNING", "message": f"Severe Bradycardia ({df_input['heart_rate_bpm'].iloc[0]} bpm)"})

        if df_input["gcs"].iloc[0] <= 12:
            flags.append({"level": "CRITICAL", "message": f"Altered Consciousness (GCS: {int(df_input['gcs'].iloc[0])}/15)"})

        if df_input["pain_score"].iloc[0] >= 8:
            flags.append({"level": "WARNING", "message": f"Severe Acute Pain (Score: {int(df_input['pain_score'].iloc[0])}/10)"})

        if mews >= 5:
            flags.append({"level": "CRITICAL", "message": f"High MEWS Score ({mews}) - Trigger Rapid Response Assessment"})

        # Triage Acuity Labeling & SLA target wait times
        triage_labels = {
            1: {"label": "Immediate / Resuscitation", "target_wait_min": 0, "target_sla_display": "Immediate (< 1m)", "color": "#ef4444", "badge": "ESI 1"},
            2: {"label": "Emergent / Very Urgent", "target_wait_min": 10, "target_sla_display": "<= 10 mins", "color": "#f97316", "badge": "ESI 2"},
            3: {"label": "Urgent", "target_wait_min": 30, "target_sla_display": "<= 30 mins", "color": "#eab308", "badge": "ESI 3"},
            4: {"label": "Less Urgent", "target_wait_min": 60, "target_sla_display": "<= 60 mins", "color": "#10b981", "badge": "ESI 4"},
            5: {"label": "Non-Urgent / Fast-Track", "target_wait_min": 120, "target_sla_display": "<= 120 mins", "color": "#3b82f6", "badge": "ESI 5"},
        }

        acuity_info = triage_labels.get(pred_urgency, triage_labels[3])

        return {
            "predicted_urgency_level": pred_urgency,
            "acuity_label": acuity_info["label"],
            "target_wait_minutes": acuity_info["target_wait_min"],
            "target_sla_display": acuity_info["target_sla_display"],
            "acuity_color": acuity_info["color"],
            "acuity_badge": acuity_info["badge"],
            "triage_probabilities": triage_prob_dict,
            "deterioration_risk_score": pred_risk,
            "mews_score": mews,
            "predicted_service_minutes": pred_duration,
            "admission_recommended": admission_recommended,
            "recommended_ward": pred_ward,
            "ward_probabilities": ward_prob_dict,
            "clinical_flags": flags
        }

# Global singleton instance
_predictor = None

def get_predictor() -> HospitalPredictor:
    global _predictor
    if _predictor is None:
        _predictor = HospitalPredictor()
    return _predictor

if __name__ == "__main__":
    predictor = get_predictor()
    sample_patient = {
        "age": 63,
        "gender": "Male",
        "department": "Cardiology",
        "heart_rate_bpm": 118,
        "systolic_bp": 172,
        "diastolic_bp": 102,
        "respiratory_rate": 24,
        "spo2_percent": 91,
        "temperature_c": 37.8,
        "pain_score": 8,
        "gcs": 15,
        "comorbidity_count": 2,
        "mobility": "Wheelchair"
    }
    result = predictor.predict(sample_patient)
    print("Inference Result:\n", json.dumps(result, indent=2))
