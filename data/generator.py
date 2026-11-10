"""
Synthetic OPD Dataset Generator
Calibrated with clinical guidelines (Emergency Severity Index - ESI, MEWS)
and public hospital data flow distributions (NHAMCS & MIMIC-IV ED flow benchmarks).
"""

import os
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

DEPARTMENTS = [
    "Emergency / Trauma",
    "Cardiology",
    "Pulmonology",
    "General Medicine",
    "Orthopedics",
    "Pediatrics",
    "Neurology"
]

CHIEF_COMPLAINTS = {
    "Emergency / Trauma": [
        ("Multiple Trauma / Road Incident", 1, 0.85),
        ("Uncontrolled Bleeding / Laceration", 2, 0.45),
        ("Severe Chemical Burn", 2, 0.60),
        ("Head Injury with Loss of Consciousness", 1, 0.90),
        ("Acute Open Fracture", 2, 0.70),
        ("Minor Contusion / Superficial Wound", 4, 0.05)
    ],
    "Cardiology": [
        ("Severe Crushing Chest Pain radiating to arm", 1, 0.88),
        ("Palpitations with Near-Syncope", 2, 0.55),
        ("Shortness of breath on exertion / Orthopnea", 3, 0.40),
        ("Uncontrolled Severe Hypertension (SBP > 200)", 2, 0.65),
        ("Mild Ankle Edema / Chronic Follow-up", 4, 0.10),
        ("Routine BP Check / Medication Refill", 5, 0.02)
    ],
    "Pulmonology": [
        ("Acute Respiratory Distress / Cyanosis", 1, 0.92),
        ("Severe Wheezing / Acute Asthma Exacerbation", 2, 0.60),
        ("Hemoptysis (Coughing blood)", 2, 0.75),
        ("Productive Cough with High Fever", 3, 0.35),
        ("Mild Chronic Cough / Smoking Review", 4, 0.08),
        ("Post-Infection Fatigue Checkup", 5, 0.02)
    ],
    "General Medicine": [
        ("Suspected Sepsis / Rigors & Lethargy", 1, 0.90),
        ("Severe Dehydration / Intractable Vomiting", 2, 0.50),
        ("High Grade Fever for 5 Days with Chills", 3, 0.30),
        ("Abdominal Pain (Suspected Appendicitis)", 2, 0.80),
        ("Mild Viral Infection / Sore Throat", 4, 0.05),
        ("General Weakness / Routine Blood Work", 5, 0.01)
    ],
    "Orthopedics": [
        ("Crush Injury with Neurovascular Compromise", 1, 0.85),
        ("Dislocated Shoulder with Severe Pain", 2, 0.40),
        ("Closed Ankle Fracture / Deformity", 3, 0.35),
        ("Acute Lumbar Sprain with Sciatica", 3, 0.15),
        ("Mild Joint Stiffness / Chronic Osteoarthritis", 4, 0.05),
        ("Cast Removal / Rehabilitation Follow-up", 5, 0.01)
    ],
    "Pediatrics": [
        ("Infant Febrile Convulsion / Lethargy", 1, 0.95),
        ("Severe Stridor / Croup with Retractions", 2, 0.70),
        ("Acute Gastroenteritis with Sunken Eyes", 2, 0.55),
        ("Moderate Fever (39°C) with Earache", 3, 0.20),
        ("Mild Diaper Rash / Teething", 4, 0.02),
        ("Developmental Milestone / Vaccination", 5, 0.01)
    ],
    "Neurology": [
        ("Acute Ischemic Stroke Signs (FAST positive)", 1, 0.95),
        ("Status Epilepticus / Recurring Seizures", 1, 0.92),
        ("Worst Headache of Life (Thunderclap)", 2, 0.80),
        ("Unilateral Weakness / Numbness (Transient)", 2, 0.60),
        ("Chronic Tension-Type Headache", 4, 0.05),
        ("Tremor Follow-up / Parkinson Review", 5, 0.02)
    ]
}

COMORBIDITIES_LIST = [
    "Hypertension",
    "Type 2 Diabetes",
    "COPD / Chronic Lung Disease",
    "Coronary Artery Disease",
    "Chronic Kidney Disease",
    "Immunocompromised / Oncology",
    "Prior Stroke / TIA"
]

WARDS = [
    "ICU (Intensive Care Unit)",
    "HDU / Telemetry Step-Down",
    "General Medical Ward",
    "Surgical Inpatient Ward",
    "Pediatric Ward",
    "Day Observation Bay",
    "Outpatient Discharge"
]


def generate_patient(patient_id: int, base_time: datetime) -> dict:
    """Generate a single realistic patient encounter record."""
    dept = random.choices(
        DEPARTMENTS,
        weights=[0.18, 0.16, 0.14, 0.22, 0.12, 0.10, 0.08]
    )[0]

    # Select complaint
    complaint_data = random.choice(CHIEF_COMPLAINTS[dept])
    complaint_name, baseline_urgency, admission_bias = complaint_data

    # Demographics
    if dept == "Pediatrics":
        age = random.randint(0, 14)
    else:
        # Bimodal age distribution typical for hospital OPD
        if random.random() < 0.65:
            age = int(np.random.normal(58, 14))
        else:
            age = int(np.random.normal(30, 8))
        age = max(15, min(95, age))

    gender = random.choice(["Male", "Female"])

    # Comorbidities count & selection based on age
    comorb_prob = min(0.85, max(0.05, (age - 20) / 70.0))
    patient_comorbidities = []
    if age > 14 and random.random() < comorb_prob:
        num_comorb = random.choices([1, 2, 3, 4], weights=[0.45, 0.30, 0.18, 0.07])[0]
        patient_comorbidities = random.sample(COMORBIDITIES_LIST, min(num_comorb, len(COMORBIDITIES_LIST)))

    comorbidity_count = len(patient_comorbidities)

    # Triage Acuity (ESI: 1=Immediate, 2=Emergent, 3=Urgent, 4=Less Urgent, 5=Non-urgent)
    # Slight perturbation around baseline urgency
    noise = np.random.choice([-1, 0, 1], p=[0.12, 0.76, 0.12])
    urgency_level = int(np.clip(baseline_urgency + noise, 1, 5))

    # Physiological Vitals correlated with urgency
    if urgency_level == 1:
        heart_rate = int(random.choice([np.random.normal(138, 12), np.random.normal(42, 6)])) # Tachycardia or severe Bradycardia
        sbp = int(random.choice([np.random.normal(78, 10), np.random.normal(210, 15)]))       # Shock or Hypertensive Crisis
        dbp = int(sbp * random.uniform(0.55, 0.68))
        resp_rate = int(random.choice([np.random.normal(34, 4), np.random.normal(8, 2)]))
        spo2 = int(np.clip(np.random.normal(84, 5), 65, 93))
        temp_c = round(float(random.choice([np.random.normal(39.4, 0.6), np.random.normal(35.2, 0.4)])), 1)
        pain_score = random.randint(7, 10)
        gcs = random.choices([3, 6, 8, 11, 13], weights=[0.2, 0.25, 0.25, 0.2, 0.1])[0]
        mobility = random.choice(["Stretcher / Immobilized", "Wheelchair"])
    elif urgency_level == 2:
        heart_rate = int(np.clip(np.random.normal(114, 10), 95, 145))
        sbp = int(random.choice([np.random.normal(92, 8), np.random.normal(175, 14)]))
        dbp = int(sbp * random.uniform(0.58, 0.70))
        resp_rate = int(np.clip(np.random.normal(26, 3), 22, 36))
        spo2 = int(np.clip(np.random.normal(91, 2.5), 87, 95))
        temp_c = round(float(np.random.normal(38.6, 0.6)), 1)
        pain_score = random.randint(6, 10)
        gcs = random.choices([13, 14, 15], weights=[0.25, 0.35, 0.40])[0]
        mobility = random.choice(["Wheelchair", "Assisted Walk", "Stretcher / Immobilized"])
    elif urgency_level == 3:
        heart_rate = int(np.clip(np.random.normal(88, 9), 70, 108))
        sbp = int(np.clip(np.random.normal(136, 12), 110, 160))
        dbp = int(sbp * random.uniform(0.60, 0.72))
        resp_rate = int(np.clip(np.random.normal(19, 2), 16, 24))
        spo2 = int(np.clip(np.random.normal(96, 1.5), 93, 98))
        temp_c = round(float(np.random.normal(37.7, 0.5)), 1)
        pain_score = random.randint(4, 7)
        gcs = 15
        mobility = random.choice(["Walking Independent", "Assisted Walk"])
    elif urgency_level == 4:
        heart_rate = int(np.clip(np.random.normal(76, 7), 62, 90))
        sbp = int(np.clip(np.random.normal(122, 10), 105, 140))
        dbp = int(sbp * random.uniform(0.62, 0.72))
        resp_rate = int(np.clip(np.random.normal(16, 2), 14, 20))
        spo2 = int(np.clip(np.random.normal(98, 1.0), 96, 100))
        temp_c = round(float(np.random.normal(37.0, 0.3)), 1)
        pain_score = random.randint(1, 4)
        gcs = 15
        mobility = "Walking Independent"
    else:  # urgency 5
        heart_rate = int(np.clip(np.random.normal(72, 6), 60, 84))
        sbp = int(np.clip(np.random.normal(118, 8), 100, 130))
        dbp = int(sbp * random.uniform(0.64, 0.72))
        resp_rate = int(np.clip(np.random.normal(15, 1.5), 12, 18))
        spo2 = int(np.clip(np.random.normal(99, 0.8), 97, 100))
        temp_c = round(float(np.random.normal(36.7, 0.2)), 1)
        pain_score = random.randint(0, 2)
        gcs = 15
        mobility = "Walking Independent"

    # Clamp vitals to physiologically realistic bounds
    heart_rate = int(np.clip(heart_rate, 35, 210))
    sbp = int(np.clip(sbp, 60, 240))
    dbp = int(np.clip(dbp, 35, 140))
    resp_rate = int(np.clip(resp_rate, 6, 50))
    spo2 = int(np.clip(spo2, 60, 100))
    temp_c = float(np.clip(temp_c, 34.0, 41.5))

    # MEWS (Modified Early Warning Score) calculation approximation
    mews = 0
    if sbp <= 70: mews += 3
    elif sbp <= 80: mews += 2
    elif sbp <= 100: mews += 1
    elif sbp >= 200: mews += 2

    if heart_rate <= 40: mews += 2
    elif heart_rate <= 50: mews += 1
    elif heart_rate >= 130: mews += 3
    elif heart_rate >= 111: mews += 2
    elif heart_rate >= 101: mews += 1

    if resp_rate < 9: mews += 2
    elif resp_rate >= 30: mews += 3
    elif resp_rate >= 25: mews += 2
    elif resp_rate >= 21: mews += 1

    if temp_c < 35.0: mews += 2
    elif temp_c >= 38.5: mews += 2

    if gcs <= 8: mews += 3
    elif gcs <= 13: mews += 2
    elif gcs == 14: mews += 1

    # Continuous deterioration risk score (0 to 100%)
    risk_factor = (
        (6 - urgency_level) * 16.0 +
        mews * 5.0 +
        (comorbidity_count * 3.5) +
        (0.25 * (age if age > 60 else (15 - age if age < 5 else 0))) +
        (100 - spo2) * 1.5 +
        np.random.normal(0, 3)
    )
    deterioration_risk = round(float(np.clip(risk_factor, 1.0, 99.5)), 1)

    # Doctor Service / Consultation Duration (minutes)
    # ESI 1: Immediate resuscitation (requires longer stabilization ~ 35-60 mins)
    # ESI 2: High complexity (25-45 mins)
    # ESI 3: Moderate (15-28 mins)
    # ESI 4: Simple (8-16 mins)
    # ESI 5: Fast track (4-10 mins)
    duration_mean = {1: 42, 2: 32, 3: 20, 4: 12, 5: 7}[urgency_level]
    duration_std = {1: 8, 2: 6, 3: 4, 4: 3, 5: 2}[urgency_level]
    service_duration_min = round(float(np.clip(
        np.random.normal(duration_mean + comorbidity_count * 1.5, duration_std),
        3, 85
    )), 1)

    # Admission Outcome & Ward Allocation
    # Driven by urgency, deterioration risk, comorbidity, and admission bias
    admission_prob = (
        (6 - urgency_level) * 0.16 +
        (deterioration_risk / 100.0) * 0.32 +
        admission_bias * 0.38 +
        (comorbidity_count * 0.05)
    )
    admission_prob = float(np.clip(admission_prob, 0.02, 0.98))
    will_admit = random.random() < admission_prob

    if not will_admit:
        assigned_ward = "Outpatient Discharge"
        stay_hours = 0.0
    else:
        if urgency_level == 1 or deterioration_risk >= 82:
            assigned_ward = "ICU (Intensive Care Unit)"
            stay_hours = round(float(np.random.normal(72, 20)), 1)
        elif dept == "Pediatrics":
            assigned_ward = "Pediatric Ward"
            stay_hours = round(float(np.random.normal(36, 10)), 1)
        elif dept in ["Orthopedics", "Emergency / Trauma"] and ("Fracture" in complaint_name or "Trauma" in complaint_name or "Bleeding" in complaint_name or "Dislocated" in complaint_name):
            assigned_ward = "Surgical Inpatient Ward"
            stay_hours = round(float(np.random.normal(48, 14)), 1)
        elif urgency_level == 2 or deterioration_risk >= 60:
            assigned_ward = "HDU / Telemetry Step-Down"
            stay_hours = round(float(np.random.normal(40, 12)), 1)
        elif urgency_level == 3 and random.random() < 0.50:
            assigned_ward = "Day Observation Bay"
            stay_hours = round(float(np.random.normal(8, 2)), 1)
        else:
            assigned_ward = "General Medical Ward"
            stay_hours = round(float(np.random.normal(36, 12)), 1)

    stay_hours = max(0.0, stay_hours)

    # Arrival time modeling (Poisson arrival waves throughout 24-hr day)
    # Peak OPD hours: 08:00 - 12:00, 14:00 - 17:00
    arrival_time = base_time + timedelta(
        days=random.randint(0, 14),
        hours=random.choices(
            range(24),
            weights=[
                1, 1, 1, 1, 1, 2,  # 00:00 - 05:00
                4, 9, 14, 15, 12, 10, # 06:00 - 11:00 (Morning Peak)
                7, 6, 11, 13, 10, 8,  # 12:00 - 17:00 (Afternoon Peak)
                6, 5, 4, 3, 2, 1   # 18:00 - 23:00 (Evening)
            ]
        )[0],
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )

    return {
        "patient_id": f"PAT-{patient_id:05d}",
        "arrival_time": arrival_time.strftime("%Y-%m-%d %H:%M:%S"),
        "department": dept,
        "chief_complaint": complaint_name,
        "age": age,
        "gender": gender,
        "comorbidities": ", ".join(patient_comorbidities) if patient_comorbidities else "None",
        "comorbidity_count": comorbidity_count,
        "heart_rate_bpm": heart_rate,
        "systolic_bp": sbp,
        "diastolic_bp": dbp,
        "respiratory_rate": resp_rate,
        "spo2_percent": spo2,
        "temperature_c": temp_c,
        "pain_score": pain_score,
        "gcs": gcs,
        "mobility": mobility,
        "mews_score": mews,
        "deterioration_risk_score": deterioration_risk,
        "urgency_level": urgency_level,       # ESI 1 - 5
        "predicted_service_min": service_duration_min,
        "admission_required": 1 if will_admit else 0,
        "assigned_ward": assigned_ward,
        "estimated_stay_hours": stay_hours
    }


def generate_opd_dataset(n_samples: int = 5000, output_path: str = "data/opd_patients.csv") -> pd.DataFrame:
    """Generate and save the full synthetic OPD cohort dataset."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    base_time = datetime(2026, 10, 1, 8, 0, 0)
    
    records = [generate_patient(i + 1, base_time) for i in range(n_samples)]
    df = pd.DataFrame(records)
    
    # Sort chronologically by arrival time
    df = df.sort_values(by="arrival_time").reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {len(df)} synthetic patient records saved to {output_path}")
    return df


if __name__ == "__main__":
    generate_opd_dataset(5000)
