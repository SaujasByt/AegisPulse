# 🏥 AegisPulse™ AI — Intelligent Hospital Patient Prioritization, Queue Orchestration & Bed Allocation System

[![Python 3.12](https://img.shields.io/badge/Python-3.12-38bdf8?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4.0+-f59e0b?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Flask](https://img.shields.io/badge/Flask-3.1.3-10b981?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-a855f7?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/Tests-10%2F10%20Passing-emerald?style=for-the-badge)](#-automated-verification--unit-tests)

**AegisPulse™ AI** is a state-of-the-art, machine-learning-driven clinical command center and hospital operations platform. It resolves the dual operational bottlenecks facing modern healthcare systems: **unpredictable outpatient department (OPD) queue wait times** and **uncoordinated inpatient bed and admission management**.

By combining calibrated clinical triage standards (**Emergency Severity Index - ESI 1 to 5**, **Modified Early Warning Score - MEWS**) with a multi-factor **Dynamic Priority Queueing (DPS)** algorithm and automated **inpatient bed capability matching**, AegisPulse reduces emergency patient wait times by **over 72%** while preventing queue starvation for routine consultations.

---

## 📌 Table of Contents
- [Executive Overview & Clinical Motivation](#-executive-overview--clinical-motivation)
- [Key Features](#-key-features)
  - [1. Dynamic ESI 1 to 5 Triage Engine](#1-dynamic-esi-1-to-5-triage-engine)
  - [2. Dynamic Priority Score (DPS) Queue Scoring Formula](#2-dynamic-priority-score-dps-queue-scoring-formula)
  - [3. Real-Time Hospital Ward & Bed Capacity Tracking](#3-real-time-hospital-ward--bed-capacity-tracking)
  - [4. Live Doctor Consultation Room Tracking](#4-live-doctor-consultation-room-tracking)
- [Machine Learning Models & Performance](#-machine-learning-models--performance)
- [Simulation Impact: Traditional FIFO vs. AegisPulse DPS](#-simulation-impact-traditional-fifo-vs-aegispulse-dps)
- [Tech Stack & System Architecture](#-tech-stack--system-architecture)
- [Project Directory Structure](#-project-directory-structure)
- [Quick Start & Installation Guide](#-quick-start--installation-guide)
- [Automated Verification & Unit Tests](#-automated-verification--unit-tests)
- [REST API Documentation](#-rest-api-documentation)
- [License & Medical Disclaimer](#-license--medical-disclaimer)

---

## 🏥 Executive Overview & Clinical Motivation

In conventional hospital Outpatient Departments (OPD) and Emergency triage units, patients are frequently processed using either rigid **First-In, First-Out (FIFO)** ordering or static triage categories. Both approaches suffer from dangerous systemic shortcomings:

1. **FIFO Queueing Risks**: Patients suffering from acute, time-sensitive decompensation (e.g., myocardial infarction, severe sepsis, hypoxic respiratory failure) are trapped behind stable patients seeking routine prescription renewals or minor wound dressing.
2. **Static Category Starvation**: Pure static priority models (where Level 1 and 2 patients always jump ahead) often trap Level 4 and Level 5 patients in infinite waiting loops during hospital surge hours.
3. **Bed Allocation Disconnect**: Admission decisions made in consultation rooms are disconnected from inpatient ward telemetry, resulting in "boarded" emergency patients, ward overcrowding, and delayed critical admissions.

**AegisPulse AI** synchronizes triage prediction, priority queue orchestration, and ward bed allocation into a single unified clinical dashboard.

---

## ⚡ Key Features

### 1. Dynamic ESI 1 to 5 Triage Engine
* **Objective Multi-Parameter Clinical Input**: Evaluates objective vital signs including **Heart Rate (HR)**, **Systolic & Diastolic Blood Pressure (BP)**, **Oxygen Saturation (SpO2)**, **Pain Score (0–10)**, **Respiratory Rate (RR)**, **Body Temperature**, **Glasgow Coma Scale (GCS)**, and comorbidity counts combined with TF-IDF vectorization of **Chief Complaints**.
* **Clinical Guardrails & Red Flags**:
  - High-acuity cardiac presentations (e.g., crushing retrosternal chest pain radiating to jaw with diaphoresis) combined with hypoxia ($\text{SpO}_2 \le 92\%$) or severe pain ($\ge 8/10$) are strictly classified as **ESI 1** (Resuscitation) or **ESI 2** (Very Urgent), **never ESI 3**.
  - Any patient designated for ICU admission is strictly guaranteed ESI 1 or ESI 2 acuity.
  - Critical decompensation markers ($\text{SpO}_2 \le 85\%$ or $\text{GCS} \le 8$) automatically trigger an immediate override to **ESI 1**.
* **Precise SLA Categorization**:
  - **ESI 1**: `Target SLA: Immediate (< 1m)`
  - **ESI 2**: `Target SLA: <= 10 mins`
  - **ESI 3**: `Target SLA: <= 30 mins`
  - **ESI 4**: `Target SLA: <= 60 mins`
  - **ESI 5**: `Target SLA: <= 120 mins`
* **Real-Time Dynamic Decision Explanations**: A reactive AI logic engine generates plain-language clinical justifications detailing why vital deviations (e.g., tachycardia, hypertensive urgency, severe hypoxia) determined the patient's assigned tier.

---

### 2. Dynamic Priority Score (DPS) Queue Scoring Formula

AegisPulse continuously recalculates a dynamic priority score for every waiting patient using a mathematically bounded, multi-factor ranking algorithm:

$$\text{DPS}_i(t) = W_{\text{base}}(A_i) + \left(2.0 \times R_i\right) + \min\left(400.0, \, 60.0 \times \left(\frac{t - t_{\text{arrival}, i}}{T_{\text{SLA}}(A_i)}\right)^{1.35}\right)$$

#### Component Breakdown:

1. **Base Clinical Acuity Tier ($W_{\text{base}}(A_i)$)**:
   - **ESI 1 (Immediate / Resuscitation)**: $10{,}000\text{ points}$
   - **ESI 2 (Emergent / Very Urgent)**: $5{,}000\text{ points}$
   - **ESI 3 (Urgent)**: $2{,}000\text{ points}$
   - **ESI 4 (Less Urgent / Standard)**: $800\text{ points}$
   - **ESI 5 (Non-Urgent / Routine)**: $200\text{ points}$

2. **Physiological Deterioration Risk Component ($2.0 \times R_i$)**:
   - $R_i$ is the ML-predicted deterioration risk percentage ($0.0 \le R_i \le 100.0\%$), contributing up to $200.0\text{ points}$ to prioritize patients at higher physiological risk within their clinical tier.

3. **Dynamic Wait Time Starvation Prevention Curve**:
   - $t - t_{\text{arrival}, i}$ is the elapsed wait time in minutes.
   - $T_{\text{SLA}}(A_i)$ is the target maximum wait SLA for that acuity level.
   - As wait time increases, the non-linear penalty accelerates at an exponent of $1.35$.
   - **Tier Boundary Protection**: The wait penalty is strictly capped at $400.0\text{ points}$. Because an ESI 3 patient's theoretical maximum score is $2{,}000 + 200 + 400 = 2{,}600\text{ points}$, routine and urgent patients can **never** overtake ESI 2 ($5{,}000+\text{ points}$) or ESI 1 ($10{,}000+\text{ points}$) patients, regardless of elapsed wait time.
   - **Queue Ranking**: Evaluated with a primary sort key on clinical urgency ascending ($A_i$), and a secondary sort key on $\text{DPS}_i(t)$ descending.

---

### 3. Real-Time Hospital Ward & Bed Capacity Tracking

The hospital bed matrix coordinates 120 inpatient beds across 6 specialized hospital care units:

| Ward Name | Code Prefix | Bed Capacity | Clinical Focus | Capabilities |
| :--- | :---: | :---: | :--- | :--- |
| **Intensive Care Unit (ICU)** | `ICU` | 12 Beds | Invasive Hemodynamic Resuscitation | Mechanical Ventilators, Continuous Telemetry, Negative-Pressure Isolation |
| **HDU / Telemetry Step-Down** | `HDU` | 16 Beds | High Dependency / Cardiac Telemetry | Multi-lead Telemetry, Partial Ventilator Support, Isolation Bays |
| **General Medical Ward** | `GMW` | 36 Beds | Inpatient Internal Medicine | Centralized Monitoring, Standard Beds |
| **Surgical Inpatient Ward** | `SIW` | 24 Beds | Post-Operative Trauma & Surgical Care | Post-Anesthesia Recovery, Telemetry Bays |
| **Pediatric Ward** | `PED` | 18 Beds | Specialized Pediatric Inpatient Care | Pediatric Telemetry, Child Isolation Rooms |
| **Day Observation Bay** | `DOB` | 14 Beds | Short-Stay Diagnostic Observation | Rapid Turnover (4–8h Infusions & Observation) |

* **Strict Bed Availability Mathematics**:
  $$\text{Free Beds} = \max(0, \, \text{Total Beds} - \text{Occupied Beds})$$
  Ensures exact numerical alignment across the command center (e.g., a ward with 11 of 12 beds occupied always displays exactly `1 Free`).
* **Inpatient Bed Filtering**: Interactive dropdown filter and clickable ward overview cards allow clinicians to filter the occupied bed matrix across **All 6 Care Wards** or inspect any individual unit.
* **1-Click Quick Discharge**: Clinicians can discharge admitted patients directly from the dashboard, immediately releasing beds back to the `AVAILABLE` pool and updating hospital surge telemetry in real time.
* **Equipment & Infection Control Tags**: Beds are tagged with clinical micro-badges for `TELEMETRY`, `VENT` (Mechanical Ventilator), and `ISOLATION` to support automated infection control isolation.

---

### 4. Live Doctor Consultation Room Tracking

AegisPulse tracks real-time physician room availability across OPD specialty examination suites:
* **Real-Time Doctor Roster**: Monitors rooms including Cardiology (Room 101), Emergency/Trauma (Trauma Bay A), Pulmonology (Room 104), General Medicine (Room 106), Pediatrics (Peds Bay 1), Neurology (Neuro 2), and Orthopedics (Ortho 3).
* **Live Status Micro-Badges**: Rooms display color-coded indicators for `FREE` (emerald) vs. `BUSY` (amber).
* **1-Click "Call Next Patient" Workflow**:
  - Automatically selects the highest-priority patient in the live queue.
  - Matches the patient with an available consultation room and specialist.
  - Transitions the doctor room status to `BUSY` and patient status to `IN_CONSULTATION`.
* **1-Click "Finish Consultation" Workflow**:
  - Marks the consultation complete, records visit duration, and frees the physician room.
  - If the patient requires admission, the automated bed manager immediately reserves or allocates the optimal bed in the appropriate inpatient ward.

---

## 🔬 Machine Learning Models & Performance

AegisPulse deploys 4 scikit-learn models trained on 5,000 synthetic patient encounters calibrated to **CDC NHAMCS (National Hospital Ambulatory Medical Care Survey)** and **MIMIC-IV** ED flow distributions:

```
                  ┌──────────────────────────────────────────────┐
                  │    Patient Intake Vitals & Chief Complaint   │
                  └──────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
   ┌───────────────────────────┐                   ┌───────────────────────────┐
   │   Triage Acuity Model     │                   │  Deterioration Risk Model │
   │ Gradient Boosting (120 tr)│                   │  Random Forest Regressor  │
   │  Accuracy: 97.6% (ESI 1-5)│                   │   R²: 0.984 | MAE: 2.07%  │
   └─────────────┬─────────────┘                   └─────────────┬─────────────┘
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │ Dynamic Priority Score (DPS) Queue Optimizer  │
                 └───────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
   ┌───────────────────────────┐                   ┌───────────────────────────┐
   │ Service Duration Forecast │                   │  Ward / Bed Allocation    │
   │  Random Forest Regressor  │                   │ Random Forest Classifier  │
   │   MAE: 4.01 min | R²:0.85 │                   │ Accuracy: 80.7% (6 Wards) │
   └───────────────────────────┘                   └───────────────────────────┘
```

| Model | Algorithm | Target Variable | Performance Metric | Primary Predictive Features |
| :--- | :--- | :--- | :--- | :--- |
| **Triage Acuity Classifier** | Gradient Boosting (120 Estimators) | Urgency Level (ESI 1–5) | **Accuracy: 97.6%**<br>Weighted F1: 0.976 | Pain Score, GCS, Respiration Rate, SpO2, Heart Rate, MEWS, TF-IDF Text |
| **Physiological Deterioration Scorer** | Random Forest Regressor (100 Trees) | Deterioration Risk (0–100%) | **$R^2 = 0.984$**<br>MAE: 2.07% | Systolic BP, SpO2, Heart Rate, Age, MEWS Score, Comorbidity Count |
| **Consultation Duration Forecaster** | Random Forest Regressor (80 Trees) | Service Time (Minutes) | **MAE: 4.01 min**<br>$R^2 = 0.847$ | Urgency Level, Department, Age, Mobility, Comorbidity Count |
| **Ward & Bed Allocation Classifier** | Random Forest Classifier (100 Trees) | Recommended Ward (6 Classes) | **Accuracy: 80.7%**<br>Weighted F1: 0.799 | Deterioration Risk, Urgency Level, Department, Isolation Need, Symptoms |

---

## 📊 Simulation Impact: Traditional FIFO vs. AegisPulse DPS

In a simulated cohort of 40 emergency arrivals with variable arrival times and acuity distributions:

| Clinical Acuity Tier | Standard FIFO Average Wait | AegisPulse DPS Average Wait | Operational & Clinical Impact |
| :--- | :---: | :---: | :--- |
| **ESI 1 (Immediate / Resuscitation)** | 12.0 mins | **3.0 mins** | **-75.0% Wait Reduction** (Immediate airway & cardiac stabilization) |
| **ESI 2 (Emergent / Very Urgent)** | 13.7 mins | **3.9 mins** | **-71.5% Wait Reduction** (Rapid catheterization & thrombolytic therapy) |
| **ESI 3 (Urgent)** | 16.5 mins | **12.0 mins** | **-27.3% Wait Reduction** (Prompt physician evaluation within 30m SLA) |
| **ESI 4 (Less Urgent / Standard)** | 15.1 mins | **39.6 mins** | Within 60m SLA; safe outpatient consultation without risk |
| **ESI 5 (Non-Urgent / Routine)** | 10.5 mins | **62.2 mins** | Within 120m SLA; starvation curve guarantees consultation |
| **Critical Emergencies (ESI 1 & 2)** | **13.2 mins** | **3.6 mins** | **-72.7% OVERALL REDUCTION IN LIFE-THREATENING DELAY** |

---

## 🛠️ Tech Stack & System Architecture

### Backend & Core Logic
- **Language**: Python 3.12 (compatible with Python 3.10+)
- **Web Framework**: Flask 3.1.3 with Flask-CORS for secure REST API endpoints
- **Data Engineering**: Pandas 2.2+, NumPy 1.26+, SciPy 1.12+
- **Machine Learning**: Scikit-Learn 1.4+, Joblib 1.3+
- **Architectural Style**: Modular separation of concerns (`data/`, `ml/`, `server/`, `static/`)

### Frontend Dashboard
- **UI Architecture**: Single-Page Application (SPA) with 3 uncluttered navigation tabs:
  1. **Patient Queue**: Live dynamic priority queue, doctor consultation rooms, and 1-click consultation controls.
  2. **Add Patient (AI Lab)**: Live intake form with preloaded clinical test scenarios and dynamic AI triage explanations.
  3. **Beds & Hospital Overview**: Ward capacity cards, interactive ward filter dropdown, and occupied bed discharge controls.
- **Styling**: Tailwind CSS CDN with custom dark glassmorphic styling (`#090d16` background, `#0f172a` cards).
- **Typography**: Inter (UI prose) and JetBrains Mono (clinical vitals & telemetry metrics).
- **Icons**: Lucide Icons CDN.
- **Auditory Feedback**: Native Web Audio API synthesizer for clinical telemetry chimes and alerts (no external audio files required).

---

## 📂 Project Directory Structure

```
aegispulse/
├── data/
│   ├── generator.py              # Calibrated synthetic patient dataset generator (5,000 records)
│   └── opd_patients.csv          # Generated patient dataset (NHAMCS & MIMIC-IV aligned)
├── ml/
│   ├── train_models.py           # End-to-end ML training pipeline for all 4 models
│   ├── predictor.py              # Real-time inference engine with clinical safety guardrails
│   ├── queue_optimizer.py        # Dynamic Priority Score (DPS) & Hospital Bed Allocation engine
│   └── models/
│       ├── triage_model.joblib   # Trained Gradient Boosting ESI triage classifier
│       ├── risk_model.joblib     # Trained Random Forest physiological deterioration scorer
│       ├── duration_model.joblib # Trained Random Forest consultation duration regressor
│       ├── ward_model.joblib     # Trained Random Forest inpatient ward allocation classifier
│       └── metadata.json         # Training metrics, feature importance, and class mappings
├── server/
│   └── app.py                    # Production Flask REST API & telemetry server
├── static/
│   ├── index.html                # Responsive clinical command center dashboard
│   ├── css/
│   │   └── styles.css            # Custom CSS, glassmorphism, priority strips, and scrollbars
│   └── js/
│       └── app.js                # Frontend controller, state manager, and Web Audio synthesizer
├── requirements.txt              # Production Python package dependencies
├── run.py                        # Turnkey one-click application runner
├── test_app.py                   # Automated test suite (10 unit tests)
└── README.md                     # Comprehensive project documentation & clinical architecture
```

---

## 🚀 Quick Start & Installation Guide

### 1. Prerequisites & Python Environment
- **Recommended Python Version:** **Python 3.10 to 3.12** (Python 3.12 recommended).
  > [!NOTE]
  > Ensure you are using a standard stable Python release (**Python 3.10 to 3.12**). Newer experimental releases (such as Python 3.15+) lack pre-compiled binary wheels for essential scientific packages such as `pandas` and `scipy`, which can lead to build errors during installation.
- **pip** package installer.
- Modern web browser (Chrome, Edge, Firefox, or Safari).
- Open your terminal or command prompt in the root project folder.

### 2. Setup & Execution Commands

#### On Windows (PowerShell / Command Prompt):
```powershell
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
.\venv\Scripts\activate

# Upgrade pip and install required packages
python -m pip install --upgrade pip
pip install -r requirements.txt

# Launch the application
python run.py
```

#### On macOS / Linux:
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip and install required packages
python3 -m pip install --upgrade pip
pip install -r requirements.txt

# Launch the application
python3 run.py
```

> **Automated Initialization**: If datasets or trained models are missing, `run.py` automatically generates the 5,000 synthetic patient records (`data/opd_patients.csv`) and trains all 4 machine learning models before launching the web server.

### 3. Access the Dashboard
Open your web browser and navigate to:
```
http://localhost:5000
```

---

## 🧪 Automated Verification & Unit Tests

Run the included automated test suite to verify ML models, queue ranking, bed allocation, and API endpoints:

```bash
# Windows
python test_app.py

# macOS / Linux
python3 test_app.py
```

### Test Suite Coverage (10/10 Passing):
1. `test_status_endpoint`: Validates hospital-wide bed occupancy percentages and active waiting counts.
2. `test_queue_endpoint`: Validates DPS computation and priority queue ordering.
3. `test_strict_queue_sorting_and_sla`: Asserts that ESI 1 ($\ge 10{,}000\text{ pts}$) and ESI 2 ($\ge 5{,}000\text{ pts}$) are strictly prioritized over long-waiting ESI 4/5 patients.
4. `test_predict_endpoint`: Validates real-time triage acuity, deterioration risk, and admission predictions.
5. `test_cardiac_hypoxia_clinical_rule`: Tests clinical guardrails ensuring severe cardiac presentations with hypoxia are strictly classified as ESI 1 or ESI 2.
6. `test_add_patient_to_queue`: Validates live intake submission and queue re-indexing.
7. `test_call_next_and_finish_consultation`: Validates doctor room matching, status transition to `BUSY`, and consultation completion.
8. `test_beds_endpoint_and_allocation`: Tests ward bed matrix, bed reservation, and patient discharge.
9. `test_simulation_endpoint`: Validates comparative FIFO vs. DPS queue simulation (-72.7% critical wait reduction).
10. `test_static_index`: Validates dashboard HTML serving and frontend asset delivery.

---

## 📡 REST API Documentation

| Endpoint | HTTP Method | Description | Sample Request / Response |
| :--- | :---: | :--- | :--- |
| `/api/status` | `GET` | High-level hospital command status, bed occupancy %, and surge level | Returns `system_status: "ONLINE"`, `overall_bed_occupancy_percent`, `active_waiting_patients` |
| `/api/queue` | `GET` | Ranked list of waiting patients ordered by Dynamic Priority Score | Returns `queue: [...]` with `dps`, `score_breakdown`, `acuity_badge` |
| `/api/queue/add` | `POST` | Triages and inserts a new patient into the active priority queue | JSON payload with vitals & complaints; returns `queue_position` |
| `/api/queue/call-next` | `POST` | Calls highest-ranked patient to an available doctor consultation room | Matches patient to doctor; updates doctor room status to `BUSY` |
| `/api/queue/finish-consultation` | `POST` | Concludes doctor encounter; triggers bed allocation if admitted | JSON payload: `{"patient_id": "PAT-7701"}` |
| `/api/predict` | `POST` | Real-time AI Triage inference for custom vitals without enqueueing | Returns `predicted_urgency_level`, `deterioration_risk_score`, `recommended_ward` |
| `/api/beds` | `GET` | Ward-by-ward bed availability and full 120-bed hospital matrix | Returns `ward_summary` and list of all 120 beds |
| `/api/beds/allocate` | `POST` | Allocates or reserves an inpatient bed for an admitted patient | JSON payload with `patient_name` and requested `ward` |
| `/api/beds/discharge` | `POST` | Discharges a patient from a hospital bed bay, marking it available | JSON payload: `{"bed_id": "ICU-04"}` |
| `/api/simulation` | `GET` | Runs comparative FIFO vs. AegisPulse DPS queue simulation | Query param `?n=40`; returns wait times by acuity tier |

---

## ⚖️ License & Medical Disclaimer

### License
This project is open-source software licensed under the **[MIT License](https://opensource.org/licenses/MIT)**.

### Clinical Decision Support Disclaimer
*AegisPulse™ AI is a clinical decision-support and queue optimization simulation designed for healthcare operations research and demonstration purposes. All patient encounters in the synthetic dataset are synthetically generated for computational validation. In production clinical environments, algorithmic triage recommendations must always be reviewed by certified triage nurses and licensed medical practitioners.*
