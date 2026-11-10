"""
AegisPulse Hospital Command Center - Flask REST API & Web Server.
"""

import os
import sys

# Ensure UTF-8 console output on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
import random
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Add root directory to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml.predictor import get_predictor
from ml.queue_optimizer import DynamicQueueOptimizer, HospitalBedManager, simulate_queue_comparison

app = Flask(__name__, static_folder="../static", static_url_path="")
CORS(app)

# Initialize core services
predictor = get_predictor()
queue_optimizer = DynamicQueueOptimizer()
bed_manager = HospitalBedManager()

# Global in-memory OPD state
DOCTORS = [
    {"id": "DOC-101", "name": "Dr. Sarah Chen, MD", "dept": "Cardiology", "status": "AVAILABLE", "room": "Room 101"},
    {"id": "DOC-102", "name": "Dr. Marcus Vance, DO", "dept": "Emergency / Trauma", "status": "AVAILABLE", "room": "Trauma Bay A"},
    {"id": "DOC-103", "name": "Dr. Elena Rostova, MD", "dept": "Pulmonology", "status": "BUSY", "room": "Room 104"},
    {"id": "DOC-104", "name": "Dr. David Kim, MD", "dept": "General Medicine", "status": "AVAILABLE", "room": "Room 106"},
    {"id": "DOC-105", "name": "Dr. Aisha Patel, MBBS", "dept": "Pediatrics", "status": "AVAILABLE", "room": "Peds Bay 1"},
    {"id": "DOC-106", "name": "Dr. James Wilson, MD", "dept": "Neurology", "status": "AVAILABLE", "room": "Neuro 2"},
    {"id": "DOC-107", "name": "Dr. Robert Sterling, MD", "dept": "Orthopedics", "status": "BUSY", "room": "Ortho 3"}
]

# Realistic initial OPD waiting queue
INITIAL_PRESETS = [
    {
        "patient_id": "PAT-7701",
        "patient_name": "Eleanor Sterling",
        "age": 68,
        "gender": "Female",
        "department": "Cardiology",
        "chief_complaint": "Crushing Retrosternal Chest Pain radiating to jaw & diaphoresis",
        "heart_rate_bpm": 122,
        "systolic_bp": 188,
        "diastolic_bp": 108,
        "respiratory_rate": 26,
        "spo2_percent": 90,
        "temperature_c": 37.2,
        "pain_score": 9,
        "gcs": 14,
        "comorbidity_count": 3,
        "mobility": "Stretcher / Immobilized",
        "arrival_time": (datetime.now() - timedelta(minutes=14)).strftime("%Y-%m-%d %H:%M:%S")
    },
    {
        "patient_id": "PAT-7702",
        "patient_name": "Liam Gallagher",
        "age": 4,
        "gender": "Male",
        "department": "Pediatrics",
        "chief_complaint": "Acute Stridor & Croup with severe intercostal retractions",
        "heart_rate_bpm": 142,
        "systolic_bp": 94,
        "diastolic_bp": 58,
        "respiratory_rate": 36,
        "spo2_percent": 91,
        "temperature_c": 39.4,
        "pain_score": 8,
        "gcs": 13,
        "comorbidity_count": 0,
        "mobility": "Wheelchair",
        "arrival_time": (datetime.now() - timedelta(minutes=22)).strftime("%Y-%m-%d %H:%M:%S")
    },
    {
        "patient_id": "PAT-7703",
        "patient_name": "Carlos Mendez",
        "age": 44,
        "gender": "Male",
        "department": "Emergency / Trauma",
        "chief_complaint": "Motorcycle Spill: Compound open tibia fracture with active bleeding",
        "heart_rate_bpm": 115,
        "systolic_bp": 112,
        "diastolic_bp": 74,
        "respiratory_rate": 22,
        "spo2_percent": 96,
        "temperature_c": 36.9,
        "pain_score": 9,
        "gcs": 15,
        "comorbidity_count": 0,
        "mobility": "Stretcher / Immobilized",
        "arrival_time": (datetime.now() - timedelta(minutes=18)).strftime("%Y-%m-%d %H:%M:%S")
    },
    {
        "patient_id": "PAT-7704",
        "patient_name": "Grace Abernathy",
        "age": 72,
        "gender": "Female",
        "department": "Pulmonology",
        "chief_complaint": "COPD Exacerbation with marked dyspnea & productive purulent sputum",
        "heart_rate_bpm": 104,
        "systolic_bp": 146,
        "diastolic_bp": 88,
        "respiratory_rate": 28,
        "spo2_percent": 89,
        "temperature_c": 38.3,
        "pain_score": 5,
        "gcs": 15,
        "comorbidity_count": 2,
        "mobility": "Wheelchair",
        "arrival_time": (datetime.now() - timedelta(minutes=35)).strftime("%Y-%m-%d %H:%M:%S")
    },
    {
        "patient_id": "PAT-7705",
        "patient_name": "Jonathan Reed",
        "age": 31,
        "gender": "Male",
        "department": "General Medicine",
        "chief_complaint": "High fever (39°C), chills and right lower quadrant abdominal pain for 12h",
        "heart_rate_bpm": 96,
        "systolic_bp": 128,
        "diastolic_bp": 82,
        "respiratory_rate": 18,
        "spo2_percent": 98,
        "temperature_c": 39.1,
        "pain_score": 7,
        "gcs": 15,
        "comorbidity_count": 0,
        "mobility": "Walking Independent",
        "arrival_time": (datetime.now() - timedelta(minutes=45)).strftime("%Y-%m-%d %H:%M:%S")
    },
    {
        "patient_id": "PAT-7706",
        "patient_name": "Sophia Bennett",
        "age": 28,
        "gender": "Female",
        "department": "Orthopedics",
        "chief_complaint": "Twisted right ankle while jogging, lateral malleolus swelling and tenderness",
        "heart_rate_bpm": 76,
        "systolic_bp": 118,
        "diastolic_bp": 76,
        "respiratory_rate": 16,
        "spo2_percent": 99,
        "temperature_c": 36.8,
        "pain_score": 4,
        "gcs": 15,
        "comorbidity_count": 0,
        "mobility": "Assisted Walk",
        "arrival_time": (datetime.now() - timedelta(minutes=58)).strftime("%Y-%m-%d %H:%M:%S")
    },
    {
        "patient_id": "PAT-7707",
        "patient_name": "Arthur Pendelton",
        "age": 60,
        "gender": "Male",
        "department": "General Medicine",
        "chief_complaint": "Routine 3-month blood pressure and glycemic check; prescription renewal",
        "heart_rate_bpm": 70,
        "systolic_bp": 122,
        "diastolic_bp": 78,
        "respiratory_rate": 14,
        "spo2_percent": 99,
        "temperature_c": 36.6,
        "pain_score": 0,
        "gcs": 15,
        "comorbidity_count": 1,
        "mobility": "Walking Independent",
        "arrival_time": (datetime.now() - timedelta(minutes=72)).strftime("%Y-%m-%d %H:%M:%S")
    }
]

# Process initial queue through ML predictor and optimizer
ACTIVE_QUEUE = []
for p in INITIAL_PRESETS:
    preds = predictor.predict(p)
    merged = {**p, **preds, "status": "WAITING"}
    ACTIVE_QUEUE.append(merged)

ACTIVE_QUEUE = queue_optimizer.rank_queue(ACTIVE_QUEUE)

# Serving history log
CONSULTATION_HISTORY = []
ADMISSION_HISTORY = []

# -------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------

@app.route("/api/status", methods=["GET"])
def get_system_status():
    """System status and high-level hospital command telemetry."""
    active_in_queue = len([p for p in ACTIVE_QUEUE if p["status"] == "WAITING"])
    in_consultation = len([p for p in ACTIVE_QUEUE if p["status"] == "IN_CONSULTATION"])
    ward_summary = bed_manager.get_ward_summary()
    
    total_beds = sum(w["total_beds"] for w in ward_summary)
    occupied_beds = sum(w["occupied_beds"] for w in ward_summary)
    available_beds = max(0, total_beds - occupied_beds)
    overall_bed_occupancy = round(occupied_beds / max(1, total_beds) * 100.0, 1)

    # Surge Status determination
    if overall_bed_occupancy >= 92 or active_in_queue >= 18:
        surge_level = "CODE AMBER - HOSPITAL SURGE LEVEL 2"
        surge_badge = "bg-rose-500/20 text-rose-400 border-rose-500/40"
    elif overall_bed_occupancy >= 80 or active_in_queue >= 10:
        surge_level = "SURGE LEVEL 1 - ELEVATED LOAD"
        surge_badge = "bg-amber-500/20 text-amber-400 border-amber-500/40"
    else:
        surge_level = "NORMAL CLINICAL OPERATIONS"
        surge_badge = "bg-emerald-500/20 text-emerald-400 border-emerald-500/40"

    # Average wait time
    waits = [p.get("elapsed_wait_min", 0) for p in ACTIVE_QUEUE if p["status"] == "WAITING"]
    avg_wait = round(sum(waits) / max(1, len(waits)), 1) if waits else 0.0

    return jsonify({
        "system_name": "AegisPulse AI Hospital Command Center",
        "system_status": "ONLINE",
        "surge_level": surge_level,
        "surge_badge": surge_badge,
        "active_waiting_patients": active_in_queue,
        "in_consultation": in_consultation,
        "average_wait_time_minutes": avg_wait,
        "total_hospital_beds": total_beds,
        "occupied_hospital_beds": occupied_beds,
        "available_hospital_beds": available_beds,
        "overall_bed_occupancy_percent": overall_bed_occupancy,
        "doctors_on_duty": len(DOCTORS),
        "doctors_available": sum(1 for d in DOCTORS if d["status"] == "AVAILABLE"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route("/api/queue", methods=["GET"])
def get_queue():
    """Retrieve ranked queue sorted by Dynamic Priority Score."""
    global ACTIVE_QUEUE
    ACTIVE_QUEUE = queue_optimizer.rank_queue(ACTIVE_QUEUE)
    
    # Enrich with SLA status
    for p in ACTIVE_QUEUE:
        target = p.get("target_wait_minutes", 30)
        elapsed = p.get("elapsed_wait_min", 0)
        if elapsed > target * 1.25:
            p["sla_status"] = "BREACHED"
            p["sla_badge"] = "bg-rose-500/20 text-rose-400 border-rose-500/30"
        elif elapsed > target * 0.8:
            p["sla_status"] = "AT_RISK"
            p["sla_badge"] = "bg-amber-500/20 text-amber-400 border-amber-500/30"
        else:
            p["sla_status"] = "ON_TRACK"
            p["sla_badge"] = "bg-emerald-500/20 text-emerald-400 border-emerald-500/30"

    return jsonify({
        "queue_count": len(ACTIVE_QUEUE),
        "queue": ACTIVE_QUEUE,
        "doctors": DOCTORS
    })

@app.route("/api/queue/add", methods=["POST"])
def add_patient_to_queue():
    """Triage and register a new patient, placing them into the dynamic queue."""
    global ACTIVE_QUEUE
    data = request.json or {}

    # Validate essential fields
    patient_id = data.get("patient_id") or f"PAT-{random.randint(8000, 9999)}"
    patient_name = data.get("patient_name") or f"Patient {patient_id[-4:]}"
    
    patient_record = {
        "patient_id": patient_id,
        "patient_name": patient_name,
        "age": float(data.get("age", 40)),
        "gender": str(data.get("gender", "Female")),
        "department": str(data.get("department", "General Medicine")),
        "chief_complaint": str(data.get("chief_complaint", "General consultation")),
        "heart_rate_bpm": float(data.get("heart_rate_bpm", 76)),
        "systolic_bp": float(data.get("systolic_bp", 120)),
        "diastolic_bp": float(data.get("diastolic_bp", 80)),
        "respiratory_rate": float(data.get("respiratory_rate", 16)),
        "spo2_percent": float(data.get("spo2_percent", 98)),
        "temperature_c": float(data.get("temperature_c", 36.8)),
        "pain_score": float(data.get("pain_score", 0)),
        "gcs": float(data.get("gcs", 15)),
        "comorbidity_count": int(data.get("comorbidity_count", 0)),
        "mobility": str(data.get("mobility", "Walking Independent")),
        "arrival_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "WAITING"
    }

    # Run ML inference
    predictions = predictor.predict(patient_record)
    patient_record.update(predictions)

    # Insert into queue and re-rank
    ACTIVE_QUEUE.append(patient_record)
    ACTIVE_QUEUE = queue_optimizer.rank_queue(ACTIVE_QUEUE)

    # Find new position
    pos = next((i + 1 for i, p in enumerate(ACTIVE_QUEUE) if p["patient_id"] == patient_id), 1)

    return jsonify({
        "success": True,
        "message": f"Patient {patient_name} successfully triaged as {predictions['acuity_badge']} and enqueued at position #{pos}.",
        "patient": patient_record,
        "queue_position": pos
    })

@app.route("/api/queue/call-next", methods=["POST"])
def call_next_patient():
    """Call the highest-ranked waiting patient into consultation with an available doctor."""
    global ACTIVE_QUEUE
    ACTIVE_QUEUE = queue_optimizer.rank_queue(ACTIVE_QUEUE)
    
    waiting = [p for p in ACTIVE_QUEUE if p["status"] == "WAITING"]
    if not waiting:
        return jsonify({"success": False, "message": "No patients currently waiting in the OPD queue."})

    # Pick top patient
    candidate = waiting[0]

    # Find matching doctor or any available doctor
    available_docs = [d for d in DOCTORS if d["status"] == "AVAILABLE"]
    matching_docs = [d for d in available_docs if d["dept"] == candidate["department"]]
    chosen_doc = matching_docs[0] if matching_docs else (available_docs[0] if available_docs else None)

    if not chosen_doc:
        return jsonify({
            "success": False,
            "message": "All clinical consultation rooms are currently occupied. Please complete an active consultation first."
        })

    candidate["status"] = "IN_CONSULTATION"
    candidate["doctor_assigned"] = chosen_doc["name"]
    candidate["room_assigned"] = chosen_doc["room"]
    candidate["consultation_start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    chosen_doc["status"] = "BUSY"
    chosen_doc["current_patient"] = candidate["patient_name"]

    CONSULTATION_HISTORY.append({
        "patient_id": candidate["patient_id"],
        "patient_name": candidate["patient_name"],
        "doctor": chosen_doc["name"],
        "department": candidate["department"],
        "urgency_level": candidate["predicted_urgency_level"],
        "wait_minutes": candidate.get("elapsed_wait_min", 0),
        "called_at": candidate["consultation_start_time"]
    })

    return jsonify({
        "success": True,
        "message": f"Called {candidate['patient_name']} to {chosen_doc['room']} ({chosen_doc['name']}).",
        "patient": candidate,
        "doctor": chosen_doc
    })

@app.route("/api/queue/finish-consultation", methods=["POST"])
def finish_consultation():
    """Complete a consultation, freeing up the doctor and handling admission routing."""
    global ACTIVE_QUEUE
    data = request.json or {}
    patient_id = data.get("patient_id")

    # Find patient
    patient = next((p for p in ACTIVE_QUEUE if p["patient_id"] == patient_id and p["status"] == "IN_CONSULTATION"), None)
    if not patient:
        # If no specific patient specified, pick first in consultation
        in_consult = [p for p in ACTIVE_QUEUE if p["status"] == "IN_CONSULTATION"]
        if in_consult:
            patient = in_consult[0]

    if not patient:
        return jsonify({"success": False, "message": "No patient currently in consultation."})

    # Free doctor
    doc_name = patient.get("doctor_assigned")
    for d in DOCTORS:
        if d["name"] == doc_name:
            d["status"] = "AVAILABLE"
            d["current_patient"] = None

    # Handle Bed allocation if admission recommended
    bed_result = None
    if patient.get("admission_recommended", False):
        bed_result = bed_manager.allocate_bed(patient)
        patient["bed_allocation"] = bed_result

    patient["status"] = "COMPLETED"
    ACTIVE_QUEUE = [p for p in ACTIVE_QUEUE if p["patient_id"] != patient["patient_id"]]

    return jsonify({
        "success": True,
        "message": f"Consultation for {patient['patient_name']} completed.",
        "patient": patient,
        "bed_result": bed_result
    })

@app.route("/api/predict", methods=["POST"])
def predict_only():
    """Instant AI Triage Lab inference endpoint."""
    data = request.json or {}
    result = predictor.predict(data)
    return jsonify({
        "success": True,
        "input": data,
        "prediction": result
    })

@app.route("/api/beds", methods=["GET"])
def get_beds():
    """Get full hospital bed matrix and ward capacities."""
    summary = bed_manager.get_ward_summary()
    return jsonify({
        "ward_summary": summary,
        "beds": list(bed_manager.beds.values())
    })

@app.route("/api/beds/allocate", methods=["POST"])
def allocate_bed_endpoint():
    """Allocate or reserve a bed for a patient."""
    data = request.json or {}
    patient = data.get("patient") or {
        "patient_id": data.get("patient_id", f"PAT-ADM-{random.randint(100, 999)}"),
        "patient_name": data.get("patient_name", "Emergency Admitted Patient"),
        "recommended_ward": data.get("ward", "General Medical Ward"),
        "predicted_service_minutes": 60
    }
    ward = data.get("ward")
    result = bed_manager.allocate_bed(patient, requested_ward=ward)
    return jsonify(result)

@app.route("/api/beds/discharge", methods=["POST"])
def discharge_bed_endpoint():
    """Discharge a patient from a hospital bed."""
    data = request.json or {}
    bed_id = data.get("bed_id")
    if not bed_id:
        return jsonify({"success": False, "message": "bed_id is required."})
    result = bed_manager.discharge_bed(bed_id)
    return jsonify(result)

@app.route("/api/simulation", methods=["GET", "POST"])
def get_simulation():
    """Run comparative queue simulation (FIFO vs AegisPulse DPS)."""
    n_patients = int(request.args.get("n", 50))
    results = simulate_queue_comparison(n_patients=n_patients)
    return jsonify({
        "success": True,
        "simulated_cohort_size": n_patients,
        "results": results
    })

@app.route("/api/analytics", methods=["GET"])
def get_analytics():
    """Retrieve ML model metrics, top feature importances, and distribution graphs."""
    metadata = predictor.metadata
    return jsonify({
        "success": True,
        "model_metadata": metadata,
        "consultation_history_count": len(CONSULTATION_HISTORY),
        "active_doctors": DOCTORS
    })

@app.route("/api/preset-patients", methods=["GET"])
def get_preset_patients():
    """Return clinical test presets for 1-click evaluation."""
    return jsonify({
        "presets": [
            {
                "id": "cardiac_arrest",
                "title": "Severe Crushing Chest Pain (Acute STEMI)",
                "department": "Cardiology",
                "age": 64,
                "gender": "Male",
                "chief_complaint": "Severe crushing substernal chest pain radiating to left jaw & diaphoresis",
                "heart_rate_bpm": 128,
                "systolic_bp": 195,
                "diastolic_bp": 112,
                "respiratory_rate": 28,
                "spo2_percent": 88,
                "temperature_c": 37.1,
                "pain_score": 10,
                "gcs": 14,
                "comorbidity_count": 3,
                "mobility": "Stretcher / Immobilized"
            },
            {
                "id": "respiratory_failure",
                "title": "Acute Respiratory Failure / Severe Hypoxia",
                "department": "Pulmonology",
                "age": 71,
                "gender": "Female",
                "chief_complaint": "Acute respiratory distress with cyanosis and severe wheezing",
                "heart_rate_bpm": 132,
                "systolic_bp": 160,
                "diastolic_bp": 94,
                "respiratory_rate": 34,
                "spo2_percent": 82,
                "temperature_c": 38.6,
                "pain_score": 7,
                "gcs": 13,
                "comorbidity_count": 2,
                "mobility": "Wheelchair"
            },
            {
                "id": "pediatric_seizure",
                "title": "Pediatric Febrile Convulsion",
                "department": "Pediatrics",
                "age": 3,
                "gender": "Male",
                "chief_complaint": "Infant febrile convulsion lasting 4 minutes; post-ictal drowsiness",
                "heart_rate_bpm": 145,
                "systolic_bp": 90,
                "diastolic_bp": 54,
                "respiratory_rate": 32,
                "spo2_percent": 94,
                "temperature_c": 39.8,
                "pain_score": 6,
                "gcs": 12,
                "comorbidity_count": 0,
                "mobility": "Wheelchair"
            },
            {
                "id": "acute_appendicitis",
                "title": "Suspected Acute Appendicitis",
                "department": "Emergency / Trauma",
                "age": 22,
                "gender": "Female",
                "chief_complaint": "Sharp right lower quadrant pain with rebound tenderness & nausea",
                "heart_rate_bpm": 102,
                "systolic_bp": 124,
                "diastolic_bp": 78,
                "respiratory_rate": 20,
                "spo2_percent": 98,
                "temperature_c": 38.5,
                "pain_score": 8,
                "gcs": 15,
                "comorbidity_count": 0,
                "mobility": "Assisted Walk"
            },
            {
                "id": "ankle_sprain",
                "title": "Mild Inversion Ankle Sprain",
                "department": "Orthopedics",
                "age": 27,
                "gender": "Male",
                "chief_complaint": "Right ankle inversion sprain; able to bear partial weight",
                "heart_rate_bpm": 74,
                "systolic_bp": 120,
                "diastolic_bp": 78,
                "respiratory_rate": 16,
                "spo2_percent": 99,
                "temperature_c": 36.7,
                "pain_score": 3,
                "gcs": 15,
                "comorbidity_count": 0,
                "mobility": "Walking Independent"
            },
            {
                "id": "routine_checkup",
                "title": "Hypertension Medication Refill",
                "department": "General Medicine",
                "age": 55,
                "gender": "Female",
                "chief_complaint": "Routine checkup and blood pressure prescription renewal; asymptomatic",
                "heart_rate_bpm": 68,
                "systolic_bp": 124,
                "diastolic_bp": 80,
                "respiratory_rate": 14,
                "spo2_percent": 99,
                "temperature_c": 36.6,
                "pain_score": 0,
                "gcs": 15,
                "comorbidity_count": 1,
                "mobility": "Walking Independent"
            }
        ]
    })

# Serve Frontend static assets
@app.route("/")
def index():
    return send_from_directory("../static", "index.html")

@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory("../static", path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n==================================================================")
    print(f" AegisPulse AI Hospital Command Center Server Running")
    print(f" URL: http://localhost:{port} or http://127.0.0.1:{port}")
    print(f"==================================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)
