"""
Dynamic Priority Queue Optimizer and Real-time Hospital Bed Allocation Engine.
"""

import os
import sys
import json
import time
import math
import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.predictor import get_predictor

class DynamicQueueOptimizer:
    """
    Orchestrates the outpatient department queue using Machine Learning predictions,
    dynamic priority scoring (DPS), and clinical SLA starvation-prevention curves.
    """

    # Clinical SLA target maximum waiting times in minutes
    SLA_TARGET_WAIT = {
        1: 0,     # Immediate (< 1m)
        2: 10,    # 10 minutes
        3: 30,    # 30 minutes
        4: 60,    # 60 minutes
        5: 120    # 120 minutes
    }

    # Strict Clinical Acuity Tier Weights
    # Insurmountable tier gaps guarantee ESI 1 & ESI 2 remain strictly at the top of the queue
    ACUITY_WEIGHTS = {
        1: 10000.0,  # ESI 1: Immediate / Resuscitation
        2: 5000.0,   # ESI 2: Emergent / Very Urgent
        3: 2000.0,   # ESI 3: Urgent
        4: 800.0,    # ESI 4: Less Urgent / Standard
        5: 200.0     # ESI 5: Non-Urgent / Routine
    }

    def __init__(self):
        self.predictor = get_predictor()

    def calculate_dps(self, patient: dict, current_time: Optional[datetime] = None) -> float:
        """
        Compute the Dynamic Priority Score (DPS) for a waiting patient.
        Formula:
          DPS = Base_Acuity_Tier + (Risk_Score * 2.0) + Capped_Wait_Penalty
        Guarantees:
          ESI 1 (~10,000+ pts) > ESI 2 (~5,000+ pts) > ESI 3 (~2,000+ pts) > ESI 4/5 (< 1,300 pts)
        """
        if current_time is None:
            current_time = datetime.now()

        urgency = int(patient.get("urgency_level", patient.get("predicted_urgency_level", 3)))
        patient["urgency_level"] = urgency
        risk = float(patient.get("deterioration_risk_score", 20.0))

        # Calculate elapsed wait minutes
        arrival_str = patient.get("arrival_time")
        if isinstance(arrival_str, str):
            try:
                arrival_dt = datetime.strptime(arrival_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                arrival_dt = current_time - timedelta(minutes=int(patient.get("simulated_wait_min", 15)))
        elif isinstance(arrival_str, datetime):
            arrival_dt = arrival_str
        else:
            arrival_dt = current_time - timedelta(minutes=int(patient.get("simulated_wait_min", 15)))

        elapsed_wait_min = max(0.0, (current_time - arrival_dt).total_seconds() / 60.0)
        patient["elapsed_wait_min"] = round(elapsed_wait_min, 1)

        # 1. Base Acuity Tier Component (Strict Clinical Priority)
        base_acuity = self.ACUITY_WEIGHTS.get(urgency, 800.0)

        # 2. Physiological Deterioration Risk Component (up to 200 pts)
        risk_comp = risk * 2.0

        # 3. Dynamic Wait Time Starvation Prevention Curve
        target_wait = max(1, self.SLA_TARGET_WAIT.get(urgency, 30))
        wait_ratio = elapsed_wait_min / target_wait
        raw_penalty = 60.0 * math.pow(wait_ratio, 1.35)
        # Cap wait penalty at 400 pts to preserve strict clinical tier boundaries
        wait_penalty = min(400.0, raw_penalty)

        # Total DPS
        dps = round(base_acuity + risk_comp + wait_penalty, 1)
        patient["dps"] = dps
        patient["base_acuity_score"] = int(base_acuity)
        patient["risk_score_component"] = round(risk_comp, 1)
        patient["wait_penalty"] = round(wait_penalty, 1)
        patient["score_breakdown"] = f"Base: {int(base_acuity)} + Risk: {round(risk_comp, 1)} + Wait: {round(wait_penalty, 1)}"
        return dps

    def rank_queue(self, queue: List[dict], current_time: Optional[datetime] = None) -> List[dict]:
        """
        Rank waiting patients:
        ESI 1 and ESI 2 patients are strictly placed at the top above all ESI 3/4/5.
        Within each tier, patients are sorted by DPS (risk + wait time).
        """
        if current_time is None:
            current_time = datetime.now()

        for patient in queue:
            self.calculate_dps(patient, current_time)

        # Sort: Primary key is urgency_level ascending (1, 2, 3, 4, 5)
        # Secondary key is DPS descending (higher score first within urgency tier)
        ranked = sorted(queue, key=lambda p: (p.get("urgency_level", 3), -p.get("dps", 0)))
        for idx, p in enumerate(ranked):
            p["queue_position"] = idx + 1
        return ranked


class HospitalBedManager:
    """
    Real-time Bed Allocation and Capacity Management Engine across hospital wards.
    """

    WARD_CONFIGS = {
        "ICU (Intensive Care Unit)": {"prefix": "ICU", "capacity": 12, "type": "Critical Care"},
        "HDU / Telemetry Step-Down": {"prefix": "HDU", "capacity": 16, "type": "High Dependency"},
        "General Medical Ward": {"prefix": "GMW", "capacity": 36, "type": "Inpatient Medical"},
        "Surgical Inpatient Ward": {"prefix": "SIW", "capacity": 24, "type": "Surgical"},
        "Pediatric Ward": {"prefix": "PED", "capacity": 18, "type": "Pediatric"},
        "Day Observation Bay": {"prefix": "DOB", "capacity": 14, "type": "Short Stay"}
    }

    def __init__(self):
        self.beds = self._initialize_beds()

    def _initialize_beds(self) -> Dict[str, dict]:
        """Initialize hospital beds with realistic baseline occupancy and telemetry."""
        beds = {}
        # Pre-seed realistic hospital occupancy (~75-85% full)
        names = [
            "Arthur Vance", "Eleanor Davies", "Robert Chen", "Maria Santos",
            "Harold Finch", "Clara Oswald", "David Kim", "Beatrice Gomez",
            "Julian Walsh", "Sarah Jenkins", "Thomas Mueller", "Amina Al-Mansoor",
            "Lucas Moreau", "Priya Nair", "William Foster", "Elena Rostova"
        ]

        bed_counter = 1
        for ward_name, config in self.WARD_CONFIGS.items():
            capacity = config["capacity"]
            prefix = config["prefix"]

            for i in range(1, capacity + 1):
                bed_id = f"{prefix}-{i:02d}"
                # Realistic occupancy probability
                is_occupied = random.random() < (0.82 if "ICU" in ward_name or "HDU" in ward_name else 0.70)
                is_turnover = False if is_occupied else (random.random() < 0.20)

                if is_occupied:
                    pt_name = names[(bed_counter) % len(names)]
                    pt_id = f"PAT-ADM-{bed_counter:04d}"
                    status = "OCCUPIED"
                    hours_rem = round(random.uniform(4.0, 48.0), 1)
                elif is_turnover:
                    pt_name = None
                    pt_id = None
                    status = "TURNOVER_CLEANING"
                    hours_rem = 0.5
                else:
                    pt_name = None
                    pt_id = None
                    status = "AVAILABLE"
                    hours_rem = 0.0

                bed_counter += 1
                beds[bed_id] = {
                    "bed_id": bed_id,
                    "ward": ward_name,
                    "ward_type": config["type"],
                    "status": status,  # AVAILABLE, OCCUPIED, RESERVED, TURNOVER_CLEANING
                    "patient_id": pt_id,
                    "patient_name": pt_name,
                    "hours_until_discharge": hours_rem,
                    "has_ventilator": "ICU" in ward_name or (("HDU" in ward_name) and (i <= 4)),
                    "has_telemetry": "ICU" in ward_name or "HDU" in ward_name or (i <= 8),
                    "is_isolation": (i == 1 or i == capacity)
                }
        return beds

    def get_ward_summary(self) -> List[dict]:
        """Aggregate ward-level bed statistics."""
        summary = []
        for ward_name, config in self.WARD_CONFIGS.items():
            ward_beds = [b for b in self.beds.values() if b["ward"] == ward_name]
            total = len(ward_beds)
            occupied = sum(1 for b in ward_beds if b["status"] == "OCCUPIED" or b["status"] == "RESERVED")
            # Strict bed availability: available = total - occupied
            available = max(0, total - occupied)
            occ_rate = round(occupied / max(1, total) * 100.0, 1)

            summary.append({
                "ward_name": ward_name,
                "prefix": config["prefix"],
                "ward_type": config["type"],
                "total_beds": total,
                "occupied_beds": occupied,
                "reserved_beds": sum(1 for b in ward_beds if b["status"] == "RESERVED"),
                "cleaning_beds": sum(1 for b in ward_beds if b["status"] == "TURNOVER_CLEANING"),
                "available_beds": available,
                "occupancy_rate_percent": occ_rate,
                "status_code": "CRITICAL" if occ_rate >= 92 else ("SURGE" if occ_rate >= 80 else "NORMAL")
            })
        return summary

    def allocate_bed(self, patient: dict, requested_ward: Optional[str] = None) -> dict:
        """
        Allocate or reserve the optimal bed for an admitted patient based on
        clinical acuity, telemetry needs, and ward availability.
        """
        target_ward = requested_ward or patient.get("recommended_ward", "General Medical Ward")
        
        # If target ward is Outpatient Discharge, no bed required
        if target_ward == "Outpatient Discharge":
            return {
                "success": False,
                "message": "Patient is designated for outpatient discharge; inpatient bed allocation not required.",
                "allocated_bed": None
            }

        # Candidate beds in preferred ward
        candidates = [
            b for b in self.beds.values()
            if b["ward"] == target_ward and b["status"] == "AVAILABLE"
        ]

        # Overflow / escalation if preferred ward is 100% full
        escalated = False
        if not candidates:
            escalated = True
            if target_ward == "ICU (Intensive Care Unit)":
                # Fallback to HDU telemetry
                candidates = [b for b in self.beds.values() if b["ward"] == "HDU / Telemetry Step-Down" and b["status"] == "AVAILABLE"]
            elif target_ward == "HDU / Telemetry Step-Down":
                # Fallback to General Medical
                candidates = [b for b in self.beds.values() if b["ward"] == "General Medical Ward" and b["status"] == "AVAILABLE"]
            else:
                # Any available inpatient bed
                candidates = [b for b in self.beds.values() if b["status"] == "AVAILABLE" and b["ward"] != "Day Observation Bay"]

        if not candidates:
            # Check for beds about to be discharged within 1 hour
            near_turnover = [
                b for b in self.beds.values()
                if b["ward"] == target_ward and (b["status"] == "TURNOVER_CLEANING" or (b["status"] == "OCCUPIED" and b["hours_until_discharge"] <= 1.0))
            ]
            if near_turnover:
                bed = near_turnover[0]
                bed["status"] = "RESERVED"
                bed["reserved_for"] = patient.get("patient_id")
                return {
                    "success": True,
                    "allocated_bed": bed["bed_id"],
                    "ward": bed["ward"],
                    "status": "RESERVED_IMMINENT_TURNOVER",
                    "estimated_availability_min": 25,
                    "message": f"Reserved {bed['bed_id']} ({bed['ward']}) pending imminent turnover discharge."
                }
            return {
                "success": False,
                "message": f"Bed Alert: All beds in {target_ward} and fallback units are currently occupied. Patient placed on Emergency Bed Waitlist.",
                "allocated_bed": None
            }

        # Select best candidate
        # Prefer isolation if patient has infection/respiratory flags
        needs_isolation = "Cyanosis" in patient.get("chief_complaint", "") or "Sepsis" in patient.get("chief_complaint", "")
        if needs_isolation:
            iso_beds = [b for b in candidates if b["is_isolation"]]
            chosen = iso_beds[0] if iso_beds else candidates[0]
        else:
            chosen = candidates[0]

        # Allocate bed
        chosen["status"] = "OCCUPIED"
        chosen["patient_id"] = patient.get("patient_id")
        chosen["patient_name"] = patient.get("patient_name", "Admitted Patient")
        chosen["hours_until_discharge"] = round(float(patient.get("predicted_service_minutes", 30)) / 60.0 + random.uniform(12.0, 36.0), 1)

        return {
            "success": True,
            "allocated_bed": chosen["bed_id"],
            "ward": chosen["ward"],
            "status": chosen["status"],
            "escalated_from_target": escalated,
            "has_ventilator": chosen["has_ventilator"],
            "has_telemetry": chosen["has_telemetry"],
            "is_isolation": chosen["is_isolation"],
            "message": f"Successfully allocated Bed {chosen['bed_id']} in {chosen['ward']}."
        }

    def discharge_bed(self, bed_id: str) -> dict:
        """Mark a bed as discharged and move to turnover cleaning."""
        if bed_id not in self.beds:
            return {"success": False, "message": "Bed ID not found"}
        bed = self.beds[bed_id]
        prev_patient = bed.get("patient_name") or bed.get("patient_id")
        bed["status"] = "AVAILABLE"
        bed["patient_id"] = None
        bed["patient_name"] = None
        bed["hours_until_discharge"] = 0.0
        return {"success": True, "message": f"Bed {bed_id} discharged from {prev_patient} and is now AVAILABLE."}


def simulate_queue_comparison(n_patients: int = 40) -> dict:
    """
    Simulate a busy hospital OPD shift comparing:
    1. Standard First-Come First-Served (FIFO)
    2. Static ESI Urgency Priority
    3. AegisPulse ML Dynamic Priority Score (DPS)
    """
    random.seed(101)
    np.random.seed(101)
    
    # Generate random queue mix
    patients = []
    urgency_dist = [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5]
    
    base_time = datetime(2026, 11, 9, 8, 0)
    current_time = base_time
    
    for i in range(n_patients):
        u = random.choice(urgency_dist)
        arrival_offset = random.randint(0, 180) # arrived within last 3 hours
        p_arrival = base_time + timedelta(minutes=arrival_offset)
        service_time = {1: 40, 2: 30, 3: 20, 4: 12, 5: 8}[u]
        
        patients.append({
            "id": f"P-{i+1:03d}",
            "urgency": u,
            "arrival_min": arrival_offset,
            "service_min": service_time,
            "risk": round(float((6 - u) * 16.0 + random.uniform(5, 18)), 1)
        })

    # 1. FIFO Simulation (served purely by arrival_min)
    fifo_sorted = sorted(patients, key=lambda p: p["arrival_min"])
    doc_free_at = [0, 0, 0, 0] # 4 doctors on duty
    fifo_waits = {1: [], 2: [], 3: [], 4: [], 5: []}
    
    for p in fifo_sorted:
        earliest_doc_idx = int(np.argmin(doc_free_at))
        start_service = max(p["arrival_min"], doc_free_at[earliest_doc_idx])
        wait = start_service - p["arrival_min"]
        doc_free_at[earliest_doc_idx] = start_service + p["service_min"]
        fifo_waits[p["urgency"]].append(wait)

    # 2. AegisPulse DPS Simulation
    # Dynamic calculation with wait penalty to avoid low-acuity starvation
    dps_sorted = []
    # Dynamic queue simulator step
    time_cursor = 0
    remaining = list(patients)
    doc_free_at_dps = [0, 0, 0, 0]
    dps_waits = {1: [], 2: [], 3: [], 4: [], 5: []}

    while remaining:
        earliest_doc_idx = int(np.argmin(doc_free_at_dps))
        time_cursor = max(time_cursor, doc_free_at_dps[earliest_doc_idx])
        
        # Available patients arrived up to time_cursor
        arrived = [p for p in remaining if p["arrival_min"] <= time_cursor]
        if not arrived:
            # Advance time to next arrival
            time_cursor = min(p["arrival_min"] for p in remaining)
            arrived = [p for p in remaining if p["arrival_min"] <= time_cursor]

        # Calculate DPS for each arrived patient
        best_p = None
        best_score = -1e9
        for p in arrived:
            elapsed = time_cursor - p["arrival_min"]
            acuity_w = {1: 10000, 2: 5000, 3: 2000, 4: 800, 5: 200}[p["urgency"]]
            target_wait = {1: 1, 2: 10, 3: 30, 4: 60, 5: 120}[p["urgency"]]
            wait_penalty = min(400.0, 60.0 * math.pow(elapsed / target_wait, 1.35))
            score = acuity_w + (p["risk"] * 2.0) + wait_penalty
            if score > best_score:
                best_score = score
                best_p = p

        # Serve best_p
        start_service = time_cursor
        wait = start_service - best_p["arrival_min"]
        doc_free_at_dps[earliest_doc_idx] = start_service + best_p["service_min"]
        dps_waits[best_p["urgency"]].append(wait)
        remaining.remove(best_p)

    def avg(lst):
        return round(float(np.mean(lst)), 1) if lst else 0.0

    return {
        "fifo": {
            "overall_avg_wait_min": avg([w for sub in fifo_waits.values() for w in sub]),
            "esi_1_wait_min": avg(fifo_waits[1]),
            "esi_2_wait_min": avg(fifo_waits[2]),
            "esi_3_wait_min": avg(fifo_waits[3]),
            "esi_4_wait_min": avg(fifo_waits[4]),
            "esi_5_wait_min": avg(fifo_waits[5]),
            "critical_acuity_avg_min": avg(fifo_waits[1] + fifo_waits[2]),
            "max_low_acuity_wait_min": max(fifo_waits[4] + fifo_waits[5]) if (fifo_waits[4] + fifo_waits[5]) else 0
        },
        "aegispulse_dps": {
            "overall_avg_wait_min": avg([w for sub in dps_waits.values() for w in sub]),
            "esi_1_wait_min": avg(dps_waits[1]),
            "esi_2_wait_min": avg(dps_waits[2]),
            "esi_3_wait_min": avg(dps_waits[3]),
            "esi_4_wait_min": avg(dps_waits[4]),
            "esi_5_wait_min": avg(dps_waits[5]),
            "critical_acuity_avg_min": avg(dps_waits[1] + dps_waits[2]),
            "max_low_acuity_wait_min": max(dps_waits[4] + dps_waits[5]) if (dps_waits[4] + dps_waits[5]) else 0
        },
        "critical_wait_reduction_percent": round(
            (avg(fifo_waits[1] + fifo_waits[2]) - avg(dps_waits[1] + dps_waits[2])) /
            max(0.1, avg(fifo_waits[1] + fifo_waits[2])) * 100.0, 1
        )
    }

if __name__ == "__main__":
    sim_res = simulate_queue_comparison()
    print("Simulation Results:\n", json.dumps(sim_res, indent=2))
