"""
Comprehensive Test Suite for AegisPulse Backend & Machine Learning Services.
"""

import os
import sys
import json
import unittest

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from server.app import app
from ml.predictor import get_predictor
from ml.queue_optimizer import DynamicQueueOptimizer, HospitalBedManager, simulate_queue_comparison

class TestAegisPulse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.predictor = get_predictor()
        cls.queue_optimizer = DynamicQueueOptimizer()
        cls.bed_manager = HospitalBedManager()

    def test_status_endpoint(self):
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["system_status"], "ONLINE")
        self.assertIn("overall_bed_occupancy_percent", data)
        self.assertIn("active_waiting_patients", data)
        print(f"Status Endpoint: OK (Waiting: {data['active_waiting_patients']}, Bed Occupancy: {data['overall_bed_occupancy_percent']}%)")

    def test_queue_endpoint(self):
        res = self.client.get("/api/queue")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(len(data["queue"]) > 0)
        top = data["queue"][0]
        self.assertIn("dps", top)
        self.assertIn("acuity_badge", top)
        print(f"Queue Endpoint: OK (Top patient: {top['patient_name']}, Badge: {top['acuity_badge']}, DPS: {top['dps']})")

    def test_strict_queue_sorting_and_sla(self):
        from datetime import datetime, timedelta
        old_time = (datetime.now() - timedelta(minutes=180)).strftime("%Y-%m-%d %H:%M:%S")
        patient_routine = {
            "patient_name": "Routine Long Waiter",
            "urgency_level": 5,
            "deterioration_risk_score": 10.0,
            "arrival_time": old_time
        }
        new_time = (datetime.now() - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
        patient_emergency = {
            "patient_name": "New Emergency Cardiac",
            "urgency_level": 1,
            "deterioration_risk_score": 95.0,
            "arrival_time": new_time
        }
        patient_urgent = {
            "patient_name": "New Emergent Trauma",
            "urgency_level": 2,
            "deterioration_risk_score": 85.0,
            "arrival_time": new_time
        }

        queue = [patient_routine, patient_urgent, patient_emergency]
        ranked = self.queue_optimizer.rank_queue(queue)

        self.assertEqual(ranked[0]["patient_name"], "New Emergency Cardiac")
        self.assertEqual(ranked[1]["patient_name"], "New Emergent Trauma")
        self.assertEqual(ranked[2]["patient_name"], "Routine Long Waiter")
        self.assertTrue(ranked[0]["dps"] >= 10000.0)
        self.assertTrue(ranked[1]["dps"] >= 5000.0)
        self.assertTrue(ranked[2]["dps"] < 1500.0)
        self.assertIn("Base: 10000", ranked[0]["score_breakdown"])
        print(f"Strict Queue Sorting: OK (ESI 1 {ranked[0]['dps']} > ESI 2 {ranked[1]['dps']} > ESI 5 {ranked[2]['dps']})")

    def test_predict_endpoint(self):
        payload = {
            "age": 65,
            "gender": "Male",
            "department": "Cardiology",
            "chief_complaint": "Severe crushing chest pain radiating to jaw",
            "heart_rate_bpm": 130,
            "systolic_bp": 190,
            "diastolic_bp": 110,
            "respiratory_rate": 28,
            "spo2_percent": 88,
            "temperature_c": 37.2,
            "pain_score": 10,
            "gcs": 14,
            "comorbidity_count": 2,
            "mobility": "Stretcher / Immobilized"
        }
        res = self.client.post("/api/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        pred = data["prediction"]
        self.assertIn(pred["predicted_urgency_level"], [1, 2]) # High acuity (ESI 1 or 2)
        self.assertTrue(pred["deterioration_risk_score"] > 80)
        self.assertTrue(pred["admission_recommended"])
        print(f"Predict Endpoint: OK (Acuity: {pred['acuity_badge']}, Risk: {pred['deterioration_risk_score']}%, Rec Ward: {pred['recommended_ward']})")

    def test_cardiac_hypoxia_clinical_rule(self):
        # High-acuity cardiac presentation with hypoxia (SpO2 91%) and severe pain (9/10)
        payload = {
            "age": 62,
            "gender": "Male",
            "department": "Cardiology",
            "chief_complaint": "Crushing chest pain radiating to jaw and diaphoresis",
            "heart_rate_bpm": 115,
            "systolic_bp": 155,
            "diastolic_bp": 95,
            "respiratory_rate": 22,
            "spo2_percent": 91,
            "temperature_c": 37.1,
            "pain_score": 9,
            "gcs": 15,
            "comorbidity_count": 2,
            "mobility": "Wheelchair"
        }
        res = self.client.post("/api/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        pred = res.get_json()["prediction"]
        self.assertIn(pred["predicted_urgency_level"], [1, 2])
        self.assertNotEqual(pred["predicted_urgency_level"], 3)
        if "ICU" in pred["recommended_ward"]:
            self.assertIn(pred["predicted_urgency_level"], [1, 2])
        print(f"Cardiac Hypoxia Rule: OK (Acuity strictly {pred['acuity_badge']}, Ward: {pred['recommended_ward']})")

    def test_add_patient_to_queue(self):
        payload = {
            "patient_name": "Test Urgent Patient",
            "age": 52,
            "gender": "Female",
            "department": "Pulmonology",
            "chief_complaint": "Severe wheezing, hypoxia and cyanosis",
            "heart_rate_bpm": 125,
            "systolic_bp": 150,
            "diastolic_bp": 95,
            "respiratory_rate": 30,
            "spo2_percent": 85,
            "temperature_c": 38.2,
            "pain_score": 8,
            "gcs": 13,
            "comorbidity_count": 1,
            "mobility": "Wheelchair"
        }
        res = self.client.post("/api/queue/add", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("queue_position", data)
        print(f"Add Patient: OK (Patient enqueued at position #{data['queue_position']})")

    def test_call_next_and_finish_consultation(self):
        res_call = self.client.post("/api/queue/call-next")
        self.assertEqual(res_call.status_code, 200)
        data_call = res_call.get_json()
        self.assertTrue(data_call["success"])
        called_patient = data_call["patient"]
        print(f"Call Next: OK (Called {called_patient['patient_name']} to {data_call['doctor']['room']})")

        # Now finish consultation
        res_finish = self.client.post("/api/queue/finish-consultation", json={"patient_id": called_patient["patient_id"]})
        self.assertEqual(res_finish.status_code, 200)
        data_finish = res_finish.get_json()
        self.assertTrue(data_finish["success"])
        print(f"Finish Consultation: OK ({data_finish['message']})")

    def test_beds_endpoint_and_allocation(self):
        res = self.client.get("/api/beds")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(len(data["ward_summary"]) >= 6)
        # Verify strict bed capacity math: available = total - occupied
        for w in data["ward_summary"]:
            self.assertEqual(w["available_beds"], max(0, w["total_beds"] - w["occupied_beds"]))
        print(f"Beds Matrix: OK ({len(data['beds'])} total beds across {len(data['ward_summary'])} wards, Math Verified: free == total - occupied)")

        # Test Allocation
        alloc_res = self.client.post("/api/beds/allocate", json={
            "patient_name": "Emergency Trauma Patient",
            "ward": "ICU (Intensive Care Unit)"
        })
        self.assertEqual(alloc_res.status_code, 200)
        alloc_data = alloc_res.get_json()
        self.assertTrue(alloc_data["success"])
        print(f"Bed Allocation: OK ({alloc_data['message']})")

        # Test Discharge
        bed_id = alloc_data["allocated_bed"]
        if bed_id:
            disc_res = self.client.post("/api/beds/discharge", json={"bed_id": bed_id})
            self.assertEqual(disc_res.status_code, 200)
            print(f"Bed Discharge: OK ({disc_res.get_json()['message']})")

    def test_simulation_endpoint(self):
        res = self.client.get("/api/simulation?n=40")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        red = data["results"]["critical_wait_reduction_percent"]
        self.assertTrue(red > 50.0)
        print(f"Simulation Endpoint: OK (Critical wait reduction: -{red}%)")

    def test_static_index(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"AegisPulse", res.data)
        res.close()
        print("Frontend Index: OK (Served successfully)")

if __name__ == "__main__":
    unittest.main()
