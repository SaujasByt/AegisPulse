"""
Turnkey Runner for AegisPulse AI Hospital Prioritization & Queue Management System.
"""

import os
import sys

# Windows Unicode console safety
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def main():
    print("=" * 72)
    print("   [+] AEGISPULSE(TM) AI - HOSPITAL COMMAND CENTER & QUEUE ORCHESTRATION")
    print("   ML-Driven Triage, Dynamic DPS Prioritization & Ward Bed Allocation")
    print("=" * 72)

    # 1. Verify Dataset
    data_file = os.path.join("data", "opd_patients.csv")
    if not os.path.exists(data_file):
        print("\n[+] Generating synthetic OPD cohort dataset (5,000 patient records)...")
        from data.generator import generate_opd_dataset
        generate_opd_dataset(5000, data_file)

    # 2. Verify Models
    model_file = os.path.join("ml", "models", "triage_model.joblib")
    if not os.path.exists(model_file):
        print("\n[+] Training Machine Learning models (Triage, Risk, Duration, Ward Allocation)...")
        from ml.train_models import train_all_models
        train_all_models()

    # 3. Launch Server
    print("\n[+] Launching Hospital Command Center Web Application...")
    from server.app import app
    port = int(os.environ.get("PORT", 5000))
    print(f"\n==================================================================")
    print(f" [+] Access AegisPulse Dashboard at: http://localhost:{port}")
    print(f"==================================================================\n")
    app.run(host="0.0.0.0", port=port, debug=False)

if __name__ == "__main__":
    main()
