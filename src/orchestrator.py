import os
import sys
import datetime

# Ensure src/ directory is in Python path for absolute imports
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from data_generator import main as run_generator
from feature_engineering import run_feature_engineering
from feast_pipeline import apply_feast, materialize_feast
from drift_detection import run_drift_detection
from model_evaluation import run_model_evaluation

def run_pipeline():
    print("=" * 60)
    print("STARTING FEATURE STORE & DRIFT DETECTION PIPELINE")
    print(f"Timestamp: {datetime.datetime.now().isoformat()}")
    print("=" * 60)
    
    # Step 1: Raw Data Generation
    print("\n--- STEP 1: GENERATING SYNTHETIC RAW DATA ---")
    run_generator()
    
    # Step 2: PySpark Feature Engineering
    print("\n--- STEP 2: RUNNING PYSPARK ETL & WRITING TO DELTA LAKE ---")
    run_feature_engineering()
    
    # Step 3: Feast Feature Store Apply & Materialize
    print("\n--- STEP 3: REGISTERING & MATERIALIZING FEAST FEATURES ---")
    apply_feast()
    # Materialize features for the full range of baseline and new data timestamps
    start_date = datetime.datetime(2025, 1, 1)
    end_date = datetime.datetime(2027, 12, 31)
    materialize_feast(start_date, end_date)
    
    # Step 4: Evidently AI Drift Detection
    print("\n--- STEP 4: RUNNING DRIFT DETECTION & QUARANTINING ---")
    drift_detected = run_drift_detection()
    
    # Step 5: Model Evaluation (Before/After comparison)
    print("\n--- STEP 5: MODEL TRAINING & DRIFT PERFORMANCE EVALUATION ---")
    run_model_evaluation()
    
    print("=" * 60)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"Drift Alert Triggered: {drift_detected}")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()
