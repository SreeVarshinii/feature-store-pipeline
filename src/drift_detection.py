import os
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from config import (
    FEATURES_DELTA_PATH,
    DRIFT_REPORT_PATH,
    QUARANTINE_DIR,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES
)

def run_drift_detection() -> bool:
    """
    Reads engineered features, calculates data drift using Evidently AI,
    saves the HTML report, and quarantines data if drift is detected.
    Returns: True if drift was detected, False otherwise.
    """
    print("Loading engineered features from Delta Lake for drift analysis...")
    # Read delta table directory directly into pandas
    df = pd.read_parquet(FEATURES_DELTA_PATH)
    
    # Split baseline (reference) and new production (current) datasets
    reference_df = df[df["batch"] == "baseline"].copy()
    current_df = df[df["batch"] == "new_production"].copy()
    
    if reference_df.empty or current_df.empty:
        raise ValueError("Baseline or production data is empty in the Delta table.")
        
    cols_to_analyze = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    
    print("Running Evidently AI Data Drift report...")
    # Configure and run the drift report
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(
        reference_data=reference_df[cols_to_analyze],
        current_data=current_df[cols_to_analyze]
    )
    
    # Ensure reports directory exists
    os.makedirs(os.path.dirname(DRIFT_REPORT_PATH), exist_ok=True)
    snapshot.save_html(DRIFT_REPORT_PATH)
    print(f"Evidently HTML report generated at {DRIFT_REPORT_PATH}")
    
    # Parse metrics programmatically from Evidently 0.7 Snapshot dictionary
    snapshot_dict = snapshot.dict()
    metrics_list = snapshot_dict.get("metrics", [])
    
    drift_detected = False
    num_drifted_features = 0
    drift_by_col = {}
    
    for m in metrics_list:
        name = m.get("metric_name", "")
        val = m.get("value")
        cfg = m.get("config", {})
        
        if "DriftedColumnsCount" in name:
            num_drifted_features = val.get("count", 0)
            if num_drifted_features > 0:
                drift_detected = True
        elif "ValueDrift" in name:
            col_name = cfg.get("column")
            if not col_name:
                continue
            threshold = cfg.get("threshold", 0.1)
            method = cfg.get("method", "")
            drift_score = float(val) if val is not None else 0.0
            
            # Decide if this column has drifted based on method type
            is_col_drifted = False
            if "p_value" in method or "p-value" in method:
                is_col_drifted = drift_score < threshold
            else:
                is_col_drifted = drift_score > threshold
                
            drift_by_col[col_name] = {
                "drift_score": drift_score,
                "method": method,
                "threshold": threshold,
                "drift_detected": is_col_drifted
            }
            if is_col_drifted:
                drift_detected = True
                
    print("\n================ DRIFT REPORT SUMMARY ================")
    print(f"Dataset Drift Detected: {drift_detected}")
    print(f"Number of drifted features: {num_drifted_features} / {len(cols_to_analyze)}")
    print("-----------------------------------------------------")
    for col_name, info in drift_by_col.items():
        print(f"Feature '{col_name}': Drifted={info['drift_detected']} (Score={info['drift_score']:.4f} via {info['method']})")
    print("=====================================================\n")
    
    # Business Alert / Quarantine Layer
    if drift_detected:
        quarantine_path = os.path.join(QUARANTINE_DIR, "quarantined_batch.parquet")
        print(f"[ALERT] Data drift detected! Quarantining new production batch to {quarantine_path}")
        current_df.to_parquet(quarantine_path, index=False)
        
        # Calculate business impact text (relative drop in income)
        mean_baseline = reference_df["annual_income"].mean()
        mean_current = current_df["annual_income"].mean()
        income_mean_drop_pct = abs(mean_baseline - mean_current) / mean_baseline * 100
        print(f"[BUSINESS IMPACT] Detected {income_mean_drop_pct:.1f}% drift in income feature before model degraded.")
    else:
        print("[INFO] No significant data drift detected. Production data is stable.")
        
    return drift_detected

if __name__ == "__main__":
    run_drift_detection()
