import os

# Root directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Directories
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
DELTA_DIR = os.path.join(DATA_DIR, "delta")
ONLINE_DB_DIR = os.path.join(DATA_DIR, "online")
QUARANTINE_DIR = os.path.join(DATA_DIR, "quarantine")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# File Paths
BASELINE_RAW_PATH = os.path.join(RAW_DATA_DIR, "baseline.parquet")
NEW_RAW_PATH = os.path.join(RAW_DATA_DIR, "new_data.parquet")
DRIFT_REPORT_PATH = os.path.join(REPORTS_DIR, "drift_report.html")

# Feature Store Paths
FEAST_REPO_PATH = os.path.join(BASE_DIR, "src", "feature_store")
FEAST_ONLINE_DB_PATH = os.path.join(ONLINE_DB_DIR, "online_store.db")

# Features Delta Path
FEATURES_DELTA_PATH = os.path.join(DELTA_DIR, "user_features")

# Model Paths
MODEL_PATH = os.path.join(BASE_DIR, "models", "credit_model.joblib")

# Ensure directories exist
for path in [DATA_DIR, RAW_DATA_DIR, DELTA_DIR, ONLINE_DB_DIR, QUARANTINE_DIR, REPORTS_DIR, os.path.join(BASE_DIR, "models")]:
    os.makedirs(path, exist_ok=True)

# Schema Definitions
ENTITY_ID = "user_id"
TIMESTAMP_COL = "timestamp"

NUMERIC_FEATURES = [
    "credit_usage_ratio",
    "payment_history_score",
    "annual_income",
    "loan_amount",
    "debt_to_income_ratio",
    "rolling_avg_spending_3m",
    "lag_income_diff_1m"
]

CATEGORICAL_FEATURES = [
    "employment_status",
    "home_ownership",
    "loan_purpose"
]

TARGET_COL = "defaulted"

# Drift Thresholds
PSI_THRESHOLD = 0.2  # Population Stability Index threshold (PSI > 0.2 indicates significant drift)
