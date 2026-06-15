import os
import sys
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Ensure the DAG can locate our src files when loaded by Airflow daemon
DAGS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if DAGS_DIR not in sys.path:
    sys.path.insert(0, DAGS_DIR)

# Import the core pipeline components
# In a real Airflow environment, these paths would point to the DAG folder or python packages
try:
    from data_generator import main as run_generator
    from feature_engineering import run_feature_engineering
    from feast_pipeline import apply_feast, materialize_feast
    from drift_detection import run_drift_detection
    from model_evaluation import run_model_evaluation
except ImportError:
    # Fallback placeholders for static parsing of DAG file without executing dependencies
    def run_generator(): pass
    def run_feature_engineering(): pass
    def apply_feast(): pass
    def materialize_feast(start_date, end_date): pass
    def run_drift_detection(): return False
    def run_model_evaluation(): pass

default_args = {
    "owner": "mlops_engineer",
    "depends_on_past": False,
    "start_date": datetime(2026, 6, 1),
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

def feast_task_wrapper():
    """Wrapper task to run Feast apply and materialize."""
    apply_feast()
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2027, 12, 31)
    materialize_feast(start_date, end_date)

def drift_detection_wrapper():
    """Wrapper task to run drift detection. In a production environment,
    an email/Slack notification can be triggered if drift is detected."""
    drift_detected = run_drift_detection()
    if drift_detected:
        print("[AIRFLOW ALERT] Data drift detected! Quarantine step triggered.")
    else:
        print("[AIRFLOW INFO] Data drift checks passed.")

with DAG(
    "feature_store_drift_pipeline",
    default_args=default_args,
    description="MLOps Pipeline: PySpark ETL -> Feast Feature Store -> Evidently Drift Detection -> Retraining",
    schedule_interval=timedelta(days=1), # Daily ingestion pipeline
    catchup=False,
) as dag:

    # 1. Generate Raw Data (simulates raw batch arrival)
    generate_data = PythonOperator(
        task_id="generate_raw_data",
        python_callable=run_generator,
    )

    # 2. PySpark Feature Engineering
    pyspark_etl = PythonOperator(
        task_id="pyspark_feature_engineering",
        python_callable=run_feature_engineering,
    )

    # 3. Register and Materialize Feast Feature Views
    feast_materialization = PythonOperator(
        task_id="feast_materialize",
        python_callable=feast_task_wrapper,
    )

    # 4. Drift Detection (using Evidently AI) and Quarantine
    drift_monitoring = PythonOperator(
        task_id="drift_detection_monitoring",
        python_callable=drift_detection_wrapper,
    )

    # 5. Model Evaluation (Before/After drift comparison and retraining)
    model_re_evaluation = PythonOperator(
        task_id="model_evaluation_retraining",
        python_callable=run_model_evaluation,
    )

    # Define DAG execution order
    generate_data >> pyspark_etl >> feast_materialization >> drift_monitoring >> model_re_evaluation
