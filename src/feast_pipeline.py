import os
import subprocess
from datetime import datetime, timedelta
import pandas as pd
from feast import FeatureStore

from config import (
    FEAST_REPO_PATH,
    ENTITY_ID,
    TIMESTAMP_COL,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL
)

# Feature Names list formatted for Feast query: "view_name:feature_name"
FEAST_FEATURE_NAMES = [
    f"user_features:{feat}" for feat in NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL]
]

def get_feature_store() -> FeatureStore:
    """Returns initialized Feast FeatureStore client."""
    return FeatureStore(repo_path=FEAST_REPO_PATH)

def apply_feast():
    """Runs 'feast apply' using CLI to parse definitions and update registry."""
    print(f"Applying Feast configuration in {FEAST_REPO_PATH}...")
    result = subprocess.run(
        ["feast", "apply"],
        cwd=FEAST_REPO_PATH,
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print("Feast apply failed!")
        print("Stdout:", result.stdout)
        print("Stderr:", result.stderr)
        result.check_returncode()
    print("Feast apply completed successfully.")
    print(result.stdout)

def materialize_feast(start_date: datetime, end_date: datetime):
    """Materializes feature views to the online SQLite store."""
    store = get_feature_store()
    print(f"Materializing features to online store from {start_date} to {end_date}...")
    store.materialize(start_date, end_date)
    print("Online store materialization complete.")

def get_historical_features_df(entity_df: pd.DataFrame) -> pd.DataFrame:
    """Retrieves point-in-time correct historical features from Feast offline store."""
    store = get_feature_store()
    print(f"Retrieving historical features for {len(entity_df)} records...")
    
    # Retrieve historical features
    training_data = store.get_historical_features(
        entity_df=entity_df,
        features=FEAST_FEATURE_NAMES
    )
    return training_data.to_df()

def get_online_features_dict(user_ids: list) -> list:
    """Retrieves online features from SQLite store for real-time serving."""
    store = get_feature_store()
    entity_rows = [{ENTITY_ID: user_id} for user_id in user_ids]
    
    # Get features names without the view name prefix for the print mapping
    raw_feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    query_features = [f"user_features:{feat}" for feat in raw_feature_names]
    
    response = store.get_online_features(
        features=query_features,
        entity_rows=entity_rows
    ).to_dict()
    
    # Format query output as list of records
    records = []
    for idx in range(len(user_ids)):
        record = {ENTITY_ID: response[ENTITY_ID][idx]}
        for feat in raw_feature_names:
            record[feat] = response[feat][idx]
        records.append(record)
    return records

def test_pipeline():
    # Run apply
    apply_feast()
    
    # Materialize features (baseline dates: Jan 2026 to Dec 2026)
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2027, 12, 31)
    materialize_feast(start_date, end_date)
    
    # Test retrieve historical (point-in-time)
    test_entities = pd.DataFrame({
        ENTITY_ID: ["usr_000001", "usr_000002"],
        TIMESTAMP_COL: [pd.Timestamp(2026, 3, 1), pd.Timestamp(2026, 3, 1)]
    })
    hist_df = get_historical_features_df(test_entities)
    print("\n--- Point-in-time Historical Features Sample ---")
    print(hist_df)
    
    # Test retrieve online
    online_features = get_online_features_dict(["usr_000001", "usr_000002"])
    print("\n--- Online Low-latency Features Sample ---")
    for rec in online_features:
        print(rec)

if __name__ == "__main__":
    test_pipeline()
