import os
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from config import (
    FEATURES_DELTA_PATH,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
    MODEL_PATH
)

METRICS_JSON_PATH = os.path.join(os.path.dirname(MODEL_PATH), "model_metrics.json")

def build_preprocessing_pipeline() -> ColumnTransformer:
    """Creates a column transformer for scaling numerics and encoding categoricals."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor

def evaluate_predictions(y_true, y_pred, y_prob) -> dict:
    """Computes key classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob))
    }

def run_model_evaluation():
    print("Loading engineered features from Delta Lake for model training...")
    df = pd.read_parquet(FEATURES_DELTA_PATH)
    
    baseline_df = df[df["batch"] == "baseline"].copy()
    production_df = df[df["batch"] == "new_production"].copy()
    
    # --- 1. Baseline Model Training ---
    print("Training baseline credit risk model...")
    X = baseline_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = baseline_df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    preprocessor = build_preprocessing_pipeline()
    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42))
    ])
    
    model_pipeline.fit(X_train, y_train)
    
    # Save the baseline model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model_pipeline, MODEL_PATH)
    print(f"Baseline model trained and saved to {MODEL_PATH}")
    
    # Evaluate baseline test set
    y_test_pred = model_pipeline.predict(X_test)
    y_test_prob = model_pipeline.predict_proba(X_test)[:, 1]
    baseline_metrics = evaluate_predictions(y_test, y_test_pred, y_test_prob)
    
    # --- 2. Evaluate Drift Ignored (Baseline Model on Drifted Production Data) ---
    print("Evaluating baseline model on drifted production data (Drift Ignored)...")
    X_prod = production_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_prod = production_df[TARGET_COL]
    
    # Split production data to have a clean test set
    X_prod_train, X_prod_test, y_prod_train, y_prod_test = train_test_split(X_prod, y_prod, test_size=0.5, random_state=42)
    
    y_prod_pred = model_pipeline.predict(X_prod_test)
    y_prod_prob = model_pipeline.predict_proba(X_prod_test)[:, 1]
    drift_ignored_metrics = evaluate_predictions(y_prod_test, y_prod_pred, y_prod_prob)
    
    # --- 3. Evaluate Drift Caught & Retrained (Model Updated with Drifted Data) ---
    print("Simulating model retraining with drifted data (Drift Caught & Retrained)...")
    # Combine baseline training data and 50% of the production data (simulating retraining pool)
    X_combined = pd.concat([X_train, X_prod_train])
    y_combined = pd.concat([y_train, y_prod_train])
    
    retrained_pipeline = Pipeline(steps=[
        ("preprocessor", build_preprocessing_pipeline()),
        ("classifier", RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42))
    ])
    retrained_pipeline.fit(X_combined, y_combined)
    
    # Evaluate retrained model on the production test set
    y_prod_retrained_pred = retrained_pipeline.predict(X_prod_test)
    y_prod_retrained_prob = retrained_pipeline.predict_proba(X_prod_test)[:, 1]
    drift_retrained_metrics = evaluate_predictions(y_prod_test, y_prod_retrained_pred, y_prod_retrained_prob)
    
    # --- Print Performance comparison ---
    print("\n================= MODEL PERFORMANCE COMPARISON =================")
    print(f"1. Baseline Performance (Historical Test Data):")
    print(f"   Accuracy:  {baseline_metrics['accuracy']:.4f}  | AUC ROC:   {baseline_metrics['auc_roc']:.4f}")
    print(f"   Precision: {baseline_metrics['precision']:.4f}  | Recall:    {baseline_metrics['recall']:.4f}")
    print(f"2. Drifted Performance (Drift Ignored - Production Data):")
    print(f"   Accuracy:  {drift_ignored_metrics['accuracy']:.4f}  | AUC ROC:   {drift_ignored_metrics['auc_roc']:.4f} (Degraded)")
    print(f"   Precision: {drift_ignored_metrics['precision']:.4f}  | Recall:    {drift_ignored_metrics['recall']:.4f}")
    print(f"3. Retrained Performance (Drift Caught & Model Updated):")
    print(f"   Accuracy:  {drift_retrained_metrics['accuracy']:.4f}  | AUC ROC:   {drift_retrained_metrics['auc_roc']:.4f} (Recovered)")
    print(f"   Precision: {drift_retrained_metrics['precision']:.4f}  | Recall:    {drift_retrained_metrics['recall']:.4f}")
    print("=================================================================\n")
    
    # Save comparative metrics to JSON
    metrics_summary = {
        "baseline": baseline_metrics,
        "drift_ignored": drift_ignored_metrics,
        "drift_retrained": drift_retrained_metrics
    }
    with open(METRICS_JSON_PATH, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Performance metrics summary written to {METRICS_JSON_PATH}")

if __name__ == "__main__":
    run_model_evaluation()
