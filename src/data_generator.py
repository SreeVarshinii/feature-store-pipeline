import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import BASELINE_RAW_PATH, NEW_RAW_PATH

def generate_user_snapshots(num_users: int, start_date: datetime, drift: bool = False) -> pd.DataFrame:
    np.random.seed(42 if not drift else 100)
    
    rows = []
    for i in range(1, num_users + 1):
        user_id = f"usr_{i:06d}"
        
        # Base attributes for this user
        if not drift:
            base_income = np.random.normal(75000, 15000)
            base_credit_usage = np.random.normal(0.3, 0.08)
            base_score = np.random.normal(710, 40)
            employment_status = np.random.choice(["Employed", "Self-Employed", "Unemployed", "Retired"], p=[0.80, 0.12, 0.04, 0.04])
            home_ownership = np.random.choice(["Rent", "Mortgage", "Own"], p=[0.40, 0.50, 0.10])
        else:
            # Shift distributions: economic stress scenario
            base_income = np.random.normal(48000, 12000)
            base_credit_usage = np.random.normal(0.58, 0.12)
            base_score = np.random.normal(650, 50)
            employment_status = np.random.choice(["Employed", "Self-Employed", "Unemployed", "Retired"], p=[0.60, 0.10, 0.25, 0.05])
            home_ownership = np.random.choice(["Rent", "Mortgage", "Own"], p=[0.60, 0.30, 0.10])

        base_income = np.clip(base_income, 12000, 250000)
        base_credit_usage = np.clip(base_credit_usage, 0.01, 1.0)
        base_score = np.clip(base_score, 300, 850)
        
        loan_amount = np.random.uniform(5000, 45000)
        loan_purpose = np.random.choice(["Debt Consolidation", "Home Improvement", "Major Purchase", "Medical"], p=[0.5, 0.2, 0.2, 0.1])
        
        # Create 3 monthly snapshots for this user
        for month_offset in range(3):
            # timestamps: month_offset = 0 (current), 1 (lag 1m), 2 (lag 2m)
            # We date them backwards from the start_date
            snapshot_date = start_date - timedelta(days=30 * month_offset)
            
            # Add some monthly fluctuations/noise
            monthly_income = base_income + np.random.normal(0, 1000)
            monthly_income = np.clip(monthly_income, 10000, 260000)
            
            credit_usage = base_credit_usage + np.random.normal(0, 0.05)
            credit_usage = np.clip(credit_usage, 0.0, 1.0)
            
            score = base_score + np.random.normal(0, 10)
            score = np.clip(score, 300, 850)
            
            # Base spending for rolling calculations
            if not drift:
                monthly_spending = np.random.normal(2500, 600)
            else:
                monthly_spending = np.random.normal(3800, 1000)
            monthly_spending = np.clip(monthly_spending, 200, 10000)
            
            # Default risk formula
            risk_score = (
                0.25 * credit_usage +
                0.25 * ((850 - score) / 550.0) +
                0.20 * ((loan_amount * 0.1) / (monthly_income / 12)) +
                0.30 * (0.4 if employment_status == "Unemployed" else 0.0)
            )
            defaulted = 1 if np.random.rand() < np.clip(risk_score, 0.0, 1.0) else 0
            
            rows.append({
                "user_id": user_id,
                "timestamp": snapshot_date,
                "annual_income": float(monthly_income),
                "credit_usage_ratio": float(credit_usage),
                "payment_history_score": float(score),
                "loan_amount": float(loan_amount),
                "monthly_spending": float(monthly_spending),
                "employment_status": employment_status,
                "home_ownership": home_ownership,
                "loan_purpose": loan_purpose,
                "defaulted": int(defaulted)
            })
            
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def main():
    print("Generating baseline (reference) snapshots...")
    baseline_df = generate_user_snapshots(
        num_users=2000,
        start_date=datetime(2026, 3, 1),
        drift=False
    )
    # Save as parquet, coerce timestamps to ms to avoid PySpark timestamp issues
    baseline_df.to_parquet(BASELINE_RAW_PATH, index=False, coerce_timestamps="ms")
    print(f"Baseline data saved to {BASELINE_RAW_PATH} (Shape: {baseline_df.shape})")
    
    print("Generating new production snapshots with injected drift...")
    new_df = generate_user_snapshots(
        num_users=600,
        start_date=datetime(2026, 7, 1),
        drift=True
    )
    new_df.to_parquet(NEW_RAW_PATH, index=False, coerce_timestamps="ms")
    print(f"New data saved to {NEW_RAW_PATH} (Shape: {new_df.shape})")

if __name__ == "__main__":
    main()
