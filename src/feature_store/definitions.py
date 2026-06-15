from datetime import timedelta
from feast import (
    Entity,
    FeatureView,
    Field,
    FileSource,
)
from feast.types import Float32, Int32, String

# Define the User Entity
user = Entity(
    name="user_id", 
    join_keys=["user_id"], 
    description="User Identifier"
)

# Define the file source pointing to our engineered Delta table directory
# (Feast FileSource reads parquet files inside the Delta Lake folder)
user_features_source = FileSource(
    name="user_features_source",
    path="../../data/delta/user_features",
    event_timestamp_column="timestamp",
)

# Define the Feature View containing our numerical and categorical columns
user_features_view = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=365),
    schema=[
        Field(name="annual_income", dtype=Float32),
        Field(name="credit_usage_ratio", dtype=Float32),
        Field(name="payment_history_score", dtype=Float32),
        Field(name="loan_amount", dtype=Float32),
        Field(name="debt_to_income_ratio", dtype=Float32),
        Field(name="rolling_avg_spending_3m", dtype=Float32),
        Field(name="lag_income_diff_1m", dtype=Float32),
        Field(name="employment_status", dtype=String),
        Field(name="home_ownership", dtype=String),
        Field(name="loan_purpose", dtype=String),
        Field(name="defaulted", dtype=Int32),
    ],
    online=True,
    source=user_features_source,
)
