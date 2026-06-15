import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, lag, coalesce, lit
from pyspark.sql.window import Window
from delta import configure_spark_with_delta_pip

from config import (
    BASELINE_RAW_PATH,
    NEW_RAW_PATH,
    FEATURES_DELTA_PATH,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
    TIMESTAMP_COL,
    ENTITY_ID
)

def get_spark_session():
    """Initializes Spark session with Delta Lake configurations."""
    builder = (
        SparkSession.builder
        .appName("FeatureStorePipeline")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        # Ensure we run in local mode with minimal memory overhead
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "2")
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()

def run_feature_engineering():
    spark = get_spark_session()
    print("Spark session started successfully with Delta Lake.")
    
    # 1. Read Raw Data
    print(f"Reading raw baseline data from {BASELINE_RAW_PATH}...")
    baseline_df = spark.read.parquet(BASELINE_RAW_PATH)
    
    print(f"Reading raw new data from {NEW_RAW_PATH}...")
    new_df = spark.read.parquet(NEW_RAW_PATH)
    
    # Tag datasets before union so we can track them
    baseline_df = baseline_df.withColumn("batch", lit("baseline"))
    new_df = new_df.withColumn("batch", lit("new_production"))
    
    # Union both datasets for unified feature processing (ensures window operations run across all snapshots)
    union_df = baseline_df.union(new_df)
    
    # 2. Window definitions for lag and rolling features
    # Order by user_id and timestamp
    window_spec = Window.partitionBy(ENTITY_ID).orderBy(TIMESTAMP_COL)
    
    # Lag feature: Income change from previous month
    income_lag = lag("annual_income", 1).over(window_spec)
    lag_income_diff_1m = col("annual_income") - coalesce(income_lag, col("annual_income"))
    
    # Rolling average: 3-month spending (average of current and 2 previous snapshots)
    rolling_spending = avg("monthly_spending").over(window_spec.rowsBetween(-2, 0))
    
    # Debt to income ratio (re-calculate or verify)
    debt_to_income = (col("loan_amount") * 0.1) / (col("annual_income") / 12)
    
    # 3. Apply transformations
    engineered_df = union_df.withColumn("lag_income_diff_1m", lag_income_diff_1m) \
                            .withColumn("rolling_avg_spending_3m", rolling_spending) \
                            .withColumn("debt_to_income_ratio", debt_to_income)
    
    # Fill any null values resulting from lags
    engineered_df = engineered_df.fillna({
        "lag_income_diff_1m": 0.0,
        "rolling_avg_spending_3m": 2500.0
    })
    
    # Select columns to store in Delta Lake
    select_cols = [ENTITY_ID, TIMESTAMP_COL, "batch"] + NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL]
    final_df = engineered_df.select(*select_cols)
    
    # 4. Save to Delta Lake
    print(f"Saving engineered features to Delta table at {FEATURES_DELTA_PATH}...")
    final_df.write.format("delta").mode("overwrite").save(FEATURES_DELTA_PATH)
    
    # Verify Delta table write
    delta_df = spark.read.format("delta").load(FEATURES_DELTA_PATH)
    print(f"Delta table created successfully. Number of records: {delta_df.count()}")
    delta_df.show(5)
    
    spark.stop()

if __name__ == "__main__":
    run_feature_engineering()
