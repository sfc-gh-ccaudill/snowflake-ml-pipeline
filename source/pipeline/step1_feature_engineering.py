"""
Pipeline Step 1 — Feature Engineering & Feature Store.

Responsibilities:
  - Initialize the Snowflake Feature Store
  - Register the PATIENT entity
  - Build a Snowpark DataFrame with 4 engineered features derived from raw vitals
  - Register PATIENT_FEATURES feature view (auto-refreshes every minute)
  - Retrieve features for the training and test populations
  - Persist TRAINING_FEATURES and TEST_FEATURES tables for downstream steps

Run standalone:
    python -m source.pipeline.step1_feature_engineering
"""

import logging
import sys
import os

import snowflake.snowpark.functions as F
from snowflake.snowpark import Session

logger = logging.getLogger(__name__)


def _build_feature_dataframe(session: Session, raw_table: str):
    """
    Build a Snowpark DataFrame with all raw columns plus 4 engineered features.

    Engineered features:
        SHOCK_INDEX           = HEART_RATE / SYSTOLIC_BP
        PULSE_PRESSURE        = SYSTOLIC_BP - DIASTOLIC_BP
        BMI_CATEGORY          = categorical bucket of BMI
        VITAL_SIGNS_SEVERITY  = integer score 0-9 summing abnormal vital flags
    """
    df = session.table(raw_table)

    df = df.with_column(
        "SHOCK_INDEX",
        F.col("HEART_RATE") / F.col("SYSTOLIC_BP"),
    ).with_column(
        "PULSE_PRESSURE",
        F.col("SYSTOLIC_BP") - F.col("DIASTOLIC_BP"),
    ).with_column(
        "BMI_CATEGORY",
        F.when(F.col("BMI") < 18.5, F.lit("UNDERWEIGHT"))
         .when(F.col("BMI") < 25.0, F.lit("NORMAL"))
         .when(F.col("BMI") < 30.0, F.lit("OVERWEIGHT"))
         .otherwise(F.lit("OBESE")),
    ).with_column(
        "VITAL_SIGNS_SEVERITY",
        (
            F.when((F.col("HEART_RATE") > 100) | (F.col("HEART_RATE") < 50), F.lit(1)).otherwise(F.lit(0))
            + F.when((F.col("SYSTOLIC_BP") > 180) | (F.col("SYSTOLIC_BP") < 90), F.lit(2)).otherwise(F.lit(0))
            + F.when(F.col("OXYGEN_SATURATION") < 92, F.lit(2)).otherwise(
                F.when(F.col("OXYGEN_SATURATION") < 95, F.lit(1)).otherwise(F.lit(0))
            )
            + F.when(F.col("RESPIRATORY_RATE") > 24, F.lit(1)).otherwise(F.lit(0))
            + F.when((F.col("TEMPERATURE") > 38.5) | (F.col("TEMPERATURE") < 36.0), F.lit(1)).otherwise(F.lit(0))
        ),
    )

    return df


def run(config, session: Session) -> dict:
    """
    Execute feature engineering and persist TRAINING_FEATURES / TEST_FEATURES.

    Args:
        config: PipelineConfig loaded from config.yaml.
        session: Active Snowpark session.

    Returns:
        dict with keys: status, training_rows, test_rows.
    """
    from source.framework.feature_store import FeatureStoreManager

    db = config.snowflake.database
    schema = config.snowflake.schema_name
    warehouse = config.snowflake.warehouse
    raw_table = f"{db}.{schema}.RAW_PATIENT_DATA"
    training_table = f"{db}.{schema}.TRAINING_FEATURES"
    test_table = f"{db}.{schema}.TEST_FEATURES"

    logger.info("=== Step 1: Feature Engineering & Feature Store ===")

    fs_manager = FeatureStoreManager(
        session=session,
        database=db,
        schema_name=schema,
        warehouse=warehouse,
    )

    fs = fs_manager.initialize_feature_store()

    entity = fs_manager.create_entity(
        entity_name="PATIENT",
        join_keys=["PATIENT_ID"],
        description="Hospital patient identified by PATIENT_ID",
    )

    feature_df = _build_feature_dataframe(session, raw_table)

    feature_view = fs_manager.create_feature_view(
        feature_view_name="PATIENT_FEATURES",
        entities=[entity],
        features_df=feature_df,
        version="v1",
        timestamp_column="TIMESTAMP",
        refresh_freq="1 minute",
        description=(
            "Raw vitals + 4 engineered features: "
            "SHOCK_INDEX, PULSE_PRESSURE, BMI_CATEGORY, VITAL_SIGNS_SEVERITY"
        ),
    )

    logger.info("Feature view registered: PATIENT_FEATURES v1")

    # Spine carries only the join key + timestamp; the feature view (built from
    # the full raw table) supplies every other column including the target.
    spine_df = session.table(raw_table).select("PATIENT_ID", "TIMESTAMP")

    training_df = fs.retrieve_feature_values(
        spine_df=spine_df.sample(frac=0.8),
        features=[feature_view],
        spine_timestamp_col="TIMESTAMP",
    )

    test_df = fs.retrieve_feature_values(
        spine_df=spine_df.sample(frac=0.2),
        features=[feature_view],
        spine_timestamp_col="TIMESTAMP",
    )

    training_df.write.mode("overwrite").save_as_table(training_table)
    test_df.write.mode("overwrite").save_as_table(test_table)

    training_rows = session.table(training_table).count()
    test_rows = session.table(test_table).count()

    logger.info("TRAINING_FEATURES: %d rows -> %s", training_rows, training_table)
    logger.info("TEST_FEATURES:     %d rows -> %s", test_rows, test_table)
    logger.info("=== Step 1 complete ===")

    return {
        "status": "success",
        "training_table": training_table,
        "training_rows": training_rows,
        "test_table": test_table,
        "test_rows": test_rows,
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from source.configs import get_config
    from source.utils import get_session

    config = get_config("source/config.yaml")
    session = get_session(config.snowflake.connection_name)
    session.use_database(config.snowflake.database)
    session.use_schema(config.snowflake.schema_name)
    session.use_warehouse(config.snowflake.warehouse)

    result = run(config, session)
    logger.info("Result: %s", result)


if __name__ == "__main__":
    main()
