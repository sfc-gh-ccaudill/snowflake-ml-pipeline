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

from datetime import datetime, timezone
import logging
import os
import sys

from snowflake.snowpark import Session
import snowflake.snowpark.functions as F

from source.framework.feature_store import FeatureStoreManager
from source.pipeline.pipeline_utils import PipelineState

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

    df = (
        df.with_column(
            "SHOCK_INDEX",
            F.col("HEART_RATE") / F.col("SYSTOLIC_BP"),
        )
        .with_column(
            "PULSE_PRESSURE",
            F.col("SYSTOLIC_BP") - F.col("DIASTOLIC_BP"),
        )
        .with_column(
            "BMI_CATEGORY",
            F.when(F.col("BMI") < 18.5, F.lit("UNDERWEIGHT"))
            .when(F.col("BMI") < 25.0, F.lit("NORMAL"))
            .when(F.col("BMI") < 30.0, F.lit("OVERWEIGHT"))
            .otherwise(F.lit("OBESE")),
        )
        .with_column(
            "VITAL_SIGNS_SEVERITY",
            (
                F.when(
                    (F.col("HEART_RATE") > 100) | (F.col("HEART_RATE") < 50), F.lit(1)
                ).otherwise(F.lit(0))
                + F.when(
                    (F.col("SYSTOLIC_BP") > 180) | (F.col("SYSTOLIC_BP") < 90), F.lit(2)
                ).otherwise(F.lit(0))
                + F.when(F.col("OXYGEN_SATURATION") < 92, F.lit(2)).otherwise(
                    F.when(F.col("OXYGEN_SATURATION") < 95, F.lit(1)).otherwise(F.lit(0))
                )
                + F.when(F.col("RESPIRATORY_RATE") > 24, F.lit(1)).otherwise(F.lit(0))
                + F.when(
                    (F.col("TEMPERATURE") > 38.5) | (F.col("TEMPERATURE") < 36.0), F.lit(1)
                ).otherwise(F.lit(0))
            ),
        )
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
    db = config.snowflake.database
    schema = config.snowflake.schema_name
    warehouse = config.snowflake.warehouse
    raw_table = config.full_raw_table
    test_table = f"{db}.{schema}.{config.tables.test_features}"

    fs_cfg = config.feature_store
    entity_name = fs_cfg.entity_name
    entity_join_keys = fs_cfg.entity_join_keys
    feature_view_name = fs_cfg.feature_view_name
    feature_view_version = fs_cfg.feature_view_version
    feature_view_refresh_freq = fs_cfg.feature_view_refresh_freq
    training_dataset_name = fs_cfg.training_dataset_name
    training_dataset_version = datetime.now(timezone.utc).strftime("v_%Y%m%d_%H%M%S")

    logger.info("=== Step 1: Feature Engineering & Feature Store ===")

    fs_manager = FeatureStoreManager(
        session=session,
        database=db,
        schema_name=schema,
        warehouse=warehouse,
    )

    fs = fs_manager.initialize_feature_store()

    entity = fs_manager.create_entity(
        entity_name=entity_name,
        join_keys=entity_join_keys,
        description=f"Hospital patient identified by {entity_join_keys[0]}",
    )

    feature_df = _build_feature_dataframe(session, raw_table)

    feature_view = fs_manager.create_feature_view(
        feature_view_name=feature_view_name,
        entities=[entity],
        features_df=feature_df,
        version=feature_view_version,
        timestamp_column="TIMESTAMP",
        refresh_freq=feature_view_refresh_freq,
        description=(
            "Raw vitals + 4 engineered features: SHOCK_INDEX, PULSE_PRESSURE, BMI_CATEGORY, VITAL_SIGNS_SEVERITY"
        ),
    )

    logger.info("Feature view registered: %s %s", feature_view_name, feature_view_version)

    state = PipelineState(session, db, schema)
    logger.info("Dataset version for this run: %s", training_dataset_version)

    state.set("feature_engineering", "training_dataset_name", training_dataset_name)
    state.set("feature_engineering", "training_dataset_version", training_dataset_version)
    logger.info("Dataset info written to pipeline state")

    spine_df = (
        session.table(raw_table)
        .select("PATIENT_ID", "TIMESTAMP")
        .with_column("_SPLIT", F.uniform(F.lit(0), F.lit(1), F.random()))
    )
    train_spine = spine_df.filter(F.col("_SPLIT") < 0.8).drop("_SPLIT")
    test_spine = spine_df.filter(F.col("_SPLIT") >= 0.8).drop("_SPLIT")

    training_dataset = fs.generate_dataset(
        spine_df=train_spine,
        features=[feature_view],
        spine_timestamp_col="TIMESTAMP",
        name=training_dataset_name,
        version=training_dataset_version,
        desc=f"Point-in-time training snapshot for PATIENT_RISK model — {training_dataset_version}",
    )
    logger.info("Training dataset %s/%s created", training_dataset_name, training_dataset_version)

    test_df = fs.retrieve_feature_values(
        spine_df=test_spine,
        features=[feature_view],
        spine_timestamp_col="TIMESTAMP",
    )
    test_df.write.mode("overwrite").save_as_table(test_table)

    training_rows = training_dataset.read.to_snowpark_dataframe().count()
    test_rows = session.table(test_table).count()

    logger.info("%s: %d rows (versioned Dataset)", training_dataset_name, training_rows)
    logger.info("%s: %d rows -> %s", config.tables.test_features, test_rows, test_table)
    logger.info("=== Step 1 complete ===")

    return {
        "status": "success",
        "training_dataset_name": training_dataset_name,
        "training_dataset_version": training_dataset_version,
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
