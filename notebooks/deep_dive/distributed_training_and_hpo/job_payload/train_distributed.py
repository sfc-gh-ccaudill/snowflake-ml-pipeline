"""
Distributed XGBoost training entrypoint using XGBEstimator.

Uses Snowflake's native distributed XGBoost API with DataConnector
for efficient data loading and automatic worker coordination.

This script runs inside an SPCS container via ML Jobs.
"""

import logging
import os

from snowflake.ml.data.data_connector import DataConnector
from snowflake.ml.model.task import Task
from snowflake.ml.modeling.distributors.xgboost import XGBEstimator, XGBScalingConfig
from snowflake.ml.registry import Registry
from snowflake.snowpark import Session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


def main():
    database = os.environ.get("DATABASE", "ML_DEMO_PIPELINE_DB")
    schema = os.environ.get("SCHEMA", "HEALTHCARE")
    model_name = os.environ.get("MODEL_NAME", "PATIENT_RISK_XGB_DISTRIBUTED")
    max_depth = int(os.environ.get("MAX_DEPTH", "8"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "0.1"))
    n_estimators = int(os.environ.get("N_ESTIMATORS", "200"))

    logger.info("=== Distributed XGBoost Training (XGBEstimator) ===")
    logger.info(
        "  max_depth: %d | lr: %.4f | n_estimators: %d", max_depth, learning_rate, n_estimators
    )

    # == Init Session ==
    session = Session.builder.getOrCreate()
    session.use_database(database)
    session.use_schema(schema)

    # == Create DataConnectors for Distributed Training ==
    from snowflake.snowpark import functions as F

    label_col = "RISK_LEVEL"
    label_encoded_col = "RISK_LEVEL_ENCODED"
    label_map = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

    label_expr = F.col(label_col)
    for label, idx in label_map.items():
        label_expr = F.when(F.col(label_col) == F.lit(label), F.lit(idx)).otherwise(label_expr)

    exclude_cols = {"PATIENT_ID", "ENCOUNTER_ID", "TIMESTAMP", label_col, label_encoded_col}
    categorical_cols = {
        "GENDER",
        "PRIMARY_DIAGNOSIS",
        "ADMISSION_TYPE",
        "INSURANCE_TYPE",
        "BMI_CATEGORY",
    }

    train_df = session.table("TRAINING_FEATURES").with_column(label_encoded_col, label_expr)
    test_df = session.table("TEST_FEATURES").with_column(label_encoded_col, label_expr)

    input_cols = [
        c for c in train_df.columns if c not in exclude_cols and c not in categorical_cols
    ]

    train_connector = DataConnector.from_dataframe(train_df)
    eval_connector = DataConnector.from_dataframe(test_df)

    logger.info("  Features: %d columns", len(input_cols))
    logger.info("  Label: %s (encoded as %s)", label_col, label_encoded_col)

    # == Init Estimator Model ==
    estimator = XGBEstimator(
        params={
            "objective": "multi:softmax",
            "num_class": 4,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "eval_metric": "mlogloss",
        },
        scaling_config=XGBScalingConfig(
            num_workers=-1,
            num_cpu_per_worker=-1,
            use_gpu=None,
        ),
    )

    # == Train Model ==
    logger.info("Fitting XGBEstimator (distributed) ...")
    booster = estimator.fit(
        dataset=train_connector,
        input_cols=input_cols,
        label_col=label_encoded_col,
        eval_set=eval_connector,
        verbose_eval=10,
    )

    # == Evaluate Model ==
    eval_results = estimator.get_eval_results()
    final_loss = eval_results["eval"]["mlogloss"][-1]
    logger.info("Final eval mlogloss: %.4f", final_loss)

    # == Register Model ==
    registry = Registry(session, database_name=database, schema_name=schema)
    from datetime import datetime

    version_name = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    sample_input = train_df.select(input_cols).limit(10)

    registry.log_model(
        model=booster,
        model_name=model_name,
        version_name=version_name,
        sample_input_data=sample_input,
        metrics={"mlogloss": final_loss, "max_depth": max_depth, "learning_rate": learning_rate},
        task=Task.TABULAR_MULTI_CLASSIFICATION,
        target_platforms=["SNOWPARK_CONTAINER_SERVICES"],
        comment=f"Distributed XGBoost via XGBEstimator — {datetime.now().isoformat()}",
    )
    logger.info("Model registered: %s / %s", model_name, version_name)
    logger.info("=== Training complete ===")


if __name__ == "__main__":
    main()
