"""
Pipeline Step 5 — Model Monitor Setup.

Responsibilities:
  - Create a Snowflake Model Monitor for the deployed model version
  - Point the monitor at an inference logs view as the live prediction source
  - Use the configured baseline table as the reference distribution
  - Configure feature-level drift detection and prediction drift
  - Enable segmented monitoring by ADMISSION_TYPE and INSURANCE_TYPE so
    drift metrics can be broken down per patient segment
  - Print monitor status to confirm successful setup

Run standalone:
    python -m source.pipeline.step5_monitor
"""

import json
import logging
import os
import sys

from snowflake.snowpark import Session

from source.framework.deploy import ModelDeployer
from source.framework.monitor import ModelMonitor
from source.pipeline.pipeline_utils import PipelineState
from source.utils import get_feature_config, get_model_version

logger = logging.getLogger(__name__)

_CATEGORICAL_COMPUTED = {"BMI_CATEGORY"}


def _create_inference_logs_view(
    session: Session, config, db: str, schema: str, model_name: str
) -> str:
    """
    Create (or replace) a flat view over INFERENCE_TABLE that exposes all
    request features and the predicted class as typed columns.

    Returns the fully-qualified view name.
    """
    feature_config = get_feature_config(config)
    raw_numeric = feature_config["raw_numeric_features"]
    categorical = feature_config["categorical_features"]
    computed = feature_config["computed_features"]

    def _numeric_col(col: str) -> str:
        return f'RECORD_ATTRIBUTES:"snow.model_serving.request.data.{col}"::FLOAT AS {col}'

    def _varchar_col(col: str) -> str:
        return f'RECORD_ATTRIBUTES:"snow.model_serving.request.data.{col}"::VARCHAR AS {col}'

    col_exprs = (
        [_numeric_col(c) for c in raw_numeric]
        + [_varchar_col(c) for c in categorical]
        + [_varchar_col(c) if c in _CATEGORICAL_COMPUTED else _numeric_col(c) for c in computed]
        + [
            'RECORD_ATTRIBUTES:"snow.model_serving.response.data.output_feature_0"::VARCHAR AS RISK_LEVEL'
        ]
    )

    view_fqn = f"{db}.{schema}.{config.monitor.inference_logs_view}"
    cols_sql = ",\n    ".join(col_exprs)

    session.sql(f"""
        CREATE OR REPLACE VIEW {view_fqn} AS
        SELECT
            MD5(TO_VARCHAR(TIMESTAMP) || RECORD_ATTRIBUTES::VARCHAR) AS RECORD_ID,
            TIMESTAMP,
            {cols_sql}
        FROM TABLE(INFERENCE_TABLE('{model_name}'))
        WHERE RECORD_ATTRIBUTES:"snow.model_serving.function.name" = 'predict'
    """).collect()

    logger.info("Created inference logs view: %s", view_fqn)
    return view_fqn


def _create_baseline_table(session: Session, config, db: str, schema: str) -> str:
    """
    Materialize a snapshot of the test features table with only the feature
    columns and types needed by the model monitor (no PATIENT_ID, all numerics
    as FLOAT to match the inference logs view).

    Returns the fully-qualified table name.
    """
    feature_config = get_feature_config(config)
    raw_numeric = feature_config["raw_numeric_features"]
    categorical = feature_config["categorical_features"]
    computed = feature_config["computed_features"]
    categorical.append("RISK_LEVEL")

    col_exprs = (
        [f"{c}::FLOAT AS {c}" for c in raw_numeric]
        + [f"{c}::VARCHAR AS {c}" for c in categorical]
        + [
            f"{c}::VARCHAR AS {c}" if c in _CATEGORICAL_COMPUTED else f"{c}::FLOAT AS {c}"
            for c in computed
        ]
    )
    cols_sql = ", ".join(col_exprs)
    table_fqn = f"{db}.{schema}.{config.monitor.baseline_table}"
    test_table = f"{db}.{schema}.{config.tables.test_features}"

    session.sql(f"""
        CREATE OR REPLACE TABLE {table_fqn} AS
        SELECT {cols_sql}
        FROM {test_table}
    """).collect()

    logger.info("Created baseline table: %s", table_fqn)
    return table_fqn


def run(config, session: Session, version_name: str = None) -> dict:
    """
    Set up model monitoring for the deployed patient risk model.

    Args:
        config: PipelineConfig loaded from config.yaml.
        session: Active Snowpark session.
        version_name: Version to monitor; None → resolves to latest.

    Returns:
        dict with keys: status, monitor_name, model_name, version_name.
    """
    db = config.snowflake.database
    schema = config.snowflake.schema_name
    model_name = config.model.model_name
    warehouse = config.snowflake.warehouse
    monitor_name = config.monitor.monitor_name

    logger.info("=== Step 5: Model Monitor Setup ===")

    if version_name is None:
        version_name = PipelineState(session, db, schema).get("training", "version_name")
    if version_name is None:
        version_name = get_model_version(session, db, schema, model_name).version_name

    source_table = _create_inference_logs_view(session, config, db, schema, model_name)
    baseline_table = _create_baseline_table(session, config, db, schema)

    logger.info(
        "Setting up monitor '%s' for %s/%s",
        monitor_name,
        model_name,
        version_name,
    )
    logger.info("  Source (inference logs view): %s", source_table)
    logger.info("  Baseline:                     %s", baseline_table)

    monitor = ModelMonitor(
        session=session,
        database=db,
        schema=schema,
    )

    monitor.create_monitor(
        monitor_name=monitor_name,
        model_name=model_name,
        version_name=version_name,
        source_table=source_table,
        baseline_table=baseline_table,
        timestamp_col="TIMESTAMP",
        prediction_col="RISK_LEVEL",
        label_col=None,
        id_columns=["RECORD_ID"],
        warehouse=warehouse,
        segment_columns=["ADMISSION_TYPE", "INSURANCE_TYPE"],
    )

    status = monitor.get_monitor_status(monitor_name)
    logger.info("Monitor status: %s", status.get("status"))

    alert_name = None
    if getattr(config.monitor, "drift_alert_enabled", False):
        alert_name = config.monitor.drift_alert_name
        logger.info(
            "Setting up drift alert '%s' to retrain on drift > %s",
            alert_name,
            config.monitor.drift_threshold,
        )
        monitor.setup_drift_alert(
            alert_name=alert_name,
            monitor_name=monitor_name,
            prediction_col="GLUCOSE_LEVEL",
            drift_metric=config.monitor.drift_metric,
            drift_threshold=config.monitor.drift_threshold,
            schedule=config.monitor.drift_alert_schedule,
            warehouse=warehouse,
            retrain_root_task=config.monitor.retrain_root_task,
        )
        logger.info(
            "Drift alert active — will EXECUTE TASK %s.%s.%s on drift",
            db,
            schema,
            config.monitor.retrain_root_task,
        )

    logger.info("=== Step 5 complete — model monitoring active ===")

    return {
        "status": "success",
        "monitor_name": monitor_name,
        "model_name": model_name,
        "version_name": version_name,
        "monitor_status": status.get("status"),
        "drift_alert": alert_name,
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

    version_name = os.getenv("MODEL_VERSION_NAME", None)

    result = run(config, session, version_name=version_name)
    logger.info("Result: %s", json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
