"""
Pipeline Step 4 — Model Monitor Setup.

Responsibilities:
  - Create a Snowflake Model Monitor for the deployed model version
  - Point the monitor at STREAMING_PATIENT_DATA as the live prediction source
  - Use TEST_FEATURES as the baseline reference distribution
  - Configure feature-level drift detection and prediction drift
  - Print monitor status to confirm successful setup

Run standalone:
    python -m source.pipeline.step4_monitor
"""

import json
import logging
import os
import sys

from snowflake.snowpark import Session

logger = logging.getLogger(__name__)

MONITOR_NAME = "PATIENT_RISK_MONITOR"


def run(config, session: Session, version_name: str = None) -> dict:
    """
    Set up model monitoring for the deployed patient risk model.

    The monitor watches STREAMING_PATIENT_DATA (live predictions) and
    compares the feature and prediction distributions against the
    TEST_FEATURES baseline captured at training time.

    Args:
        config: PipelineConfig loaded from config.yaml.
        session: Active Snowpark session.
        version_name: Version to monitor; None → resolves to latest.

    Returns:
        dict with keys: status, monitor_name, model_name, version_name.
    """
    from source.framework.deploy import ModelDeployer
    from source.framework.monitor import ModelMonitor

    db = config.snowflake.database
    schema = config.snowflake.schema_name
    model_name = config.model.model_name
    warehouse = config.snowflake.warehouse
    source_table = f"{db}.{schema}.STREAMING_PATIENT_DATA"
    baseline_table = f"{db}.{schema}.TEST_FEATURES"

    logger.info("=== Step 4: Model Monitor Setup ===")

    if version_name is None:
        deployer = ModelDeployer(
            session=session,
            registry_database=db,
            registry_schema=schema,
        )
        version_name = deployer.get_latest_version_name(model_name)

    logger.info(
        "Setting up monitor '%s' for %s/%s",
        MONITOR_NAME,
        model_name,
        version_name,
    )
    logger.info("  Source (live):   %s", source_table)
    logger.info("  Baseline:        %s", baseline_table)

    monitor = ModelMonitor(
        session=session,
        database=db,
        schema=schema,
    )

    monitor.create_monitor(
        monitor_name=MONITOR_NAME,
        model_name=model_name,
        version_name=version_name,
        source_table=source_table,
        baseline_table=baseline_table,
        timestamp_col="TIMESTAMP",
        prediction_col="PREDICTED_RISK_LEVEL",
        label_col="RISK_LEVEL",
        id_columns=["PATIENT_ID"],
        warehouse=warehouse,
    )

    status = monitor.get_monitor_status(MONITOR_NAME)
    logger.info("Monitor status: %s", status.get("status"))
    logger.info("=== Step 4 complete — model monitoring active ===")

    return {
        "status": "success",
        "monitor_name": MONITOR_NAME,
        "model_name": model_name,
        "version_name": version_name,
        "monitor_status": status.get("status"),
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
