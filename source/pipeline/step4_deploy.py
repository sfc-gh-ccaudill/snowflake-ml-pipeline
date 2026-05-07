"""
Pipeline Step 4 — Model Deployment to REST Endpoint.

Responsibilities:
  - Gate on should_promote from PipelineState (skip if evaluation failed)
  - Drop the existing service if present, then deploy the new version
  - Fire a sample inference request to confirm end-to-end health

Run standalone:
    python -m source.pipeline.step4_deploy
"""

import json
import logging
import os
import sys

import pandas as pd
from snowflake.snowpark import Session

from source.framework.deploy import ModelDeployer
from source.pipeline.pipeline_utils import PipelineState
from source.utils import get_feature_config

logger = logging.getLogger(__name__)


def run(config, session: Session, version_name: str = None) -> dict:
    """
    Deploy the latest (or specified) model version as a SPCS REST service.

    Args:
        config: PipelineConfig loaded from config.yaml.
        session: Active Snowpark session.
        version_name: Specific version to deploy; None → use latest.

    Returns:
        dict with keys: status, service_name, model_name, version_name,
                        sample_predictions.
    """
    db = config.snowflake.database
    schema = config.snowflake.schema_name
    model_name = config.model.model_name
    compute_pool = config.compute.compute_pool
    service_name = config.deploy.service_name
    min_instances = config.deploy.min_instances
    max_instances = config.deploy.max_instances
    auto_suspend_secs = config.deploy.auto_suspend_secs
    test_table = f"{db}.{schema}.{config.tables.test_features}"

    state = PipelineState(session, db, schema)
    should_promote = state.get("evaluation", "should_promote", default="false").lower() == "true"
    if not should_promote:
        logger.info("Skipping deployment — evaluation did not pass promotion thresholds")
        return {"status": "skipped", "reason": "should_promote=false"}

    logger.info("=== Step 4: Model Deployment to REST Endpoint ===")

    deployer = ModelDeployer(
        session=session,
        registry_database=db,
        registry_schema=schema,
    )

    if version_name is None:
        version_name = state.get("training", "version_name")
    if version_name is None:
        version_name = deployer.get_latest_version_name(model_name)

    # --- Step 1: drop existing service if present -------------------------
    if deployer.service_exists(service_name):
        logger.info("Dropping existing service '%s'", service_name)
        deployer.drop_service(service_name)

    # --- Step 2: configure compute pool to prevent idle suspension ---------
    # deployer.configure_compute_pool_auto_suspend(compute_pool, auto_suspend_secs)

    # --- Step 3: deploy new service and wait until RUNNING ----------------
    logger.info("Deploying %s/%s as service '%s'", model_name, version_name, service_name)
    deployer.deploy(
        model_name=model_name,
        version_name=version_name,
        service_name=service_name,
        compute_pool=compute_pool,
        min_instances=min_instances,
        max_instances=max_instances,
        auto_suspend_secs=auto_suspend_secs,
    )
    logger.info("Service '%s' is RUNNING", service_name)

    # --- Step 4: update the registry default version ---------------------
    deployer.set_default_version(model_name, version_name)

    # --- Step 5: validate end-to-end with sample inference ----------------
    feature_config = get_feature_config(config)
    feature_columns = [
        c.upper()
        for c in feature_config["all_numeric_features"] + feature_config["all_categorical_features"]
    ]

    sample_df = session.table(test_table).limit(3).to_pandas()
    sample_df.columns = [c.upper() for c in sample_df.columns]

    predictions = deployer.predict(
        model_name=model_name,
        service_name=service_name,
        version_name=version_name,
        features_df=sample_df[feature_columns],
        function_name="predict",
    )

    sample_preds = (
        predictions.to_dict(orient="records")
        if isinstance(predictions, pd.DataFrame)
        else str(predictions)
    )
    logger.info("Sample predictions: %s", sample_preds)
    logger.info("=== Step 4 complete — REST endpoint live, default version updated ===")

    return {
        "status": "success",
        "service_name": service_name,
        "model_name": model_name,
        "version_name": version_name,
        "sample_predictions": sample_preds,
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
