"""
Pipeline Step 3 — Model Deployment to REST Endpoint.

Responsibilities:
  - Retrieve the latest (promoted) model version from the registry
  - Deploy it as a scalable SPCS REST inference service
  - Wait for the service to reach RUNNING state
  - Fire a sample inference request to confirm end-to-end health
  - Log service metadata for Step 4 (monitor) to consume

Run standalone:
    python -m source.pipeline.step3_deploy
"""

import json
import logging
import os
import sys

import pandas as pd
from snowflake.snowpark import Session

logger = logging.getLogger(__name__)

SERVICE_NAME = "PATIENT_RISK_SERVICE"


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
    from source.framework.deploy import ModelDeployer
    from source.utils import get_feature_config

    db = config.snowflake.database
    schema = config.snowflake.schema_name
    model_name = config.model.model_name
    compute_pool = config.compute.compute_pool

    logger.info("=== Step 3: Model Deployment to REST Endpoint ===")

    deployer = ModelDeployer(
        session=session,
        registry_database=db,
        registry_schema=schema,
    )

    if version_name is None:
        version_name = deployer.get_latest_version_name(model_name)

    current_status = deployer.get_service_status(SERVICE_NAME)
    if current_status == "RUNNING":
        logger.info(
            "Service '%s' already RUNNING — dropping and re-deploying for fresh state",
            SERVICE_NAME,
        )
        deployer.drop_service(SERVICE_NAME)

    deployer.deploy(
        model_name=model_name,
        version_name=version_name,
        service_name=SERVICE_NAME,
        compute_pool=compute_pool,
        min_instances=1,
        max_instances=config.compute.max_nodes,
    )

    logger.info("Service '%s' is RUNNING — validating with sample inference", SERVICE_NAME)

    feature_config = get_feature_config(config)
    numeric_cols = feature_config["all_numeric_features"]
    categorical_cols = feature_config["all_categorical_features"]
    feature_columns = [c.upper() for c in numeric_cols + categorical_cols]

    test_table = f"{db}.{schema}.TEST_FEATURES"
    sample_df = session.table(test_table).limit(3).to_pandas()
    sample_df.columns = [c.upper() for c in sample_df.columns]
    sample_features = sample_df[feature_columns]

    predictions = deployer.predict(
        model_name=model_name,
        version_name=version_name,
        features_df=sample_features,
        function_name="predict",
    )

    sample_preds = predictions.to_dict(orient="records") if isinstance(predictions, pd.DataFrame) else str(predictions)
    logger.info("Sample predictions: %s", sample_preds)
    logger.info("=== Step 3 complete — REST endpoint live ===")

    return {
        "status": "success",
        "service_name": SERVICE_NAME,
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
