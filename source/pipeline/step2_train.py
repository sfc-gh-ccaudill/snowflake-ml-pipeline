"""
Pipeline Step 2 — Distributed Remote Training.

Responsibilities:
  - Submit training to SPCS via ML Jobs (num_instances controls distributed nodes)
  - Wait for completion and stream logs
  - Retrieve and return the newly registered model version name

Run standalone:
    python -m source.pipeline.step2_train
"""

import json
import logging
import os
import sys

from snowflake.snowpark import Session

from source.configs import config_to_dict, get_config
from source.framework.train import RemoteTrainer
from source.pipeline.pipeline_utils import PipelineState
from source.utils import get_model_version, get_session

logger = logging.getLogger(__name__)


def run(config, session: Session) -> dict:
    """
    Submit a distributed training job and return the registered model version.

    The number of training nodes is read from config.compute.max_nodes so
    it stays in sync with the compute pool capacity configured in config.yaml.

    Args:
        config: PipelineConfig loaded from config.yaml.
        session: Active Snowpark session.

    Returns:
        dict with keys: status, model_name, version_name, num_instances_used.
    """

    num_instances = config.train.num_nodes
    db = config.snowflake.database
    schema = config.snowflake.schema_name
    model_name = config.model.model_name
    compute_pool = config.compute.compute_pool

    state = PipelineState(session, db, schema)
    dataset_name = (
        state.get("feature_engineering", "training_dataset_name")
        or config.feature_store.training_dataset_name
    )
    dataset_version = state.get("feature_engineering", "training_dataset_version")

    stage = f"{db}.{schema}.{config.stages.job_payloads}"

    logger.info("=== Step 2: Distributed Remote Training ===")
    logger.info("Distributed training: %d nodes on compute pool '%s'", num_instances, compute_pool)
    logger.info("Training dataset: %s / %s", dataset_name, dataset_version)

    trainer = RemoteTrainer(
        session=session,
        compute_pool=compute_pool,
        stage=stage,
        source_dir="source",
    )

    job = trainer.submit(
        entrypoint="train.py",
        num_instances=num_instances,
        env_vars={
            "ML_PIPELINE_CONFIG":        json.dumps(config_to_dict(config)),
            "TRAINING_DATASET_NAME":     dataset_name,
            "TRAINING_DATASET_VERSION":  dataset_version or "",
        },
    )

    logger.info("ML Job submitted: %s", job.id)
    trainer.wait_and_log(job)

    logger.info("Training complete — retrieving latest model version")

    latest_version = get_model_version(session, db, schema, model_name)
    version_name = latest_version.version_name
    logger.info("Registered version: %s/%s", model_name, version_name)

    state.set("training", "version_name", version_name)
    logger.info("Version name written to pipeline state")

    logger.info("=== Step 2 complete ===")

    return {
        "status": "success",
        "model_name": model_name,
        "version_name": version_name,
        "num_instances_used": num_instances,
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    config = get_config("source/config.yaml")
    session = get_session(config.snowflake.connection_name)
    session.use_database(config.snowflake.database)
    session.use_schema(config.snowflake.schema_name)
    session.use_warehouse(config.snowflake.warehouse)

    num_instances = int(os.getenv("NUM_TRAINING_INSTANCES", str(config.train.num_nodes)))

    result = run(config, session)
    logger.info("Result: %s", json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
