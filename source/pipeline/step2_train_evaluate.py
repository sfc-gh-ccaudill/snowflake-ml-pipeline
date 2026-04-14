"""
Pipeline Step 2 — Distributed Remote Training & Evaluation.

Responsibilities:
  - Submit training to SPCS via ML Jobs (num_instances controls distributed nodes)
  - Wait for completion and stream logs
  - Evaluate the newly registered model version against TEST_FEATURES
  - Check promotion criteria thresholds
  - Log all metrics to MODEL_METRICS table
  - Persist the promoted version name for Step 3

Run standalone:
    python -m source.pipeline.step2_train_evaluate
"""

import json
import logging
import os
import sys

from snowflake.snowpark import Session

logger = logging.getLogger(__name__)

PROMOTION_THRESHOLDS = {
    "accuracy": 0.80,
    "f1_macro": 0.75,
}


def run(config, session: Session, num_instances: int = 3) -> dict:
    """
    Submit a distributed training job, evaluate the result, and check promotion.

    Args:
        config: PipelineConfig loaded from config.yaml.
        session: Active Snowpark session.
        num_instances: Number of SPCS nodes for distributed training.
                       1 = single-node remote, 3 = distributed (default for demo).

    Returns:
        dict with keys: status, model_name, version_name, metrics,
                        should_promote, promotion_checks.
    """
    from source.framework.evaluator import Evaluator
    from source.framework.train import RemoteTrainer
    from source.utils import get_feature_config

    db = config.snowflake.database
    schema = config.snowflake.schema_name
    model_name = config.model.model_name
    test_table = f"{db}.{schema}.TEST_FEATURES"
    metrics_table = f"{db}.{schema}.MODEL_METRICS"
    compute_pool = config.compute.compute_pool
    stage = f"{db}.{schema}.JOB_PAYLOADS"

    logger.info("=== Step 2: Distributed Remote Training & Evaluation ===")
    logger.info("Distributed training: %d nodes on compute pool '%s'", num_instances, compute_pool)

    trainer = RemoteTrainer(
        session=session,
        compute_pool=compute_pool,
        stage=stage,
        source_dir="source",
    )

    job = trainer.submit(
        entrypoint="train.py",
        num_instances=num_instances,
    )

    logger.info("ML Job submitted: %s", job.id)

    trainer.wait_and_log(job)

    logger.info("Training complete — retrieving latest model version")

    from snowflake.ml.registry import Registry

    registry = Registry(session, database_name=db, schema_name=schema)
    model = registry.get_model(model_name)
    versions = model.versions()
    if not versions:
        raise RuntimeError(f"No versions found for model '{model_name}' after training")

    latest_version = versions[-1]
    version_name = latest_version.version_name
    logger.info("Evaluating version: %s/%s", model_name, version_name)

    feature_config = get_feature_config(config)
    numeric_cols = feature_config["all_numeric_features"]
    categorical_cols = feature_config["all_categorical_features"]
    feature_columns = [c.upper() for c in numeric_cols + categorical_cols]
    target_column = feature_config["target_column"].upper()
    class_labels = feature_config["class_labels"]

    evaluator = Evaluator(session=session)
    metrics = evaluator.evaluate_from_registry(
        model_name=model_name,
        model_version=version_name,
        registry_database=db,
        registry_schema=schema,
        test_table=test_table,
        feature_columns=feature_columns,
        target_column=target_column,
        class_labels=class_labels,
    )

    evaluator.log_metrics(
        metrics=metrics,
        metrics_table=metrics_table,
        model_name=model_name,
        model_version=version_name,
    )

    promotion_result = evaluator.check_promotion_criteria(
        metrics=metrics,
        thresholds=PROMOTION_THRESHOLDS,
    )

    report = evaluator.generate_report(metrics, promotion_result)
    logger.info("\n%s", report)

    if promotion_result["should_promote"]:
        logger.info("Model APPROVED for promotion to REST endpoint")
    else:
        failed = [k for k, v in promotion_result["checks"].items() if not v["passed"]]
        logger.warning("Model NOT approved — failed checks: %s", failed)

    logger.info("=== Step 2 complete ===")

    return {
        "status": "success",
        "model_name": model_name,
        "version_name": version_name,
        "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float, str))},
        "should_promote": promotion_result["should_promote"],
        "promotion_checks": promotion_result["checks"],
        "num_instances_used": num_instances,
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

    num_instances = int(os.getenv("NUM_TRAINING_INSTANCES", "3"))

    result = run(config, session, num_instances=num_instances)
    logger.info("Result: %s", json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
