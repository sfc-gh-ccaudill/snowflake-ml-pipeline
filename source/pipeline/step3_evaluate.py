"""
Pipeline Step 3 — Model Evaluation & Promotion Gate.

Responsibilities:
  - Retrieve the latest registered model version produced by Step 2 training
  - Evaluate against TEST_FEATURES
  - Log evaluation metrics to the MODEL_METRICS table
  - Log evaluation metrics to the model version in the Snowflake ML Registry
  - Check promotion criteria thresholds
  - Return should_promote to gate the deploy task via the DAG WHEN clause

Run standalone:
    python -m source.pipeline.step2c_evaluate
"""

import json
import logging
import os
import sys

from snowflake.snowpark import Session

from source.framework.evaluator import Evaluator
from source.pipeline.pipeline_utils import PipelineState
from source.utils import get_feature_config, get_model_version

logger = logging.getLogger(__name__)


def run(config, session: Session) -> dict:
    """
    Evaluate the latest model version and check promotion criteria.

    Args:
        config: PipelineConfig loaded from config.yaml.
        session: Active Snowpark session.

    Returns:
        dict with keys: status, model_name, version_name, metrics,
                        should_promote, promotion_checks.
    """
    db = config.snowflake.database
    schema = config.snowflake.schema_name
    model_name = config.model.model_name
    test_table = f"{db}.{schema}.{config.tables.test_features}"
    metrics_table = f"{db}.{schema}.{config.tables.metrics_table}"
    promotion_thresholds = {
        "accuracy": config.evaluation.accuracy_threshold,
        "f1_macro": config.evaluation.f1_macro_threshold,
    }

    logger.info("=== Step 3: Model Evaluation & Promotion Gate ===")

    mv = get_model_version(session, db, schema, model_name)
    version_name = mv.version_name
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

    logger.info("Logging evaluation metrics to model version in registry")
    scalar_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    for metric_name, metric_value in scalar_metrics.items():
        try:
            mv.log_metric(metric_name, metric_value)
        except Exception as e:
            logger.warning("Could not log metric '%s' to registry: %s", metric_name, e)
    logger.info(
        "Logged %d metrics to model version %s/%s in registry",
        len(scalar_metrics),
        model_name,
        version_name,
    )

    promotion_result = evaluator.check_promotion_criteria(
        metrics=metrics,
        thresholds=promotion_thresholds,
    )

    report = evaluator.generate_report(metrics, promotion_result)
    logger.info("\n%s", report)

    if promotion_result["should_promote"]:
        logger.info("Model APPROVED for promotion to REST endpoint")
    else:
        failed = [k for k, v in promotion_result["checks"].items() if not v["passed"]]
        logger.warning("Model NOT approved — failed checks: %s", failed)

    PipelineState(session, db, schema).set(
        "evaluation", "should_promote", str(promotion_result["should_promote"]).lower()
    )

    logger.info("=== Step 3 complete ===")

    return {
        "status": "success",
        "should_promote": promotion_result["should_promote"],
        "promotion_checks": promotion_result.get("checks", {}),
        "metrics": metrics,
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
    logger.info("Result: %s", json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
