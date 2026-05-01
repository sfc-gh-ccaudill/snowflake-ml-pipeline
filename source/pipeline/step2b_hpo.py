"""
Pipeline Step 2b — Optional Hyperparameter Tuning via Ray Tune on SPCS.

Submits a Ray Tune HPO job using RemoteTrainer.submit_hpo(). The search
space targets the XGBoost / GradientBoosting classifier in train_hpo.py.
Best results are written to the HPO_RESULTS table for Step 2 to consume.

Skipped automatically when TUNE_HPO = FALSE in PIPELINE_RUN_CONFIG.

Run standalone:
    python -m source.pipeline.step2b_hpo
"""

import json
import logging
import os
import sys

from snowflake.snowpark import Session

from source.configs import config_to_dict
from source.framework.train import RayHPOConfig, RemoteTrainer

logger = logging.getLogger(__name__)


def run(config, session: Session) -> dict:
    """
    Submit a Ray Tune HPO job and wait for completion.

    Returns {"status": "skipped"} immediately when config.pipeline.tune_hpo is False.

    Args:
        config: PipelineConfig (pipeline.hpo_* fields control the search).
        session: Active Snowpark session.

    Returns:
        dict with keys: status, job_id, best_score, best_params  — or
        {"status": "skipped", "reason": "tune_hpo=false"} when skipped.
    """
    if not config.tune.enabled:
        logger.info("tune_hpo is False — skipping hyperparameter tuning")
        return {"status": "skipped", "reason": "tune_hpo=false"}

    db = config.snowflake.database
    schema = config.snowflake.schema_name
    compute_pool = config.compute.compute_pool
    stage = f"{db}.{schema}.{config.stages.job_payloads}"
    tune_cfg = config.tune

    logger.info("=== Step 2b: Hyperparameter Tuning ===")
    logger.info(
        "search_alg=%s  scheduler=%s  num_samples=%d  num_instances=%d",
        tune_cfg.search_alg,
        tune_cfg.scheduler,
        tune_cfg.num_samples,
        tune_cfg.num_instances,
    )

    hpo_config = RayHPOConfig(
        search_space=tune_cfg.search_space,
        metric=tune_cfg.metric,
        mode=tune_cfg.mode,
        num_samples=tune_cfg.num_samples,
        scheduler=tune_cfg.scheduler,
        search_alg=tune_cfg.search_alg,
    )

    trainer = RemoteTrainer(
        session=session,
        compute_pool=compute_pool,
        stage=stage,
        source_dir="source",
    )

    job = trainer.submit_hpo(
        hpo_config=hpo_config,
        entrypoint="train_hpo.py",
        num_instances=tune_cfg.num_instances,
        env_vars={"ML_PIPELINE_CONFIG": json.dumps(config_to_dict(config))},
    )

    logger.info("HPO job submitted: %s", job.id)
    trainer.wait_and_log(job)

    best_row = session.sql(f"""
        SELECT BEST_PARAMS, BEST_SCORE
        FROM {db}.{schema}.HPO_RESULTS
        WHERE JOB_ID = '{job.id}'
        LIMIT 1
    """).collect()

    best_params = json.loads(best_row[0]["BEST_PARAMS"]) if best_row else {}
    best_score = float(best_row[0]["BEST_SCORE"]) if best_row else 0.0

    logger.info("HPO complete — best %s: %.4f  params: %s", config.tune.metric, best_score, best_params)
    logger.info("=== Step 2b complete ===")

    return {
        "status": "success",
        "job_id": job.id,
        "best_score": best_score,
        "best_params": best_params,
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
