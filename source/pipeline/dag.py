"""
Task DAG Creation Script — Orchestrates the full ML pipeline via Snowflake Tasks.

This script creates four Python Stored Procedures (one per pipeline step) and
wires them into a chained Snowflake Task DAG:

    ROOT_TASK (Step 1: Feature Engineering)
        └── TRAIN_TASK (Step 2: Distributed Training + Evaluation)
              └── DEPLOY_TASK (Step 3: REST Endpoint Deployment)
                    └── MONITOR_TASK (Step 4: Model Monitor Setup)

The root task is scheduled weekly (Sunday 02:00 AM PT) but can be triggered
manually at any time with:
    EXECUTE TASK <DB>.<SCHEMA>.PIPELINE_ROOT_TASK;

Teardown:
    python -m source.pipeline.dag --teardown

Usage:
    python -m source.pipeline.dag           # create / update DAG
    python -m source.pipeline.dag --run     # create DAG + trigger immediately
    python -m source.pipeline.dag --teardown
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import zipfile
from pathlib import Path

from snowflake.snowpark import Session

logger = logging.getLogger(__name__)

PACKAGES = "('snowflake-ml-python', 'scikit-learn', 'pandas', 'numpy')"
RUNTIME = "3.11"

_STEP_PROCEDURES = [
    {
        "proc_name": "RUN_FEATURE_ENGINEERING",
        "task_name": "PIPELINE_ROOT_TASK",
        "imports": "source.pipeline.step1_feature_engineering",
        "step_func": "run",
        "description": "Step 1 — Feature Engineering & Feature Store",
        "after": None,
        "schedule": "USING CRON 0 2 * * 0 America/Los_Angeles",
        "when": None,
    },
    {
        "proc_name": "RUN_HPO",
        "task_name": "PIPELINE_HPO_TASK",
        "imports": "source.pipeline.step2b_hpo",
        "step_func": "run",
        "description": "Step 2b — Hyperparameter Tuning (skips internally if tune_hpo=false)",
        "after": "PIPELINE_ROOT_TASK",
        "schedule": None,
        "when": None,
    },
    {
        "proc_name": "RUN_TRAIN_EVALUATE",
        "task_name": "PIPELINE_TRAIN_TASK",
        "imports": "source.pipeline.step2_train_evaluate",
        "step_func": "run",
        "description": "Step 2 — Distributed Training & Evaluation",
        "after": "PIPELINE_HPO_TASK",
        "schedule": None,
        "when": None,
    },
    {
        "proc_name": "RUN_DEPLOY",
        "task_name": "PIPELINE_DEPLOY_TASK",
        "imports": "source.pipeline.step3_deploy",
        "step_func": "run",
        "description": "Step 3 — REST Endpoint Deployment",
        "after": "PIPELINE_TRAIN_TASK",
        "schedule": None,
        "when": None,
    },
    {
        "proc_name": "RUN_MONITOR_SETUP",
        "task_name": "PIPELINE_MONITOR_TASK",
        "imports": "source.pipeline.step4_monitor",
        "step_func": "run",
        "description": "Step 4 — Model Monitor Setup",
        "after": "PIPELINE_DEPLOY_TASK",
        "schedule": None,
        "when": None,
    },
]


def _upload_config_yaml(session: Session, stage: str) -> None:
    """Upload config.yaml to the stage so SPCS jobs can read it."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    config_path = str(repo_root / "source" / "config.yaml")
    logger.info("Uploading config.yaml to @%s", stage)
    session.file.put(config_path, f"@{stage}", auto_compress=False, overwrite=True)
    logger.info("config.yaml uploaded successfully")


def _upload_source_zip(session: Session, stage: str) -> None:
    """Zip the source/ package and upload it to the stage."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    source_dir = repo_root / "source"

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "source.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in source_dir.rglob("*.py"):
                zf.write(file, file.relative_to(repo_root))

        logger.info("Uploading source.zip to @%s", stage)
        session.file.put(
            zip_path,
            f"@{stage}",
            auto_compress=False,
            overwrite=True,
        )
        logger.info("source.zip uploaded successfully")


def _serialize_config(config) -> str:
    d = {
        "snowflake": {
            "connection_name": config.snowflake.connection_name,
            "database": config.snowflake.database,
            "schema": config.snowflake.schema_name,
            "warehouse": config.snowflake.warehouse,
        },
        "compute": {
            "compute_pool": config.compute.compute_pool,
            "instance_family": config.compute.instance_family,
            "min_nodes": config.compute.min_nodes,
            "max_nodes": config.compute.max_nodes,
        },
        "model": {
            "model_name": config.model.model_name,
            "target_platforms": config.model.target_platforms,
        },
            "pipeline": {
                "tune_hpo": config.pipeline.tune_hpo,
                "hpo_num_samples": config.pipeline.hpo_num_samples,
                "hpo_search_alg": config.pipeline.hpo_search_alg,
                "hpo_scheduler": config.pipeline.hpo_scheduler,
                "hpo_num_instances": config.pipeline.hpo_num_instances,
            },
        "tables": {"raw_data": config.tables.raw_data},
        "feature_config": {
            "raw_numeric_features": config.feature_config.raw_numeric_features,
            "categorical_features": config.feature_config.categorical_features,
            "computed_features": config.feature_config.computed_features,
            "target_column": config.feature_config.target_column,
            "class_labels": config.feature_config.class_labels,
        },
    }
    return json.dumps(d)


def _build_handler(step_import: str, step_func: str, config_json: str, predecessor_task: str = None) -> str:
    pred_block = ""
    if predecessor_task:
        pred_block = (
            "    try:\n"
            f"        raw = session.sql(\"SELECT SYSTEM$GET_PREDECESSOR_RETURN_VALUE('{predecessor_task}')\").collect()[0][0]\n"
            "        if raw:\n"
            "            pred = json.loads(raw) if isinstance(raw, str) else raw\n"
            "            if isinstance(pred, dict) and '_pipeline_config' in pred:\n"
            "                config_dict = pred['_pipeline_config']\n"
            "    except Exception:\n"
            "        pass\n"
        )
    result_block = (
        f"    result = {step_func}(config, session)\n"
        "    if not isinstance(result, dict):\n"
        "        result = dict(result=result)\n"
        "    result['_pipeline_config'] = config_dict\n"
        "    return result\n"
    )
    return (
        "import sys\n"
        "def handler(session):\n"
        "    import json\n"
        "    import logging\n"
        "    logging.basicConfig(level=logging.INFO,\n"
        "                        format='%(asctime)s %(levelname)s %(name)s - %(message)s')\n"
        f"    from {step_import} import {step_func}\n"
        "    from source.configs import get_config_from_dict\n"
        f"    config_dict = json.loads('{config_json}')\n"
        + pred_block +
        "    try:\n"
        "        raw = session.sql(\"SELECT SYSTEM$TASK_RUNTIME_INFO('CURRENT_TASK_CONFIG')\").collect()[0][0]\n"
        "        if raw:\n"
        "            overrides = json.loads(raw)\n"
        "            def _merge(base, patch):\n"
        "                for k, v in patch.items():\n"
        "                    if isinstance(v, dict) and isinstance(base.get(k), dict):\n"
        "                        _merge(base[k], v)\n"
        "                    else:\n"
        "                        base[k] = v\n"
        "            _merge(config_dict, overrides)\n"
        "    except Exception:\n"
        "        pass\n"
        "    config = get_config_from_dict(config_dict)\n"
        + result_block
    )


def _create_stored_procedure(
    session: Session,
    db: str,
    schema: str,
    warehouse: str,
    proc_name: str,
    step_import: str,
    step_func: str,
    stage: str,
    config_json: str,
    predecessor_task: str = None,
) -> None:
    """Create or replace a stored procedure that wraps one pipeline step."""
    full_proc = f"{db}.{schema}.{proc_name}"
    logger.info("Creating stored procedure: %s", full_proc)

    handler_body = _build_handler(step_import, step_func, config_json, predecessor_task)

    session.sql(f"""
        CREATE OR REPLACE PROCEDURE {full_proc}()
        RETURNS VARIANT
        LANGUAGE PYTHON
        RUNTIME_VERSION = '{RUNTIME}'
        PACKAGES = {PACKAGES}
        IMPORTS = ('@{stage}/source.zip')
        HANDLER = 'handler'
        EXECUTE AS CALLER
        AS $$
{handler_body}
$$
    """).collect()

    logger.info("Stored procedure %s created", full_proc)


def _create_task(
    session: Session,
    db: str,
    schema: str,
    warehouse: str,
    task_name: str,
    proc_name: str,
    after: str,
    schedule: str,
    when_col: str = None,
) -> None:
    """Create or replace a task that calls a stored procedure."""
    full_task = f"{db}.{schema}.{task_name}"
    full_proc = f"{db}.{schema}.{proc_name}"

    if after:
        full_after = f"{db}.{schema}.{after}"
        after_clause = f"AFTER {full_after}"
        schedule_clause = ""
        overlap_clause = ""
    else:
        after_clause = ""
        schedule_clause = f"SCHEDULE = '{schedule}'"
        overlap_clause = "ALLOW_OVERLAPPING_EXECUTION = FALSE"

    when_clause = f"WHEN {when_col}" if when_col is not None else ""

    logger.info("Creating task: %s", full_task)

    session.sql(f"""
        CREATE OR REPLACE TASK {full_task}
            WAREHOUSE = {warehouse}
            {schedule_clause}
            {after_clause}
            {overlap_clause}
            {when_clause}
        AS
            CALL {full_proc}()
    """).collect()

    logger.info("Task %s created", full_task)


def _ensure_task_privileges(session: Session) -> None:
    """Grant EXECUTE TASK on the account to the current role."""
    role = session.sql("SELECT CURRENT_ROLE()").collect()[0][0]
    try:
        session.sql(f"GRANT EXECUTE TASK ON ACCOUNT TO ROLE {role}").collect()
        logger.info("Granted EXECUTE TASK ON ACCOUNT to role %s", role)
    except Exception as e:
        logger.warning(
            "Could not auto-grant EXECUTE TASK (need ACCOUNTADMIN). "
            "Run manually: GRANT EXECUTE TASK ON ACCOUNT TO ROLE %s;  Error: %s",
            role, e,
        )


def create_dag(session: Session, config) -> None:
    """
    Create all stored procedures and tasks, then resume the root task.

    After calling this function the pipeline will run on the configured
    weekly schedule, or can be triggered immediately with EXECUTE TASK.

    Args:
        session: Active Snowpark session.
        config: PipelineConfig loaded from config.yaml.
    """
    db = config.snowflake.database
    schema = config.snowflake.schema_name
    warehouse = config.snowflake.warehouse
    stage = f"{db}.{schema}.JOB_PAYLOADS"

    logger.info("=== Creating ML Pipeline Task DAG ===")
    logger.info("  Database:  %s", db)
    logger.info("  Schema:    %s", schema)
    logger.info("  Warehouse: %s", warehouse)

    _upload_source_zip(session, stage)
    _upload_config_yaml(session, stage)

    config_json = _serialize_config(config)

    for step in _STEP_PROCEDURES:
        _create_stored_procedure(
            session=session,
            db=db,
            schema=schema,
            warehouse=warehouse,
            proc_name=step["proc_name"],
            step_import=step["imports"],
            step_func=step["step_func"],
            stage=stage,
            config_json=config_json,
            predecessor_task=step.get("after"),
        )

    for step in _STEP_PROCEDURES:
        when_val = step.get("when")
        _create_task(
            session=session,
            db=db,
            schema=schema,
            warehouse=warehouse,
            task_name=step["task_name"],
            proc_name=step["proc_name"],
            after=step["after"],
            schedule=step.get("schedule"),
            when_col=when_val(config) if callable(when_val) else when_val,
        )

    _ensure_task_privileges(session)

    for step in _STEP_PROCEDURES:
        if step["after"] is not None:
            full_task = f"{db}.{schema}.{step['task_name']}"
            logger.info("Resuming child task: %s", full_task)
            session.sql(f"ALTER TASK {full_task} RESUME").collect()

    logger.info("Resuming root task to activate schedule")
    session.sql(
        f"ALTER TASK {db}.{schema}.PIPELINE_ROOT_TASK RESUME"
    ).collect()

    logger.info("=== Task DAG created and scheduled ===")
    _print_dag_summary(session, db, schema)


def _print_dag_summary(session: Session, db: str, schema: str) -> None:
    """Log a human-readable summary of the task DAG."""
    rows = session.sql(f"SHOW TASKS IN SCHEMA {db}.{schema}").collect()
    logger.info("\nTask DAG summary:")
    logger.info("%-40s %-12s %-30s", "TASK", "STATE", "AFTER / SCHEDULE")
    logger.info("-" * 85)
    for row in rows:
        d = row.as_dict()
        name = d.get("name", "")
        if not any(name.upper() == s["task_name"] for s in _STEP_PROCEDURES):
            continue
        state = d.get("state", "")
        schedule = d.get("schedule", "") or d.get("predecessors", "")
        logger.info("%-40s %-12s %-30s", name, state, schedule)


def execute_dag(session: Session, config) -> None:
    """Trigger the root task immediately (runs full pipeline now)."""
    db = config.snowflake.database
    schema = config.snowflake.schema_name
    logger.info("Triggering pipeline execution: EXECUTE TASK PIPELINE_ROOT_TASK")
    session.sql(
        f"EXECUTE TASK {db}.{schema}.PIPELINE_ROOT_TASK"
    ).collect()
    logger.info("Pipeline triggered — monitor progress in Snowsight > Tasks")


def teardown_dag(session: Session, config) -> None:
    """
    Drop all pipeline tasks and stored procedures.

    Use this to cleanly remove the DAG before re-creating with updated code.
    """
    db = config.snowflake.database
    schema = config.snowflake.schema_name

    logger.info("=== Tearing down ML Pipeline Task DAG ===")

    for step in reversed(_STEP_PROCEDURES):
        full_task = f"{db}.{schema}.{step['task_name']}"
        full_proc = f"{db}.{schema}.{step['proc_name']}"
        logger.info("Dropping task: %s", full_task)
        session.sql(f"DROP TASK IF EXISTS {full_task}").collect()
        logger.info("Dropping procedure: %s", full_proc)
        session.sql(f"DROP PROCEDURE IF EXISTS {full_proc}()").collect()

    logger.info("=== Teardown complete ===")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Manage ML pipeline Task DAG")
    parser.add_argument("--build", action="store_true", help="Create DAG", default=True)
    parser.add_argument("--run", action="store_true", help="Trigger DAG immediately")
    parser.add_argument("--teardown", action="store_true", help="Drop all tasks and procedures")
    args = parser.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from source.configs import get_config
    from source.utils import get_session

    config = get_config("source/config.yaml")
    session = get_session(config.snowflake.connection_name)
    session.use_database(config.snowflake.database)
    session.use_schema(config.snowflake.schema_name)
    session.use_warehouse(config.snowflake.warehouse)

    if args.teardown:
        teardown_dag(session, config)
    else:
        if args.build:
            create_dag(session, config)
        if args.run:
            execute_dag(session, config)


if __name__ == "__main__":
    main()
