"""
Task DAG Creation Script — Orchestrates the full ML pipeline via Snowflake Tasks.

This script creates Python Stored Procedures (one per pipeline step) and
wires them into a chained Snowflake Task DAG:

    PIPELINE_FEATURE_ENG_TASK  (Step 1: Feature Engineering)
        └── HPO_TASK       (Step 2b: Hyperparameter Tuning — optional)
              └── TRAIN_TASK     (Step 2:  Distributed Training)
                    └── EVALUATE_TASK  (Step 2c: Evaluation & Promotion Gate)
                          └── DEPLOY_TASK    (Step 3:  REST Endpoint — runs only when should_promote=true)
                                └── MONITOR_TASK   (Step 4:  Model Monitor Setup)

The root task is scheduled weekly (Sunday 02:00 AM PT) but can be triggered
manually at any time with:
    EXECUTE TASK <DB>.<SCHEMA>.PIPELINE_FEATURE_ENG_TASK;

Teardown:
    python -m source.pipeline.dag --teardown

Usage:
    python -m source.pipeline.dag           # create / update DAG
    python -m source.pipeline.dag --run     # create DAG + trigger immediately
    python -m source.pipeline.dag --teardown
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import zipfile
from pathlib import Path

from snowflake.snowpark import Session
from snowflake.snowpark.types import VariantType

from source.configs import config_to_dict
from source.pipeline.step_handler import make_handler

logger = logging.getLogger(__name__)

PACKAGES = ["snowflake-ml-python", "scikit-learn", "pandas", "numpy"]

_STEPS: list[dict] = [
    {
        "proc_name": "RUN_FEATURE_ENGINEERING",
        "task_name": "PIPELINE_FEATURE_ENG_TASK",
        "task_key":  "feature_engineering",
        "imports":   "source.pipeline.step1_feature_engineering",
        "step_func": "run",
        "description": "Step 1 — Feature Engineering & Feature Store",
        "after":     None,
        "schedule":  "USING CRON 0 2 * * 0 America/Los_Angeles",
        "when":      None,
        "is_final":  False,
    },
    {
        "proc_name": "RUN_HPO",
        "task_name": "PIPELINE_HPO_TASK",
        "task_key":  "hpo",
        "imports":   "source.pipeline.step2b_hpo",
        "step_func": "run",
        "description": "Step 2b — Hyperparameter Tuning (skips internally if tune.enabled=false)",
        "after":     "PIPELINE_FEATURE_ENG_TASK",
        "schedule":  None,
        "when":      None,
        "is_final":  False,
    },
    {
        "proc_name": "RUN_TRAIN",
        "task_name": "PIPELINE_TRAIN_TASK",
        "task_key":  "training",
        "imports":   "source.pipeline.step2_train",
        "step_func": "run",
        "description": "Step 2 — Distributed Training",
        "after":     "PIPELINE_HPO_TASK",
        "schedule":  None,
        "when":      None,
        "is_final":  False,
    },
    {
        "proc_name": "RUN_EVALUATE",
        "task_name": "PIPELINE_EVALUATE_TASK",
        "task_key":  "evaluation",
        "imports":   "source.pipeline.step3_evaluate",
        "step_func": "run",
        "description": "Step 3 — Model Evaluation & Promotion Gate",
        "after":     "PIPELINE_TRAIN_TASK",
        "schedule":  None,
        "when":      None,
        "is_final":  False,
    },
    {
        "proc_name": "RUN_DEPLOY",
        "task_name": "PIPELINE_DEPLOY_TASK",
        "task_key":  "deployment",
        "imports":   "source.pipeline.step4_deploy",
        "step_func": "run",
        "description": "Step 4 — REST Endpoint Deployment",
        "after":     "PIPELINE_EVALUATE_TASK",
        "schedule":  None,
        "when":      None,
        "is_final":  False,
    },
    {
        "proc_name": "RUN_MONITOR_SETUP",
        "task_name": "PIPELINE_MONITOR_TASK",
        "task_key":  "monitoring",
        "imports":   "source.pipeline.step5_monitor",
        "step_func": "run",
        "description": "Step 5 — Model Monitor Setup",
        "after":     "PIPELINE_DEPLOY_TASK",
        "schedule":  None,
        "when":      None,
        "is_final":  True,
    },
]


class PipelineDAG:
    """Manages the lifecycle of the ML pipeline Snowflake Task DAG.

    Attributes:
        session:   Active Snowpark session.
        config:    PipelineConfig loaded from config.yaml.
        db:        Target Snowflake database.
        schema:    Target Snowflake schema.
        warehouse: Warehouse used by all tasks.
        stage:     Fully-qualified stage name for uploads and SP handler files.
    """

    def __init__(self, session: Session, config) -> None:
        self.session = session
        self.config = config
        self.db = config.snowflake.database
        self.schema = config.snowflake.schema_name
        self.warehouse = config.snowflake.warehouse
        self.stage = f"{self.db}.{self.schema}.{config.stages.job_payloads}"

    def build(self) -> None:
        """Create all stored procedures and tasks, then resume the root task.

        After calling this the pipeline will run on the configured weekly
        schedule, or can be triggered immediately via ``run()``.
        """
        logger.info("=== Creating ML Pipeline Task DAG ===")
        logger.info("  Database:  %s", self.db)
        logger.info("  Schema:    %s", self.schema)
        logger.info("  Warehouse: %s", self.warehouse)

        source_zip_path, deploy_ts = self._upload_source_zip()
        self._upload_config_yaml()

        config_json = json.dumps(config_to_dict(self.config))

        for step in _STEPS:
            self._register_step_procedure(step, source_zip_path, config_json, deploy_ts)


        for step in _STEPS:
            self._create_task(step)

        self._ensure_task_privileges()

        for step in _STEPS:
            if step["after"] is not None:
                full_task = f"{self.db}.{self.schema}.{step['task_name']}"
                logger.info("Resuming child task: %s", full_task)
                self.session.sql(f"ALTER TASK {full_task} RESUME").collect()

        logger.info("Resuming root task to activate schedule")
        self.session.sql(
            f"ALTER TASK {self.db}.{self.schema}.PIPELINE_FEATURE_ENG_TASK RESUME"
        ).collect()

        self._prune_old_deploys()
        logger.info("=== Task DAG created and scheduled ===")
        self._print_dag_summary()

    def run(self) -> None:
        """Trigger the root task immediately (runs the full pipeline now)."""
        logger.info("Triggering pipeline execution: EXECUTE TASK PIPELINE_FEATURE_ENG_TASK")
        self.session.sql(
            f"EXECUTE TASK {self.db}.{self.schema}.PIPELINE_FEATURE_ENG_TASK"
        ).collect()
        logger.info("Pipeline triggered — monitor progress in Snowsight > Tasks")

    def teardown(self) -> None:
        """Drop all pipeline tasks and stored procedures in reverse order."""
        logger.info("=== Tearing down ML Pipeline Task DAG ===")
        for step in reversed(_STEPS):
            full_task = f"{self.db}.{self.schema}.{step['task_name']}"
            full_proc = f"{self.db}.{self.schema}.{step['proc_name']}"
            logger.info("Dropping task: %s", full_task)
            self.session.sql(f"DROP TASK IF EXISTS {full_task}").collect()
            logger.info("Dropping procedure: %s", full_proc)
            self.session.sql(f"DROP PROCEDURE IF EXISTS {full_proc}()").collect()
        logger.info("=== Teardown complete ===")

    def _upload_source_zip(self) -> str:
        """Zip the source/ package and upload it to a timestamped stage path.

        Returns the fully-qualified stage path of the uploaded zip so callers
        can embed the exact path in IMPORTS — bypassing Snowflake's per-path
        stage-file cache that would otherwise serve a stale version.
        """
        import time

        repo_root = Path(__file__).resolve().parent.parent.parent
        source_dir = repo_root / "source"
        deploy_ts = int(time.time())
        stage_subdir = f"deploys/{deploy_ts}"

        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "source.zip")
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file in source_dir.rglob("*.py"):
                    zf.write(file, file.relative_to(repo_root))
                config_yaml = source_dir / "config.yaml"
                if config_yaml.exists():
                    zf.write(config_yaml, config_yaml.relative_to(repo_root))

            stage_path = f"@{self.stage}/{stage_subdir}"
            logger.info("Uploading source.zip to %s", stage_path)
            self.session.file.put(
                zip_path,
                stage_path,
                auto_compress=False,
                overwrite=True,
            )
            logger.info("source.zip uploaded to %s/source.zip", stage_path)

        return f"@{self.stage}/{stage_subdir}/source.zip", deploy_ts

    def _upload_config_yaml(self) -> None:
        """Upload config.yaml to the stage so SPCS jobs can read it."""
        repo_root = Path(__file__).resolve().parent.parent.parent
        config_path = str(repo_root / "source" / "config.yaml")
        logger.info("Uploading config.yaml to @%s", self.stage)
        self.session.file.put(config_path, f"@{self.stage}", auto_compress=False, overwrite=True)
        logger.info("config.yaml uploaded successfully")

    def _prune_old_deploys(self, keep: int = 3) -> None:
        """Remove stale deploy archives and SP handler zips from the stage.

        Keeps the ``keep`` most recent timestamped directories under both
        ``deploys/`` (source.zip archives) and ``procs/<name>/`` (handler zips)
        so the stage does not accumulate unbounded artifacts over time.
        """
        try:
            rows = self.session.sql(f"LIST @{self.stage}/deploys/").collect()
            paths = sorted(
                {r["name"].split("/deploys/")[1].split("/")[0] for r in rows},
                reverse=True,
            )
            for old_ts in paths[keep:]:
                self.session.sql(f"REMOVE @{self.stage}/deploys/{old_ts}/").collect()
                logger.info("Pruned old deploy archive: deploys/%s", old_ts)
        except Exception as e:
            logger.warning("Could not prune deploy archives: %s", e)

        for step in _STEPS:
            proc_dir = step["proc_name"].lower()
            try:
                rows = self.session.sql(f"LIST @{self.stage}/procs/{proc_dir}/").collect()
                ts_dirs = sorted(
                    {r["name"].split(f"/procs/{proc_dir}/")[1].split("/")[0] for r in rows},
                    reverse=True,
                )
                for old_ts in ts_dirs[keep:]:
                    self.session.sql(f"REMOVE @{self.stage}/procs/{proc_dir}/{old_ts}/").collect()
                    logger.info("Pruned old handler zip: procs/%s/%s", proc_dir, old_ts)
            except Exception as e:
                logger.warning("Could not prune handler zips for %s: %s", proc_dir, e)

    def _register_step_procedure(
        self,
        step: dict,
        source_zip_path: str,
        config_json: str,
        deploy_ts: int,
    ) -> None:
        """Register a permanent stored procedure that wraps one pipeline step.

        Uses a timestamped ``stage_location`` (``procs/<proc_name>/<deploy_ts>/``)
        so each deployment lands at a new stage path, forcing Snowflake warehouse
        workers to download the fresh handler rather than serving a cached copy.
        """
        proc_name = step["proc_name"]
        full_proc = f"{self.db}.{self.schema}.{proc_name}"
        logger.info("Registering stored procedure: %s", full_proc)

        handler = make_handler(
            step_import=step["imports"],
            step_func=step["step_func"],
            config_json=config_json,
            task_key=step["task_key"],
            task_name=step["task_name"],
            is_final=step.get("is_final", False),
        )

        self.session.sproc.register(
            func=handler,
            return_type=VariantType(),
            name=full_proc,
            is_permanent=True,
            stage_location=f"@{self.stage}/procs/{proc_name.lower()}/{deploy_ts}/",
            imports=[source_zip_path],
            packages=PACKAGES,
            replace=True,
            execute_as="owner",
        )

        logger.info("Stored procedure %s registered", full_proc)

    def _create_task(self, step: dict) -> None:
        """Create or replace a Snowflake task that calls a stored procedure."""
        task_name = step["task_name"]
        proc_name = step["proc_name"]
        full_task = f"{self.db}.{self.schema}.{task_name}"
        full_proc = f"{self.db}.{self.schema}.{proc_name}"

        clauses = [f"WAREHOUSE = {self.warehouse}"]

        if step["after"]:
            clauses.append(f"AFTER {self.db}.{self.schema}.{step['after']}")
        else:
            clauses.append(f"SCHEDULE = '{step['schedule']}'")
            clauses.append("ALLOW_OVERLAPPING_EXECUTION = FALSE")

        when_val = step.get("when")
        if when_val:
            when_str = when_val(self.config) if callable(when_val) else when_val
            clauses.append(f"WHEN {when_str}")

        props = "\n    ".join(clauses)
        logger.info("Creating task: %s", full_task)
        self.session.sql(f"""
            CREATE OR REPLACE TASK {full_task}
                {props}
            AS
                CALL {full_proc}()
        """).collect()
        logger.info("Task %s created", full_task)

    def _ensure_task_privileges(self) -> None:
        """Grant EXECUTE TASK and Feature Store tag privileges to the current role."""
        role = self.session.sql("SELECT CURRENT_ROLE()").collect()[0][0]
        grants = [
            (
                f"GRANT EXECUTE TASK ON ACCOUNT TO ROLE {role}",
                f"EXECUTE TASK ON ACCOUNT to role {role}",
                f"Run manually: GRANT EXECUTE TASK ON ACCOUNT TO ROLE {role};",
            ),
            (
                f"GRANT APPLY TAG ON ACCOUNT TO ROLE {role}",
                f"APPLY TAG ON ACCOUNT to role {role}",
                f"Run manually: GRANT APPLY TAG ON ACCOUNT TO ROLE {role};",
            ),
        ]
        for sql, description, hint in grants:
            try:
                self.session.sql(sql).collect()
                logger.info("Granted %s", description)
            except Exception as e:
                logger.warning(
                    "Could not auto-grant %s (need ACCOUNTADMIN). %s  Error: %s",
                    description, hint, e,
                )

    def _print_dag_summary(self) -> None:
        """Log a human-readable summary of the deployed task DAG."""
        rows = self.session.sql(f"SHOW TASKS IN SCHEMA {self.db}.{self.schema}").collect()
        task_names = {s["task_name"] for s in _STEPS}
        logger.info("\nTask DAG summary:")
        logger.info("%-40s %-12s %-30s", "TASK", "STATE", "AFTER / SCHEDULE")
        logger.info("-" * 85)
        for row in rows:
            d = row.as_dict()
            name = d.get("name", "")
            if name.upper() not in task_names:
                continue
            state = d.get("state", "")
            schedule = d.get("schedule", "") or d.get("predecessors", "")
            logger.info("%-40s %-12s %-30s", name, state, schedule)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Manage ML pipeline Task DAG")
    parser.add_argument("--build", action="store_true", help="Create / update DAG")
    parser.add_argument("--run", action="store_true", help="Trigger DAG immediately after build")
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

    dag = PipelineDAG(session, config)

    if args.teardown:
        dag.teardown()
    else:
        if args.build:
            dag.build()
        if args.run:
            dag.run()


if __name__ == "__main__":
    main()
