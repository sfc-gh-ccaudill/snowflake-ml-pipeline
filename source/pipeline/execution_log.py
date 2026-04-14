"""
Pipeline execution logging — one row per DAG run in PIPELINE_EXECUTIONS.

Schema
------
PIPELINE_EXECUTIONS
  PIPELINE_RUN_ID      VARCHAR  PK   — SYSTEM$TASK_RUNTIME_INFO graph run group ID
  PIPELINE_START_TIME  TIMESTAMP_NTZ — set when the first task starts
  PIPELINE_END_TIME    TIMESTAMP_NTZ — set when the final task finishes (or any task fails)
  PIPELINE_STATUS      VARCHAR       — RUNNING | SUCCESS | FAILED
  TASK_STEPS           VARIANT       — JSON object keyed by task_key, e.g.:
                                       {
                                         "feature_engineering": {
                                           "task_name": "PIPELINE_ROOT_TASK",
                                           "start_time": "2024-01-01T02:00:00.000+00:00",
                                           "end_time":   "2024-01-01T02:05:12.123+00:00",
                                           "duration_secs": 312.1,
                                           "status": "success",
                                           "details": {...}
                                         },
                                         "hpo": { ..., "status": "skipped", "skip_reason": "tune_hpo=false" },
                                         ...
                                       }
  CREATED_AT           TIMESTAMP_NTZ
  UPDATED_AT           TIMESTAMP_NTZ
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from snowflake.snowpark import Session

logger = logging.getLogger(__name__)

_DDL = """\
CREATE TABLE IF NOT EXISTS {table} (
    PIPELINE_RUN_ID      VARCHAR         NOT NULL,
    PIPELINE_START_TIME  TIMESTAMP_NTZ,
    PIPELINE_END_TIME    TIMESTAMP_NTZ,
    PIPELINE_STATUS      VARCHAR         DEFAULT 'RUNNING',
    TASK_STEPS           VARIANT,
    CREATED_AT           TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP(),
    UPDATED_AT           TIMESTAMP_NTZ   DEFAULT CURRENT_TIMESTAMP(),
    CONSTRAINT PK_PIPELINE_EXECUTIONS PRIMARY KEY (PIPELINE_RUN_ID)
)"""


def _safe_details(d: dict) -> dict:
    result = {}
    for k, v in d.items():
        if k.startswith("_"):
            continue
        try:
            json.dumps(v)
            result[k] = v
        except (TypeError, ValueError):
            result[k] = str(v)
    return result


class PipelineExecutionLogger:
    """
    Writes per-task timing, status, and details into a single PIPELINE_EXECUTIONS
    row per DAG run.

    Typical usage inside a task handler
    ------------------------------------
    log = PipelineExecutionLogger(session, db, schema)
    run_id, t0 = log.log_task_start("feature_engineering", "PIPELINE_ROOT_TASK")
    try:
        result = run(config, session)
        log.log_task_end(run_id, "feature_engineering", t0, "success", details=result)
    except Exception as exc:
        log.log_task_end(run_id, "feature_engineering", t0, "failed",
                         details={"error": str(exc)}, is_final=True)
        raise
    """

    def __init__(self, session: Session, db: str, schema: str) -> None:
        self.session = session
        self.table = f"{db}.{schema}.PIPELINE_EXECUTIONS"

    @classmethod
    def ensure_table(cls, session: Session, db: str, schema: str) -> None:
        table = f"{db}.{schema}.PIPELINE_EXECUTIONS"
        session.sql(_DDL.format(table=table)).collect()
        logger.info("Ensured table %s exists", table)

    def _run_id(self) -> str:
        try:
            result = self.session.sql(
                "SELECT SYSTEM$TASK_RUNTIME_INFO('CURRENT_TASK_GRAPH_RUN_GROUP_ID')"
            ).collect()[0][0]
            if result:
                return result
        except Exception:
            pass
        import uuid
        return f"local-{uuid.uuid4()}"

    def _read_steps(self, run_id: str) -> dict:
        rows = self.session.sql(
            f"SELECT TASK_STEPS FROM {self.table} "
            f"WHERE PIPELINE_RUN_ID = '{run_id}'"
        ).collect()
        if rows and rows[0][0] is not None:
            raw = rows[0][0]
            return json.loads(raw) if isinstance(raw, str) else raw
        return {}

    def _write_steps(
        self,
        run_id: str,
        steps: dict,
        pipeline_status: Optional[str] = None,
        mark_end: bool = False,
    ) -> None:
        steps_json = json.dumps(steps, default=str)
        end_clause = ", PIPELINE_END_TIME = CURRENT_TIMESTAMP()" if mark_end else ""
        status_clause = (
            f", PIPELINE_STATUS = '{pipeline_status}'" if pipeline_status else ""
        )
        self.session.sql(
            f"UPDATE {self.table} "
            f"SET TASK_STEPS = PARSE_JSON($json${steps_json}$json$), "
            f"    UPDATED_AT = CURRENT_TIMESTAMP() "
            f"    {end_clause} "
            f"    {status_clause} "
            f"WHERE PIPELINE_RUN_ID = '{run_id}'"
        ).collect()

    def log_task_start(self, task_key: str, task_name: str) -> tuple:
        """
        Upsert the pipeline row and record that this task has started.
        Returns (run_id, monotonic_start) — pass both to log_task_end.
        """
        run_id = self._run_id()
        wall_start = time.monotonic()
        start_iso = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

        step_init = {
            "task_name": task_name,
            "start_time": start_iso,
            "status": "running",
        }
        step_json = json.dumps(step_init)
        init_json = json.dumps({task_key: step_init})

        self.session.sql(
            f"MERGE INTO {self.table} t "
            f"USING (SELECT '{run_id}' AS run_id, CURRENT_TIMESTAMP() AS ts) s "
            f"ON t.PIPELINE_RUN_ID = s.run_id "
            f"WHEN MATCHED THEN UPDATE SET "
            f"    TASK_STEPS = OBJECT_INSERT(COALESCE(t.TASK_STEPS, PARSE_JSON('{{}}')), "
            f"        '{task_key}', PARSE_JSON($json${step_json}$json$), TRUE), "
            f"    UPDATED_AT = s.ts "
            f"WHEN NOT MATCHED THEN INSERT "
            f"    (PIPELINE_RUN_ID, PIPELINE_START_TIME, PIPELINE_STATUS, TASK_STEPS, CREATED_AT, UPDATED_AT) "
            f"VALUES "
            f"    (s.run_id, s.ts, 'RUNNING', PARSE_JSON($json${init_json}$json$), s.ts, s.ts)"
        ).collect()

        logger.info("Logged task start: %s (run_id=%s)", task_key, run_id)
        return run_id, wall_start

    def log_task_end(
        self,
        run_id: str,
        task_key: str,
        wall_start: float,
        status: str,
        details: Optional[dict] = None,
        is_final: bool = False,
    ) -> None:
        """
        Record task completion, merging end_time/duration/status into the existing
        start entry.  Set is_final=True on the last task to seal the row with
        PIPELINE_END_TIME and final PIPELINE_STATUS.
        """
        duration = round(time.monotonic() - wall_start, 2)
        end_iso = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

        steps = self._read_steps(run_id)
        step = steps.get(task_key, {})
        step.update(
            end_time=end_iso,
            duration_secs=duration,
            status=status,
        )
        if details:
            step["details"] = _safe_details(details)
        steps[task_key] = step

        pipeline_status: Optional[str] = None
        if status == "failed":
            pipeline_status = "FAILED"
        elif is_final:
            pipeline_status = "SUCCESS"

        self._write_steps(
            run_id,
            steps,
            pipeline_status=pipeline_status,
            mark_end=is_final or status == "failed",
        )
        logger.info(
            "Logged task end: %s status=%s duration=%.1fs (run_id=%s)",
            task_key, status, duration, run_id,
        )

    def log_task_skipped(
        self,
        task_key: str,
        task_name: str,
        reason: str = "",
    ) -> str:
        """
        Record a skipped task (no start/end split needed).
        Returns run_id.
        """
        run_id = self._run_id()
        now_iso = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

        steps = self._read_steps(run_id)
        steps[task_key] = {
            "task_name": task_name,
            "start_time": now_iso,
            "end_time": now_iso,
            "duration_secs": 0,
            "status": "skipped",
            "skip_reason": reason,
        }
        self._write_steps(run_id, steps)
        logger.info("Logged task skipped: %s reason=%r (run_id=%s)", task_key, reason, run_id)
        return run_id
