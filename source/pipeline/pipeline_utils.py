"""
Pipeline utilities — execution logging and inter-step state.

Two classes, one module:

PipelineExecutionLogger
    Writes per-task timing, status, and result details into a single
    PIPELINE_EXECUTIONS row per DAG run (one row, VARIANT column for steps).

PipelineState
    Lightweight key/value store for passing typed values between tasks
    (dataset names, model version names, etc.) via a PIPELINE_STATE table.

Both classes use the same run-ID resolution logic: prefer
SYSTEM$TASK_RUNTIME_INFO('CURRENT_TASK_GRAPH_RUN_GROUP_ID') so runs
triggered by the task scheduler share a consistent ID; fall back to a
local UUID for standalone / test invocations.

Table schemas
-------------
PIPELINE_EXECUTIONS
  PIPELINE_RUN_ID      VARCHAR  PK
  PIPELINE_START_TIME  TIMESTAMP_NTZ
  PIPELINE_END_TIME    TIMESTAMP_NTZ
  PIPELINE_STATUS      VARCHAR        RUNNING | SUCCESS | FAILED
  TASK_STEPS           VARIANT        JSON object keyed by task_key
  CREATED_AT           TIMESTAMP_NTZ
  UPDATED_AT           TIMESTAMP_NTZ

PIPELINE_STATE
  RUN_ID      VARCHAR  NOT NULL  }
  STEP        VARCHAR  NOT NULL  } composite PK
  KEY         VARCHAR  NOT NULL  }
  VALUE       VARCHAR
  UPDATED_AT  TIMESTAMP_NTZ
"""

from datetime import datetime, timezone
import json
import logging
import time
from typing import Optional
import uuid

from snowflake.snowpark import Session

logger = logging.getLogger(__name__)


def _get_run_id(session: Session) -> str:
    """Return the current task-graph run ID, or a local UUID for standalone runs."""
    try:
        result = session.sql(
            "SELECT SYSTEM$TASK_RUNTIME_INFO('CURRENT_TASK_GRAPH_RUN_GROUP_ID')"
        ).collect()[0][0]
        if result:
            return result
    except Exception:
        pass
    return f"local-{uuid.uuid4()}"


def _safe_details(d: dict) -> dict:
    """Strip private keys and coerce non-serialisable values to strings."""
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
    """Writes per-task timing and status into a single PIPELINE_EXECUTIONS row per run.

    Usage
    -----
    log = PipelineExecutionLogger(session, db, schema)
    run_id, t0 = log.log_task_start("feature_engineering", "PIPELINE_FEATURE_ENG_TASK")
    try:
        result = run(config, session)
        log.log_task_end(run_id, "feature_engineering", t0, "success", details=result)
    except Exception as exc:
        log.log_task_end(run_id, "feature_engineering", t0, "failed",
                         details={"error": str(exc)}, is_final=True)
        raise
    """

    _TABLE = "PIPELINE_EXECUTIONS"

    def __init__(self, session: Session, db: str, schema: str) -> None:
        self.session = session
        self.table = f"{db}.{schema}.{self._TABLE}"

    def _run_id(self) -> str:
        return _get_run_id(self.session)

    def _read_steps(self, run_id: str) -> dict:
        rows = self.session.sql(
            f"SELECT TASK_STEPS FROM {self.table} WHERE PIPELINE_RUN_ID = '{run_id}'"
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
        steps_json = json.dumps(steps, default=str).replace("'", "''")
        end_clause = ", PIPELINE_END_TIME = CURRENT_TIMESTAMP()" if mark_end else ""
        status_clause = f", PIPELINE_STATUS = '{pipeline_status}'" if pipeline_status else ""
        self.session.sql(
            f"UPDATE {self.table} "
            f"SET TASK_STEPS = PARSE_JSON('{steps_json}'), "
            f"    UPDATED_AT = CURRENT_TIMESTAMP() "
            f"    {end_clause} "
            f"    {status_clause} "
            f"WHERE PIPELINE_RUN_ID = '{run_id}'"
        ).collect()

    def log_task_start(self, task_key: str, task_name: str) -> tuple:
        """Upsert the pipeline row and record that this task has started.

        Returns (run_id, monotonic_start) — pass both to log_task_end.
        """
        run_id = self._run_id()
        wall_start = time.monotonic()
        start_iso = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

        step_init = {"task_name": task_name, "start_time": start_iso, "status": "running"}
        step_json = json.dumps(step_init).replace("'", "''")
        init_json = json.dumps({task_key: step_init}).replace("'", "''")

        self.session.sql(
            f"MERGE INTO {self.table} t "
            f"USING (SELECT '{run_id}' AS run_id, CURRENT_TIMESTAMP() AS ts) s "
            f"ON t.PIPELINE_RUN_ID = s.run_id "
            f"WHEN MATCHED THEN UPDATE SET "
            f"    TASK_STEPS = OBJECT_INSERT(COALESCE(t.TASK_STEPS, PARSE_JSON('{{}}')), "
            f"        '{task_key}', PARSE_JSON('{step_json}'), TRUE), "
            f"    UPDATED_AT = s.ts "
            f"WHEN NOT MATCHED THEN INSERT "
            f"    (PIPELINE_RUN_ID, PIPELINE_START_TIME, PIPELINE_STATUS, TASK_STEPS, CREATED_AT, UPDATED_AT) "
            f"VALUES "
            f"    (s.run_id, s.ts, 'RUNNING', PARSE_JSON('{init_json}'), s.ts, s.ts)"
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
        """Record task completion and optionally seal the pipeline row.

        Set is_final=True on the last task to write PIPELINE_END_TIME and
        the final PIPELINE_STATUS.
        """
        duration = round(time.monotonic() - wall_start, 2)
        end_iso = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

        steps = self._read_steps(run_id)
        step = steps.get(task_key, {})
        step.update(end_time=end_iso, duration_secs=duration, status=status)
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
            task_key,
            status,
            duration,
            run_id,
        )

    def log_task_skipped(self, task_key: str, task_name: str, reason: str = "") -> str:
        """Record a skipped task (no start/end split needed). Returns run_id."""
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


class PipelineState:
    """Key/value store for passing typed values between pipeline tasks.

    Usage
    -----
    state = PipelineState(session, db, schema)
    state.set("feature_engineering", "training_dataset_version", "v1")
    version = state.get("training", "version_name", default="latest")
    """

    _TABLE = "PIPELINE_STATE"

    def __init__(self, session: Session, db: str, schema: str) -> None:
        self.session = session
        self.table = f"{db}.{schema}.{self._TABLE}"

    def _run_id(self) -> str:
        return _get_run_id(self.session)

    def set(self, step: str, key: str, value: str, run_id: Optional[str] = None) -> None:
        if run_id is None:
            run_id = self._run_id()
        val_escaped = str(value).replace("'", "''")
        self.session.sql(f"""
            MERGE INTO {self.table} t
            USING (SELECT '{run_id}' AS r, '{step}' AS s, '{key}' AS k) src
            ON t.RUN_ID = src.r AND t.STEP = src.s AND t.KEY = src.k
            WHEN MATCHED THEN UPDATE SET
                VALUE      = '{val_escaped}',
                UPDATED_AT = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED THEN INSERT (RUN_ID, STEP, KEY, VALUE)
                VALUES ('{run_id}', '{step}', '{key}', '{val_escaped}')
        """).collect()
        logger.debug("State set: %s/%s = %r (run_id=%s)", step, key, value, run_id)

    def get(
        self,
        step: str,
        key: str,
        run_id: Optional[str] = None,
        default: Optional[str] = None,
    ) -> Optional[str]:
        if run_id is None:
            run_id = self._run_id()
        rows = self.session.sql(f"""
            SELECT VALUE FROM {self.table}
            WHERE RUN_ID = '{run_id}' AND STEP = '{step}' AND KEY = '{key}'
        """).collect()
        return rows[0][0] if rows else default
