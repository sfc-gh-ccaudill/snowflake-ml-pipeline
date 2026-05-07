"""
Generic stored-procedure handler factory for the ML pipeline Task DAG.

Each pipeline step SP is registered via session.sproc.register() using a
closure produced by make_handler().  All per-step parameters are captured
at registration time so the closure is fully self-contained when Snowpark
serialises it with cloudpickle.

Key behaviours
--------------
* sys.modules is cleared of all 'source.*' entries on every invocation so
  that the freshly-uploaded source.zip is always used — never a stale copy
  cached in the warehouse-worker process.
* handler.__name__ is set to the lowercase task name so Snowpark uploads
  each SP's serialised closure to a unique filename, preventing cross-SP
  stage-file collisions.
* Deep-merge of SYSTEM$TASK_RUNTIME_INFO overrides on top of the baked-in
  config_json, preserving the ability to tweak pipeline parameters at
  execution time without redeploying.
"""

from __future__ import annotations


def _json_safe(obj):
    """Recursively replace non-JSON-serializable values with a type label string."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return f"<{type(obj).__name__}>"


def make_handler(
    step_import: str,
    step_func: str,
    config_json: str,
    task_key: str,
    task_name: str,
    is_final: bool,
):
    """Return a Snowpark stored-procedure handler for one pipeline step.

    Args:
        step_import:  Dotted module path, e.g. 'source.pipeline.step1_feature_engineering'.
        step_func:    Name of the callable inside that module, e.g. 'run'.
        config_json:  JSON-serialised PipelineConfig snapshot baked in at deploy time.
        task_key:     Short key used for execution-log rows, e.g. 'feature_engineering'.
        task_name:    Snowflake task name, e.g. 'PIPELINE_FEATURE_ENG_TASK'.
        is_final:     True only for the last task; triggers pipeline-level log finalisation.

    Returns:
        A callable ``handler(session)`` suitable for ``session.sproc.register(func=...)``.
    """

    def handler(session):
        import importlib
        import json
        import logging
        import sys

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        )

        for _k in [k for k in sys.modules if k == "source" or k.startswith("source.")]:
            del sys.modules[_k]

        from source.configs import get_config_from_dict
        from source.pipeline.pipeline_utils import PipelineExecutionLogger

        config_dict = json.loads(config_json)

        try:
            raw = session.sql("SELECT SYSTEM$TASK_RUNTIME_INFO('CURRENT_TASK_CONFIG')").collect()[
                0
            ][0]
            if raw:
                overrides = json.loads(raw)

                def _merge(base, patch):
                    for k, v in patch.items():
                        if isinstance(v, dict) and isinstance(base.get(k), dict):
                            _merge(base[k], v)
                        else:
                            base[k] = v

                _merge(config_dict, overrides)
        except Exception:
            pass

        config = get_config_from_dict(config_dict)
        db = config_dict.get("snowflake", {}).get("database", "")
        sc = config_dict.get("snowflake", {}).get("schema", "")

        _log = PipelineExecutionLogger(session, db, sc)
        _run_id, _t0 = _log.log_task_start(task_key, task_name)

        try:
            mod = importlib.import_module(step_import)
            step_fn = getattr(mod, step_func)
            result = step_fn(config, session)
            if not isinstance(result, dict):
                result = {"result": result}
            _status = result.get("status", "success")
            if _status not in ("skipped", "failed"):
                _status = "success"
            _log.log_task_end(
                _run_id,
                task_key,
                _t0,
                _status,
                details=_json_safe(result),
                is_final=is_final,
            )
        except Exception as _exc:
            try:
                _log.log_task_end(
                    _run_id,
                    task_key,
                    _t0,
                    "failed",
                    details={"error": str(_exc)},
                    is_final=is_final,
                )
            except Exception:
                pass
            raise

        return result

    handler.__name__ = task_name.lower()
    return handler
