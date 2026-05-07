"""
Distributed Ray Tune HPO framework for SPCS ML Jobs.

Handles all Ray cluster lifecycle, Tune orchestration, and result persistence
so entrypoint scripts can focus exclusively on model-specific business logic.

Architecture
------------
When an ML Job runs with num_instances > 1, Snowflake sets RANK and WORLD_SIZE
environment variables on every container.  init_cluster() uses these to form a
Ray cluster:

    RANK 0          → Ray head node + Tune coordinator (runs Tuner.fit())
    RANK 1 … N-1   → Ray worker nodes; contribute CPUs for parallel trials
                      (block inside init_cluster() — never return to caller)

Typical entrypoint usage
------------------------

    from framework.hpo import RayTuneRunner

    def build_trainable(train_df, test_df):
        def trainable(config):
            from ray import tune as ray_tune
            model.fit(X_train, y_train)
            ray_tune.report({"f1_macro": score})
        return trainable

    def main():
        runner = RayTuneRunner.from_env()
        runner.init_cluster()                          # workers block here
        train_df, test_df = load_data()
        results    = runner.run(build_trainable(train_df, test_df))
        cfg, score = runner.best_result(results)
        runner.write_results(session, db, schema, cfg, score)
        runner.shutdown()
"""

import json
import logging
import os
import subprocess
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

_RAY_HEAD_PORT = 6379
_RAY_DASHBOARD_PORT = 8265
_RAY_WORKER_WAIT_S = 15
_RAY_HEAD_STARTUP_S = 10

_SEARCH_SPACE_BUILDERS = {
    "randint": lambda s: __import__("ray").tune.randint(s["lower"], s["upper"]),
    "uniform": lambda s: __import__("ray").tune.uniform(s["lower"], s["upper"]),
    "loguniform": lambda s: __import__("ray").tune.loguniform(s["lower"], s["upper"]),
    "choice": lambda s: __import__("ray").tune.choice(s["values"]),
    "grid_search": lambda s: __import__("ray").tune.grid_search(s["values"]),
}


class RayTuneRunner:
    """
    Manages Ray cluster setup and Tune orchestration for SPCS HPO jobs.

    Reads configuration from the HPO_CONFIG_JSON environment variable
    serialised by RayHPOConfig.to_env_vars().  Workers block inside
    init_cluster(); only rank-0 (the head) returns and continues to
    run the Tuner.

    Args:
        cfg: Deserialised HPO configuration dict (keys match RayHPOConfig fields).

    Example:
        >>> runner = RayTuneRunner.from_env()
        >>> runner.init_cluster()
        >>> results = runner.run(my_trainable)
        >>> best_config, best_score = runner.best_result(results)
        >>> runner.write_results(session, db, schema, best_config, best_score)
        >>> runner.shutdown()
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.metric = cfg["metric"]
        self.mode = cfg.get("mode", "max")
        self.num_samples = int(cfg.get("num_samples", 20))
        self.max_concurrent_trials = cfg.get("max_concurrent_trials")
        self.scheduler_name = cfg.get("scheduler", "asha")
        self.search_alg_name = cfg.get("search_alg", "random")
        self.grace_period = int(cfg.get("grace_period", 1))
        self.reduction_factor = int(cfg.get("reduction_factor", 2))
        self._raw_search_space = cfg["search_space"]

    @classmethod
    def from_env(cls) -> "RayTuneRunner":
        """Construct from the HPO_CONFIG_JSON environment variable."""
        raw = os.environ.get("HPO_CONFIG_JSON") or "{}"
        cfg = json.loads(raw)
        if not cfg:
            raise ValueError(
                "HPO_CONFIG_JSON is not set or empty — was RemoteTrainer.submit_hpo() used to launch this job?"
            )
        return cls(cfg)

    def _log_dashboard_url(self) -> None:
        """Log the Ray dashboard URL after the cluster is initialised."""
        import ray

        url = ray.get_dashboard_url()
        if url:
            logger.info("Ray dashboard URL : http://%s", url)
            logger.info(
                "NOTE: dashboard is accessible within the SPCS network only. "
                "To view it externally, forward port %d from the head container.",
                _RAY_DASHBOARD_PORT,
            )
        else:
            logger.info("Ray dashboard URL not available (still starting up)")

    def init_cluster(self) -> int:
        """
        Initialise the Ray cluster using RANK / WORLD_SIZE env vars.

        Rank-0 starts the Ray head node and returns 0 to the caller so
        execution can continue.  All other ranks start worker daemons,
        join the cluster, and then block indefinitely — they never return.

        Returns:
            rank (int): always 0 for the head node.
        """
        import ray

        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        master_addr = os.environ.get("MASTER_ADDR", "localhost")

        if world_size > 1:
            if rank == 0:
                logger.info("Starting Ray head node (port=%d)", _RAY_HEAD_PORT)
                subprocess.Popen(
                    [
                        "ray",
                        "start",
                        "--head",
                        f"--port={_RAY_HEAD_PORT}",
                        "--dashboard-host=0.0.0.0",
                        f"--dashboard-port={_RAY_DASHBOARD_PORT}",
                    ]
                )
                time.sleep(_RAY_HEAD_STARTUP_S)
                ray.init(f"ray://localhost:{_RAY_HEAD_PORT}", ignore_reinit_error=True)
                self._log_dashboard_url()
                logger.info("Ray head ready — waiting for %d worker(s)", world_size - 1)
            else:
                logger.info(
                    "Worker (rank=%d) connecting to head at %s:%d",
                    rank,
                    master_addr,
                    _RAY_HEAD_PORT,
                )
                time.sleep(_RAY_WORKER_WAIT_S)
                subprocess.Popen(
                    [
                        "ray",
                        "start",
                        f"--address={master_addr}:{_RAY_HEAD_PORT}",
                    ]
                )
                time.sleep(_RAY_HEAD_STARTUP_S)
                ray.init(address="auto", ignore_reinit_error=True)
                logger.info("Worker rank=%d ready — waiting for trial dispatch", rank)
                while True:
                    time.sleep(30)
        else:
            ray.init(ignore_reinit_error=True)
            self._log_dashboard_url()

        return rank

    @staticmethod
    def reconstruct_search_space(raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a serialised search-space dict into Ray Tune primitives.

        Each parameter spec must have a ``type`` key matching one of:
        ``randint``, ``uniform``, ``loguniform``, ``choice``, ``grid_search``.

        Args:
            raw: Dict mapping parameter names to spec dicts, e.g.
                 ``{"lr": {"type": "loguniform", "lower": 1e-4, "upper": 0.3}}``.

        Returns:
            Dict mapping parameter names to ``tune.*`` sample objects.

        Raises:
            ValueError: If an unknown type is encountered.
        """
        from ray import tune

        builders = {
            "randint": lambda s: tune.randint(s["lower"], s["upper"]),
            "uniform": lambda s: tune.uniform(s["lower"], s["upper"]),
            "loguniform": lambda s: tune.loguniform(s["lower"], s["upper"]),
            "choice": lambda s: tune.choice(s["values"]),
            "grid_search": lambda s: tune.grid_search(s["values"]),
        }
        out: Dict[str, Any] = {}
        for key, spec in raw.items():
            t = spec.get("type")
            if t not in builders:
                raise ValueError(
                    f"Unknown search-space type {t!r} for parameter {key!r}. Valid types: {list(builders)}"
                )
            out[key] = builders[t](spec)
        return out

    def _build_scheduler(self) -> Optional[Any]:
        if self.scheduler_name == "asha":
            from ray.tune.schedulers import ASHAScheduler

            return ASHAScheduler(
                grace_period=self.grace_period,
                reduction_factor=self.reduction_factor,
            )
        return None

    def _build_search_alg(self) -> Optional[Any]:
        if self.search_alg_name == "optuna":
            from ray.tune.search.optuna import OptunaSearch

            return OptunaSearch(metric=self.metric, mode=self.mode)
        if self.search_alg_name == "hyperopt":
            from ray.tune.search.hyperopt import HyperOptSearch

            return HyperOptSearch(metric=self.metric, mode=self.mode)
        return None

    def run(self, trainable) -> Any:
        """
        Execute the Tune search and return the full ResultGrid.

        Reconstructs the search space, builds the scheduler and search
        algorithm, constructs a ``tune.Tuner``, and calls ``.fit()``.

        Args:
            trainable: A Ray-compatible callable that accepts a ``config``
                       dict and calls ``ray.tune.report({metric: value})``.

        Returns:
            ``ray.tune.ResultGrid`` — pass to ``best_result()`` to extract
            the winning configuration.
        """
        from ray import tune

        param_space = self.reconstruct_search_space(self._raw_search_space)
        scheduler = self._build_scheduler()
        search_alg = self._build_search_alg()

        logger.info(
            "Starting Tune search: %d trials, alg=%s, scheduler=%s, metric=%s (%s)",
            self.num_samples,
            self.search_alg_name,
            self.scheduler_name,
            self.metric,
            self.mode,
        )

        tuner = tune.Tuner(
            trainable,
            param_space=param_space,
            tune_config=tune.TuneConfig(
                metric=self.metric,
                mode=self.mode,
                num_samples=self.num_samples,
                max_concurrent_trials=self.max_concurrent_trials,
                scheduler=scheduler,
                search_alg=search_alg,
            ),
        )
        return tuner.fit()

    def best_result(self, results: Any) -> Tuple[Dict[str, Any], float]:
        """
        Extract the best (config, score) pair from a ResultGrid.

        Args:
            results: ``ray.tune.ResultGrid`` returned by ``run()``.

        Returns:
            Tuple of (best_config dict, best_score float).
        """
        best = results.get_best_result(metric=self.metric, mode=self.mode)
        score = float(best.metrics.get(self.metric, 0.0))
        logger.info("Best %s: %.4f  config: %s", self.metric, score, best.config)
        return best.config, score

    def write_results(
        self,
        session: Any,
        db: str,
        schema: str,
        best_config: Dict[str, Any],
        best_score: float,
        job_id: Optional[str] = None,
        table_name: str = "HPO_RESULTS",
    ) -> None:
        """
        Persist the best trial configuration to a Snowflake table.

        Creates the table if it does not already exist.  The table schema::

            JOB_ID      VARCHAR
            BEST_PARAMS VARCHAR   (JSON)
            BEST_SCORE  FLOAT
            METRIC_NAME VARCHAR
            CREATED_AT  TIMESTAMP_LTZ

        Args:
            session:     Active Snowpark session.
            db:          Target database name.
            schema:      Target schema name.
            best_config: Best hyperparameter configuration dict.
            best_score:  Best metric value achieved.
            job_id:      Job identifier; defaults to SNOWFLAKE_JOB_ID env var
                         or a timestamp-based fallback.
            table_name:  Table name within db.schema (default: HPO_RESULTS).
        """
        if job_id is None:
            job_id = os.environ.get("SNOWFLAKE_JOB_ID", f"hpo_{int(time.time())}")

        fq_table = f"{db}.{schema}.{table_name}"
        session.sql(f"""
            CREATE TABLE IF NOT EXISTS {fq_table} (
                JOB_ID      VARCHAR,
                BEST_PARAMS VARCHAR,
                BEST_SCORE  FLOAT,
                METRIC_NAME VARCHAR,
                CREATED_AT  TIMESTAMP_LTZ DEFAULT CURRENT_TIMESTAMP()
            )
        """).collect()

        params_json = json.dumps(best_config).replace("'", "''")
        job_id_safe = job_id.replace("'", "''")
        session.sql(f"""
            INSERT INTO {fq_table} (JOB_ID, BEST_PARAMS, BEST_SCORE, METRIC_NAME)
            VALUES ('{job_id_safe}', '{params_json}', {best_score}, '{self.metric}')
        """).collect()
        logger.info("Results written to %s (job=%s, score=%.4f)", fq_table, job_id, best_score)

    def shutdown(self) -> None:
        """Shut down the Ray cluster."""
        import ray

        ray.shutdown()
        logger.info("Ray cluster shut down")
