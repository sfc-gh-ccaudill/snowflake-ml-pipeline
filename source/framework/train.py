"""
Remote training framework for the Healthcare ML Pipeline.

Wraps Snowflake ML Jobs (submit_directory) to execute training on SPCS
compute pools. Setting num_instances > 1 triggers distributed multi-node
training where each node runs the entrypoint concurrently under its own
RANK / WORLD_SIZE environment variables.

Hyperparameter tuning via Ray Tune is available through RayHPOConfig +
RemoteTrainer.submit_hpo().  The entrypoint receives the full search
configuration as the HPO_CONFIG_JSON environment variable and is
responsible for initialising the Ray cluster (rank-0 = head node,
rank > 0 = worker nodes) and calling tune.Tuner(...).fit().
"""

import json
import logging
import os
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from snowflake.snowpark import Session

logger = logging.getLogger(__name__)


def _resolve_source_dir(source_dir: str) -> str:
    """
    Return a local filesystem path to source_dir.

    When running inside a Snowflake stored procedure the working directory
    does not contain the source tree; instead it is available as source.zip
    on sys.path (via IMPORTS).  In that case we extract the zip to a temp
    directory and return the extracted path.
    """
    if os.path.isdir(source_dir):
        return source_dir
    for p in sys.path:
        if p.endswith("source.zip") and os.path.isfile(p):
            tmpdir = tempfile.mkdtemp()
            with zipfile.ZipFile(p, "r") as zf:
                zf.extractall(tmpdir)
            resolved = os.path.join(tmpdir, source_dir)
            if os.path.isdir(resolved):
                logger.info("Resolved source_dir from zip: %s", resolved)
                return resolved
    raise FileNotFoundError(
        f"'{source_dir}' not found as a local directory or inside source.zip on sys.path"
    )


_JOB_POLL_INTERVAL_SECS = 15
_JOB_TIMEOUT_SECS = 3600

_RAY_PACKAGES: Dict[str, List[str]] = {
    "random":   ["ray[tune]"],
    "hyperopt": ["ray[tune]", "hyperopt"],
    "optuna":   ["ray[tune]", "optuna"],
}


@dataclass
class RayHPOConfig:
    """
    Configuration for a Ray Tune hyperparameter search.

    Pass an instance to RemoteTrainer.submit_hpo().  All fields are
    serialised to JSON and forwarded to the SPCS job as the
    HPO_CONFIG_JSON environment variable, where the entrypoint script
    deserialises them and constructs the Tuner.

    Args:
        search_space: Ray Tune search-space dict.  Values must be
            serialisable (use plain dicts for tune.* primitives; the
            entrypoint reconstructs them from the config).
        metric: Metric name reported by the trainable to optimise.
        mode: "max" to maximise metric, "min" to minimise.
        num_samples: Total number of hyperparameter trials to run.
        max_concurrent_trials: Concurrent trials cap.  None lets Ray
            decide based on available resources.
        scheduler: Early-stopping scheduler — "asha" (default),
            "pbt", or "fifo".
        search_alg: Search algorithm — "random" (default),
            "hyperopt", or "optuna".  Non-random algorithms require
            the matching package to be installed.
        grace_period: Minimum epochs/iterations before ASHA may stop
            a trial.  Ignored for non-ASHA schedulers.
        reduction_factor: Halving factor for ASHA.  Ignored for
            non-ASHA schedulers.

    Example:
        >>> cfg = RayHPOConfig(
        ...     search_space={
        ...         "n_estimators":  {"type": "randint",   "lower": 50,  "upper": 500},
        ...         "max_depth":     {"type": "randint",   "lower": 3,   "upper": 20},
        ...         "learning_rate": {"type": "loguniform","lower": 1e-4,"upper": 0.3},
        ...     },
        ...     metric="f1_macro",
        ...     mode="max",
        ...     num_samples=40,
        ...     scheduler="asha",
        ...     search_alg="optuna",
        ... )
    """

    search_space: Dict[str, Any]
    metric: str
    mode: str = "max"
    num_samples: int = 20
    max_concurrent_trials: Optional[int] = None
    scheduler: str = "asha"
    search_alg: str = "random"
    grace_period: int = 1
    reduction_factor: int = 2

    def __post_init__(self) -> None:
        if self.mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode!r}")
        if self.search_alg not in _RAY_PACKAGES:
            raise ValueError(
                f"search_alg must be one of {list(_RAY_PACKAGES)}, "
                f"got {self.search_alg!r}"
            )

    def to_env_vars(self) -> Dict[str, str]:
        """Serialise the config into the env-var dict expected by the entrypoint."""
        return {
            "HPO_CONFIG_JSON": json.dumps(
                {
                    "search_space": self.search_space,
                    "metric": self.metric,
                    "mode": self.mode,
                    "num_samples": self.num_samples,
                    "max_concurrent_trials": self.max_concurrent_trials,
                    "scheduler": self.scheduler,
                    "search_alg": self.search_alg,
                    "grace_period": self.grace_period,
                    "reduction_factor": self.reduction_factor,
                }
            )
        }

    @property
    def pip_packages(self) -> List[str]:
        """Ray packages required for the configured search algorithm."""
        return _RAY_PACKAGES[self.search_alg]


class RemoteTrainer:
    """
    Submits and manages remote ML training jobs via Snowflake ML Jobs.

    Wraps snowflake.ml.jobs.submit_directory so pipeline steps stay
    agnostic of the underlying job mechanics.  Pass num_instances > 1 to
    enable distributed multi-node training.

    Args:
        session: Active Snowpark session.
        compute_pool: Name of the SPCS compute pool to run the job on.
        stage: Fully-qualified stage name used for job payload upload
               (e.g. "DB.SCHEMA.JOB_PAYLOADS").
        source_dir: Local path to the directory that contains the training
                    entrypoint and all its imports.  Defaults to "source".

    Example:
        >>> trainer = RemoteTrainer(session, "ML_DEMO_COMPUTE_POOL",
        ...                         "ML_DEMO_PIPELINE_DB.HEALTHCARE.JOB_PAYLOADS")
        >>> job = trainer.submit(num_instances=3)
        >>> status = trainer.wait_and_log(job)
    """

    def __init__(
        self,
        session: Session,
        compute_pool: str,
        stage: str,
        source_dir: str = "source",
    ):
        self.session = session
        self.compute_pool = compute_pool
        self.stage = stage
        self.source_dir = _resolve_source_dir(source_dir)

    def submit(
        self,
        entrypoint: str = "train.py",
        num_instances: int = 1,
        env_vars: Optional[dict] = None,
        pip_requirements: Optional[list] = None,
        external_access_integrations: Optional[List[str]] = None,
    ):
        """
        Submit a training job to SPCS via ML Jobs.

        When num_instances > 1 Snowflake provisions that many nodes and
        sets RANK (0-based node index) and WORLD_SIZE environment variables
        inside each container, enabling distributed coordination.

        Args:
            entrypoint: Python file inside source_dir to execute.
            num_instances: Number of SPCS nodes.  1 = single-node remote,
                           >1 = distributed multi-node.
            env_vars: Additional environment variables to inject.
            pip_requirements: Extra pip packages beyond snowflake-ml-python.
            external_access_integrations: List of external access integration
                names to attach to the job (e.g. ["PYPI_ACCESS_INTEGRATION"]).
                Required when pip_requirements includes packages not present
                in the SPCS local wheel cache.

        Returns:
            MLJob object (use wait_and_log to block until completion).
        """
        from snowflake.ml.jobs import submit_directory

        env_vars = env_vars or {}
        pip_requirements = pip_requirements or []

        logger.info(
            "Submitting ML Job: entrypoint=%s, compute_pool=%s, num_instances=%d",
            entrypoint,
            self.compute_pool,
            num_instances,
        )
        if external_access_integrations:
            logger.info("External access integrations: %s", external_access_integrations)

        submit_kwargs: Dict[str, Any] = dict(
            dir_path=self.source_dir,
            entrypoint=entrypoint,
            compute_pool=self.compute_pool,
            stage_name=self.stage,
            target_instances=num_instances,
            env_vars=env_vars,
            pip_requirements=pip_requirements,
        )
        if external_access_integrations:
            submit_kwargs["external_access_integrations"] = external_access_integrations

        job = submit_directory(**submit_kwargs)

        logger.info("Job submitted: id=%s", job.id)
        return job

    def submit_hpo(
        self,
        hpo_config: RayHPOConfig,
        entrypoint: str = "train_hpo.py",
        num_instances: int = 2,
        env_vars: Optional[dict] = None,
        pip_requirements: Optional[list] = None,
        external_access_integrations: Optional[List[str]] = None,
    ):
        """
        Submit a Ray Tune hyperparameter optimisation job to SPCS.

        Provisions a Ray cluster across ``num_instances`` SPCS nodes.
        Rank-0 acts as the Ray head node; all other ranks join as workers.
        The entrypoint receives the full HPO configuration via the
        ``HPO_CONFIG_JSON`` environment variable.

        Typical entrypoint skeleton::

            import json, os, ray
            from ray import tune

            cfg = json.loads(os.environ["HPO_CONFIG_JSON"])
            rank = int(os.environ.get("RANK", 0))

            if rank == 0:
                ray.init()
            else:
                head_ip = os.environ["RAY_HEAD_IP"]   # set by rank-0 side-car
                ray.init(address=f"ray://{head_ip}:10001")

            def trainable(config):
                # build & train model, then tune.report(f1_macro=...)
                ...

            tuner = tune.Tuner(
                trainable,
                param_space=cfg["search_space"],
                tune_config=tune.TuneConfig(
                    metric=cfg["metric"],
                    mode=cfg["mode"],
                    num_samples=cfg["num_samples"],
                ),
            )
            results = tuner.fit()
            print("Best config:", results.get_best_result().config)

        Args:
            hpo_config: RayHPOConfig defining the search space, scheduler,
                and search algorithm.
            entrypoint: Python file inside source_dir that reads
                HPO_CONFIG_JSON and drives tune.Tuner(...).fit().
            num_instances: Total SPCS nodes (rank-0 head + n-1 workers).
                Must be >= 1; use >= 2 for true distributed search.
            env_vars: Additional environment variables to inject alongside
                HPO_CONFIG_JSON.
            pip_requirements: Extra pip packages appended after Ray packages.
            external_access_integrations: List of external access integration
                names to attach to the job (e.g. ["PYPI_ACCESS_INTEGRATION"]).
                Required for non-random search algorithms (optuna, hyperopt)
                that must be pip-installed at job start time.

        Returns:
            MLJob object (use wait_and_log to block until completion).

        Raises:
            ValueError: If num_instances < 1.
        """
        from snowflake.ml.jobs import submit_directory

        if num_instances < 1:
            raise ValueError(f"num_instances must be >= 1, got {num_instances}")

        merged_env = {**(env_vars or {}), **hpo_config.to_env_vars()}

        extra_pip = [p for p in (pip_requirements or []) if p not in hpo_config.pip_packages]
        merged_pip = hpo_config.pip_packages + extra_pip

        logger.info(
            "Submitting Ray HPO job: entrypoint=%s compute_pool=%s "
            "num_instances=%d metric=%s mode=%s num_samples=%d "
            "scheduler=%s search_alg=%s search_space_keys=%s",
            entrypoint,
            self.compute_pool,
            num_instances,
            hpo_config.metric,
            hpo_config.mode,
            hpo_config.num_samples,
            hpo_config.scheduler,
            hpo_config.search_alg,
            list(hpo_config.search_space.keys()),
        )
        if external_access_integrations:
            logger.info("External access integrations: %s", external_access_integrations)

        submit_kwargs: Dict[str, Any] = dict(
            dir_path=self.source_dir,
            entrypoint=entrypoint,
            compute_pool=self.compute_pool,
            stage_name=self.stage,
            num_instances=num_instances,
            env_vars=merged_env,
            pip_requirements=merged_pip,
        )
        if external_access_integrations:
            submit_kwargs["external_access_integrations"] = external_access_integrations

        job = submit_directory(**submit_kwargs)

        logger.info("HPO job submitted: id=%s", job.id)
        return job

    def wait_and_log(self, job, timeout_secs: int = _JOB_TIMEOUT_SECS) -> str:
        """
        Block until the job finishes (or times out), streaming logs.

        Args:
            job: MLJob returned by submit().
            timeout_secs: Maximum seconds to wait before raising TimeoutError.

        Returns:
            Final job status string ("DONE", "FAILED", etc.).

        Raises:
            TimeoutError: If the job has not finished within timeout_secs.
            RuntimeError: If the job reaches a terminal failure state.
        """
        logger.info("Waiting for job %s to complete (timeout=%ds)", job.id, timeout_secs)

        start = time.time()
        last_logged_status = None

        while True:
            status = job.status

            if status != last_logged_status:
                logger.info("Job %s status: %s", job.id, status)
                last_logged_status = status

            if status in ("DONE", "FAILED", "CANCELLED", "INTERNAL_ERROR"):
                break

            elapsed = time.time() - start
            if elapsed > timeout_secs:
                raise TimeoutError(
                    f"Job {job.id} did not complete within {timeout_secs}s. "
                    f"Last status: {status}"
                )

            time.sleep(_JOB_POLL_INTERVAL_SECS)

        logs = job.get_logs() or ""
        tail = logs[-4000:] if len(logs) > 4000 else logs
        logger.info("=== Job logs (tail) ===\n%s", tail)

        if status != "DONE":
            raise RuntimeError(f"Job {job.id} finished with status: {status}")

        logger.info("Job %s completed successfully", job.id)
        return status

    def get_job_logs(self, job, tail_chars: int = 4000) -> str:
        """Return the trailing portion of job logs for display."""
        logs = job.get_logs() or ""
        return logs[-tail_chars:] if len(logs) > tail_chars else logs
