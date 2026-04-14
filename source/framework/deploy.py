"""
Model deployment framework for the Healthcare ML Pipeline.

Wraps Snowflake Model Registry's create_service() to deploy a registered
model version as a REST inference endpoint on SPCS, and provides a
predict() helper to validate the endpoint post-deployment.
"""

import logging
import time
from typing import Optional

import pandas as pd
from snowflake.snowpark import Session

logger = logging.getLogger(__name__)

_SERVICE_POLL_INTERVAL_SECS = 15
_SERVICE_READY_TIMEOUT_SECS = 600


class ModelDeployer:
    """
    Deploys Snowflake Registry model versions as SPCS REST endpoints.

    Abstracts the Registry → create_service() workflow so pipeline steps
    can deploy a model with a single call and get back a validated service.

    Args:
        session: Active Snowpark session.
        registry_database: Database that contains the model registry.
        registry_schema: Schema that contains the model registry.

    Example:
        >>> deployer = ModelDeployer(session, "ML_DEMO_PIPELINE_DB", "HEALTHCARE")
        >>> service_name = deployer.deploy(
        ...     model_name="PATIENT_RISK_MODEL",
        ...     version_name="v_20260413_120000",
        ...     service_name="PATIENT_RISK_SERVICE",
        ...     compute_pool="ML_DEMO_COMPUTE_POOL",
        ... )
        >>> result = deployer.predict(service_name, sample_df)
    """

    def __init__(
        self,
        session: Session,
        registry_database: str,
        registry_schema: str,
    ):
        self.session = session
        self.registry_database = registry_database
        self.registry_schema = registry_schema
        self._registry = None

    @property
    def registry(self):
        if self._registry is None:
            from snowflake.ml.registry import Registry

            self._registry = Registry(
                self.session,
                database_name=self.registry_database,
                schema_name=self.registry_schema,
            )
        return self._registry

    def get_latest_version_name(self, model_name: str) -> str:
        """Return the most recently created version name for a model."""
        model = self.registry.get_model(model_name)
        versions = model.versions()
        if not versions:
            raise ValueError(f"No versions found for model '{model_name}'")
        latest = versions[-1]
        version_name = latest.version_name
        logger.info("Latest version of %s: %s", model_name, version_name)
        return version_name

    def deploy(
        self,
        model_name: str,
        service_name: str,
        compute_pool: str,
        version_name: Optional[str] = None,
        min_instances: int = 1,
        max_instances: int = 3,
        gpu_requests: Optional[str] = None,
        timeout_secs: int = _SERVICE_READY_TIMEOUT_SECS,
    ) -> str:
        """
        Deploy a model version as a SPCS REST inference service.

        If version_name is None the latest version is used automatically,
        which lets the pipeline always deploy whatever training just produced.

        Args:
            model_name: Registry model name.
            service_name: Name for the SPCS service to create.
            compute_pool: SPCS compute pool to host the service.
            version_name: Specific version to deploy; None → latest.
            min_instances: Minimum service replicas.
            max_instances: Maximum service replicas (enables auto-scaling).
            gpu_requests: GPU resource request string (e.g. "1"), or None.
            timeout_secs: Seconds to wait for the service to reach RUNNING.

        Returns:
            service_name after the service is confirmed RUNNING.

        Raises:
            TimeoutError: If the service does not become RUNNING in time.
        """
        if version_name is None:
            version_name = self.get_latest_version_name(model_name)

        logger.info(
            "Deploying %s/%s as service '%s' on pool '%s' (min=%d, max=%d)",
            model_name,
            version_name,
            service_name,
            compute_pool,
            min_instances,
            max_instances,
        )

        model = self.registry.get_model(model_name)
        mv = model.version(version_name)

        kwargs = dict(
            service_name=service_name,
            compute_pool=compute_pool,
            min_instances=min_instances,
            max_instances=max_instances,
        )
        if gpu_requests:
            kwargs["gpu_requests"] = gpu_requests

        mv.create_service(**kwargs)

        logger.info("Service '%s' created — waiting for RUNNING state", service_name)
        self._wait_for_service(service_name, timeout_secs)

        return service_name

    def _wait_for_service(self, service_name: str, timeout_secs: int) -> None:
        """Poll until the service reaches RUNNING or timeout."""
        start = time.time()
        last_status = None

        while True:
            rows = self.session.sql(
                f"SHOW SERVICES LIKE '{service_name}'"
            ).collect()

            status = None
            for row in rows:
                row_dict = row.as_dict()
                if row_dict.get("name", "").upper() == service_name.upper():
                    status = row_dict.get("status", "UNKNOWN").upper()
                    break

            if status != last_status:
                logger.info("Service '%s' status: %s", service_name, status)
                last_status = status

            if status == "RUNNING":
                logger.info("Service '%s' is RUNNING", service_name)
                return

            if status in ("FAILED", "DELETING", "DELETED"):
                raise RuntimeError(
                    f"Service '{service_name}' reached terminal state: {status}"
                )

            elapsed = time.time() - start
            if elapsed > timeout_secs:
                raise TimeoutError(
                    f"Service '{service_name}' did not reach RUNNING within "
                    f"{timeout_secs}s. Last status: {status}"
                )

            time.sleep(_SERVICE_POLL_INTERVAL_SECS)

    def predict(
        self,
        model_name: str,
        version_name: str,
        features_df: pd.DataFrame,
        function_name: str = "predict",
    ) -> pd.DataFrame:
        """
        Run inference against a deployed (or registry) model version.

        Uses the registry run() method which routes through the deployed
        service when one is active, providing an end-to-end validation
        of the REST endpoint without requiring manual HTTP calls.

        Args:
            model_name: Registry model name.
            version_name: Version to call.
            features_df: DataFrame of feature values (columns must match
                         the schema the model was trained on).
            function_name: Model function to invoke ("predict",
                           "predict_proba", etc.).

        Returns:
            DataFrame with prediction results.
        """
        model = self.registry.get_model(model_name)
        mv = model.version(version_name)

        logger.info(
            "Running inference: model=%s/%s, function=%s, rows=%d",
            model_name,
            version_name,
            function_name,
            len(features_df),
        )

        result = mv.run(features_df, function_name=function_name)
        logger.info("Inference returned %d rows", len(result))
        return result

    def get_service_status(self, service_name: str) -> str:
        """Return the current status of a deployed service."""
        rows = self.session.sql(f"SHOW SERVICES LIKE '{service_name}'").collect()
        for row in rows:
            row_dict = row.as_dict()
            if row_dict.get("name", "").upper() == service_name.upper():
                return row_dict.get("status", "NOT_FOUND").upper()
        return "NOT_FOUND"

    def drop_service(self, service_name: str) -> None:
        """Drop a running service (used in teardown / re-deploy scenarios)."""
        logger.info("Dropping service '%s'", service_name)
        self.session.sql(f"DROP SERVICE IF EXISTS {service_name}").collect()
        logger.info("Service '%s' dropped", service_name)
