"""
Model monitoring framework for the Healthcare ML Pipeline.

Wraps Snowflake ML Model Monitor to set up feature drift and prediction
drift detection against a registered model version, using a static
baseline table as the reference distribution.
"""

import logging
from typing import Optional

from snowflake.snowpark import Session

logger = logging.getLogger(__name__)


class ModelMonitor:
    """
    Creates and manages Snowflake Model Monitors for drift detection.

    Sets up column-level feature drift and prediction-class drift monitoring
    against a registered model version.  Uses the TEST_FEATURES table as the
    baseline (reference) distribution.

    Args:
        session: Active Snowpark session.
        database: Database that contains both the model and the monitoring tables.
        schema: Schema used for monitoring objects.

    Example:
        >>> monitor = ModelMonitor(session, "ML_DEMO_PIPELINE_DB", "HEALTHCARE")
        >>> monitor.create_monitor(
        ...     monitor_name="PATIENT_RISK_MONITOR",
        ...     model_name="PATIENT_RISK_MODEL",
        ...     version_name="v_20260413_120000",
        ...     source_table="ML_DEMO_PIPELINE_DB.HEALTHCARE.STREAMING_PATIENT_DATA",
        ...     baseline_table="ML_DEMO_PIPELINE_DB.HEALTHCARE.TEST_FEATURES",
        ...     timestamp_col="TIMESTAMP",
        ...     prediction_col="PREDICTED_RISK_LEVEL",
        ...     label_col="RISK_LEVEL",
        ... )
    """

    def __init__(
        self,
        session: Session,
        database: str,
        schema: str,
    ):
        self.session = session
        self.database = database
        self.schema = schema

    def create_monitor(
        self,
        monitor_name: str,
        model_name: str,
        version_name: str,
        source_table: str,
        baseline_table: str,
        timestamp_col: str = "TIMESTAMP",
        prediction_col: str = "PREDICTED_RISK_LEVEL",
        label_col: Optional[str] = "RISK_LEVEL",
        id_columns: Optional[list] = None,
        warehouse: Optional[str] = None,
    ) -> None:
        """
        Create a Model Monitor for drift and accuracy tracking.

        The monitor compares live predictions from source_table against the
        distribution captured in baseline_table.  Both feature drift and
        prediction drift are tracked at the column level.

        Args:
            monitor_name: Unique name for this monitor.
            model_name: Registry model name being monitored.
            version_name: Specific version being monitored.
            source_table: Fully-qualified table receiving live predictions.
            baseline_table: Fully-qualified table used as the reference
                            distribution (typically TEST_FEATURES).
            timestamp_col: Column used to window monitoring periods.
            prediction_col: Column that holds the model's output prediction.
            label_col: Column with ground-truth labels (enables accuracy
                       tracking); None disables label-based metrics.
            id_columns: List of entity ID columns (e.g. ["PATIENT_ID"]).
            warehouse: Warehouse for monitor refresh jobs.
        """
        from snowflake.ml.monitoring import MonitorClient
        from snowflake.ml.monitoring.entities.model_monitor_config import (
            ModelMonitorConfig,
            ModelMonitorSourceConfig,
        )
        from snowflake.ml.registry import Registry

        id_columns = id_columns or ["PATIENT_ID"]

        logger.info(
            "Creating monitor '%s' for %s/%s", monitor_name, model_name, version_name
        )

        registry = Registry(
            self.session,
            database_name=self.database,
            schema_name=self.schema,
        )
        model = registry.get_model(model_name)
        mv = model.version(version_name)

        source_config = ModelMonitorSourceConfig(
            source=self.session.table(source_table),
            prediction_score_columns=[prediction_col],
            label_columns=[label_col] if label_col else [],
            id_columns=id_columns,
            timestamp_column=timestamp_col,
            baseline=self.session.table(baseline_table),
        )

        monitor_config = ModelMonitorConfig(
            model=mv,
            model_function_name="predict",
            background_compute_warehouse_name=(
                warehouse or self.session.get_current_warehouse()
            ),
        )

        client = MonitorClient(session=self.session)

        try:
            client.add_monitor(
                name=monitor_name,
                source_config=source_config,
                model_monitor_config=monitor_config,
            )
            logger.info("Monitor '%s' created successfully", monitor_name)
        except Exception as exc:
            if "already exists" in str(exc).lower():
                logger.info("Monitor '%s' already exists, skipping creation", monitor_name)
            else:
                raise

    def get_monitor_status(self, monitor_name: str) -> dict:
        """
        Return a summary dict describing the current monitor state.

        Args:
            monitor_name: Name of the monitor to inspect.

        Returns:
            Dict with keys: name, status, last_refresh, and dashboard_url.
        """
        from snowflake.ml.monitoring import MonitorClient

        client = MonitorClient(session=self.session)

        try:
            monitor = client.get_monitor(monitor_name)
            return {
                "name": monitor_name,
                "status": "active",
                "monitor": monitor,
            }
        except Exception as exc:
            logger.warning("Could not retrieve monitor '%s': %s", monitor_name, exc)
            return {"name": monitor_name, "status": "not_found", "error": str(exc)}

    def list_monitors(self) -> list:
        """Return a list of all monitor names visible to the current session."""
        from snowflake.ml.monitoring import MonitorClient

        client = MonitorClient(session=self.session)
        try:
            monitors = client.list_monitors()
            return [m.name for m in monitors]
        except Exception as exc:
            logger.warning("Could not list monitors: %s", exc)
            return []

    def drop_monitor(self, monitor_name: str) -> None:
        """Delete a monitor (used in teardown / re-create scenarios)."""
        from snowflake.ml.monitoring import MonitorClient

        client = MonitorClient(session=self.session)
        try:
            client.delete_monitor(monitor_name)
            logger.info("Monitor '%s' deleted", monitor_name)
        except Exception as exc:
            logger.warning("Could not delete monitor '%s': %s", monitor_name, exc)
