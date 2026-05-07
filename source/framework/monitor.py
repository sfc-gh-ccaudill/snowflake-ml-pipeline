"""
Model monitoring framework for the Healthcare ML Pipeline.

Wraps Snowflake ML Model Monitor to set up feature drift and prediction
drift detection against a registered model version, using a static
baseline table as the reference distribution.
"""

import logging
from typing import List, Optional

from snowflake.snowpark import Session

logger = logging.getLogger(__name__)


def _get_registry(session: Session, database: str, schema: str):
    from snowflake.ml.registry import Registry

    return Registry(session, database_name=database, schema_name=schema)


class ModelMonitor:
    """
    Creates and manages Snowflake Model Monitors for drift detection.

    Sets up column-level feature drift and prediction-class drift monitoring
    against a registered model version.  Uses the TEST_FEATURES table as the
    baseline (reference) distribution.  Optionally enables segmented
    monitoring so drift metrics can be broken down by categorical
    dimensions (e.g. ADMISSION_TYPE, INSURANCE_TYPE).

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
        ...     segment_columns=["ADMISSION_TYPE", "INSURANCE_TYPE"],
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
        segment_columns: Optional[List[str]] = None,
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
            segment_columns: Optional list of categorical columns to enable
                grouped (segmented) monitoring.  When provided, drift and
                accuracy metrics are computed per-segment in addition to
                the overall aggregate, allowing drill-down by dimensions
                such as admission type or insurance type.
        """
        from snowflake.ml.monitoring.entities.model_monitor_config import (
            ModelMonitorConfig,
            ModelMonitorSourceConfig,
        )

        id_columns = id_columns or ["PATIENT_ID"]

        logger.info("Creating monitor '%s' for %s/%s", monitor_name, model_name, version_name)

        registry = _get_registry(self.session, self.database, self.schema)
        mv = registry.get_model(model_name).version(version_name)

        source_config = ModelMonitorSourceConfig(
            source=source_table,
            timestamp_column=timestamp_col,
            id_columns=id_columns,
            prediction_class_columns=[prediction_col],
            actual_class_columns=[label_col] if label_col else None,
            baseline=baseline_table,
            segment_columns=segment_columns,
        )

        monitor_config = ModelMonitorConfig(
            model_version=mv,
            model_function_name="predict",
            background_compute_warehouse_name=(warehouse or self.session.get_current_warehouse()),
        )

        try:
            registry.add_monitor(
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
            Dict with keys: name, status, monitor.
        """
        registry = _get_registry(self.session, self.database, self.schema)
        try:
            monitor = registry.get_monitor(name=monitor_name)
            return {"name": monitor_name, "status": "active", "monitor": monitor}
        except Exception as exc:
            logger.warning("Could not retrieve monitor '%s': %s", monitor_name, exc)
            return {"name": monitor_name, "status": "not_found", "error": str(exc)}

    def list_monitors(self) -> list:
        """Return a list of all monitor names visible to the current session."""
        registry = _get_registry(self.session, self.database, self.schema)
        try:
            rows = registry.show_model_monitors()
            return [r.as_dict().get("name", "") for r in rows]
        except Exception as exc:
            logger.warning("Could not list monitors: %s", exc)
            return []

    def drop_monitor(self, monitor_name: str) -> None:
        """Delete a monitor (used in teardown / re-create scenarios)."""
        registry = _get_registry(self.session, self.database, self.schema)
        try:
            registry.delete_monitor(name=monitor_name)
            logger.info("Monitor '%s' deleted", monitor_name)
        except Exception as exc:
            logger.warning("Could not delete monitor '%s': %s", monitor_name, exc)

    def setup_drift_alert(
        self,
        alert_name: str,
        monitor_name: str,
        prediction_col: str,
        drift_metric: str = "POPULATION_STABILITY_INDEX",
        drift_threshold: float = 0.25,
        schedule: str = "USING CRON 0 6 * * * America/Los_Angeles",
        warehouse: Optional[str] = None,
        retrain_root_task: str = "PIPELINE_FEATURE_ENG_TASK",
    ) -> None:
        """
        Create a Snowflake Alert that checks the model monitor for drift and
        re-triggers the training pipeline task DAG when drift exceeds the
        configured threshold.

        The alert queries MODEL_MONITOR_DRIFT_METRIC for the prediction column
        over the last 24 hours. If the max drift score exceeds the threshold,
        the alert action executes the root task of the pipeline DAG.

        Args:
            alert_name: Name for the alert object.
            monitor_name: Monitor to query drift metrics from.
            prediction_col: Column to check drift on (e.g. RISK_LEVEL).
            drift_metric: Drift metric name (PSI, JSD, etc.).
            drift_threshold: Threshold above which retraining is triggered.
            schedule: Alert evaluation schedule (cron or interval).
            warehouse: Warehouse for alert evaluation.
            retrain_root_task: Root task name to EXECUTE when drift detected.
        """
        wh = warehouse or self.session.get_current_warehouse()
        fq_alert = f"{self.database}.{self.schema}.{alert_name}"
        fq_task = f"{self.database}.{self.schema}.{retrain_root_task}"

        condition_sql = f"""
            SELECT MAX(metric_value) AS MAX_DRIFT
            FROM TABLE(MODEL_MONITOR_DRIFT_METRIC(
                '{monitor_name}',
                '{drift_metric}',
                '{prediction_col}',
                '1 DAY',
                DATEADD('DAY', -1, CURRENT_TIMESTAMP())::TIMESTAMP_NTZ,
                CURRENT_TIMESTAMP()::TIMESTAMP_NTZ
            ))
            WHERE metric_value > {drift_threshold}
        """.strip()

        action_sql = f"EXECUTE TASK {fq_task}"

        create_sql = f"""
            CREATE OR REPLACE ALERT {fq_alert}
                WAREHOUSE = {wh}
                SCHEDULE = '{schedule}'
                COMMENT = 'Re-triggers training pipeline when {drift_metric} drift on {prediction_col} exceeds {drift_threshold}'
            IF(EXISTS(
                {condition_sql}
            ))
            THEN
                {action_sql}
        """

        try:
            self.session.sql(create_sql).collect()
            logger.info("Drift alert '%s' created", fq_alert)

            self.session.sql(f"ALTER ALERT {fq_alert} RESUME").collect()
            logger.info("Drift alert '%s' resumed", fq_alert)
        except Exception as exc:
            logger.error("Failed to create drift alert '%s': %s", fq_alert, exc)
            raise

    def drop_drift_alert(self, alert_name: str) -> None:
        """Drop a drift alert (used in teardown / re-create scenarios)."""
        fq_alert = f"{self.database}.{self.schema}.{alert_name}"
        try:
            self.session.sql(f"DROP ALERT IF EXISTS {fq_alert}").collect()
            logger.info("Drift alert '%s' dropped", fq_alert)
        except Exception as exc:
            logger.warning("Could not drop drift alert '%s': %s", fq_alert, exc)
