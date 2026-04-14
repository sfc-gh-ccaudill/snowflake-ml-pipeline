"""
Model evaluation utilities for Healthcare ML Pipeline.

This module provides the Evaluator class for computing classification metrics,
evaluating models from the registry, logging metrics to Snowflake tables,
and determining promotion eligibility based on configurable thresholds.

Key Features:
    - Multiclass classification metrics (accuracy, precision, recall, F1)
    - Per-class metric breakdown
    - Model registry integration for evaluation
    - Metrics logging to Snowflake tables
    - Promotion criteria checking with configurable thresholds
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from snowflake.snowpark import Session

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluates ML models and manages metrics for promotion decisions.

    This class provides comprehensive model evaluation including:
    - Computing standard classification metrics
    - Per-class performance breakdown
    - Integration with Snowflake Model Registry
    - Metrics persistence to Snowflake tables
    - Promotion eligibility checking against thresholds

    Args:
        session: Active Snowpark session for Snowflake operations.

    Example:
        >>> evaluator = Evaluator(session)
        >>> metrics = evaluator.evaluate_from_registry(
        ...     model_name="MY_MODEL",
        ...     model_version="v1",
        ...     registry_database="ML_DB",
        ...     registry_schema="MODELS",
        ...     test_table="ML_DB.DATA.TEST_DATA",
        ...     feature_columns=["age", "bmi"],
        ...     target_column="risk_level",
        ... )
        >>> evaluator.log_metrics(metrics, "ML_DB.DATA.MODEL_METRICS")
    """

    def __init__(self, session: Session):
        """
        Initialize the Evaluator.

        Args:
            session: Active Snowpark session for Snowflake operations.
        """
        self.session = session

    def compute_multiclass_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        class_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute comprehensive metrics for multiclass classification.

        Calculates both macro-averaged and weighted-averaged metrics,
        plus optional per-class breakdown and log loss.

        Args:
            y_true: Ground truth labels as numpy array.
            y_pred: Predicted labels as numpy array.
            y_proba: Optional probability predictions (shape: n_samples x n_classes).
            class_labels: Optional list of class names for per-class metrics.

        Returns:
            Dict containing:
                - accuracy: Overall accuracy
                - precision_macro: Unweighted mean precision across classes
                - recall_macro: Unweighted mean recall across classes
                - f1_macro: Unweighted mean F1 across classes
                - precision_weighted: Class-frequency weighted precision
                - recall_weighted: Class-frequency weighted recall
                - f1_weighted: Class-frequency weighted F1
                - confusion_matrix: Confusion matrix as nested list
                - log_loss: Log loss (if y_proba provided)
                - per_class: Per-class metrics dict (if class_labels provided)
        """
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            log_loss,
            precision_score,
            recall_score,
        )

        # Overall accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # Macro-averaged metrics (unweighted mean across classes)
        precision_macro = precision_score(
            y_true, y_pred, average="macro", zero_division=0
        )
        recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # Weighted-averaged metrics (weighted by class frequency)
        precision_weighted = precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        recall_weighted = recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        )
        f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            "accuracy": float(accuracy),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(precision_weighted),
            "recall_weighted": float(recall_weighted),
            "f1_weighted": float(f1_weighted),
            "confusion_matrix": cm.tolist(),
        }

        # Log loss if probabilities are available
        if y_proba is not None:
            try:
                logloss = log_loss(y_true, y_proba)
                metrics["log_loss"] = float(logloss)
            except Exception as e:
                logger.warning(f"Could not compute log loss: {e}")

        # Per-class metrics if labels are provided
        if class_labels:
            per_class_metrics = {}
            for _, label in enumerate(class_labels):
                # Convert to binary classification for this class
                y_true_binary = (y_true == label).astype(int)
                y_pred_binary = (y_pred == label).astype(int)

                per_class_metrics[label] = {
                    "precision": float(
                        precision_score(y_true_binary, y_pred_binary, zero_division=0)
                    ),
                    "recall": float(
                        recall_score(y_true_binary, y_pred_binary, zero_division=0)
                    ),
                    "f1": float(
                        f1_score(y_true_binary, y_pred_binary, zero_division=0)
                    ),
                    "support": int(np.sum(y_true == label)),
                }

            metrics["per_class"] = per_class_metrics

        logger.info(
            f"Computed metrics: accuracy={accuracy:.4f}, f1_macro={f1_macro:.4f}"
        )

        return metrics

    def evaluate_model(
        self,
        model: Any,
        test_df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str,
        class_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a local model object on test data.

        Args:
            model: Trained model with predict() and optionally predict_proba() methods.
            test_df: Pandas DataFrame containing test data.
            feature_columns: List of feature column names.
            target_column: Name of the target/label column.
            class_labels: Optional list of class names for per-class metrics.

        Returns:
            Dict containing all computed metrics plus:
                - test_size: Number of test samples
                - evaluated_at: ISO timestamp of evaluation
        """
        logger.info("Evaluating model on test data")

        features = test_df[feature_columns]
        y_true = test_df[target_column].values

        # Get predictions
        y_pred = model.predict(features)

        # Try to get probability predictions
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(features)
            except Exception as e:
                logger.warning(f"Could not get probability predictions: {e}")

        # Infer class labels from model if not provided
        if class_labels is None and hasattr(model, "classes_"):
            class_labels = list(model.classes_)

        metrics = self.compute_multiclass_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            class_labels=class_labels,
        )

        metrics["test_size"] = len(test_df)
        metrics["evaluated_at"] = datetime.now().isoformat()

        return metrics

    def evaluate_from_registry(
        self,
        model_name: str,
        model_version: str,
        registry_database: str,
        registry_schema: str,
        test_table: str,
        feature_columns: List[str],
        target_column: str,
        class_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a model from the Snowflake Model Registry.

        Loads the model from the registry, retrieves test data from Snowflake,
        runs inference, and computes evaluation metrics.

        Args:
            model_name: Name of the model in the registry.
            model_version: Version string of the model to evaluate.
            registry_database: Database containing the model registry.
            registry_schema: Schema containing the model registry.
            test_table: Fully qualified name of the test data table.
            feature_columns: List of feature column names.
            target_column: Name of the target/label column.
            class_labels: Optional list of class names for per-class metrics.

        Returns:
            Dict containing all computed metrics plus:
                - model_name: Name of the evaluated model
                - model_version: Version that was evaluated
                - test_size: Number of test samples
                - evaluated_at: ISO timestamp of evaluation
        """
        from snowflake.ml.registry import Registry

        logger.info(f"Loading model {model_name}/{model_version} from registry")

        registry = Registry(
            self.session,
            database_name=registry_database,
            schema_name=registry_schema,
        )

        model = registry.get_model(model_name)
        model_version_obj = model.version(model_version)

        # Load test data from Snowflake
        logger.info(f"Loading test data from {test_table}")
        test_df = self.session.table(test_table).to_pandas()
        test_df.columns = [c.upper() for c in test_df.columns]

        features = test_df[feature_columns]
        y_true = test_df[target_column].values

        # Run inference using the registered model
        logger.info("Running inference")
        predictions_df = model_version_obj.run(features, function_name="predict")
        y_pred = predictions_df["output_feature_0"].values

        # Try to get probability predictions
        y_proba = None
        try:
            proba_df = model_version_obj.run(features, function_name="predict_proba")
            y_proba = proba_df.values
        except Exception as e:
            logger.warning(f"Could not get probability predictions: {e}")

        metrics = self.compute_multiclass_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            class_labels=class_labels,
        )

        metrics["model_name"] = model_name
        metrics["model_version"] = model_version
        metrics["test_size"] = len(test_df)
        metrics["evaluated_at"] = datetime.now().isoformat()

        return metrics

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        metrics_table: str,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> None:
        """
        Log evaluation metrics to a Snowflake table.

        Persists all numeric metrics, the confusion matrix, and per-class
        metrics to the specified metrics table for tracking and comparison.

        Args:
            metrics: Dict of metrics from evaluate_model() or evaluate_from_registry().
            metrics_table: Fully qualified name of the metrics table.
            model_name: Model name override (uses metrics dict value if not provided).
            model_version: Model version override (uses metrics dict value if not provided).
        """
        logger.info(f"Logging metrics to {metrics_table}")

        # Log scalar metrics
        records = []
        for metric_name, metric_value in metrics.items():
            # Skip non-scalar metrics (handled separately)
            if metric_name in ["confusion_matrix", "per_class", "evaluated_at"]:
                continue

            if isinstance(metric_value, (int, float)):
                records.append(
                    {
                        "METRIC_ID": f"M{uuid.uuid4().hex[:8].upper()}",
                        "MODEL_NAME": model_name or metrics.get("model_name"),
                        "MODEL_VERSION": model_version or metrics.get("model_version"),
                        "METRIC_NAME": metric_name,
                        "METRIC_VALUE": float(metric_value),
                        "METRIC_DETAILS": None,
                        "EVALUATED_AT": datetime.now(),
                    }
                )

        if records:
            df = pd.DataFrame(records)
            snowpark_df = self.session.create_dataframe(df)
            snowpark_df.write.mode("append").save_as_table(metrics_table)
            logger.info(f"Logged {len(records)} metrics to {metrics_table}")

        # Log confusion matrix as JSON
        if "confusion_matrix" in metrics:
            import json

            cm_json = json.dumps(metrics["confusion_matrix"])
            self.session.sql(
                f"""
                INSERT INTO {metrics_table}
                (METRIC_ID, MODEL_NAME, MODEL_VERSION, METRIC_NAME, METRIC_VALUE, METRIC_DETAILS, EVALUATED_AT)
                SELECT
                    'M' || SUBSTR(UUID_STRING(), 1, 8),
                    '{model_name or metrics.get("model_name")}',
                    '{model_version or metrics.get("model_version")}',
                    'confusion_matrix',
                    NULL,
                    PARSE_JSON('{cm_json}'),
                    CURRENT_TIMESTAMP()
            """
            ).collect()

        # Log per-class metrics
        if "per_class" in metrics:
            per_class_records = []
            for class_label, class_metrics in metrics["per_class"].items():
                for metric_name, metric_value in class_metrics.items():
                    per_class_records.append(
                        {
                            "METRIC_ID": f"M{uuid.uuid4().hex[:8].upper()}",
                            "MODEL_NAME": model_name or metrics.get("model_name"),
                            "MODEL_VERSION": model_version
                            or metrics.get("model_version"),
                            "METRIC_NAME": f"{class_label}_{metric_name}",
                            "METRIC_VALUE": float(metric_value),
                            "METRIC_DETAILS": None,
                            "EVALUATED_AT": datetime.now(),
                        }
                    )
            if per_class_records:
                df = pd.DataFrame(per_class_records)
                snowpark_df = self.session.create_dataframe(df)
                snowpark_df.write.mode("append").save_as_table(metrics_table)
                logger.info(f"Logged {len(per_class_records)} per-class metrics")

    def check_promotion_criteria(
        self,
        metrics: Dict[str, Any],
        thresholds: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Check if metrics meet promotion thresholds.

        Compares each metric against its threshold to determine if the model
        should be promoted to production.

        Args:
            metrics: Dict of computed metrics.
            thresholds: Dict mapping metric names to minimum required values.
                Example: {"accuracy": 0.85, "f1_macro": 0.80}

        Returns:
            Dict containing:
                - should_promote: True if all thresholds are met
                - checks: Dict of per-metric results with:
                    - passed: Whether this threshold was met
                    - threshold: Required value
                    - actual: Actual metric value
                    - reason: Failure reason if applicable
        """
        logger.info("Checking promotion criteria")

        result = {
            "should_promote": True,
            "checks": {},
        }

        for metric_name, threshold in thresholds.items():
            actual_value = metrics.get(metric_name)

            # Check if metric exists
            if actual_value is None:
                result["checks"][metric_name] = {
                    "passed": False,
                    "reason": "metric_not_found",
                }
                result["should_promote"] = False
                continue

            # Check if metric meets threshold
            passed = actual_value >= threshold
            result["checks"][metric_name] = {
                "passed": passed,
                "threshold": threshold,
                "actual": actual_value,
            }

            if not passed:
                result["should_promote"] = False
                logger.warning(
                    f"Promotion check failed for {metric_name}: "
                    f"{actual_value:.4f} < {threshold:.4f}"
                )

        if result["should_promote"]:
            logger.info("All promotion criteria passed")
        else:
            logger.info("Promotion criteria not met")

        return result

    def generate_report(
        self,
        metrics: Dict[str, Any],
        promotion_result: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a human-readable evaluation report.

        Creates a formatted text report summarizing model performance,
        including overall metrics, per-class breakdown, and promotion status.

        Args:
            metrics: Dict of computed metrics.
            promotion_result: Optional promotion check results from
                check_promotion_criteria().

        Returns:
            Formatted string report suitable for logging or display.
        """
        report = []
        report.append("=" * 60)
        report.append("MODEL EVALUATION REPORT")
        report.append("=" * 60)

        # Model identification
        if "model_name" in metrics:
            report.append(f"\nModel: {metrics['model_name']}")
        if "model_version" in metrics:
            report.append(f"Version: {metrics['model_version']}")
        if "test_size" in metrics:
            report.append(f"Test samples: {metrics['test_size']}")
        if "evaluated_at" in metrics:
            report.append(f"Evaluated at: {metrics['evaluated_at']}")

        # Overall metrics section
        report.append("\n" + "-" * 40)
        report.append("OVERALL METRICS")
        report.append("-" * 40)

        overall_metrics = [
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
            "log_loss",
        ]

        for metric in overall_metrics:
            if metric in metrics:
                report.append(f"{metric}: {metrics[metric]:.4f}")

        # Per-class metrics section
        if "per_class" in metrics:
            report.append("\n" + "-" * 40)
            report.append("PER-CLASS METRICS")
            report.append("-" * 40)

            for class_label, class_metrics in metrics["per_class"].items():
                report.append(f"\n{class_label}:")
                for metric_name, value in class_metrics.items():
                    report.append(f"  {metric_name}: {value:.4f}")

        # Promotion criteria section
        if promotion_result:
            report.append("\n" + "-" * 40)
            report.append("PROMOTION CRITERIA")
            report.append("-" * 40)

            for metric_name, check in promotion_result.get("checks", {}).items():
                status = "PASSED" if check.get("passed") else "FAILED"
                threshold = check.get("threshold", "N/A")
                actual = check.get("actual", "N/A")
                report.append(
                    f"{metric_name}: {status} (threshold: {threshold}, actual: {actual})"
                )

            overall_status = (
                "APPROVED" if promotion_result.get("should_promote") else "REJECTED"
            )
            report.append(f"\nPromotion Status: {overall_status}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)
