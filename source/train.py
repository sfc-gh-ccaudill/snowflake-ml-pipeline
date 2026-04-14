"""
Patient Risk Stratification Training Pipeline.
"""

import logging
import os
import pickle
from typing import Any, Dict, List, Optional
import time

from snowflake.snowpark import Session
import numpy as np
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from snowflake.ml.registry import Registry
from snowflake.ml.model.task import Task
from snowflake.ml.experiment import ExperimentTracking

try:
    from configs import get_config
    from utils import get_session, get_feature_config
except ModuleNotFoundError:
    from source.configs import get_config
    from source.utils import get_session, get_feature_config

logger = logging.getLogger(__name__)


class PatientRiskTraining:
    """Training step for the Patient Risk Stratification pipeline."""

    def __init__(self, database: str, schema_name: str):
        self.session = get_session()
        self.database = database
        self.schema_name = schema_name

    def create_training_pipeline(
        self, numeric_columns, categorical_columns, model_params
    ):

        numeric_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_transformer = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="constant", fill_value="Unknown"),
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_columns),
                ("cat", categorical_transformer, categorical_columns),
            ],
            remainder="drop",
        )

        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", RandomForestClassifier(**model_params)),
            ]
        )

        return pipeline

    def calculate_metrics(self, y_test, y_pred):

        metrics = dict()
        metrics["test_accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["test_precision"] = float(
            precision_score(y_test, y_pred, average="weighted", zero_division=0)
        )
        metrics["test_recall"] = float(
            recall_score(y_test, y_pred, average="weighted", zero_division=0)
        )
        metrics["test_f1"] = float(
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        )

        return metrics

    def get_data(self, table_name):
        """Load Training Data"""
        df = self.session.table(table_name).to_pandas()
        df.columns = [c.upper() for c in df.columns]
        return df

    def train(
        self,
        train_table: str,
        test_table: str,
        feature_config: Dict,
        log_experiment: bool = True,
        register_model: bool = True,
        save_artifacts: bool = True,
        model_name: str = "PATIENT_RISK_MODEL",
        target_platforms: Optional[List[str]] = None,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:

        # == Get Feature Configuration ==
        numeric_columns = feature_config["all_numeric_features"]
        categorical_columns = feature_config["all_categorical_features"]
        feature_columns = numeric_columns + categorical_columns
        target_column = feature_config["target_column"]

        # == Get Training Data ==
        train_df = self.get_data(train_table)
        X_train = train_df[feature_columns]
        y_train = train_df[target_column]

        logger.info(f"Training data shape: X={X_train.shape}, y={y_train.shape}")

        if model_params is None:
            model_params = {
                "n_estimators": 100,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
            }

        # == Construct Model Pipeline ==
        model = self.create_training_pipeline(
            numeric_columns, categorical_columns, model_params
        )

        # == Train Model ==
        logger.info("Training model...")
        model.fit(X_train, y_train)

        # == Generate Training Predictions ==
        train_pred = model.predict(X_train)

        # == Calculate Training Accuracy ==
        train_accuracy = accuracy_score(y_train, train_pred)
        metrics = {
            "train_accuracy": float(train_accuracy),
        }

        # == Load Test Data ==
        logger.info(f"Loading test data from {test_table}")
        test_df = self.get_data(test_table)
        X_test = test_df[feature_columns]
        y_test = test_df[target_column]

        # == Calculate Test Metrics ==
        logger.info("Evaluating on test data...")
        y_pred = model.predict(X_test)
        test_metrics = self.calculate_metrics(y_test, y_pred)
        metrics = metrics | test_metrics
        logger.info(f"Test accuracy: {metrics['test_accuracy']:.4f}")
        logger.info(f"Test F1: {metrics['test_f1']:.4f}")

        if log_experiment:
            self.log_experiment(model_name, metrics, model_params)

        if save_artifacts:
            self.save_artifacts(model, feature_columns, metrics)

        if register_model:
            self.register_model(model, model_name, X_train, metrics, target_platforms)

    def log_experiment(self, model_name, metrics, model_params):

        exp = ExperimentTracking(
            session=self.session,
            database_name=self.database,
            schema_name=self.schema_name,
        )
        EXPERIMENT_NAME = str(f"{model_name}_EXPERIMENT").upper()
        exp.set_experiment(EXPERIMENT_NAME)
        logger.info(f"Experiment: {EXPERIMENT_NAME}")
        run_name = f"baseline_{int(time.time())}"
        with exp.start_run(run_name):
            exp.log_params({**model_params, "run_type": "baseline"})
            exp.log_metrics(
                {
                    "test_accuracy": metrics["test_accuracy"],
                    "test_precision": metrics["test_precision"],
                    "test_recall": metrics["test_recall"],
                    "test_f1": metrics["test_f1"],
                }
            )
        logger.info(f"\nLogged run '{run_name}' to experiment {EXPERIMENT_NAME}")

    def register_model(self, model, model_name, train_data, metrics, target_platforms):
        logger.info(f"Registering model: {model_name}")
        registry = Registry(
            self.session,
            database_name=self.database,
            schema_name=self.schema_name,
        )

        sample_data = train_data.head(10).fillna(0)
        for col in sample_data.select_dtypes(include=["object"]).columns:
            sample_data[col] = sample_data[col].fillna("Unknown")

        model_version = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        registry.log_model(
            model=model,
            model_name=model_name,
            version_name=model_version,
            sample_input_data=sample_data,
            metrics=metrics,
            task=Task.TABULAR_MULTI_CLASSIFICATION,
            target_platforms=target_platforms,
            options={"enable_explainability": True},
            comment=f"Trained via ML Jobs at {datetime.now().isoformat()}",
        )
        logger.info(f"Model registered: {model_name}/{model_version}")

    def save_artifacts(self, model, feature_columns, metrics):

        MODEL_FILE = "risk_model.pkl"
        METRICS_FILE = "metrics.pkl"
        MODEL_STAGE = f"{self.database}.{self.schema_name}.MODEL_ARTIFACTS"
        os.makedirs("/tmp/model", exist_ok=True)
        model_path = f"/tmp/model/{MODEL_FILE}"
        metrics_path = f"/tmp/model/{METRICS_FILE}"

        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "features": feature_columns}, f)

        with open(metrics_path, "wb") as f:
            pickle.dump(metrics, f)

        self.session.file.put(
            model_path, MODEL_STAGE, auto_compress=False, overwrite=True
        )
        self.session.file.put(
            metrics_path, MODEL_STAGE, auto_compress=False, overwrite=True
        )

        logger.info(f"\nModel saved to {MODEL_STAGE}/{model_path}")
        logger.info(f"Metrics saved to {MODEL_STAGE}/{metrics_path}")

        return


def main():

    config = get_config("config.yaml")

    DB = config.snowflake.database
    SCHEMA = config.snowflake.schema_name
    feature_config = get_feature_config(config)

    trainer = PatientRiskTraining(database=DB, schema_name=SCHEMA)

    trainer.train(
        train_table=f"{DB}.{SCHEMA}.TRAINING_FEATURES",
        feature_config=feature_config,
        test_table=f"{DB}.{SCHEMA}.TEST_FEATURES",
        register_model=True,
        model_name=config.model.model_name,
        target_platforms=config.model.target_platforms,
    )


if __name__ == "__main__":
    main()
