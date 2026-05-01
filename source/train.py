"""
Patient Risk Stratification Training Pipeline.
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from snowflake.ml.experiment import ExperimentTracking
from snowflake.ml.model.task import Task
from snowflake.ml.registry import Registry

try:
    from configs import get_config, get_config_from_dict
    from snowflake.ml.dataset import load_dataset
    from utils import get_feature_config, get_session
except ModuleNotFoundError:
    from snowflake.ml.dataset import load_dataset

    from source.configs import get_config, get_config_from_dict
    from source.utils import get_feature_config, get_session

logger = logging.getLogger(__name__)


class PatientRiskTraining:
    """Training step for the Patient Risk Stratification pipeline."""

    def __init__(self, database: str, schema_name: str):
        self.session = get_session()
        self.database = database
        self.schema_name = schema_name

    def create_training_pipeline(self, numeric_columns, categorical_columns, model_params):
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_columns),
                ("cat", categorical_transformer, categorical_columns),
            ],
            remainder="drop",
        )
        return Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(**model_params)),
        ])

    def calculate_metrics(self, y_test, y_pred):
        return {
            "test_accuracy":  float(accuracy_score(y_test, y_pred)),
            "test_precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "test_recall":    float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "test_f1":        float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        }

    def train(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_config: Dict,
        model_name: str,
        model_params: Dict[str, Any],
        target_platforms: Optional[List[str]] = None,
        log_experiment: bool = True,
        register_model: bool = True,
    ) -> Dict[str, Any]:

        numeric_columns = feature_config["all_numeric_features"]
        categorical_columns = feature_config["all_categorical_features"]
        feature_columns = numeric_columns + categorical_columns
        target_column = feature_config["target_column"]

        X_train = train_df[feature_columns]
        y_train = train_df[target_column]
        logger.info("Training data shape: X=%s, y=%s", X_train.shape, y_train.shape)

        model = self.create_training_pipeline(numeric_columns, categorical_columns, model_params)

        logger.info("Training model...")
        model.fit(X_train, y_train)

        metrics = {"train_accuracy": float(accuracy_score(y_train, model.predict(X_train)))}

        X_test = test_df[feature_columns]
        y_test = test_df[target_column]

        logger.info("Evaluating on test data...")
        metrics = metrics | self.calculate_metrics(y_test, model.predict(X_test))
        logger.info("Test accuracy: %.4f", metrics["test_accuracy"])
        logger.info("Test F1: %.4f", metrics["test_f1"])

        if log_experiment:
            self.log_experiment(model_name, metrics, model_params)

        if register_model:
            self.register_model(model, model_name, X_train, metrics, target_platforms)

        return metrics

    def log_experiment(self, model_name, metrics, model_params):
        exp = ExperimentTracking(
            session=self.session,
            database_name=self.database,
            schema_name=self.schema_name,
        )
        experiment_name = f"{model_name}_EXPERIMENT".upper()
        exp.set_experiment(experiment_name)
        logger.info("Experiment: %s", experiment_name)
        run_name = f"baseline_{int(time.time())}"
        with exp.start_run(run_name):
            exp.log_params({**model_params, "run_type": "baseline"})
            exp.log_metrics({
                "test_accuracy":  metrics["test_accuracy"],
                "test_precision": metrics["test_precision"],
                "test_recall":    metrics["test_recall"],
                "test_f1":        metrics["test_f1"],
            })
        logger.info("Logged run '%s' to experiment %s", run_name, experiment_name)

    def register_model(self, model, model_name, train_data, metrics, target_platforms):
        logger.info("Registering model: %s", model_name)
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
        logger.info("Model registered: %s/%s", model_name, model_version)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    _ml_cfg = os.environ.get("ML_PIPELINE_CONFIG")
    if _ml_cfg:
        config = get_config_from_dict(json.loads(_ml_cfg))
    else:
        config = get_config("config.yaml")

    db = config.snowflake.database
    schema = config.snowflake.schema_name
    session = get_session()
    session.use_database(db)
    session.use_schema(schema)
    session.use_warehouse(config.snowflake.warehouse)

    feature_config = get_feature_config(config)

    dataset_name    = os.environ.get("TRAINING_DATASET_NAME") or config.feature_store.training_dataset_name
    dataset_version = os.environ.get("TRAINING_DATASET_VERSION") or None

    logger.info("Loading training dataset: %s / %s", dataset_name, dataset_version)
    training_ds = load_dataset(session, f"{db}.{schema}.{dataset_name}", version=dataset_version)
    train_df = training_ds.read.to_pandas()
    train_df.columns = [c.upper() for c in train_df.columns]
    logger.info("Training rows: %d", len(train_df))

    test_table = f"{db}.{schema}.{config.tables.test_features}"
    logger.info("Loading test data from: %s", test_table)
    test_df = session.table(test_table).to_pandas()
    test_df.columns = [c.upper() for c in test_df.columns]
    logger.info("Test rows: %d", len(test_df))

    trainer = PatientRiskTraining(database=db, schema_name=schema)
    trainer.train(
        train_df=train_df,
        test_df=test_df,
        feature_config=feature_config,
        model_name=config.model.model_name,
        target_platforms=config.model.target_platforms,
        model_params={
            "n_estimators": config.model.params.n_estimators,
            "class_weight":  config.model.params.class_weight,
            "random_state":  config.model.params.random_state,
            "n_jobs":        config.model.params.n_jobs,
        },
    )


if __name__ == "__main__":
    main()
