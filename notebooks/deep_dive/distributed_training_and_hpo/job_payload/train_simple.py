"""
Simple training entrypoint for ML Jobs (submit_directory demo).

Loads patient data from a Feature Store Dataset (for lineage),
trains a RandomForestClassifier, evaluates it, and registers the
model in the Snowflake Model Registry.

This script is designed to run inside an SPCS container via ML Jobs.
"""

import logging
import os

from modules.data_loader import DataLoader
from modules.preprocess import Preprocessor
from modules.train import Trainer
from sklearn.ensemble import RandomForestClassifier
from snowflake.snowpark import Session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


def main():
    # == Gather Config Params ==
    database = os.environ.get("DATABASE", "ML_DEMO_PIPELINE_DB")
    schema = os.environ.get("SCHEMA", "HEALTHCARE")
    model_name = os.environ.get("MODEL_NAME", "PATIENT_RISK_MODEL_SESSION3")
    dataset_name = os.environ.get("TRAINING_DATASET_NAME", "PATIENT_TRAINING_DATASET")
    dataset_version = os.environ.get("TRAINING_DATASET_VERSION", None)
    n_estimators = int(os.environ.get("N_ESTIMATORS", "150"))
    max_depth = int(os.environ.get("MAX_DEPTH", "10"))
    random_state = int(os.environ.get("RANDOM_STATE", "42"))

    logger.info("=== Simple RandomForest Training ===")
    logger.info("  n_estimators: %d | max_depth: %d", n_estimators, max_depth)

    # == Init Session ==
    session = Session.builder.getOrCreate()
    session.use_database(database)
    session.use_schema(schema)

    # == Load Data ==
    loader = DataLoader(session, database, schema)
    train_df, test_df = loader.load_train_test(
        train_dataset_name=dataset_name,
        train_dataset_version=dataset_version,
    )

    # == Preprocess Data ==
    preprocessor = Preprocessor()
    X_train, y_train, X_test, y_test = preprocessor.fit_transform(train_df, test_df)

    # == Init Model ==
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    # == Execute Training via ML Jobs ==
    trainer = Trainer(session, model, column_transformer=preprocessor.column_transformer)
    trainer.train(X_train, y_train)
    trainer.evaluate(X_test, y_test)
    trainer.register(
        model_name=model_name,
        database=database,
        schema=schema,
        sample_input=preprocessor.sample_input(train_df),
    )
    logger.info("=== Training complete ===")


if __name__ == "__main__":
    main()
