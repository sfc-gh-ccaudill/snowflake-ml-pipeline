"""
XGBoost training entrypoint for Dockerfile-based ML Jobs demo.

This script is packaged inside a Docker image and submitted via
submit_directory with a Dockerfile. It demonstrates how to use a
custom container for training when you need full control over the
runtime environment.
"""

from datetime import datetime
import logging
import os

from modules.data_loader import DataLoader
from modules.preprocess import Preprocessor
from modules.train import Trainer
from snowflake.snowpark import Session
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


def main():
    database = os.environ.get("DATABASE", "ML_DEMO_PIPELINE_DB")
    schema = os.environ.get("SCHEMA", "HEALTHCARE")
    model_name = os.environ.get("MODEL_NAME", "PATIENT_RISK_XGB_DOCKER")
    n_estimators = int(os.environ.get("N_ESTIMATORS", "200"))
    max_depth = int(os.environ.get("MAX_DEPTH", "8"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "0.1"))

    logger.info("=== Dockerfile-based XGBoost Training ===")
    logger.info(
        "  n_estimators: %d | max_depth: %d | lr: %.4f", n_estimators, max_depth, learning_rate
    )

    session = Session.builder.getOrCreate()
    session.use_database(database)
    session.use_schema(schema)

    loader = DataLoader(session, database, schema)
    train_df, test_df = loader.load_train_test(
        train_table_name="TRAINING_FEATURES",
    )

    preprocessor = Preprocessor()
    X_train, y_train, X_test, y_test = preprocessor.fit_transform(train_df, test_df)

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    trainer = Trainer(session, model, column_transformer=preprocessor.column_transformer)
    trainer.train(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    metrics = trainer.evaluate(X_test, y_test)
    metrics.update(
        {"n_estimators": n_estimators, "max_depth": max_depth, "learning_rate": learning_rate}
    )
    trainer.register(
        model_name=model_name,
        database=database,
        schema=schema,
        sample_input=preprocessor.sample_input(train_df),
        metrics=metrics,
        comment=f"Dockerfile-based XGBoost training — {datetime.now().isoformat()}",
    )
    logger.info("=== Training complete ===")


if __name__ == "__main__":
    main()
