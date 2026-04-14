"""
Hyperparameter Tuning Entrypoint — executed remotely by RemoteTrainer.submit_hpo().

Business logic for the Healthcare Patient Risk Model HPO job:
  - Load TRAINING_FEATURES / TEST_FEATURES from Snowflake
  - Build the XGBoost preprocessing + classification pipeline
  - Expose a Ray Tune trainable that reports f1_macro

All Ray cluster management, scheduler / search-algorithm wiring, and result
persistence are delegated to RayTuneRunner (source/framework/hpo.py).
"""

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

try:
    from configs import get_config
    from framework.hpo import RayTuneRunner
    from utils import get_feature_config, get_session
except ModuleNotFoundError:
    from source.configs import get_config
    from source.framework.hpo import RayTuneRunner
    from source.utils import get_feature_config, get_session

logger = logging.getLogger(__name__)

_XGB_PARAM_KEYS = {
    "n_estimators", "max_depth", "learning_rate",
    "subsample", "colsample_bytree", "reg_alpha", "reg_lambda",
}
_GBM_PARAM_KEYS = {"n_estimators", "max_depth", "learning_rate", "subsample"}


def _build_pipeline(numeric_cols: list, categorical_cols: list, params: dict) -> Pipeline:
    """Preprocessing + classifier pipeline for patient risk scoring.

    Uses XGBoost when available, falls back to sklearn GradientBoostingClassifier.
    Target labels must be integer-encoded before calling fit/predict.
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ], remainder="drop")

    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            **{k: v for k, v in params.items() if k in _XGB_PARAM_KEYS},
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(
            **{k: v for k, v in params.items() if k in _GBM_PARAM_KEYS},
            random_state=42,
        )

    return Pipeline([("preprocessor", preprocessor), ("model", clf)])


def build_trainable(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_config: dict):
    """Return a Ray Tune trainable closed over the prepared training data.

    String target labels (LOW / MEDIUM / HIGH / CRITICAL) are encoded once
    at construction time so every trial reuses the same encoder without
    refitting it.  Any trial-level exception is caught and reported as
    f1_macro=0.0 so Ray always receives a metric.
    """
    numeric_cols     = [c.upper() for c in feature_config["all_numeric_features"]]
    categorical_cols = [c.upper() for c in feature_config["all_categorical_features"]]
    target_col       = feature_config["target_column"].upper()

    X_train = train_df[numeric_cols + categorical_cols]
    X_test  = test_df[numeric_cols + categorical_cols]

    le      = LabelEncoder()
    y_train = le.fit_transform(train_df[target_col])
    y_test  = le.transform(test_df[target_col])

    def trainable(config):
        import logging as _logging
        from ray import tune as ray_tune

        _log = _logging.getLogger("train_hpo.trainable")
        try:
            pipeline = _build_pipeline(numeric_cols, categorical_cols, config)
            pipeline.fit(X_train, y_train)
            y_pred  = pipeline.predict(X_test)
            f1_mac  = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
            ray_tune.report({"f1_macro": f1_mac})
        except Exception as exc:
            _log.error("Trial failed (config=%s): %s", config, exc, exc_info=True)
            ray_tune.report({"f1_macro": 0.0})

    return trainable


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    runner = RayTuneRunner.from_env()
    runner.init_cluster()

    config  = get_config("config.yaml")
    session = get_session()
    session.use_database(config.snowflake.database)
    session.use_schema(config.snowflake.schema_name)
    session.use_warehouse(config.snowflake.warehouse)

    db     = config.snowflake.database
    schema = config.snowflake.schema_name

    feature_config = get_feature_config(config)

    logger.info("Loading TRAINING_FEATURES and TEST_FEATURES ...")
    train_df = session.table(f"{db}.{schema}.TRAINING_FEATURES").to_pandas()
    test_df  = session.table(f"{db}.{schema}.TEST_FEATURES").to_pandas()
    train_df.columns = [c.upper() for c in train_df.columns]
    test_df.columns  = [c.upper() for c in test_df.columns]
    logger.info("Data loaded: train=%d rows, test=%d rows", len(train_df), len(test_df))

    trainable               = build_trainable(train_df, test_df, feature_config)
    results                 = runner.run(trainable)
    best_config, best_score = runner.best_result(results)

    logger.info("Best f1_macro : %.4f", best_score)

    runner.write_results(session, db, schema, best_config, best_score)
    runner.shutdown()
    logger.info("train_hpo finished.")


if __name__ == "__main__":
    main()
