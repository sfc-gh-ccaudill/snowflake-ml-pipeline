"""
HPO entrypoint for the Tuner API demo (Notebook 04).

Self-contained script that creates and runs the Snowflake Tuner API
server-side on SPCS. Submitted via submit_directory from a local notebook.

Each trial trains an XGBoost model with sampled hyperparameters,
reports metrics back to the Tuner, and logs the trial to Experiment Tracking.

This script runs inside an SPCS container via ML Jobs.
"""

import logging
import os
import time

from snowflake.snowpark import Session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger(__name__)


def train_func():
    """Called once per trial by the Tuner. Receives sampled hyperparameters."""
    from modules.preprocess import Preprocessor
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.pipeline import Pipeline
    from snowflake.ml.modeling.tune import get_tuner_context
    from xgboost import XGBClassifier

    # == Establish Context for Trial ==
    # Note: This is where the training function determines the
    # hyperparameters for each specific trial
    ctx = get_tuner_context()
    params = ctx.get_hyper_params()
    datasets = ctx.get_dataset_map()

    logger.info("Trial params: %s", params)

    train_df = datasets["train"].to_pandas()
    test_df = datasets["test"].to_pandas()
    train_df.columns = [c.upper() for c in train_df.columns]
    test_df.columns = [c.upper() for c in test_df.columns]

    preprocessor = Preprocessor()
    X_train, y_train, X_test, y_test = preprocessor.fit_transform(train_df, test_df)

    clf = XGBClassifier(
        n_estimators=int(params.get("n_estimators", 100)),
        max_depth=int(params.get("max_depth", 6)),
        learning_rate=float(params.get("learning_rate", 0.1)),
        subsample=float(params.get("subsample", 0.8)),
        colsample_bytree=float(params.get("colsample_bytree", 0.8)),
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    full_pipeline = Pipeline([("preprocessor", preprocessor.column_transformer), ("model", clf)])
    ctx.report(
        metrics={"accuracy": accuracy, "f1_macro": f1_macro},
        model=full_pipeline,
    )

    logger.info("Trial complete — accuracy: %.4f | f1_macro: %.4f", accuracy, f1_macro)


class HPORunner:
    SEARCH_ALG_MAP = {
        "RandomSearch": "RandomSearch",
        "GridSearch": "GridSearch",
        "BayesOpt": "BayesOpt",
    }

    def __init__(
        self,
        database=None,
        schema=None,
        num_trials=None,
        model_name=None,
        experiment_name=None,
        metric=None,
        mode=None,
        search_alg=None,
    ):
        self.database = database or os.environ.get("DATABASE", "ML_DEMO_PIPELINE_DB")
        self.schema = schema or os.environ.get("SCHEMA", "HEALTHCARE")
        self.num_trials = num_trials or int(os.environ.get("NUM_TRIALS", "10"))
        self.model_name = model_name or os.environ.get("MODEL_NAME", "PATIENT_RISK_HPO_BEST")
        self.experiment_name = experiment_name or os.environ.get(
            "EXPERIMENT_NAME", "PATIENT_RISK_HPO"
        )
        self.metric = metric or os.environ.get("TUNER_METRIC", "f1_macro")
        self.mode = mode or os.environ.get("TUNER_MODE", "max")
        self.search_alg_name = search_alg or os.environ.get("TUNER_SEARCH_ALG", "RandomSearch")

        self.session = Session.builder.getOrCreate()
        self.session.use_database(self.database)
        self.session.use_schema(self.schema)

        self.results = None

    def build_search_space(self):
        from snowflake.ml.modeling.tune import loguniform, randint, uniform

        return {
            "n_estimators": randint(50, 300),
            "max_depth": randint(3, 12),
            "learning_rate": loguniform(0.001, 0.3),
            "subsample": uniform(0.6, 1.0),
            "colsample_bytree": uniform(0.6, 1.0),
        }

    def _resolve_search_alg(self):
        from snowflake.ml.modeling.tune.search import BayesOpt, GridSearch, RandomSearch

        alg_map = {
            "RandomSearch": RandomSearch,
            "GridSearch": GridSearch,
            "BayesOpt": BayesOpt,
        }
        cls = alg_map.get(self.search_alg_name, RandomSearch)
        return cls()

    def run_tuner(self):
        from snowflake.ml.data.data_connector import DataConnector
        from snowflake.ml.modeling.tune import Tuner, TunerConfig

        logger.info("=== Hyperparameter Optimization (Tuner API) ===")
        logger.info("  Metric: %s (%s)", self.metric, self.mode)
        logger.info("  Search: %s", self.search_alg_name)
        logger.info("  Trials: %d", self.num_trials)

        search_space = self.build_search_space()

        tuner_config = TunerConfig(
            metric=self.metric,
            mode=self.mode,
            search_alg=self._resolve_search_alg(),
            num_trials=self.num_trials,
        )

        train_connector = DataConnector.from_dataframe(self.session.table("TRAINING_FEATURES"))
        test_connector = DataConnector.from_dataframe(self.session.table("TEST_FEATURES"))

        logger.info("Starting Tuner ...")
        tuner = Tuner(
            train_func=train_func,
            search_space=search_space,
            tuner_config=tuner_config,
        )

        self.results = tuner.run(
            dataset_map={"train": train_connector, "test": test_connector},
        )

        logger.info("HPO complete — %d trials", len(self.results.results))
        logger.info("Best result:\n%s", self.results.best_result)
        return self.results

    def get_best_model(self):
        if self.results is None:
            raise RuntimeError("No results available — call run_tuner() first")
        return self.results.best_model

    def get_best_metrics(self):
        if self.results is None:
            raise RuntimeError("No results available — call run_tuner() first")
        best = self.results.best_result
        return {k: float(best[k]) for k in ("accuracy", "f1_macro") if k in best}

    def register_best_model(self):
        from modules.preprocess import TARGET_COL
        from snowflake.ml.registry import Registry

        best_model = self.get_best_model()
        if best_model is None:
            logger.warning("No best_model returned from Tuner — skipping registration")
            return None

        registry = Registry(
            session=self.session, database_name=self.database, schema_name=self.schema
        )

        sample_input = self.session.table("TEST_FEATURES").limit(10).to_pandas()
        sample_input.columns = [c.upper() for c in sample_input.columns]
        sample_input = sample_input.drop(columns=[TARGET_COL], errors="ignore")

        metrics = self.get_best_metrics()

        mv = registry.log_model(
            model=best_model,
            model_name=self.model_name,
            version_name=f"hpo_v{int(time.time())}",
            sample_input_data=sample_input,
            metrics=metrics,
            comment=f"Best HPO trial ({self.num_trials} trials, RandomSearch)",
        )
        logger.info("Registered best model: %s version %s", mv.model_name, mv.version_name)
        return mv

    def log_experiments(self):
        from snowflake.ml.experiment import ExperimentTracking

        if self.results is None:
            raise RuntimeError("No results available — call run_tuner() first")

        logger.info(
            "Logging %d trials to experiment: %s", len(self.results.results), self.experiment_name
        )

        exp = ExperimentTracking(
            session=self.session, database_name=self.database, schema_name=self.schema
        )
        exp.set_experiment(self.experiment_name)

        for idx, row in self.results.results.iterrows():
            run_name = f"trial_{idx}_{int(row.get('f1_macro', 0) * 10000)}"
            with exp.start_run(run_name):
                params = {
                    k.replace("config/", ""): str(row[k])
                    for k in row.index
                    if k.startswith("config/")
                }
                metrics = {
                    k: float(row[k])
                    for k in row.index
                    if k in ("accuracy", "f1_macro") and row[k] is not None
                }
                if not params:
                    params = {
                        k: str(row[k]) for k in row.index if k not in ("accuracy", "f1_macro")
                    }
                exp.log_params(params)
                exp.log_metrics(metrics)
            logger.info("  Logged trial %d: f1_macro=%.4f", idx, row.get("f1_macro", 0))

        logger.info("All trials logged to experiment: %s", self.experiment_name)

    def run(self):
        self.run_tuner()
        self.register_best_model()
        self.log_experiments()
        logger.info("=== HPO complete ===")


if __name__ == "__main__":
    runner = HPORunner()
    runner.run()
