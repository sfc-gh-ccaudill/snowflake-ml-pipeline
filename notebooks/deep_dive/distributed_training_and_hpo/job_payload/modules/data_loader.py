"""
Shared data loading module for all training entrypoints.

Handles loading training data from either a Feature Store Dataset
(preserving lineage) or a plain Snowflake table, plus test data
from a table. Normalizes column names to uppercase.
"""

import logging

import pandas as pd
from snowflake.snowpark import Session

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads train/test DataFrames from Snowflake."""

    def __init__(self, session: Session, database: str, schema: str):
        self.session = session
        self.database = database
        self.schema = schema

    def _fqn(self, name: str) -> str:
        return f"{self.database}.{self.schema}.{name}"

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [c.upper() for c in df.columns]
        return df

    def _get_latest_dataset_version(self, dataset_name: str) -> str:
        fqn = self._fqn(dataset_name)
        rows = self.session.sql(f"SHOW VERSIONS IN DATASET {fqn}").collect()
        if not rows:
            raise ValueError(f"No versions found for dataset {fqn}")
        latest = rows[-1]["version"]
        logger.info("Resolved latest dataset version: %s", latest)
        return latest

    def _load_dataset(self, dataset_name: str, version: str = None) -> pd.DataFrame:
        from snowflake.ml.dataset import load_dataset

        if version is None:
            version = self._get_latest_dataset_version(dataset_name)

        logger.info("Loading dataset: %s (version=%s)", dataset_name, version)
        ds = load_dataset(self.session, self._fqn(dataset_name), version=version)
        df = self._normalize(ds.read.to_pandas())
        logger.info("  Rows: %d", len(df))
        return df

    def load_table(self, table_name: str) -> pd.DataFrame:
        logger.info("Loading table: %s", table_name)
        df = self._normalize(self.session.table(table_name).to_pandas())
        logger.info("  Rows: %d", len(df))
        return df

    def load_train_test(
        self,
        train_dataset_name: str = None,
        train_dataset_version: str = None,
        train_table_name: str = None,
        test_table_name: str = "TEST_FEATURES",
    ) -> tuple:
        if train_dataset_name:
            train_df = self._load_dataset(train_dataset_name, version=train_dataset_version)
        elif train_table_name:
            train_df = self.load_table(train_table_name)
        else:
            raise ValueError("Provide either train_dataset_name or train_table_name")

        test_df = self.load_table(test_table_name)
        return train_df, test_df
