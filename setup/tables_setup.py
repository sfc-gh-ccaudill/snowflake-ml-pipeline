"""
Table creation for Healthcare ML Pipeline.
"""

import logging
from typing import List

from snowflake.snowpark import Session

logger = logging.getLogger(__name__)

PATIENT_DATA_COLUMNS = """
    PATIENT_ID VARCHAR(50) NOT NULL,
    ENCOUNTER_ID VARCHAR(50) NOT NULL,
    TIMESTAMP TIMESTAMP_NTZ NOT NULL,
    AGE NUMBER(3),
    GENDER VARCHAR(10),
    BMI FLOAT,
    HEART_RATE NUMBER(3),
    SYSTOLIC_BP NUMBER(3),
    DIASTOLIC_BP NUMBER(3),
    TEMPERATURE FLOAT,
    RESPIRATORY_RATE NUMBER(2),
    OXYGEN_SATURATION FLOAT,
    GLUCOSE_LEVEL FLOAT,
    CREATININE FLOAT,
    HEMOGLOBIN FLOAT,
    WBC_COUNT FLOAT,
    PRIMARY_DIAGNOSIS VARCHAR(20),
    COMORBIDITY_COUNT NUMBER(2),
    ADMISSION_TYPE VARCHAR(20),
    INSURANCE_TYPE VARCHAR(20),
    PREVIOUS_ADMISSIONS NUMBER(3),
    MEDICATION_COUNT NUMBER(3),
    RISK_LEVEL VARCHAR(20)
"""


class TablesSetup:
    def __init__(self, session: Session, database: str, schema_name: str):
        self.session = session
        self.database = database
        self.schema_name = schema_name
        self.full_schema = f"{database}.{schema_name}"

    def _execute(self, sql: str) -> None:
        self.session.sql(sql).collect()

    def create_raw_patient_table(self) -> None:
        table_name = f"{self.full_schema}.RAW_PATIENT_DATA"
        logger.info(f"Creating table: {table_name}")

        self._execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {PATIENT_DATA_COLUMNS},
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
        """
        )
        logger.info(f"Table {table_name} ready")

    def create_streaming_patient_table(self) -> None:
        table_name = f"{self.full_schema}.STREAMING_PATIENT_DATA"
        logger.info(f"Creating table: {table_name}")

        self._execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {PATIENT_DATA_COLUMNS},
                INGESTED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
        """
        )
        logger.info(f"Table {table_name} ready")

    def create_metrics_table(self) -> None:
        table_name = f"{self.full_schema}.MODEL_METRICS"
        logger.info(f"Creating table: {table_name}")

        self._execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                METRIC_ID VARCHAR(50) NOT NULL,
                MODEL_NAME VARCHAR(100),
                MODEL_VERSION VARCHAR(100),
                METRIC_NAME VARCHAR(100),
                METRIC_VALUE FLOAT,
                METRIC_DETAILS VARIANT,
                EVALUATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
        """
        )
        logger.info(f"Table {table_name} ready")

    def create_baseline_table(self) -> None:
        table_name = f"{self.full_schema}.BASELINE_PATIENT_DATA"
        logger.info(f"Creating table: {table_name}")

        self._execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {PATIENT_DATA_COLUMNS},
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
        """
        )
        logger.info(f"Table {table_name} ready")

    def create_test_data_table(self) -> None:
        table_name = f"{self.full_schema}.TEST_PATIENT_DATA"
        logger.info(f"Creating table: {table_name}")

        self._execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {PATIENT_DATA_COLUMNS},
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
        """
        )
        logger.info(f"Table {table_name} ready")

    def create_dynamic_streaming_table(self, warehouse: str) -> None:
        table_name = f"{self.full_schema}.STREAMING_FEATURES"
        source_table = f"{self.full_schema}.STREAMING_PATIENT_DATA"
        logger.info(f"Creating dynamic table: {table_name}")

        try:
            self._execute(
                f"""
                CREATE OR REPLACE DYNAMIC TABLE {table_name}
                TARGET_LAG = '1 minute'
                WAREHOUSE = {warehouse}
                AS
                SELECT
                    PATIENT_ID,
                    ENCOUNTER_ID,
                    TIMESTAMP,
                    AGE,
                    GENDER,
                    BMI,
                    HEART_RATE,
                    SYSTOLIC_BP,
                    DIASTOLIC_BP,
                    TEMPERATURE,
                    RESPIRATORY_RATE,
                    OXYGEN_SATURATION,
                    GLUCOSE_LEVEL,
                    CREATININE,
                    HEMOGLOBIN,
                    WBC_COUNT,
                    PRIMARY_DIAGNOSIS,
                    COMORBIDITY_COUNT,
                    ADMISSION_TYPE,
                    INSURANCE_TYPE,
                    PREVIOUS_ADMISSIONS,
                    MEDICATION_COUNT,
                    RISK_LEVEL,
                    INGESTED_AT
                FROM {source_table}
            """
            )
            logger.info(f"Dynamic table {table_name} ready")
        except Exception as e:
            logger.warning(
                f"Could not create dynamic table (may require source data): {e}"
            )

    def get_table_list(self) -> List[str]:
        return [
            f"{self.full_schema}.RAW_PATIENT_DATA",
            f"{self.full_schema}.STREAMING_PATIENT_DATA",
            f"{self.full_schema}.MODEL_METRICS",
            f"{self.full_schema}.BASELINE_PATIENT_DATA",
            f"{self.full_schema}.TEST_PATIENT_DATA",
        ]

    def run(self, warehouse: str = None) -> dict:
        logger.info("Running tables setup")

        self.create_raw_patient_table()
        self.create_streaming_patient_table()
        self.create_metrics_table()
        self.create_baseline_table()
        self.create_test_data_table()

        if warehouse:
            self.create_dynamic_streaming_table(warehouse)

        tables = self.get_table_list()
        logger.info(f"Tables setup complete. Created {len(tables)} tables.")

        return {
            "tables": tables,
            "status": "success",
        }


def main():
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from configs import get_config
    from source.utils import get_session

    logging.basicConfig(level=logging.INFO)

    config = get_config()
    session = get_session(config.snowflake.connection_name)
    session.use_database(config.snowflake.database)
    session.use_schema(config.snowflake.schema_name)

    setup = TablesSetup(
        session=session,
        database=config.snowflake.database,
        schema_name=config.snowflake.schema_name,
    )

    result = setup.run(warehouse=config.snowflake.warehouse)
    logger.info(f"Setup result: {result}")
    return result


if __name__ == "__main__":
    main()
