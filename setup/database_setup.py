"""
Database and schema setup for Healthcare ML Pipeline.
"""

import logging
from typing import Optional

from snowflake.snowpark import Session

logger = logging.getLogger(__name__)


class DatabaseSetup:
    def __init__(
        self, session: Session, database: str, schema_name: str, warehouse: str
    ):
        self.session = session
        self.database = database
        self.schema_name = schema_name
        self.warehouse = warehouse

    def create_database(self) -> None:
        logger.info(f"Creating database: {self.database}")
        self.session.sql(f"CREATE DATABASE IF NOT EXISTS {self.database}").collect()
        logger.info(f"Database {self.database} ready")

    def create_schema(self) -> None:
        logger.info(f"Creating schema: {self.database}.{self.schema_name}")
        self.session.sql(
            f"CREATE SCHEMA IF NOT EXISTS {self.database}.{self.schema_name}"
        ).collect()
        logger.info(f"Schema {self.database}.{self.schema_name} ready")

    def create_warehouse(self, size: str = "MEDIUM", auto_suspend: int = 300) -> None:
        logger.info(f"Creating warehouse: {self.warehouse}")
        self.session.sql(
            f"""
            CREATE WAREHOUSE IF NOT EXISTS {self.warehouse}
            WITH WAREHOUSE_SIZE = '{size}'
            AUTO_SUSPEND = {auto_suspend}
            AUTO_RESUME = TRUE
            INITIALLY_SUSPENDED = FALSE
        """
        ).collect()
        logger.info(f"Warehouse {self.warehouse} ready")

    def grant_permissions(self, role: Optional[str] = None) -> None:
        if role is None:
            role = self.session.get_current_role()

        logger.info(f"Granting permissions to role: {role}")

        grants = [
            f"GRANT USAGE ON DATABASE {self.database} TO ROLE {role}",
            f"GRANT ALL ON SCHEMA {self.database}.{self.schema_name} TO ROLE {role}",
            f"GRANT USAGE ON WAREHOUSE {self.warehouse} TO ROLE {role}",
        ]

        for grant in grants:
            try:
                self.session.sql(grant).collect()
            except Exception as e:
                logger.warning(
                    f"Grant may already exist or insufficient privileges: {e}"
                )

    def set_context(self) -> None:
        logger.info("Setting session context")
        self.session.use_database(self.database)
        self.session.use_schema(self.schema_name)
        self.session.use_warehouse(self.warehouse)

    def run(self, warehouse_size: str = "MEDIUM") -> dict:
        logger.info("Running database setup")

        self.create_database()
        self.create_schema()
        self.create_warehouse(size=warehouse_size)
        self.grant_permissions()
        self.set_context()

        logger.info("Database setup complete")

        return {
            "database": self.database,
            "schema": self.schema_name,
            "warehouse": self.warehouse,
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

    setup = DatabaseSetup(
        session=session,
        database=config.snowflake.database,
        schema_name=config.snowflake.schema_name,
        warehouse=config.snowflake.warehouse,
    )

    result = setup.run()
    logger.info(f"Setup result: {result}")
    return result


if __name__ == "__main__":
    main()
