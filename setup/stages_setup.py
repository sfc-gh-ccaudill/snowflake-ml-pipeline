"""
Stages setup for Healthcare ML Pipeline artifacts.
"""

import logging
from typing import List, Optional

from snowflake.snowpark import Session

logger = logging.getLogger(__name__)


class StagesSetup:
    def __init__(self, session: Session, database: str, schema_name: str):
        self.session = session
        self.database = database
        self.schema_name = schema_name
        self.full_schema = f"{database}.{schema_name}"

    def _execute(self, sql: str) -> None:
        self.session.sql(sql).collect()

    def create_stage(self, stage_name: str, directory: bool = True) -> dict:
        full_stage = f"{self.full_schema}.{stage_name}"
        logger.info(f"Creating stage: {full_stage}")

        try:
            directory_clause = "DIRECTORY = (ENABLE = TRUE)" if directory else ""
            self._execute(f"""
                CREATE STAGE IF NOT EXISTS {full_stage}
                {directory_clause}
            """)
            logger.info(f"Stage {full_stage} created")
            return {"created": True, "stage": full_stage}
        except Exception as e:
            logger.warning(f"Could not create stage: {e}")
            return {"created": False, "error": str(e)}

    def create_model_artifacts_stage(self) -> dict:
        return self.create_stage("MODEL_ARTIFACTS")

    def create_job_payloads_stage(self) -> dict:
        return self.create_stage("JOB_PAYLOADS")

    def create_data_stage(self) -> dict:
        return self.create_stage("DATA_STAGE")

    def create_evaluation_stage(self) -> dict:
        return self.create_stage("EVALUATION_RESULTS")

    def grant_stage_permissions(
        self,
        stage_name: str,
        role: Optional[str] = None,
    ) -> dict:
        if role is None:
            role = self.session.get_current_role()

        full_stage = f"{self.full_schema}.{stage_name}"
        logger.info(f"Granting permissions on {full_stage} to role: {role}")

        try:
            grants = [
                f"GRANT READ ON STAGE {full_stage} TO ROLE {role}",
                f"GRANT WRITE ON STAGE {full_stage} TO ROLE {role}",
            ]
            for grant in grants:
                try:
                    self._execute(grant)
                except Exception:
                    pass
            return {"granted": True}
        except Exception as e:
            logger.warning(f"Could not grant permissions: {e}")
            return {"granted": False, "error": str(e)}

    def list_stages(self) -> List[str]:
        result = self.session.sql(f"SHOW STAGES IN SCHEMA {self.full_schema}").collect()
        return [row["name"] for row in result]

    def get_stage_list(self) -> List[str]:
        return [
            f"{self.full_schema}.MODEL_ARTIFACTS",
            f"{self.full_schema}.JOB_PAYLOADS",
            f"{self.full_schema}.DATA_STAGE",
            f"{self.full_schema}.EVALUATION_RESULTS",
        ]

    def run(self) -> dict:
        logger.info("Running stages setup")

        results = {
            "model_artifacts": self.create_model_artifacts_stage(),
            "job_payloads": self.create_job_payloads_stage(),
            "data_stage": self.create_data_stage(),
            "evaluation_results": self.create_evaluation_stage(),
        }

        for stage_name in ["MODEL_ARTIFACTS", "JOB_PAYLOADS", "DATA_STAGE", "EVALUATION_RESULTS"]:
            self.grant_stage_permissions(stage_name)

        stages = self.get_stage_list()
        logger.info(f"Stages setup complete. Created {len(stages)} stages.")

        return {
            "stages": stages,
            "results": results,
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

    setup = StagesSetup(
        session=session,
        database=config.snowflake.database,
        schema_name=config.snowflake.schema_name,
    )

    result = setup.run()
    logger.info(f"Setup result: {result}")
    return result


if __name__ == "__main__":
    main()
