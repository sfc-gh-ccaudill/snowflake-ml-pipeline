"""
Compute Pool setup for Healthcare ML Pipeline SPCS workloads.
"""

import logging
from typing import Optional

from snowflake.snowpark import Session

logger = logging.getLogger(__name__)


class ComputePoolSetup:
    def __init__(
        self,
        session: Session,
        compute_pool_name: str,
        instance_family: str = "CPU_X64_S",
        min_nodes: int = 1,
        max_nodes: int = 3,
    ):
        self.session = session
        self.compute_pool_name = compute_pool_name
        self.instance_family = instance_family
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes

    def create_compute_pool(self) -> dict:
        logger.info(f"Creating compute pool: {self.compute_pool_name}")
        logger.info(f"  Instance family: {self.instance_family}")
        logger.info(f"  Min nodes: {self.min_nodes}, Max nodes: {self.max_nodes}")

        try:
            self.session.sql(f"""
                CREATE COMPUTE POOL IF NOT EXISTS {self.compute_pool_name}
                MIN_NODES = {self.min_nodes}
                MAX_NODES = {self.max_nodes}
                INSTANCE_FAMILY = {self.instance_family}
                AUTO_SUSPEND_SECS = 300
                AUTO_RESUME = TRUE
            """).collect()

            logger.info(f"Compute pool {self.compute_pool_name} created")
            return {"created": True, "name": self.compute_pool_name}
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"Compute pool {self.compute_pool_name} already exists")
                return {"created": False, "exists": True, "name": self.compute_pool_name}
            logger.error(f"Failed to create compute pool: {e}")
            return {"created": False, "error": str(e)}

    def get_compute_pool_status(self) -> dict:
        logger.info(f"Getting status for compute pool: {self.compute_pool_name}")

        try:
            result = self.session.sql(f"""
                SHOW COMPUTE POOLS LIKE '{self.compute_pool_name}'
            """).collect()

            if result:
                row = result[0]
                return {
                    "name": row["name"],
                    "state": row["state"],
                    "instance_family": row["instance_family"],
                    "min_nodes": row["min_nodes"],
                    "max_nodes": row["max_nodes"],
                    "active_nodes": row["active_nodes"],
                    "idle_nodes": row["idle_nodes"],
                }
            return {"error": "Compute pool not found"}
        except Exception as e:
            logger.error(f"Failed to get compute pool status: {e}")
            return {"error": str(e)}

    def resume_compute_pool(self) -> dict:
        logger.info(f"Resuming compute pool: {self.compute_pool_name}")

        try:
            self.session.sql(f"ALTER COMPUTE POOL {self.compute_pool_name} RESUME").collect()
            logger.info("Compute pool resumed")
            return {"resumed": True}
        except Exception as e:
            logger.warning(f"Could not resume compute pool: {e}")
            return {"resumed": False, "error": str(e)}

    def suspend_compute_pool(self) -> dict:
        logger.info(f"Suspending compute pool: {self.compute_pool_name}")

        try:
            self.session.sql(f"ALTER COMPUTE POOL {self.compute_pool_name} SUSPEND").collect()
            logger.info("Compute pool suspended")
            return {"suspended": True}
        except Exception as e:
            logger.warning(f"Could not suspend compute pool: {e}")
            return {"suspended": False, "error": str(e)}

    def grant_usage(self, role: Optional[str] = None) -> dict:
        if role is None:
            role = self.session.get_current_role()

        logger.info(f"Granting usage on compute pool to role: {role}")

        try:
            self.session.sql(f"""
                GRANT USAGE ON COMPUTE POOL {self.compute_pool_name} TO ROLE {role}
            """).collect()
            logger.info("Usage granted")
            return {"granted": True}
        except Exception as e:
            logger.warning(f"Could not grant usage: {e}")
            return {"granted": False, "error": str(e)}

    def run(self, resume: bool = True) -> dict:
        logger.info("Running compute pool setup")

        create_result = self.create_compute_pool()

        if resume and create_result.get("created") or create_result.get("exists"):
            self.resume_compute_pool()

        self.grant_usage()

        status = self.get_compute_pool_status()

        logger.info("Compute pool setup complete")

        return {
            "compute_pool": self.compute_pool_name,
            "instance_family": self.instance_family,
            "status": status,
            "result": "success",
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

    setup = ComputePoolSetup(
        session=session,
        compute_pool_name=config.compute.compute_pool,
        instance_family=config.compute.instance_family,
        min_nodes=config.compute.min_nodes,
        max_nodes=config.compute.max_nodes,
    )

    result = setup.run()
    logger.info(f"Setup result: {result}")
    return result


if __name__ == "__main__":
    main()
