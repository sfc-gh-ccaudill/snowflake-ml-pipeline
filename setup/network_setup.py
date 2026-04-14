"""
Network egress setup for SPCS ML Jobs.

Creates the network rule and external access integration that allow
pip to reach PyPI (and optionally other hosts) from inside SPCS
containers. The integration name must be passed as
`external_access_integrations` when submitting ML Jobs so that
containers inherit the egress policy.
"""

import logging
from typing import Optional

from snowflake.snowpark import Session

logger = logging.getLogger(__name__)

_DEFAULT_RULE        = "PYPI_NETWORK_RULE"
_DEFAULT_INTEGRATION = "PYPI_ACCESS_INTEGRATION"
_PYPI_HOSTS = (
    "pypi.org",
    "pypi.python.org",
    "pythonhosted.org",
    "files.pythonhosted.org",
    "raw.githubusercontent.com",
)


class NetworkSetup:
    """
    Creates a HOST_PORT egress network rule covering PyPI and associates
    it with an external access integration that can be attached to
    SPCS ML Jobs via ``external_access_integrations``.

    Args:
        session:          Active Snowpark session.
        database:         Database that owns the network rule.
        schema:           Schema that owns the network rule.
        network_rule:     Name for the network rule object.
        integration_name: Name for the external access integration.
    """

    def __init__(
        self,
        session: Session,
        database: str,
        schema: str,
        network_rule: str = _DEFAULT_RULE,
        integration_name: str = _DEFAULT_INTEGRATION,
    ):
        self.session          = session
        self.database         = database
        self.schema           = schema
        self.network_rule     = network_rule
        self.integration_name = integration_name
        self._fq_rule         = f"{database}.{schema}.{network_rule}"

    def create_network_rule(self) -> dict:
        hosts = ", ".join(f"'{h}'" for h in _PYPI_HOSTS)
        logger.info("Creating network rule: %s", self._fq_rule)
        try:
            self.session.sql(f"""
                CREATE OR REPLACE NETWORK RULE {self._fq_rule}
                    MODE       = EGRESS
                    TYPE       = HOST_PORT
                    VALUE_LIST = ({hosts})
            """).collect()
            logger.info("Network rule created: %s", self._fq_rule)
            return {"created": True, "name": self._fq_rule}
        except Exception as exc:
            logger.error("Failed to create network rule: %s", exc)
            return {"created": False, "error": str(exc)}

    def admin_sql(self, grant_to_role: Optional[str] = None) -> str:
        """
        Return the SQL an ACCOUNTADMIN must run once to create the integration
        and grant usage.  Useful when the current role lacks CREATE INTEGRATION.
        """
        role = grant_to_role or self.session.get_current_role()
        return "\n".join([
            f"-- Run as ACCOUNTADMIN",
            f"CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION {self.integration_name}",
            f"    ALLOWED_NETWORK_RULES = ({self._fq_rule})",
            f"    ENABLED = TRUE;",
            f"",
            f"GRANT USAGE ON INTEGRATION {self.integration_name} TO ROLE {role};",
        ])

    def create_external_access_integration(self) -> dict:
        logger.info("Creating external access integration: %s", self.integration_name)
        original_role = self.session.get_current_role()

        def _create() -> None:
            self.session.sql(f"""
                CREATE OR REPLACE EXTERNAL ACCESS INTEGRATION {self.integration_name}
                    ALLOWED_NETWORK_RULES = ({self._fq_rule})
                    ENABLED = TRUE
            """).collect()

        # First attempt: current role
        try:
            _create()
            logger.info("External access integration created: %s", self.integration_name)
            return {"created": True, "name": self.integration_name, "needs_admin": False,
                    "role_used": original_role}
        except Exception as first_exc:
            if "insufficient privileges" not in str(first_exc).lower() \
                    and "access control error" not in str(first_exc).lower():
                logger.error("Failed to create external access integration: %s", first_exc)
                return {"created": False, "needs_admin": False, "error": str(first_exc)}

        # Second attempt: escalate to ACCOUNTADMIN, then restore original role
        logger.info(
            "Current role (%s) lacks CREATE INTEGRATION — retrying as ACCOUNTADMIN",
            original_role,
        )
        try:
            self.session.sql("USE ROLE ACCOUNTADMIN").collect()
            _create()
            logger.info(
                "External access integration created as ACCOUNTADMIN: %s",
                self.integration_name,
            )
            return {"created": True, "name": self.integration_name, "needs_admin": False,
                    "role_used": "ACCOUNTADMIN"}
        except Exception as admin_exc:
            if "insufficient privileges" in str(admin_exc).lower() \
                    or "access control error" in str(admin_exc).lower():
                logger.warning(
                    "ACCOUNTADMIN also unavailable. Run the following SQL manually:\n\n%s",
                    self.admin_sql(original_role),
                )
                return {"created": False, "needs_admin": True,
                        "admin_sql": self.admin_sql(original_role)}
            logger.error("Failed even as ACCOUNTADMIN: %s", admin_exc)
            return {"created": False, "needs_admin": False, "error": str(admin_exc)}
        finally:
            # Always restore the original role
            try:
                self.session.sql(f"USE ROLE {original_role}").collect()
                logger.info("Restored role to %s", original_role)
            except Exception as restore_exc:
                logger.warning("Could not restore role to %s: %s", original_role, restore_exc)

    def grant_usage(self, role: Optional[str] = None) -> dict:
        role = role or self.session.get_current_role()
        logger.info("Granting USAGE on integration %s to role %s",
                    self.integration_name, role)
        try:
            self.session.sql(f"""
                GRANT USAGE ON INTEGRATION {self.integration_name} TO ROLE {role}
            """).collect()
            return {"granted": True, "role": role}
        except Exception as exc:
            logger.warning("Could not grant usage: %s", exc)
            return {"granted": False, "error": str(exc)}

    def get_status(self) -> dict:
        try:
            rows = self.session.sql(f"""
                SHOW EXTERNAL ACCESS INTEGRATIONS LIKE '{self.integration_name}'
            """).collect()
            if rows:
                row = rows[0]
                return {
                    "name":    row["name"],
                    "enabled": row["enabled"],
                    "comment": row.get("comment", ""),
                }
            return {"error": "integration not found"}
        except Exception as exc:
            return {"error": str(exc)}

    def run(self, role: Optional[str] = None) -> dict:
        """
        Idempotent end-to-end setup.

        If the current role lacks CREATE INTEGRATION privilege, the network
        rule is still created and the result will contain ``needs_admin=True``
        plus ``admin_sql`` with the statement an ACCOUNTADMIN must run once.

        Returns a result dict with keys:
            network_rule, integration, grant, status, integration_name
        """
        logger.info("=== Network egress setup ===")
        rule_result  = self.create_network_rule()
        eai_result   = self.create_external_access_integration()

        # Only attempt grant if integration was created successfully
        if eai_result.get("created"):
            grant_result = self.grant_usage(role)
        else:
            grant_result = {"granted": False, "reason": "integration not created"}

        status = self.get_status()

        logger.info("=== Network egress setup complete ===")
        return {
            "network_rule":     rule_result,
            "integration":      eai_result,
            "grant":            grant_result,
            "status":           status,
            "integration_name": self.integration_name,
        }
