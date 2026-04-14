"""
Common utilities for Snowflake ML framework.
"""

import os
from typing import Any, Dict, Optional

from snowflake.snowpark import Session
from snowflake.snowpark.context import get_active_session


def get_env(key: str, default: Optional[str] = None) -> str:
    """
    Get environment variable or raise if missing.

    Args:
        key: Environment variable name
        default: Default value if not set

    Returns:
        Environment variable value

    Raises:
        ValueError: If variable not set and no default provided
    """
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"Missing required environment variable: {key}")
    return value


def get_session(connection_name: Optional[str] = None) -> Session:
    """
    Create Snowpark session.

    Args:
        connection_name: Snowflake connection name (defaults to env or config)

    Returns:
        Snowpark Session
    """
    try:
        # Try to get an existing session (Snowflake Notebooks)
        session = get_active_session()
    except Exception:
        # Use a specific connection (Local Notebook)
        print("Creating Session...")
        if connection_name is None:
            connection_name = os.getenv("SNOWFLAKE_CONNECTION_NAME")
        session = Session.builder.config("connection_name", connection_name).create()

    return session


def get_session_from_config() -> Session:
    """Create Snowpark session using config settings."""
    from configs import get_config

    config = get_config()
    session = get_session(config.snowflake.connection_name)
    session.use_database(config.snowflake.database)
    session.use_schema(config.snowflake.schema_name)
    session.use_warehouse(config.snowflake.warehouse)

    return session


def get_feature_config(config: Dict) -> Dict[str, Any]:
    """
    Get the feature configuration for the pipeline.

    Returns:
        Dict containing feature lists and metadata.
    """

    fc = config.feature_config

    return {
        "raw_numeric_features": fc.raw_numeric_features,
        "categorical_features": fc.categorical_features,
        "computed_features": fc.computed_features,
        "all_numeric_features": fc.raw_numeric_features
        + ["SHOCK_INDEX", "PULSE_PRESSURE", "VITAL_SIGNS_SEVERITY"],
        "all_categorical_features": fc.categorical_features + ["BMI_CATEGORY"],
        "target_column": fc.target_column,
        "class_labels": fc.class_labels,
        "id_columns": ["PATIENT_ID", "ENCOUNTER_ID"],
        "timestamp_column": "TIMESTAMP",
    }
