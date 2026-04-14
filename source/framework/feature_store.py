"""
Feature Store setup for Healthcare ML Pipeline.

This module sets up the Snowflake Feature Store with:
- Patient entity definition
- Raw feature view (passthrough from source)
- Engineered feature view (computed features)

The engineered features demonstrate the value of a Feature Store by providing
pre-computed, reusable features that encode domain knowledge.
"""

import logging
from typing import Any, List

from snowflake.snowpark import Session
import snowflake.snowpark.functions as F

logger = logging.getLogger(__name__)


class FeatureStoreManager:
    def __init__(
        self,
        session: Session,
        database: str,
        schema_name: str,
        warehouse: str,
    ):
        self.session = session
        self.database = database
        self.schema_name = schema_name
        self.warehouse = warehouse
        self.fs = None

    def initialize_feature_store(self) -> Any:
        from snowflake.ml.feature_store import FeatureStore, CreationMode

        logger.info(f"Initializing Feature Store in {self.database}.{self.schema_name}")

        self.fs = FeatureStore(
            session=self.session,
            database=self.database,
            name=self.schema_name,
            default_warehouse=self.warehouse,
            creation_mode=CreationMode.CREATE_IF_NOT_EXIST,
        )

        logger.info("Feature Store initialized")
        return self.fs

    def create_entity(
        self, entity_name: str, join_keys: List, description: str = None
    ) -> Any:
        from snowflake.ml.feature_store import Entity

        logger.info(f"Creating entity: {entity_name} with join keys: {join_keys}")

        entity = Entity(
            name=entity_name,
            join_keys=join_keys,
            desc=description,
        )

        try:
            registered_entity = self.fs.register_entity(entity)
            logger.info(f"Entity {entity_name} registered")
            return registered_entity
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(f"Entity {entity_name} already exists, retrieving")
                return self.fs.get_entity(entity_name)
            raise

    def create_feature_view(
        self,
        feature_view_name: str,
        entities: List[Any],
        features_df: Any,
        version: str = "v1",
        timestamp_column: str = "TIMESTAMP",
        refresh_freq: str = "1 minute",
        description: str = None,
    ) -> Any:
        """Create feature view with computed/engineered features."""
        from snowflake.ml.feature_store import FeatureView

        logger.info(
            f"Creating feature view: {feature_view_name} with computed features"
        )

        fv = FeatureView(
            name=feature_view_name,
            entities=entities,
            feature_df=features_df,
            timestamp_col=timestamp_column,
            refresh_freq=refresh_freq,
            desc=description,
        )

        try:
            registered_fv = self.fs.register_feature_view(
                feature_view=fv,
                version=version,
                block=True,
            )
            logger.info(
                f"Feature view {feature_view_name} registered with computed features"
            )
            return registered_fv
        except Exception as e:
            if "already exists" in str(e).lower():
                logger.info(
                    f"Feature view {feature_view_name} already exists, retrieving"
                )
                return self.fs.get_feature_view(feature_view_name, version)
            raise

    def enable_online_serving(self, feature_view: Any) -> dict:
        logger.info("Enabling online serving for feature view")

        try:
            from snowflake.ml.feature_store import OnlineConfig

            self.fs.update_feature_view(
                name=feature_view,
                online_config=OnlineConfig(
                    enable=True,
                    target_lag="1 minutes",
                ),
            )
            logger.info("Online serving enabled")
            return {"enabled": True}
        except Exception as e:
            if "already enabled" in str(e).lower():
                logger.info("Online serving already enabled")
                return {"enabled": True, "already_enabled": True}
            logger.warning(f"Could not enable online serving: {e}")
            return {"enabled": False, "error": str(e)}

    def list_entities(self) -> List[str]:
        if self.fs is None:
            self.initialize_feature_store()

        entities = self.fs.list_entities()
        return [e.name for e in entities.collect()]

    def list_feature_views(self) -> List[str]:
        if self.fs is None:
            self.initialize_feature_store()

        fvs = self.fs.list_feature_views()
        return [fv.name for fv in fvs.collect()]



