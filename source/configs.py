"""
Configuration dataclasses for Healthcare ML Pipeline.
"""

import dataclasses
from dataclasses import dataclass, field
import os
from typing import Any, Dict, List

import yaml


class BaseConfig:
    """Mixin that provides generic from_dict and to_dict classmethods for all config dataclasses."""

    @classmethod
    def from_dict(cls, d: dict):
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}


@dataclass
class SnowflakeConfig(BaseConfig):
    connection_name: str
    database: str
    schema_name: str
    warehouse: str

    @classmethod
    def from_dict(cls, d: dict) -> "SnowflakeConfig":
        return cls(
            connection_name=d.get("connection_name", ""),
            database=d.get("database"),
            schema_name=d.get("schema"),
            warehouse=d.get("warehouse"),
        )

    def to_dict(self) -> dict:
        return {
            "connection_name": self.connection_name,
            "database": self.database,
            "schema": self.schema_name,
            "warehouse": self.warehouse,
        }


@dataclass
class ComputeConfig(BaseConfig):
    compute_pool: str
    instance_family: str
    min_nodes: int
    max_nodes: int


@dataclass
class ModelParams(BaseConfig):
    n_estimators: int = 100
    class_weight: str = "balanced"
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class ModelConfig(BaseConfig):
    model_name: str
    target_platforms: List[str]
    params: ModelParams

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(
            model_name=d.get("model_name"),
            target_platforms=d.get("target_platforms"),
            params=ModelParams.from_dict(d.get("params", {})),
        )

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "target_platforms": self.target_platforms,
            "params": self.params.to_dict(),
        }


@dataclass
class TableConfig(BaseConfig):
    raw_data: str
    test_features: str = "TEST_FEATURES"
    metrics_table: str = "MODEL_METRICS"


@dataclass
class FeatureStoreConfig(BaseConfig):
    entity_name: str
    entity_join_keys: List[str]
    feature_view_name: str
    feature_view_version: str
    feature_view_refresh_freq: str
    training_dataset_name: str


@dataclass
class DeployConfig(BaseConfig):
    service_name: str
    min_instances: int = 1
    max_instances: int = 3
    auto_suspend_secs: int = 3600


@dataclass
class DriftAlertConfig(BaseConfig):
    alert_name: str = "PREDICTION_DRIFT_ALERT"
    column: str = "RISK_LEVEL"
    drift_metric: str = "POPULATION_STABILITY_INDEX"
    drift_threshold: float = 0.25
    schedule: str = "USING CRON 0 6 * * * America/Los_Angeles"

    @classmethod
    def from_dict(cls, d: dict) -> "DriftAlertConfig":
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass
class MonitorConfig(BaseConfig):
    monitor_name: str = "PATIENT_RISK_MONITOR"
    inference_logs_view: str = "INFERENCE_LOGS_VIEW"
    baseline_table: str = "MONITOR_BASELINE"
    drift_alert_enabled: bool = True
    retrain_root_task: str = "PIPELINE_FEATURE_ENG_TASK"
    drift_alerts: List[DriftAlertConfig] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "MonitorConfig":
        alerts_raw = d.get("drift_alerts", [])
        alerts = [DriftAlertConfig.from_dict(a) for a in alerts_raw]
        return cls(
            monitor_name=d.get("monitor_name", "PATIENT_RISK_MONITOR"),
            inference_logs_view=d.get("inference_logs_view", "INFERENCE_LOGS_VIEW"),
            baseline_table=d.get("baseline_table", "MONITOR_BASELINE"),
            drift_alert_enabled=d.get("drift_alert_enabled", True),
            retrain_root_task=d.get("retrain_root_task", "PIPELINE_FEATURE_ENG_TASK"),
            drift_alerts=alerts,
        )

    def to_dict(self) -> dict:
        return {
            "monitor_name": self.monitor_name,
            "inference_logs_view": self.inference_logs_view,
            "baseline_table": self.baseline_table,
            "drift_alert_enabled": self.drift_alert_enabled,
            "retrain_root_task": self.retrain_root_task,
            "drift_alerts": [a.to_dict() for a in self.drift_alerts],
        }


@dataclass
class StagesConfig(BaseConfig):
    job_payloads: str = "JOB_PAYLOADS"


@dataclass
class EvaluationConfig(BaseConfig):
    accuracy_threshold: float = 0.80
    f1_macro_threshold: float = 0.75


@dataclass
class FeatureConfig(BaseConfig):
    raw_numeric_features: List[str]
    categorical_features: List[str]
    computed_features: List[str]
    target_column: str
    class_labels: List[str]


@dataclass
class TrainConfig(BaseConfig):
    num_nodes: int = 2


@dataclass
class TuneConfig(BaseConfig):
    enabled: bool = False
    num_samples: int = 20
    search_alg: str = "random"
    scheduler: str = "asha"
    num_instances: int = 2
    metric: str = "f1_macro"
    mode: str = "max"
    search_space: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig(BaseConfig):
    snowflake: SnowflakeConfig
    compute: ComputeConfig
    model: ModelConfig
    tables: TableConfig
    features: FeatureConfig
    feature_store: FeatureStoreConfig = None
    deploy: DeployConfig = None
    stages: StagesConfig = None
    evaluation: EvaluationConfig = None
    train: TrainConfig = None
    tune: TuneConfig = None
    monitor: MonitorConfig = None

    @property
    def full_schema(self) -> str:
        return f"{self.snowflake.database}.{self.snowflake.schema_name}"

    @property
    def full_raw_table(self) -> str:
        return f"{self.full_schema}.{self.tables.raw_data}"

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        return cls(
            snowflake=SnowflakeConfig.from_dict(d.get("snowflake", {})),
            compute=ComputeConfig.from_dict(d.get("compute", {})),
            model=ModelConfig.from_dict(d.get("model", {})),
            tables=TableConfig.from_dict(d.get("tables", {})),
            features=FeatureConfig.from_dict(d.get("features", {})),
            feature_store=FeatureStoreConfig.from_dict(d.get("feature_store", {})),
            deploy=DeployConfig.from_dict(d.get("deploy", {})),
            stages=StagesConfig.from_dict(d.get("stages", {})),
            evaluation=EvaluationConfig.from_dict(d.get("evaluation", {})),
            train=TrainConfig.from_dict(d.get("train", {})),
            tune=TuneConfig.from_dict(d.get("tune", {})),
            monitor=MonitorConfig.from_dict(d.get("monitor", {})),
        )

    def to_dict(self) -> dict:
        return {
            "snowflake": self.snowflake.to_dict(),
            "compute": self.compute.to_dict(),
            "model": self.model.to_dict(),
            "tables": self.tables.to_dict(),
            "features": self.features.to_dict(),
            "feature_store": self.feature_store.to_dict(),
            "deploy": self.deploy.to_dict(),
            "stages": self.stages.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "train": self.train.to_dict(),
            "tune": self.tune.to_dict(),
            "monitor": self.monitor.to_dict(),
        }


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_config_from_dict(config_dict: dict) -> PipelineConfig:
    return PipelineConfig.from_dict(config_dict)


def get_config(config_path: str = "config.yaml") -> PipelineConfig:
    config_dict = load_config(config_path)
    snowflake = config_dict.setdefault("snowflake", {})
    if not snowflake.get("connection_name"):
        snowflake["connection_name"] = os.getenv("SNOWFLAKE_CONNECTION_NAME", "default")
    return get_config_from_dict(config_dict)


def config_to_dict(config: PipelineConfig) -> dict:
    """Serialize a PipelineConfig to the canonical dict format accepted by get_config_from_dict."""
    return config.to_dict()
