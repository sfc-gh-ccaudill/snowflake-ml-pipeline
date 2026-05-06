# Production-Grade ML on Snowflake

A demonstration of an end-to-end, production-ready machine learning pipeline built entirely within Snowflake — no external MLOps tools required.

**Use case**: Patient Risk Stratification — classifying hospital patients as `LOW`, `MEDIUM`, `HIGH`, or `CRITICAL` risk from clinical vitals and lab values using a distributed RandomForest classifier.

**Key message**: Every component — data, feature store, distributed training, model registry, REST endpoint, drift monitoring, automated retraining alerts, and pipeline orchestration — runs inside a single Snowflake account.

---

## Architecture

```
Raw Patient Data (EHR)
        │
        ▼
┌───────────────────────┐
│  STEP 1               │   Snowflake Feature Store
│  Feature Engineering  │   Entity: PATIENT (join key: PATIENT_ID)
│                       │   Feature View: PATIENT_FEATURES (1-min refresh)
│  Engineered features: │
│  SHOCK_INDEX          │
│  PULSE_PRESSURE       │
│  BMI_CATEGORY         │
│  VITAL_SIGNS_SEVERITY │
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  STEP 2b (optional)   │   Snowflake ML Jobs (SPCS)
│  Hyperparameter       │   Ray Tune + ASHA scheduler
│  Optimization         │   Configurable via tune.enabled in config
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  STEP 2               │   Snowflake ML Jobs (SPCS)
│  Distributed Training │   Multi-node distributed training on CPU_X64_S
│                       │   Model registered in Snowflake Model Registry
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  STEP 3               │   Snowflake Model Registry
│  Evaluation &         │   Metrics: accuracy, f1_macro, per-class F1
│  Promotion Gate       │   Promotion thresholds:
│                       │     accuracy ≥ 0.80, f1_macro ≥ 0.75
└───────────────────────┘
        │ (only if promoted)
        ▼
┌───────────────────────┐
│  STEP 4               │   Snowflake Model Serving (SPCS)
│  REST Endpoint        │   Auto-suspend after 1 hour idle
│  Deployment           │   Zero external infrastructure
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  STEP 5               │   Snowflake Model Monitor
│  Drift Detection &    │   Feature drift + prediction drift
│  Automated Retraining │   Baseline: TEST_FEATURES (training distribution)
│                       │   Segmented by ADMISSION_TYPE, INSURANCE_TYPE
│                       │
│  Drift Alert:         │   Snowflake Alert checks PSI daily
│  PSI > 0.25 triggers  │   → EXECUTE TASK re-runs full pipeline
│  automatic retrain    │
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  ORCHESTRATION        │   Snowflake Tasks (DAG)
│  Task DAG             │   Weekly CRON schedule (Sun 02:00 AM PT)
│                       │   FEATURE_ENG → HPO → TRAIN → EVALUATE
│                       │     → DEPLOY → MONITOR
└───────────────────────┘
```

---

## Directory Structure

```
snowflake-ml-prod/
├── source/
│   ├── config.yaml                         # All config: DB, schema, compute, features, thresholds
│   ├── configs.py                          # PipelineConfig dataclasses
│   ├── utils.py                            # get_session(), get_feature_config()
│   ├── train.py                            # Training entrypoint (ML Jobs runs this)
│   ├── train_hpo.py                        # HPO training entrypoint
│   ├── framework/
│   │   ├── feature_store.py                # FeatureStoreManager
│   │   ├── train.py                        # RemoteTrainer (ML Jobs wrapper)
│   │   ├── evaluator.py                    # Evaluator (metrics + promotion gate)
│   │   ├── deploy.py                       # ModelDeployer (REST endpoint)
│   │   ├── hpo.py                          # Hyperparameter optimization
│   │   └── monitor.py                      # ModelMonitor (drift detection + alerts)
│   └── pipeline/
│       ├── step1_feature_engineering.py    # Feature Store setup + feature tables
│       ├── step2_train.py                  # Distributed training
│       ├── step2b_hpo.py                   # Optional HPO step
│       ├── step3_evaluate.py               # Model evaluation + promotion gate
│       ├── step4_deploy.py                 # REST endpoint deployment
│       ├── step5_monitor.py                # Model monitor + drift alert creation
│       ├── dag.py                          # Task DAG creation + management
│       ├── step_handler.py                 # Stored procedure handler factory
│       └── pipeline_utils.py              # PipelineState + shared utilities
├── data/
│   ├── historical.py                       # Synthetic patient data generator
│   └── simulator.py                        # Streaming data simulator
├── setup/                                  # One-time infrastructure setup
│   ├── setup.sql                           # Snowflake object DDL
│   ├── database_setup.py
│   ├── stages_setup.py
│   ├── compute_pool_setup.py
│   ├── network_setup.py
│   └── tables_setup.py
├── notebooks/
│   ├── 01_setup_infrastructure.ipynb
│   ├── 02_data_generation.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_model_deployment.ipynb
│   ├── 06_model_monitoring.ipynb
│   ├── 07_streaming_inference.ipynb
│   ├── 08_cleanup.ipynb
│   └── deep_dive/
│       └── feature_and_experiment/         # Advanced Feature Store + experiment notebooks
└── docs/
    └── DEMO_GUIDE.md                       # Live demo script and talking points
```

---

## Prerequisites

- Snowflake account with SPCS (Snowpark Container Services) enabled
- Snowflake CLI (`snow`) installed and configured
- Python 3.10+
- `snowflake-ml-python` package

---

## Setup

### 1. Configure your connection

Edit `source/config.yaml` and set `connection_name` to match your Snowflake CLI connection:

```yaml
snowflake:
  connection_name: DEMO  # your connection name here
```

### 2. Provision infrastructure (one-time)

```bash
python -m setup.database_setup
python -m setup.stages_setup
python -m setup.compute_pool_setup
python -m setup.tables_setup
```

### 3. Generate synthetic patient data

```bash
python -m data.historical  # seeds ~50,000 patient records
```

---

## Running the Pipeline

### Run each step individually

```bash
# Step 1: Feature engineering + Feature Store registration
python -m source.pipeline.step1_feature_engineering

# Step 2b (optional): Hyperparameter optimization
python -m source.pipeline.step2b_hpo

# Step 2: Distributed remote training
python -m source.pipeline.step2_train

# Step 3: Model evaluation + promotion gate
python -m source.pipeline.step3_evaluate

# Step 4: Deploy REST endpoint
python -m source.pipeline.step4_deploy

# Step 5: Set up model monitor + drift alert
python -m source.pipeline.step5_monitor
```

### Create and run the Task DAG

```bash
# Create the DAG (does not execute immediately)
python -m source.pipeline.dag --build

# Create DAG + trigger a full pipeline run
python -m source.pipeline.dag --run

# Remove all tasks and stored procedures
python -m source.pipeline.dag --teardown
```

### Trigger via SQL

```sql
EXECUTE TASK ML_DEMO_PIPELINE_DB.HEALTHCARE.PIPELINE_FEATURE_ENG_TASK;
```

### Task DAG Flow

```
PIPELINE_FEATURE_ENG_TASK  (weekly CRON: Sun 02:00 AM PT)
  └── PIPELINE_HPO_TASK
        └── PIPELINE_TRAIN_TASK
              └── PIPELINE_EVALUATE_TASK
                    └── PIPELINE_DEPLOY_TASK
                          └── PIPELINE_MONITOR_TASK
```

---

## Drift Alerting & Automated Retraining

The monitoring step (Step 5) creates a Snowflake Alert (`PATIENT_RISK_DRIFT_ALERT`) that:

1. Runs daily at 6 AM PT (configurable via `drift_alert_schedule`)
2. Queries `MODEL_MONITOR_DRIFT_METRIC` for PSI on the prediction column
3. If PSI exceeds the threshold (default `0.25`), triggers `EXECUTE TASK` on the root task to re-run the full pipeline

Configure in `source/config.yaml`:

```yaml
monitor:
  drift_alert_enabled: true
  drift_alert_name: PATIENT_RISK_DRIFT_ALERT
  drift_metric: POPULATION_STABILITY_INDEX
  drift_threshold: 0.25
  drift_alert_schedule: "USING CRON 0 6 * * * America/Los_Angeles"
  retrain_root_task: PIPELINE_FEATURE_ENG_TASK
```

---

## Configuration

All pipeline behavior is controlled by `source/config.yaml`:

| Section | Key settings |
|---|---|
| `snowflake` | Connection name, database, schema, warehouse |
| `compute` | Compute pool name, instance family, min/max nodes |
| `model` | Model name, target platforms (WAREHOUSE / SPCS), model params |
| `tables` | Raw data, test features, metrics table names |
| `features` | Numeric features, categorical features, computed features, target column |
| `feature_store` | Entity, join keys, feature view name/version, refresh frequency |
| `train` | Number of distributed training nodes |
| `tune` | HPO toggle, num samples, search algorithm, scheduler, search space |
| `evaluation` | Accuracy and F1 macro promotion thresholds |
| `deploy` | Service name, min/max instances, auto-suspend timeout |
| `monitor` | Monitor name, drift alert config, threshold, schedule |
| `stages` | Stage name for job payloads and artifacts |

---

## Troubleshooting

| Issue | Fix |
|---|---|
| Compute pool in IDLE state | `ALTER COMPUTE POOL ML_DEMO_COMPUTE_POOL RESUME;` |
| ML Job stays PENDING | `DESCRIBE COMPUTE POOL ML_DEMO_COMPUTE_POOL;` to check node capacity |
| Service not reaching RUNNING | `SHOW SERVICES;` → check `status_message` for image pull errors |
| Feature Store "already exists" error | Safe to ignore — uses `overwrite=True` for idempotent runs |
| Task stuck SUSPENDED | `ALTER TASK <TASK_NAME> RESUME;` then re-execute |
| Monitor creation fails | Ensure inference logs view exists and has rows |
| `ModuleNotFoundError: dotenv` in SP | `deploy.py` guards this import; redeploy DAG to pick up fix |
| `EXECUTE ALERT privilege` error | `GRANT EXECUTE ALERT ON ACCOUNT TO ROLE SYSADMIN;` (auto-granted on DAG build) |
| Stale handler after redeploy | DAG uses timestamped stage paths to bust cache; run `--teardown` then `--build` |

---

## Additional Resources

- [Demo guide with live script and Q&A prep](docs/DEMO_GUIDE.md)
- [Snowflake ML documentation](https://docs.snowflake.com/en/developer-guide/snowflake-ml/overview)
