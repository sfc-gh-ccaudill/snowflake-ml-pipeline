# Production-Grade ML on Snowflake

A demonstration of an end-to-end, production-ready machine learning pipeline built entirely within Snowflake — no external MLOps tools required.

**Use case**: Patient Risk Stratification — classifying hospital patients as `LOW`, `MEDIUM`, `HIGH`, or `CRITICAL` risk from clinical vitals and lab values using a distributed RandomForest classifier.

**Key message**: Every component — data, feature store, distributed training, model registry, REST endpoint, drift monitoring, and pipeline orchestration — runs inside a single Snowflake account.

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
│  STEP 2               │   Snowflake ML Jobs (SPCS)
│  Distributed Training │   3-node distributed training on CPU_X64_S
│  + Evaluation         │   Experiment tracking + Model Registry
│                       │
│  Promotion gate:      │
│  accuracy ≥ 0.80      │
│  f1_macro ≥ 0.75      │
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  STEP 3               │   Snowflake Model Serving (SPCS)
│  REST Endpoint        │   Auto-scales 1–3 replicas
│  Deployment           │   Zero external infrastructure
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  STEP 4               │   Snowflake Model Monitor
│  Drift Detection      │   Feature drift + prediction drift
│                       │   Baseline: TEST_FEATURES (training distribution)
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  ORCHESTRATION        │   Snowflake Tasks (DAG)
│  Task DAG             │   Weekly CRON schedule (Sun 02:00 AM PT)
│                       │   ROOT → TRAIN → DEPLOY → MONITOR
└───────────────────────┘
```

---

## Directory Structure

```
snowflake-ml-prod/
├── source/
│   ├── config.yaml                         # All config: DB, schema, compute pool, features
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
│   │   └── monitor.py                      # ModelMonitor (drift detection)
│   └── pipeline/
│       ├── step1_feature_engineering.py    # Feature Store setup + feature tables
│       ├── step2_train_evaluate.py         # Distributed training + evaluation
│       ├── step2b_hpo.py                   # Optional HPO step
│       ├── step3_deploy.py                 # REST endpoint deployment
│       ├── step4_monitor.py                # Model monitor creation
│       ├── dag.py                          # Task DAG creation + management
│       └── execution_log.py               # Pipeline run logging
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
│   ├── pipeline_master.ipynb               # Full pipeline walkthrough
│   ├── 03_preprocessing.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_model_deployment.ipynb
│   ├── 07_model_monitoring.ipynb
│   └── setup/
│       ├── 01_setup_infrastructure.ipynb
│       └── 02_data_generation.ipynb
└── docs/
    ├── DEMO_GUIDE.md                       # Live demo script and talking points
    └── presentation.html                   # Slide deck
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

# Step 2: Distributed remote training + evaluation (3 nodes)
NUM_TRAINING_INSTANCES=3 python -m source.pipeline.step2_train_evaluate

# Step 2b (optional): Hyperparameter optimization
python -m source.pipeline.step2b_hpo

# Step 3: Deploy REST endpoint
python -m source.pipeline.step3_deploy

# Step 4: Set up model monitor
python -m source.pipeline.step4_monitor
```

### Create and run the Task DAG

```bash
# Create the DAG (does not execute immediately)
python -m source.pipeline.dag

# Trigger a full pipeline run on demand
python -m source.pipeline.dag --run

# Remove the DAG
python -m source.pipeline.dag --teardown
```

### Trigger via SQL

```sql
EXECUTE TASK ML_DEMO_PIPELINE_DB.HEALTHCARE.PIPELINE_ROOT_TASK;
```

---

## Configuration

All pipeline behavior is controlled by `source/config.yaml`:

| Section | Key settings |
|---|---|
| `snowflake` | Connection name, database, schema, warehouse |
| `compute` | Compute pool name, instance family, min/max nodes |
| `model` | Model name, target platforms (WAREHOUSE / SPCS) |
| `pipeline` | HPO toggle, HPO samples, search algorithm |
| `feature_config` | Numeric features, categorical features, computed features, target column |

---

## Troubleshooting

| Issue | Fix |
|---|---|
| Compute pool in IDLE state | `ALTER COMPUTE POOL ML_DEMO_COMPUTE_POOL RESUME;` |
| ML Job stays PENDING | `DESCRIBE COMPUTE POOL ML_DEMO_COMPUTE_POOL;` to check node capacity |
| Service not reaching RUNNING | `SHOW SERVICES;` → check `status_message` for image pull errors |
| Feature Store "already exists" error | Safe to ignore — uses `CREATE_IF_NOT_EXIST` |
| Task stuck SUSPENDED | `ALTER TASK <TASK_NAME> RESUME;` then re-execute |
| Monitor creation fails | Ensure `STREAMING_PATIENT_DATA` table exists and has rows |

---

## Additional Resources

- [Demo guide with live script and Q&A prep](docs/DEMO_GUIDE.md)
- [Snowflake ML documentation](https://docs.snowflake.com/en/developer-guide/snowflake-ml/overview)
