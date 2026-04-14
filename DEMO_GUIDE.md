# Demo Guide: End-to-End Production ML on Snowflake

**Use Case**: Patient Risk Stratification — classifying hospital patients as LOW / MEDIUM / HIGH / CRITICAL risk using clinical vitals and lab values.

**Audience**: 100+ technical and business stakeholders

**Duration**: 45-60 minutes (30 min live demo + 15 min Q&A)

---

## Pre-Demo Checklist

Run these steps the day before the demo to ensure a clean environment.

```bash
# 1. Verify Snowflake connection
snow connection test --connection DEMO

# 2. Verify infrastructure (compute pool, stages, tables)
python -m setup.database_setup
python -m setup.stages_setup
python -m setup.compute_pool_setup
python -m setup.tables_setup

# 3. Seed synthetic patient data (50,000 records)
python -m data.historical

# 4. Create the Task DAG (does not execute it yet)
python -m source.pipeline.dag

# 5. Optional: Pre-warm the compute pool to avoid cold-start during demo
snow sql -q "ALTER COMPUTE POOL ML_DEMO_COMPUTE_POOL RESUME IF SUSPENDED"
```

**Snowsight pre-flight**:
- Open Snowsight in a browser tab and keep it on the **Tasks** view
- Open a second tab on **AI & ML > Feature Store**
- Open a third tab on **AI & ML > Model Registry**
- Open a fourth tab on **AI & ML > Model Monitoring**

---

## Architecture Overview (Slide Talking Points)

```
Raw Patient Data (EHR)
        │
        ▼
┌───────────────────────┐
│  STEP 1               │   Snowflake Feature Store
│  Feature Engineering  │   - Entity: PATIENT (join key: PATIENT_ID)
│                       │   - Feature View: PATIENT_FEATURES (auto-refresh 1 min)
│  4 Computed Features: │   - 4 engineered signals from raw vitals
│  SHOCK_INDEX          │   "The Feature Store ensures every model trains on
│  PULSE_PRESSURE       │    the same feature logic — no training/serving skew"
│  BMI_CATEGORY         │
│  VITAL_SIGNS_SEVERITY │
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  STEP 2               │   Snowflake ML Jobs (SPCS)
│  Distributed Training │   - 3-node distributed training on CPU_X64_S
│  + Evaluation         │   - RANK / WORLD_SIZE env vars for coordination
│                       │   - Experiment Tracking: runs, params, metrics
│  Model: RandomForest  │   - Model Registry: versioned, with metrics & lineage
│  Promotion gate:      │   "Training never leaves your Snowflake account —
│  accuracy ≥ 0.80      │    data + compute + model all in one governance boundary"
│  f1_macro ≥ 0.75      │
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  STEP 3               │   Snowflake Model Serving (SPCS)
│  REST Endpoint        │   - Auto-scales 1-3 replicas
│  Deployment           │   - Zero external infrastructure
│                       │   "POST /predict — any app, any language,
│  PATIENT_RISK_SERVICE │    sub-100ms latency"
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  STEP 4               │   Snowflake Model Monitor
│  Drift Detection      │   - Baseline: TEST_FEATURES (training distribution)
│                       │   - Live source: STREAMING_PATIENT_DATA
│  Feature drift        │   "Continuous drift alerts without leaving Snowflake —
│  Prediction drift     │    no external MLOps tooling required"
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  ORCHESTRATION        │   Snowflake Tasks (DAG)
│  Task DAG             │   - Weekly schedule (CRON Sun 02:00 AM PT)
│                       │   - Each task calls a stored procedure
│  ROOT → TRAIN →       │   "Fully automated retraining pipeline —
│  DEPLOY → MONITOR     │    one SQL command to trigger end-to-end"
└───────────────────────┘
```

**Key message**: Every component — data, feature store, compute, model registry, endpoint, monitoring, orchestration — lives inside Snowflake. No external MLOps tools required.

---

## Step-by-Step Demo Script

### Step 1: Feature Engineering & Feature Store (~8 minutes)

**What to run**:
```bash
python -m source.pipeline.step1_feature_engineering
```

**What to show** (switch to Snowsight → AI & ML → Feature Store):
- PATIENT entity with join key `PATIENT_ID`
- PATIENT_FEATURES feature view showing all columns including the 4 computed ones
- Refresh frequency: 1 minute (show the "auto-refresh" badge)

**Talking points**:
- "Feature Store solves the training/serving skew problem — when we call the REST endpoint later, it will compute SHOCK_INDEX the exact same way"
- "SHOCK_INDEX = Heart Rate / Systolic BP — a classic hemodynamic instability signal used in clinical scoring systems"
- "The feature view refreshes every minute, so streaming data always has up-to-date engineered features"
- "A data scientist defines features once; every model in the organization reuses them"

**Expected output**:
```
Step 1: Feature Engineering & Feature Store
Feature Store initialized in ML_DEMO_PIPELINE_DB.HEALTHCARE
Entity PATIENT registered
Feature view PATIENT_FEATURES registered with computed features
TRAINING_FEATURES: ~40,000 rows
TEST_FEATURES:     ~10,000 rows
Step 1 complete
```

---

### Step 2: Distributed Remote Training + Evaluation (~12 minutes)

**What to run**:
```bash
NUM_TRAINING_INSTANCES=3 python -m source.pipeline.step2_train_evaluate
```

**What to show** (Snowsight → AI & ML → ML Jobs):
- Job appears immediately with status PENDING → RUNNING
- 3 worker nodes shown (distributed training)
- Live log streaming in the UI
- When complete: switch to Model Registry → PATIENT_RISK_MODEL → latest version

**What to show in Model Registry**:
- Version name with timestamp
- Metrics tab: accuracy, f1_macro, precision, recall, confusion matrix
- Experiment runs (click "View Experiments")

**Talking points**:
- "We're running this across 3 SPCS nodes simultaneously — each node gets a RANK (0, 1, 2) and a WORLD_SIZE of 3"
- "The training job runs entirely on Snowflake compute — the training data never left the platform"
- "Every run is logged to the Experiment Tracker automatically — you can compare 50 runs side by side"
- "The promotion gate checks accuracy ≥ 80% and F1 ≥ 75%. If the model doesn't meet the bar, the pipeline stops here"
- "The Model Registry stores the serialized model, its schema, sample input data, and all metrics in one versioned artifact"

**Distributed training talking point** (if asked how it works):
> "Snowflake ML Jobs runs each node as a container on the compute pool. The framework sets `RANK` (the node's index, 0-2) and `WORLD_SIZE` (total nodes, 3) as environment variables inside each container. For RandomForest, each node processes a partition of the data and the coordinator aggregates. You can swap in PyTorch DDP or XGBoost distributed with the same API — just change the entrypoint."

**Expected output**:
```
ML Job submitted: job-abc123
Job job-abc123 status: RUNNING (3 instances)
...
Test accuracy: 0.8731
Test F1: 0.8642
Promotion check PASSED: accuracy 0.8731 ≥ 0.80
Promotion check PASSED: f1_macro 0.8642 ≥ 0.75
Model APPROVED for promotion to REST endpoint
Step 2 complete
```

---

### Step 3: REST Endpoint Deployment (~8 minutes)

**What to run**:
```bash
python -m source.pipeline.step3_deploy
```

**What to show** (Snowsight → AI & ML → Model Serving or SHOW SERVICES):
```sql
SHOW SERVICES LIKE 'PATIENT_RISK_SERVICE';
```
- Service transitioning from PENDING → RUNNING
- Min 1 / Max 3 replicas (auto-scaling)

**Live inference demo** (run in a notebook or terminal):
```python
import requests, json

payload = {
    "data": [[65, "M", 28.3, 92, 145, 85, 37.2, 18, 96.0, 
              140.0, 1.1, 13.5, 8.2, "I10", 2, "Emergency", 
              "Medicare", 1, 5, 0.63, 60, "OVERWEIGHT", 3]]
}

response = requests.post(
    "https://<account>.snowflakecomputing.com/api/v2/endpoints/PATIENT_RISK_SERVICE/predict",
    headers={"Authorization": f"Bearer {token}"},
    json=payload
)
print(response.json())  # {"predictions": ["HIGH"]}
```

**Talking points**:
- "The service spins up in about 3 minutes — that's the container pull from the Snowflake image registry"
- "Auto-scaling: if we get a burst of 500 concurrent requests, Snowflake automatically scales to 3 replicas"
- "The endpoint is fully within the Snowflake security perimeter — it inherits your network policies and authentication"
- "Any application — web app, mobile app, downstream SQL pipeline — can call this endpoint"

---

### Step 4: Model Monitor Setup (~5 minutes)

**What to run**:
```bash
python -m source.pipeline.step4_monitor
```

**What to show** (Snowsight → AI & ML → Model Monitoring):
- PATIENT_RISK_MONITOR created
- Baseline vs. source configuration
- Feature drift columns listed
- Prediction drift tracking enabled

**Talking points**:
- "The monitor compares the distribution of incoming patient data against the distribution we trained on"
- "If SHOCK_INDEX starts drifting — say, a hospital changes how they record heart rates — we get an alert before the model degrades"
- "Prediction drift catches model decay even when we don't have labels yet — if the model suddenly predicts 60% CRITICAL when it used to predict 7%, that's a signal"
- "The baseline is the TEST_FEATURES table — the held-out data from training, so it perfectly represents the expected distribution"

---

### Orchestration: Task DAG (~5 minutes)

**What to show** (Snowsight → Data → Tasks):

```sql
SHOW TASKS IN SCHEMA ML_DEMO_PIPELINE_DB.HEALTHCARE;
```

- 4 tasks visible: ROOT_TASK → TRAIN_TASK → DEPLOY_TASK → MONITOR_TASK
- DAG view showing the chained dependencies
- ROOT_TASK scheduled: CRON 0 2 * * 0 (weekly, Sunday 2 AM)

**Trigger a full run live**:
```sql
EXECUTE TASK ML_DEMO_PIPELINE_DB.HEALTHCARE.PIPELINE_ROOT_TASK;
```

**Talking points**:
- "This is a fully automated retraining pipeline — every Sunday the system retrains on the latest week of patient data and redeploys if the model passes the quality gate"
- "If Step 2 training fails the promotion criteria, Steps 3 and 4 are skipped automatically — the existing endpoint stays live"
- "One SQL command triggers the full pipeline on demand — useful for ad-hoc retraining after a data schema change"

---

## Key Demo Moments (Audience Engagement Cues)

| Moment | What to Say | Why It Lands |
|---|---|---|
| Feature Store UI | "This is the feature catalog — notice the lineage from RAW_PATIENT_DATA" | Connects to data governance story |
| 3 nodes in ML Jobs | "We're running distributed training right now in this account" | Tangible scale, live proof |
| Metrics in Registry | "Every experiment is tracked, every version is reproducible" | Addresses audit/reproducibility concerns |
| REST endpoint live | "That's a production ML API — deployed in 3 minutes" | Speed to value |
| Monitor baseline | "This is how you catch silent model decay" | Risk mitigation story |
| Task DAG view | "Zero external tools — Airflow, MLflow, Kubernetes — none of it" | Total cost of ownership |

---

## Distributed Training Deep-Dive (For Technical Audiences)

When asked how distributed training works in more detail:

1. **Job submission**: `submit_directory()` with `num_instances=3` provisions 3 SPCS containers simultaneously
2. **Coordination**: Snowflake sets `RANK` (0, 1, 2) and `WORLD_SIZE` (3) in each container's environment
3. **Data parallelism**: The training entrypoint (`source/train.py`) detects `RANK` and `WORLD_SIZE`, partitions the training dataset across nodes, trains local model on each partition
4. **Aggregation**: The rank-0 (coordinator) node aggregates results and writes the final model to the registry
5. **Scalability**: The same `RemoteTrainer.submit(num_instances=N)` call works for 1 node (single-node remote) to 10+ nodes (large-scale distributed) — just change the number

**For PyTorch / XGBoost**: Replace `source/train.py` with a PyTorch DDP or XGBoost distributed script — the `RemoteTrainer` framework class doesn't change.

---

## Troubleshooting Quick Reference

| Issue | Fix |
|---|---|
| Compute pool in IDLE state | `ALTER COMPUTE POOL ML_DEMO_COMPUTE_POOL RESUME;` |
| ML Job stays PENDING | Check pool node capacity: `DESCRIBE COMPUTE POOL ML_DEMO_COMPUTE_POOL;` |
| Service not reaching RUNNING | Check SPCS image pull logs via `SHOW SERVICES;` → status_message |
| Feature Store "already exists" | Safe to ignore — `FeatureStoreManager` uses `CREATE_IF_NOT_EXIST` |
| Model version not found | Run `SHOW MODELS IN SCHEMA ML_DEMO_PIPELINE_DB.HEALTHCARE;` to confirm |
| Monitor creation fails | Ensure STREAMING_PATIENT_DATA table exists and has data |
| Task stuck SUSPENDED | `ALTER TASK <TASK_NAME> RESUME;` then re-execute |

---

## Anticipated Q&A

**Q: What ML frameworks are supported?**
> sklearn, XGBoost, LightGBM, PyTorch, TensorFlow — anything that can be serialized and run in a Python container. The `log_model()` API in the registry handles all of them.

**Q: How does this compare to SageMaker / Vertex AI / Databricks?**
> Those require separate services for storage, compute, model registry, endpoints, and monitoring — plus the data has to move out of your warehouse to train. With Snowflake, the data never moves. One governance boundary, one bill, one security model.

**Q: Can we bring our own models (BYO)?**
> Yes — `registry.log_model(your_sklearn_model, ...)` works with any serializable Python object. You can also log PyTorch state dicts and ONNX models.

**Q: What does distributed training actually buy us here?**
> On 50,000 records a single node finishes in ~2 minutes. On 50 million records (a realistic production dataset) distributing across 3-10 nodes cuts training time from hours to minutes. The same code runs at both scales.

**Q: How is the REST endpoint secured?**
> The service inherits Snowflake authentication (OAuth, key pair, or Snowflake token). Network policies and private connectivity (PrivateLink) apply at the account level — no separate VPC or firewall rules needed.

**Q: What happens when a new model fails the quality gate?**
> The Task DAG stops at Step 2 (TRAIN_TASK). The existing PATIENT_RISK_SERVICE keeps running with the last approved version. An alert can be configured on task failure via Snowflake Alerts.

**Q: How does the Feature Store prevent training/serving skew?**
> The same Python function that computes SHOCK_INDEX during training is called at inference time via `retrieve_feature_values()`. There's one canonical definition, version-controlled in the feature view.

---

## Commands Cheat Sheet

```bash
# Full pipeline (each step individually)
python -m source.pipeline.step1_feature_engineering
NUM_TRAINING_INSTANCES=3 python -m source.pipeline.step2_train_evaluate
python -m source.pipeline.step3_deploy
python -m source.pipeline.step4_monitor

# Create Task DAG
python -m source.pipeline.dag

# Trigger DAG immediately
python -m source.pipeline.dag --run

# Teardown DAG
python -m source.pipeline.dag --teardown

# Check running services
snow sql -q "SHOW SERVICES IN SCHEMA ML_DEMO_PIPELINE_DB.HEALTHCARE"

# Check model versions
snow sql -q "SHOW MODELS IN SCHEMA ML_DEMO_PIPELINE_DB.HEALTHCARE"

# Check task DAG status
snow sql -q "SHOW TASKS IN SCHEMA ML_DEMO_PIPELINE_DB.HEALTHCARE"
```

---

## File Structure Reference

```
snowflake-ml-prod/
├── source/
│   ├── config.yaml                         # All config: DB, schema, compute pool, features
│   ├── configs.py                          # PipelineConfig dataclasses
│   ├── utils.py                            # get_session(), get_feature_config()
│   ├── train.py                            # Training entrypoint (ML Jobs runs this)
│   ├── framework/
│   │   ├── __init__.py                     # Public API exports
│   │   ├── feature_store.py                # FeatureStoreManager
│   │   ├── train.py                        # RemoteTrainer (ML Jobs wrapper)
│   │   ├── evaluator.py                    # Evaluator (metrics + promotion)
│   │   ├── deploy.py                       # ModelDeployer (REST endpoint)
│   │   └── monitor.py                      # ModelMonitor (drift detection)
│   └── pipeline/
│       ├── __init__.py
│       ├── step1_feature_engineering.py    # Feature Store setup + feature tables
│       ├── step2_train_evaluate.py         # Distributed training + evaluation
│       ├── step3_deploy.py                 # REST endpoint deployment
│       ├── step4_monitor.py                # Model monitor creation
│       └── dag.py                          # Task DAG creation + management
├── data/
│   └── historical.py                       # Synthetic patient data generator
├── setup/                                  # One-time infrastructure setup
└── notebooks/                              # Interactive walkthrough notebooks
```
