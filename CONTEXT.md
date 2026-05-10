# WaterTwinML — Context Document for AI Code Generators

> **Project**: WaterTwinML — Digital Twin for Water Distribution Networks  
> **Course**: TDSE (Universidad)  
> **Stack**: Python 3.12, AWS Serverless, scikit-learn, XGBoost, NetworkX  
> **Purpose**: Real-time anomaly detection, failure prediction, and risk propagation for water pipe networks with multi-tenant SaaS architecture.

---

## 1. Project Identity

**WaterTwinML** is an academic prototype implementing a Digital Twin for water distribution networks deployed on AWS. It simulates IoT sensor data (pressure, flow, vibration) for two municipal water networks (Bogota/Tenant-A and Medellin/Tenant-B), detects anomalies in real-time via Isolation Forest, predicts pipe failure probabilities using an RF+XGBoost ensemble, and propagates risk through a pipe-network topology graph to produce a ranked maintenance priority list.

---

## 2. Architecture Overview

```
IoT Simulator (per-tenant)
    │
    ▼  Kinesis Stream (tenant-{a,b}-stream)
    │
    ▼  Lambda: Anomaly Detection (Isolation Forest)
    │  - Cross-tenant write guard (T-07.03)
    │  - 8-feature anomaly scoring
    │  - Writes isAnomaly + anomalyScore to DynamoDB
    │
    ▼  DynamoDB: water-twin-data (processed_by_prediction=False)
    │
    ▼  Lambda: Failure Prediction (RF+XGBoost Ensemble)
    │  [Trigger: EventBridge rate(1 min)]
    │  - Queries unprocessed anomaly records
    │  - Computes P(Failure|X) = beta*P_RF + (1-beta)*P_XGB
    │  - Updates DynamoDB with p_failure
    │
    ▼  DynamoDB: water-twin-data (p_failure populated)
    │
    ▼  Lambda: Digital Twin (BFS Risk Propagation)
    │  [Trigger: EventBridge rate(1 min)]
    │  - Loads topology + latest p_failure per segment
    │  - BFS propagation: Risk(ej) = max(Risk(ej), alpha*Risk(ei))
    │  - Persists top-K ranking to DynamoDB (TTL=7 days)
    │
    ▼  DynamoDB: water-twin-data (twin_ranking_* items)
```

### Pattern: Choreography (Event-Driven)
Anomaly Lambda → DynamoDB → Prediction Lambda polls on schedule → DynamoDB → Digital Twin polls on schedule. No direct Lambda-to-Lambda invocation. Unprocessed records are automatically retried on next cycle (fault tolerance).

---

## 3. Complete Project Structure

```
TDSE_Proyect/
│
├── .env                              # Environment variables (44 params)
├── .gitignore
├── README.md
├── requirements.txt                  # 20 dependencies
├── measure_metrics.py                # QA evaluation (US-07, Table 4)
│
├── infra/
│   ├── setup_aws.py                  # Idempotent AWS resource creation (630 lines)
│   │   └── class WaterTwinInfrastructure
│   └── teardown_aws.py               # Safe resource destruction (262 lines)
│       └── class WaterTwinTeardown
│
├── lambdas/
│   ├── anomaly_detection/lambda_function.py    # Isolation Forest (280 lines)
│   ├── failure_prediction/lambda_function.py   # RF+XGBoost ensemble (327 lines)
│   └── digital_twin/lambda_function.py         # BFS risk propagation (301 lines)
│
├── ml/
│   ├── train_models.py               # Full ML pipeline (813 lines)
│   │   └── class WaterTwinML
│   ├── upload_models.py              # S3 upload/download utility (222 lines)
│   │   └── class ModelUploader
│   ├── test_model.py                 # Isolation Forest smoke test (182 lines)
│   ├── simple_test.py                # Simple IF test (107 lines)
│   ├── leakdb_adapter.py             # KIOS LeakDB → JSON converter (589 lines)
│   │   └── class LeakDBAdapter
│   └── battledim_adapter.py          # BattLeDIM → JSON converter (382 lines)
│       └── class BattleDIMAdapter
│
├── simulator/
│   ├── config.py                     # Tenant params + anomaly config (50 lines)
│   ├── iot_simulator.py              # Kinesis-based IoT simulator (393 lines)
│   │   └── class IoTSimulator
│   └── local_simulator.py            # File-based local simulator (418 lines)
│       └── class LocalIoTSimulator
│
├── src/simulators/
│   ├── tenant_a_simulator.py         # Standalone Tenant-A (214 lines)
│   │   └── class TenantASimulator
│   └── tenant_b_simulator.py         # Standalone Tenant-B (226 lines)
│       └── class TenantBSimulator
│
├── tests/
│   ├── test_qa5.py                   # F1 >= 0.75 ensemble quality gate
│   ├── test_qa6_bfs.py              # BFS risk propagation unit tests
│   ├── test_qa6_handler.py          # Digital Twin Lambda integration (moto)
│   └── test_qa7_isolation.py        # Multi-tenant isolation (EP-05, 9 categories)
│
├── network/
│   ├── .gitkeep
│   └── L-TOWN_v2_Real.inp           # EPANET file (169 MB, gitignored)
│
├── models/                           # Gitignored — ML model artifacts
├── data/outputs/                     # Gitignored — training data
└── layers/                           # Gitignored — Lambda layer ZIPs
```

---

## 4. Coding Conventions

### General
- **Naming**: `snake_case` for files, functions, variables. `PascalCase` for classes.
- **Imports**: stdlib → third-party → local, separated by blank line
- **Type hints**: Preferred for function signatures
- **Formatting**: `black` (line length ~100), `flake8` for linting
- **Quotes**: Double quotes `"..."` for strings

### Lambda Functions
- File: `lambdas/{name}/lambda_function.py`
- Handler: `def lambda_handler(event: dict, context: object) -> dict`
- Return: `{"statusCode": 200, "body": json.dumps(result)}`
- Models: Load from S3 on first invocation, cache in module-level `_model_cache` dict
- Config: Loaded from `os.environ` (populated by `.env` on local, Lambda env vars on AWS)

### Lambda Model Loading Pattern
```python
_model_cache = {}

def _load_model(bucket: str, prefix: str, tenant: str) -> tuple:
    cache_key = f"{tenant}/ensemble"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    # Load from S3...
    _model_cache[cache_key] = (rf_model, xgb_model, scaler, beta)
    return _model_cache[cache_key]
```

### Tests
- Framework: `pytest`
- AWS Mocking: `moto` (DynamoDB, Kinesis, S3, Lambda, EventBridge)
- Mock patches: `unittest.mock.patch` / `unittest.mock.MagicMock`
- Fixtures: `@pytest.fixture` for `aws_credentials`, `dynamodb_client`, `dynamodb_table`
- Parametrization: `@pytest.mark.parametrize("tenant", ["tenant-A", "tenant-B"])`
- Auto-skip: `pytest.skip("Missing model artifacts")` when files absent
- Assertions: `assert` statements (not `self.assertEqual`)

---

## 5. Data Models & Schemas

### 5.1 DynamoDB Table: `water-twin-data`

| Attribute | Type | Description |
|-----------|------|-------------|
| `tenantId` | String (PK) | `"tenant-A"` or `"tenant-B"` |
| `sortKey` | String (SK) | Varies by item type |
| `ttl_epoch` | Number (optional) | TTL in epoch seconds (7-day expiry for rankings) |

#### Sensor Record Item
```json
{
  "tenantId": "tenant-A",
  "sortKey": "2024-01-15T10:30:00#SEG_A_001",
  "timestamp": "2024-01-15T10:30:00",
  "segmentId": "SEG_A_001",
  "city": "Bogota",
  "pressure": 4.52,
  "flow": 0.78,
  "vibration": 2.45,
  "isAnomaly": false,
  "anomalyScore": -0.12,
  "metadata": "{\"diameter\": 203.5, \"material\": \"Ductile Iron\", \"length\": 312.0, \"age\": 22.3, \"elevation\": 2650.0}",
  "processed_by_prediction": false
}
```

After prediction Lambda processes:
```json
{
  "...": "...",
  "processed_by_prediction": true,
  "p_failure": 0.34,
  "prediction_timestamp": "2024-01-15T10:31:00"
}
```

#### Topology Config Item
```json
{
  "tenantId": "tenant-A",
  "sortKey": "topology#config",
  "nodes": ["SEG_A_000", "SEG_A_001", ..., "SEG_A_049"],
  "edges": [["SEG_A_000", "SEG_A_001"], ["SEG_A_001", "SEG_A_002"], ...],
  "n_segments": 50,
  "topology_type": "synthetic"
}
```

#### Twin Ranking Item
```json
{
  "tenantId": "tenant-A",
  "sortKey": "twin_ranking_2024-01-15T10:31:00",
  "generated_at": "2024-01-15T10:31:00",
  "top_k": 5,
  "segments": "[{\"segment_id\": \"SEG_A_015\", \"risk_propagated\": 0.85}, ...]",
  "alpha": 0.7,
  "ttl_epoch": 1705336260
}
```

### 5.2 Input Sensor JSON (from simulator / adapters)
```json
{
  "tenant_id": "tenant-A",
  "city": "Bogota",
  "segment_id": "SEG_A_001",
  "timestamp": "2024-01-15T10:30:00",
  "pressure": 4.52,
  "flow": 0.78,
  "vibration": 2.45,
  "is_anomaly": false,
  "metadata": {
    "diameter": 203.5,
    "material": "Ductile Iron",
    "length": 312.0,
    "age": 22.3,
    "elevation": 2650.0
  }
}
```

### 5.3 DynamoDB Query Patterns

**Prediction Lambda: Fetch unprocessed anomalies**
```python
response = table.query(
    KeyConditionExpression=Key('tenantId').eq(tenant) & Key('sortKey').gt(' '),
    FilterExpression=Attr('isAnomaly').eq(True) & Attr('processed_by_prediction').eq(False)
)
```

**Digital Twin: Load latest p_failure per segment**
```python
response = table.query(
    KeyConditionExpression=Key('tenantId').eq(tenant) & Key('sortKey').between(cutoff, 'twin_'),
    ProjectionExpression='segmentId, p_failure'
)
```
> ⚠️ The `between(..., 'twin_')` bound is critical to exclude ranking items. Ranking sortKeys start with `twin_ranking_`, so `sortKey < 'twin_'` captures only sensor records.

---

## 6. 8-Feature ML Vector

```
X = [pressure, flow, vibration, delta_pressure, diameter, material_enc, length, age]
```

| Feature | Source | Processing |
|---------|--------|------------|
| `pressure` | Sensor reading | Direct float value |
| `flow` | Sensor reading | Direct float value |
| `vibration` | Sensor reading | Direct float value |
| `delta_pressure` | Computed | `df.groupby('segment_id')['pressure'].diff()`, NaN → 0 |
| `diameter` | Metadata | Default 300 if missing |
| `material_enc` | Metadata | Encoded via MATERIAL_ENCODING map |
| `length` | Metadata | Default 250 if missing |
| `age` | Metadata | Default 20 if missing |

### Material Encoding Map
| Material | Code |
|----------|------|
| PVC, HDPE, Steel | 0 |
| AC (Asbestos Cement) | 1 |
| DI, Ductile Iron | 2 |
| CI, Cast Iron | 3 |

### Material Default Resolution
```python
MATERIAL_ENCODING = {
    'PVC': 0, 'HDPE': 0, 'Steel': 0,
    'AC': 1,
    'DI': 2, 'Ductile Iron': 2,
    'CI': 3, 'Cast Iron': 3,
}
UNKNOWN_MATERIAL_CODE = 1  # Conservative (AC)
```

---

## 7. Mathematical Formulas

### 7.1 Ensemble Failure Prediction (Ecuacion 2)
```
P(Failure|X) = beta * P_RF(X) + (1 - beta) * P_XGB(X)
```

| Variable | Meaning | Source |
|----------|---------|--------|
| `P_RF(X)` | Random Forest probability of failure for features X | `rf_model.predict_proba(X)[:, 1]` |
| `P_XGB(X)` | XGBoost probability of failure for features X | `xgb_model.predict_proba(X)[:, 1]` |
| `beta` | Blending coefficient | Optimized via 5-fold CV on ROC-AUC |

**Beta optimization**: Grid search over `[0.00, 0.05, ..., 1.00]` (21 values), maximize mean ROC-AUC across 5 stratified folds. SMOTE applied inside each fold to prevent data leakage.

### 7.2 BFS Risk Propagation (Ecuacion 3)
```
Risk(ej) = max(Risk(ej), alpha * Risk(ei))
```

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `alpha` | 0.7 (`RISK_PROPAGATION_ALPHA`) | Risk decay factor per hop |
| `Risk(ei)` | Current risk at source node | Initialized from `p_failure` |
| `Risk(ej)` | Risk at neighbor node | Updated in-place |

**Algorithm behavior**:
- Uses max-priority queue (Python `heapq` with negated values)
- Risk monotone non-decreasing (only ever raises values)
- Exponential decay: `risk = p_failure * (alpha ^ k)` after k hops
- Cuts off naturally: `0.7^10 ≈ 0.028`, negligible beyond ~10 hops
- Time complexity: `O((V+E) log V)`
- Graph is undirected; propagation flows both ways along edges

### 7.3 Failure Label Generation
```
Label[i] = 1  iff  Σ(anomalies in [i+1, i+W]) / W > 0.10
```
Where `W = 24 hours` of data. Implemented via prefix-sum + `np.searchsorted` for `O(n log n)` per segment.

### 7.4 SMOTE Ratio
```python
smote = SMOTE(sampling_strategy=0.2, random_state=42)
```
Produces a 5:1 majority-to-minority ratio in the training set.

---

## 8. Multi-Tenant Architecture

### 8.1 Tenant Comparison

| Aspect | Tenant-A (Bogota) | Tenant-B (Medellin) |
|--------|-------------------|---------------------|
| Kinesis Stream | `tenant-a-stream` | `tenant-b-stream` |
| Segments | 50 (`SEG_A_000`..`SEG_A_049`) | 40 (`SEG_B_000`..`SEG_B_039`) |
| Altitude | 2600m | 1500m |
| Base Pressure | 4.5 bar | 5.2 bar |
| Base Flow | 0.8 m³/s | 0.9 m³/s |
| Base Vibration | 2.5 mm/s | 2.2 mm/s |
| S3 Model Prefix | `models/tenant-a/` | `models/tenant-b/` |
| DynamoDB PK | `tenant-A` | `tenant-B` |

### 8.2 Isolation Mechanisms (EP-05)

| # | Mechanism | Implementation |
|---|-----------|----------------|
| 1 | **DynamoDB PK Isolation** | Each tenant's data under its own `tenantId` partition key |
| 2 | **Kinesis Stream Isolation** | Dedicated stream per tenant |
| 3 | **Cross-tenant Write Guard (T-07.03)** | In anomaly Lambda: validate `payload.tenant_id` matches stream's expected tenant from `eventSourceARN` |
| 4 | **Topology Isolation** | `topology#config` item stored per tenant partition |
| 5 | **Ranking Isolation** | `twin_ranking_*` items stored per tenant partition |
| 6 | **Model Isolation** | Separate ML models in S3 per tenant |

### 8.3 Kinesis Stream → Tenant Mapping (T-07.03 Cross-Tenant Guard)
```python
STREAM_TENANT_MAP = {
    "tenant-a-stream": "tenant-A",
    "tenant-b-stream": "tenant-B",
}

def _get_expected_tenant(event: dict) -> str:
    """Extract expected tenant from Kinesis event source ARN."""
    stream_arn = event['Records'][0]['eventSourceARN']
    stream_name = stream_arn.split('/')[1]
    return STREAM_TENANT_MAP[stream_name]

# Guard: silently drop if tenant_id doesn't match
if payload_tenant != expected_tenant:
    print(f"WARNING: Cross-tenant write blocked: {payload_tenant} on {stream_name}")
    return {"statusCode": 200, "body": json.dumps({"dropped": 1})}
```

---

## 9. Key Configuration Parameters (`.env`)

| Parameter | Default | Used By | Purpose |
|-----------|---------|---------|---------|
| `AWS_DEFAULT_REGION` | `us-east-1` | All Lambdas, infra | AWS region |
| `LAB_ROLE_ARN` | — | Infra setup | IAM role for Lambda execution |
| `DYNAMO_TABLE` | `water-twin-data` | All Lambdas, infra | DynamoDB table name |
| `MODEL_BUCKET` | `water-twin-{ACCOUNT_ID}` | Prediction, Anomaly Lambdas | S3 bucket for ML models |
| `TENANT_A_STREAM` | `tenant-a-stream` | Anomaly Lambda | Kinesis stream name |
| `TENANT_B_STREAM` | `tenant-b-stream` | Anomaly Lambda | Kinesis stream name |
| `ANOMALY_LAMBDA` | `water-twin-anomaly` | Infra setup | Lambda function name |
| `PREDICTION_LAMBDA` | `water-twin-prediction` | Infra setup | Lambda function name |
| `DIGITAL_TWIN_LAMBDA` | `water-twin-digital-twin` | Infra setup | Lambda function name |
| `ISOLATION_FOREST_CONTAMINATION` | `0.02` | Anomaly Lambda | Expected anomaly proportion |
| `ISOLATION_FOREST_N_ESTIMATORS` | `100` | Anomaly Lambda | Number of isolation trees |
| `ANOMALY_THRESHOLD` | `0.65` | Anomaly Lambda | Score threshold for anomaly |
| `RISK_PROPAGATION_ALPHA` | `0.7` | Digital Twin Lambda | Risk decay per hop |
| `RANKING_TOP_K` | `5` | Digital Twin Lambda | Top-K segments to rank |
| `RISK_WINDOW_HOURS` | `2` | Digital Twin Lambda | p_failure lookback window |
| `PREDICTION_BATCH_SIZE` | `100` | Prediction Lambda | Max records per batch |
| `ANOMALY_RATE` | `0.02` | Simulator | Anomaly injection probability |
| `ANOMALY_DURATION_SECONDS` | `10` | Simulator | Anomaly duration |
| `SIMULATION_FREQUENCY_HZ` | `1.0` | Simulator | Sensor readings per second |
| `TENANT_A_SEGMENTS` | `50` | Simulator, infra | Pipe segments count |
| `TENANT_B_SEGMENTS` | `40` | Simulator, infra | Pipe segments count |
| `TENANT_A_ALTITUDE` | `2600` | Simulator | Altitude in meters |
| `TENANT_B_ALTITUDE` | `1500` | Simulator | Altitude in meters |

---

## 10. Lambda Functions Reference

### 10.1 `lambdas/anomaly_detection/lambda_function.py`

| Aspect | Detail |
|--------|--------|
| **Trigger** | Kinesis stream (per-tenant) |
| **Models** | `isolation_forest.pkl`, `scaler.pkl` from S3 |
| **Config** | `ISOLATION_FOREST_CONTAMINATION`, `ISOLATION_FOREST_N_ESTIMATORS`, `ANOMALY_THRESHOLD` |
| **Output** | DynamoDB sensor record with `isAnomaly`, `anomalyScore` |
| **Key logic** | Cross-tenant write guard; 8-feature scaling + IF prediction; IF returns `-1` (anomaly) or `1` (normal) |
| **Caching** | Module-level `_model_cache` dict keyed by `"{tenant}/if"` |

### 10.2 `lambdas/failure_prediction/lambda_function.py`

| Aspect | Detail |
|--------|--------|
| **Trigger** | EventBridge scheduled rule (rate 1 minute) |
| **Models** | `rf_model.pkl`, `xgb_model.pkl`, `scaler.pkl`, `beta.json` from S3 |
| **Query** | `isAnomaly=True AND processed_by_prediction=False` |
| **Batch** | Up to `PREDICTION_BATCH_SIZE` records per tenant |
| **Formula** | `P = beta * P_RF + (1-beta) * P_XGB` |
| **Fault tolerance** | Un-updated records keep `processed_by_prediction=False` → retried next cycle |
| **Caching** | Module-level `_model_cache` keyed by `"{tenant}/ensemble"` |

### 10.3 `lambdas/digital_twin/lambda_function.py`

| Aspect | Detail |
|--------|--------|
| **Trigger** | EventBridge scheduled rule (rate 1 minute) |
| **Input data** | Topology from DynamoDB (`topology#config` → NetworkX Graph) + latest `p_failure` from last `RISK_WINDOW_HOURS` |
| **Algorithm** | BFS with max-priority queue (Ecuacion 3) |
| **Output** | `twin_ranking_{timestamp}` item with TTL=7 days |
| **Stateless** | Graph rebuilt from DynamoDB every invocation |
| **Key detail** | Ranking sortKey `twin_ranking_*` must NOT be loaded as p_failure source → `sortKey.between(cutoff, 'twin_')` bound |

---

## 11. ML Training Pipeline

### 11.1 Training Flow (detailed)
```
Raw Sensor Data (JSON with metadata)
    │
    ▼  _load_training_data()
    │   - JSON: flatten metadata fields into top-level columns
    │   - CSV: use defaults for pipe metadata
    │
    ▼  _build_feature_matrix()
    │   - Sort by [segment_id, timestamp]
    │   - Compute delta_pressure per segment (diff, NaN→0)
    │   - Encode material via MATERIAL_ENCODING
    │   - Defaults: diameter=300, length=250, age=20
    │
    ▼  _create_failure_labels()
    │   - For each segment: sliding 24h window
    │   - Label = 1 if anomaly rate in next 24h > 10%
    │   - O(n log n) via prefix-sum + searchsorted
    │
    ▼  train_test_split(test_size=0.2, stratify=y, random_state=42)
    │
    ▼  SMOTE(sampling_strategy=0.2, random_state=42)
    │   - Applied to training set only (5:1 ratio)
    │   - Prevents data leakage: applied AFTER split
    │
    ▼  StandardScaler().fit_transform(X_train), .transform(X_test)
    │
    ├───► RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    │
    ├───► XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, scale_pos_weight=20)
    │
    ▼  _find_optimal_beta()
    │   - 5-fold Stratified CV on training set
    │   - Grid: [0.00, 0.05, ..., 1.00] (21 values)
    │   - Metric: mean ROC-AUC of P_ensemble
    │   - SMOTE applied inside each fold
    │
    ▼  Save artifacts:
    │   ml/artifacts/{tenant}/
    │   ├── rf_model.pkl
    │   ├── xgb_model.pkl
    │   ├── scaler.pkl
    │   ├── beta.json
    │   └── beta_cv_results.json
```

### 11.2 CLI Usage
```bash
# Full pipeline (recommended): generate data + train
python ml/train_models.py --generate-data --train-ensemble --tenant both

# Train only (with existing data)
python ml/train_models.py --train-ensemble --tenant tenant-A

# Evaluate existing model
python ml/train_models.py --eval-ensemble --tenant both

# Upload models to S3
python ml/train_models.py --upload-models --tenant both
```

### 11.3 Model Summary

| Model | File | Purpose | Params |
|-------|------|---------|--------|
| Isolation Forest | `isolation_forest.pkl` | Real-time anomaly detection in Lambda | `contamination=0.15`, `n_estimators=100` |
| Random Forest | `rf_model.pkl` | Ensemble failure prediction component | `n_estimators=100`, `class_weight='balanced'` |
| XGBoost | `xgb_model.pkl` | Ensemble failure prediction component | `n_estimators=100`, `max_depth=5`, `scale_pos_weight=20` |
| Scaler | `scaler.pkl` | Feature standardization | `StandardScaler` fit on training data |
| Beta | `beta.json` | Ensemble blend weight | `{"beta": 0.45}` (example) |

---

## 12. Testing Reference

### 12.1 Test Suites

| Suite | File | Type | What Validates |
|-------|------|------|----------------|
| QA5 | `test_qa5.py` | Quality gate | Ensemble F1 >= 0.75 on held-out test set |
| QA6 BFS | `test_qa6_bfs.py` | Unit | BFS math: decay, max-operator, disconnected graphs, node counts |
| QA6 Handler | `test_qa6_handler.py` | Integration (moto) | Full Lambda flow, TTL, graceful degradation |
| QA7 Isolation | `test_qa7_isolation.py` | Integration + Unit | 9 isolation categories across DynamoDB, Lambda, concurrent writes |

### 12.2 How to Run
```bash
# All tests
pytest tests/ -v

# Specific suite
pytest tests/test_qa5.py -v

# With print statements visible
pytest tests/ -v -s
```

### 12.3 Test Patterns

**DynamoDB mocking (moto):**
```python
@pytest.fixture
def dynamodb_table():
    with mock_dynamodb():
        client = boto3.client('dynamodb', region_name='us-east-1')
        client.create_table(...)
        yield table
```

**Lambda handler mocking (unittest.mock):**
```python
@patch('digital_twin.lambda_function.boto3.client')
@patch('digital_twin.lambda_function._load_topology')
def test_handler_writes_ranking(mock_topology, mock_boto3):
    ...
```

**Multi-tenant isolation test pattern:**
```python
# Write tenant-A data
for item in tenant_a_data:
    table.put_item(Item=item)

# Query from tenant-A context — should only see tenant-A data
response = table.query(KeyConditionExpression=Key('tenantId').eq('tenant-A'))
assert all(item['tenantId'] == 'tenant-A' for item in response['Items'])
```

---

## 13. Sensor Simulation Logic

### 13.1 Base Reading Generation
```python
pressure_factor = 1.0 + (elevation - altitude) / 10000
flow_factor = diameter / 300
vibration_factor = 1.0 + age / 50  # Cast Iron x1.3, PVC x0.8

pressure = random.gauss(base_pressure * pressure_factor, pressure_std)
flow = random.gauss(base_flow * flow_factor, flow_std)
vibration = random.gauss(base_vibration * vibration_factor, vibration_std)
```

### 13.2 Anomaly Types

| Type | Effect |
|------|--------|
| `pressure_drop` | pressure -= 3*σ, flow *= 0.7, vibration *= 1.5 |
| `flow_anomaly` | flow *= 0.3, pressure *= 1.2 |
| `vibration_spike` | vibration *= 2.5, pressure *= 0.95 |

---

## 14. Infrastructure Setup

### 14.1 AWS Resources Created by `setup_aws.py`

| Resource | Names | Config |
|----------|-------|--------|
| Kinesis Stream | `tenant-a-stream`, `tenant-b-stream` | 1 shard each |
| DynamoDB Table | `water-twin-data` | PAY_PER_REQUEST, TTL on `ttl_epoch` |
| S3 Bucket | `water-twin-{ACCOUNT_ID}` | Folders: `models/`, `layers/` |
| Lambda (anomaly) | `water-twin-anomaly` | Python 3.12, 512MB, 60s timeout, Kinesis trigger |
| Lambda (prediction) | `water-twin-prediction` | Python 3.12, 512MB, 60s timeout, EventBridge trigger |
| Lambda (digital twin) | `water-twin-digital-twin` | Python 3.12, 512MB, 60s timeout, EventBridge trigger |
| EventBridge Rule | `water-twin-digital-twin-rule` | `rate(1 minute)` → digital twin Lambda |
| EventBridge Rule | `water-twin-prediction-rule` | `rate(1 minute)` → prediction Lambda |

### 14.2 Setup Commands
```bash
# 1. Configure .env with ACCOUNT_ID
# 2. Create all resources
python infra/setup_aws.py

# 3. Deploy Lambda code manually or via CI
# 4. Teardown when done
python infra/teardown_aws.py
```

### 14.3 Topology Generation (in setup_aws.py)
Creates a synthetic pipe network with:
- 1 main trunk line (linear chain)
- 4 lateral branches of varying lengths
- Cross-links between branches
- Tenant A: 50 nodes, Tenant B: 40 nodes
- Stored as `topology#config` DynamoDB item

---

## 15. `requirements.txt` — Dependency Reference

```text
boto3>=1.34.0, botocore>=1.34.0       # AWS SDK
pandas>=2.1.0, numpy>=1.26.0          # Data processing
scikit-learn>=1.4.2                   # Isolation Forest, RF, scaler
xgboost>=2.0.3                        # XGBoost classifier
imbalanced-learn>=0.12.0              # SMOTE oversampling
networkx>=3.3                          # Graph algorithms (BFS risk propagation)
joblib>=1.3.2                          # Model serialization (pickle)
streamlit>=1.29.0                      # Dashboard (not yet implemented)
python-dotenv>=1.0.0                   # .env file loading
tqdm>=4.66.0, colorama>=0.4.6         # CLI progress/color
wntr>=1.2.0                            # Water network analysis (battledim adapter)
openpyxl>=3.1.0, pyyaml>=6.0          # Excel/YAML parsing (adapters)
pytest>=7.4.0                          # Test framework
black>=23.0.0, flake8>=6.0.0          # Formatting/linting
moto[dynamodb]>=4.2.0,<5              # AWS DynamoDB mocking
```

---

## 16. Common Pitfalls & Rules

### DO:
- ✅ Use `tenantId` as PK and `sortKey` as SK for ALL DynamoDB queries
- ✅ Apply SMOTE inside CV folds (not before split) — prevents data leakage
- ✅ Scope ALL DynamoDB queries by `tenantId` — isolation is structural
- ✅ Validate `tenant_id` in anomaly Lambda payload vs. stream — cross-tenant guard
- ✅ Use `sortKey.between(cutoff, 'twin_')` to exclude ranking items when querying sensor records
- ✅ Cache models in module-level dict for Lambda warm starts
- ✅ Default `delta_pressure` NaN to 0 after diff
- ✅ Default missing metadata: `diameter=300`, `length=250`, `age=20`

### DON'T:
- ❌ Never hardcode `ACCOUNT_ID` — use `os.environ` or `.env`
- ❌ Never load `twin_ranking_*` items as p_failure sources (wrong `sortKey` range)
- ❌ Never skip the cross-tenant write guard in anomaly Lambda
- ❌ Never write `.pkl`, `.joblib`, `.csv`, `.zip`, `network/*.inp` to git — they're in `.gitignore`
- ❌ Never assume models exist in S3 — handle `ClientError` gracefully
- ❌ Never use `&&` for chaining in PowerShell — use `; if ($?) { }`
- ❌ Never train models on non-segment-sorted data — sort by `[segment_id, timestamp]`

---

## 17. Key Files Quick Reference

| File | What to find there |
|------|--------------------|
| `infra/setup_aws.py` | Topology generation logic, DynamoDB seed data |
| `lambdas/anomaly_detection/lambda_function.py` | Cross-tenant guard, Isolation Forest inference |
| `lambdas/failure_prediction/lambda_function.py` | Ensemble formula, DynamoDB batch processing |
| `lambdas/digital_twin/lambda_function.py` | BFS propagation, max-heap, ranking persistence |
| `ml/train_models.py` | Full ML pipeline, feature engineering, beta optimization |
| `simulator/local_simulator.py` | Sensor generation math, anomaly injection |
| `simulator/config.py` | All per-tenant constants |
| `tests/test_qa7_isolation.py` | All isolation test patterns (9 categories) |

---

## 18. Important String Constants

| Constant | Value | Used In |
|----------|-------|---------|
| `STREAM_TENANT_MAP` | `{"tenant-a-stream": "tenant-A", "tenant-b-stream": "tenant-B"}` | Anomaly Lambda |
| Tenant PKs | `"tenant-A"`, `"tenant-B"` | All Lambdas, tests, infra |
| Topology SK | `"topology#config"` | Digital Twin Lambda, infra |
| Ranking SK prefix | `"twin_ranking_"` | Digital Twin Lambda |
| S3 model paths | `"models/tenant-a/"`, `"models/tenant-b/"` | Prediction + Anomaly Lambdas |
| IF model files | `"isolation_forest.pkl"`, `"scaler.pkl"` | Anomaly Lambda |
| Ensemble model files | `"rf_model.pkl"`, `"xgb_model.pkl"`, `"scaler.pkl"`, `"beta.json"` | Prediction Lambda |
| Segment ID format | `"SEG_A_{000..049}"`, `"SEG_B_{000..039}"` | Simulator, infra, tests |
| DynamoDB table | `"water-twin-data"` | All |

---

> **Document generated for WaterTwinML** — Use this as the single source of truth when generating, modifying, or debugging code in this repository. All architectural decisions, naming conventions, data schemas, isolation rules, and ML formulas are documented here for consistent AI-assisted development.
