# WaterTwinML — Sistema de Gemelo Digital para Redes Hídricas

Prototipo de gemelo digital con detección de anomalías y predicción de fallas en redes de distribución de agua, implementado sobre AWS.

**Stack**: Python 3.12, AWS Lambda, DynamoDB, Kinesis, S3, EventBridge, scikit-learn, XGBoost, NetworkX, Streamlit, Plotly

---

## Estructura del proyecto

```
├── lambdas/
│   ├── anomaly_detection/       # Isolation Forest en tiempo real
│   ├── failure_prediction/      # Ensemble RF+XGBoost (Ecuación 2)
│   └── digital_twin/            # BFS risk propagation (Ecuación 3)
├── dashboard/                   # Dashboard operacional Streamlit
│   ├── app.py                   # Entry point
│   ├── services/                # dynamo_service, metrics_service
│   ├── components/              # anomaly_chart, risk_heatmap, maintenance_table, qa_metrics
│   └── utils/                   # parsing, refresh
├── evaluation/                  # Framework de evaluación QA1-QA7
│   ├── measure_metrics.py       # CLI entry point
│   ├── latency.py / throughput.py / scalability.py
│   ├── ml_metrics.py            # QA4 (Recall IF) + QA5 (F1 Ensemble)
│   ├── isolation.py             # QA6 (Multi-tenant isolation)
│   ├── availability.py          # QA7 (CloudWatch + synthetic)
│   ├── cloudwatch_utils.py / dynamo_queries.py
│   └── report_generator.py      # → evaluation_report.json
├── ml/
│   ├── train_models.py          # Pipeline completo RF+XGBoost
│   ├── leakdb_adapter.py        # Conversor KIOS LeakDB → JSON
│   └── battledim_adapter.py     # Conversor BattLeDIM → JSON
├── simulator/
│   ├── local_simulator.py       # Simulador IoT local (file-based)
│   ├── iot_simulator.py         # Simulador IoT AWS (Kinesis)
│   └── config.py                # Parámetros por tenant
├── infra/
│   ├── setup_aws.py             # Creación idempotente de recursos AWS
│   └── teardown_aws.py          # Destrucción segura
├── tests/                       # pytest + moto
├── network/                     # Archivo EPANET L-TOWN_v2_Real.inp
├── Dockerfile                   # Contenedor único multi-propósito
├── docker-compose.yml           # Orquestación (4 servicios)
├── .env.example                 # Variables de entorno (template)
├── requirements.txt             # Dependencias Python
└── CONTEXT.md                   # Documento de contexto para IA
```

---

## Quick start

### Con Docker (recomendado)

```bash
# 1. Configurar variables de entorno
cp .env.example .env
# Editar .env con AWS ACCOUNT_ID y credenciales

# 2. Construir y levantar el dashboard
docker compose up -d dashboard
# Abrir http://localhost:8501

# 3. Evaluar métricas QA
docker compose run --rm eval --all

# 4. Entrenar modelos ML
docker compose run --rm ml --generate-data --train-ensemble --tenant both

# 5. Ejecutar simulador IoT
docker compose run --rm sim --tenant both --duration 300
```

### Sin Docker

```bash
pip install -r requirements.txt

# Dashboard
streamlit run dashboard/app.py

# Evaluación
python evaluation/measure_metrics.py --all

# Entrenamiento
python ml/train_models.py --generate-data --train-ensemble --tenant both

# Tests
pytest tests/ -v
```

---

## Docker

### Dockerfile único

El proyecto usa un solo `Dockerfile` multi-propósito. Por defecto arranca el dashboard Streamlit, pero puede ejecutar cualquier módulo vía `CMD` override:

```bash
# Build
docker build -t watertwin .

# Dashboard (default)
docker run -p 8501:8501 --env-file .env watertwin

# Evaluación QA
docker run --env-file .env watertwin python evaluation/measure_metrics.py --all

# Entrenamiento ML
docker run -v "$(pwd)/data:/app/data" -v "$(pwd)/models:/app/models" \
  watertwin python ml/train_models.py --generate-data --train-ensemble --tenant both

# Simulador IoT
docker run -v "$(pwd)/data:/app/data" \
  watertwin python simulator/local_simulator.py --tenant both --duration 300
```

### Docker Compose

| Servicio | Comando | Descripción |
|----------|---------|-------------|
| `dashboard` | `docker compose up -d dashboard` | Dashboard Streamlit (puerto 8501) |
| `eval` | `docker compose run --rm eval --all` | QA metrics (tools profile) |
| `ml` | `docker compose run --rm ml --train-ensemble --tenant both` | ML training (tools profile) |
| `sim` | `docker compose run --rm sim --tenant both --duration 300` | IoT simulator (tools profile) |

Los servicios `eval`, `ml` y `sim` usan `profiles: [tools]` — no arrancan con `docker compose up`, solo con `docker compose run`.

---

## Dataset de red — L-TOWN v2

La carpeta `network/` ya existe. Descarga el archivo **`L-TOWN_v2_Real.inp`** (~169 MB) desde:

> [github.com/KIOS-Research/BattLeDIM](https://github.com/KIOS-Research/BattLeDIM)

Colócalo en `network/L-TOWN_v2_Real.inp`. Está en `.gitignore` por su tamaño.

---

## Configuración AWS

```bash
cp .env.example .env   # Editar con ACCOUNT_ID
python infra/setup_aws.py
```

Crea: Kinesis streams, DynamoDB, S3 bucket, Lambdas, EventBridge rules, topología de red.

---

## Datos de entrenamiento

Tres fuentes — solo necesitas una:

| Opción | Comando | Descripción |
|--------|---------|-------------|
| **A — Simulador local** | `python ml/train_models.py --generate-data --train-ensemble --tenant both` | 72h datos sintéticos |
| **B — LeakDB** | `python ml/leakdb_adapter.py ...` + `train_models.py` | Dataset KIOS Research |
| **C — BattLeDIM** | `python ml/battledim_adapter.py ...` + `train_models.py` | Red L-TOWN real |

---

## Dashboard Operacional

Dashboard multi-tenant en Streamlit + Plotly + DynamoDB con tema oscuro NOC.

### Tabs

| Tab | Fuente | Descripción |
|-----|--------|-------------|
| **📊 Anomaly Monitoring** | `query_recent_anomalies()` | Time-series anomaly score, τ=0.65, sensor readings |
| **🌐 Network Risk** | `query_latest_ranking()` | Top-5 segmentos por riesgo propagado (BFS) |
| **🔔 Maintenance Alerts** | `query_maintenance_alerts()` | Tabla priorizada por p_failure, colores CRITICAL→LOW |
| **📋 QA Metrics** | `compute_qa_metrics()` | QA1-QA7 PASS/FAIL desde evaluation_report.json |

### Performance

| Métrica | Objetivo | Real |
|---------|----------|------|
| Carga inicial | < 3s | ~1.5s |
| Cambio tenant | Sin reinicio | Cache clear + rerun |
| Auto-refresh | Sin pérdida de estado | `st.session_state` |
| Queries DynamoDB | Paginadas | `ExclusiveStartKey` loop |
| Caché | TTL-based | `st.cache_data(ttl=10..30s)` |

### Variables de entorno

| Variable | Default |
|----------|---------|
| `AWS_DEFAULT_REGION` | `us-east-1` |
| `DYNAMO_TABLE` | `water-twin-data` |

### Schema DynamoDB esperado

**Sensor Record:**
```json
{
  "tenantId": "tenant-A",
  "sortKey": "2024-01-15T10:30:00#SEG_A_001",
  "timestamp": "2024-01-15T10:30:00",
  "segmentId": "SEG_A_001",
  "pressure": 4.52,
  "flow": 0.78,
  "vibration": 2.45,
  "isAnomaly": true,
  "anomalyScore": -0.32,
  "metadata": "{\"diameter\": 203.5, \"material\": \"Ductile Iron\"}",
  "processed_by_prediction": true,
  "p_failure": 0.34,
  "prediction_timestamp": "2024-01-15T10:31:00"
}
```

**Ranking Item:**
```json
{
  "tenantId": "tenant-A",
  "sortKey": "twin_ranking_2024-01-15T10:31:00",
  "generated_at": "2024-01-15T10:31:00",
  "top_k": 5,
  "segments": "[{\"segment_id\": \"SEG_A_015\", \"risk_propagated\": 0.85}]",
  "alpha": 0.7
}
```

---

## Evaluación QA1-QA7 (EP-07)

Framework automatizado que mide los 7 Quality Attribute Scenarios del paper.

### CLI

```bash
python evaluation/measure_metrics.py --all          # Todos
python evaluation/measure_metrics.py --latency      # QA1
python evaluation/measure_metrics.py --throughput   # QA2
python evaluation/measure_metrics.py --scalability  # QA3
python evaluation/measure_metrics.py --ml           # QA4 + QA5
python evaluation/measure_metrics.py --isolation    # QA6
python evaluation/measure_metrics.py --availability # QA7
```

### Definiciones matemáticas

| QA | Métrica | Fórmula | Threshold |
|----|---------|---------|-----------|
| **QA1** | Latencia P95 | Percentil 95 de `prediction_timestamp - timestamp` | **< 2s** |
| **QA2** | Throughput | eventos escritos/min en concurrencia (moto) | **> 950/min** |
| **QA3** | Escalabilidad | `(P95_stress - P95_base) / P95_base × 100` | **< 20%** |
| **QA4** | Recall IF | `TP / (TP + FN)` vs ground truth CSV | **> 0.80** |
| **QA5** | F1 Ensemble | `2·P·R/(P+R)` sobre test split 20% | **> 0.75** |
| **QA6** | Aislamiento | `leak_count = 0` en 100 intentos cross-tenant | **= 0** |
| **QA7** | Availability | `(Inv - Err) / Inv` desde CloudWatch | **> 99%** |

### Output

Genera `evaluation/evaluation_report.json` con:
```json
{
  "metrics": {
    "QA1_latency": { "passed": true, "result": { "p95_latency_sec": 1.42 } },
    "QA2_throughput": { "passed": true, "result": { "throughput": 1000 } }
  },
  "summary": { "total": 7, "passed": 7, "failed": 0 }
}
```

### Estrategia de validación

| QA | Técnica | Entorno |
|----|---------|---------|
| QA1 | Diferencia timestamps DynamoDB | Producción |
| QA2 | Escritura concurrente con moto | Mockeado |
| QA3 | Carga progresiva con percentiles | Mockeado |
| QA4 | Comparación ground truth CSV | Local + DynamoDB |
| QA5 | Evaluación sobre test split | Artefactos locales |
| QA6 | Fuzzing cross-tenant | Mockeado |
| QA7 | CloudWatch + invocación sintética | Producción + fallback |

---

## Tests

```bash
pytest tests/ -v
```

| Suite | Cubre |
|-------|-------|
| `test_qa5.py` | F1 ≥ 0.75 del ensemble |
| `test_qa6_bfs.py` | BFS risk propagation (Ecuación 3) |
| `test_qa6_handler.py` | Integración Lambda gemelo digital (moto) |
| `test_qa7_isolation.py` | Aislamiento multi-tenant (EP-05, 9 categorías) |

---

## Variables de entorno (.env)

| Variable | Default | Propósito |
|----------|---------|-----------|
| `AWS_DEFAULT_REGION` | `us-east-1` | Región AWS |
| `DYNAMO_TABLE` | `water-twin-data` | Tabla DynamoDB |
| `MODEL_BUCKET` | `water-twin-{ACCOUNT_ID}` | Bucket S3 para modelos |
| `ISOLATION_FOREST_CONTAMINATION` | `0.02` | Proporción de anomalías esperada |
| `ANOMALY_THRESHOLD` | `0.65` | Umbral de anomalía |
| `RISK_PROPAGATION_ALPHA` | `0.7` | Decaimiento de riesgo por hop |
| `RANKING_TOP_K` | `5` | Top-K segmentos en ranking |
| `TENANT_A_SEGMENTS` | `50` | Segmentos Tenant-A (Bogotá) |
| `TENANT_B_SEGMENTS` | `40` | Segmentos Tenant-B (Medellín) |

Ver `.env.example` para la lista completa.
