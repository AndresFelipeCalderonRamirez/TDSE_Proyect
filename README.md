# WaterTwinML — Sistema de Gemelo Digital para Redes Hídricas

Prototipo de gemelo digital con detección de anomalías y predicción de fallas en redes de distribución de agua, implementado sobre AWS.

---

## Estructura del proyecto

```text
├── lambdas/
│   ├── anomaly_detection/     # Lambda de detección de anomalías (Isolation Forest)
│   ├── failure_prediction/    # Lambda de predicción de fallas (RF + XGBoost, Ecuación 2)
│   └── digital_twin/          # Lambda del gemelo digital (BFS propagación de riesgo, Ecuación 3)
├── ml/
│   └── train_models.py        # Entrenamiento local del ensemble RF+XGBoost
├── simulator/
│   └── local_simulator.py     # Simulador IoT local (datos de sensores)
├── infra/
│   └── setup_aws.py           # Creación de recursos AWS (idempotente)
├── tests/                     # Suite de tests (pytest + moto)
├── network/                   # Topología de red (ver instrucciones abajo)
├── measure_metrics.py         # Medición de métricas QA (evaluation_report.json)
└── requirements.txt
```

---

## Requisitos previos

```bash
pip install -r requirements.txt
```

---

## Dataset de red — L-TOWN v2

La carpeta `network/` ya existe en el repositorio. Solo necesitas descargar el archivo de red y colocarlo ahí.

### Pasos

1. Descarga el archivo **`L-TOWN_v2_Real.inp`** desde el repositorio oficial del dataset BattLeDIM (Battle of the Leakage Detection and Isolation Methods):

   > **BattLeDIM** — L-TOWN Water Distribution Network  
   > Disponible en: [github.com/KIOS-Research/BattLeDIM](https://github.com/KIOS-Research/BattLeDIM)

2. Coloca el archivo descargado en la carpeta `network/`:

   ```text
   network/
   └── L-TOWN_v2_Real.inp   ← aquí
   ```

El archivo pesa ~169 MB y supera el límite de GitHub (100 MB), por eso no está incluido directamente en el repositorio.

---

## Configuración AWS

1. Copia `.env` y rellena los valores de tu entorno de laboratorio:

   ```bash
   # Edita .env con tu ACCOUNT_ID y ARNs del lab
   ```

2. Ejecuta el setup de infraestructura:

   ```bash
   python infra/setup_aws.py
   ```

   Esto crea (idempotente): streams Kinesis, tabla DynamoDB, bucket S3, funciones Lambda, reglas EventBridge y siembra la topología de red.

---

## Datos de entrenamiento

Hay tres fuentes disponibles — **solo necesitas una**:

### Opción A — Simulador local (recomendada, sin descargas)

Genera 72 h de datos sintéticos directamente:

```bash
python ml/train_models.py --generate-data --train-ensemble --tenant both
```

### Opción B — LeakDB (datos reales de presiones y fugas)

1. Descarga el dataset desde [github.com/KIOS-Research/LeakDB](https://github.com/KIOS-Research/LeakDB)
2. Adapta y entrena:

```bash
python ml/leakdb_adapter.py \
    --leakdb-path /ruta/a/LeakDB \
    --tenant tenant-A \
    --output data/outputs/tenant-A_sensor_data.json

python ml/train_models.py --train-ensemble --tenant tenant-A
```

### Opción C — BattLeDIM (red L-TOWN real)

Requiere además el archivo `network/L-TOWN_v2_Real.inp` (ver sección anterior).

1. Ejecuta el generador de BattLeDIM para producir `Measurements.xlsx`
2. Adapta y entrena:

```bash
python ml/battledim_adapter.py \
    --measurements /ruta/BattLeDIM/Results/Measurements.xlsx \
    --config /ruta/BattLeDIM/dataset_configuration_72h.yalm \
    --inp network/L-TOWN_v2_Real.inp \
    --tenant tenant-A \
    --output data/outputs/tenant-A_sensor_data.json

python ml/train_models.py --train-ensemble --tenant tenant-A
```

---

## Evaluar ensemble guardado

```bash
python ml/train_models.py --eval-ensemble --tenant both
```

---

## Tests

```bash
pytest tests/ -v
```

| Suite | Cubre |
| --- | --- |
| `test_qa5.py` | F1 ≥ 0.75 del ensemble |
| `test_qa6_bfs.py` | BFS propagación de riesgo (Ecuación 3) |
| `test_qa6_handler.py` | Integración Lambda gemelo digital (moto) |
| `test_qa7_isolation.py` | Aislamiento multi-tenant (EP-05) |

---

## Métricas QA

```bash
python measure_metrics.py --metric isolation
```

Genera `evaluation_report.json` con los resultados de cada métrica del paper (Tabla 4).
