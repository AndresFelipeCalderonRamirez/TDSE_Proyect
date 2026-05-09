"""
Lambda Failure Prediction — US-05 / T-05.01 + T-05.02
Choreography pattern (paper Section 8, Newman [19])

Trigger : EventBridge scheduled rule — rate(1 minute)
           (AWS minimum; spec requests 30 s — achievable via EventBridge Scheduler)

Flow:
  1. For each tenant, query DynamoDB:
       isAnomaly = True  AND  processed_by_prediction = False
  2. For each record assemble 8-feature vector X (paper Section 5.3)
  3. Load RF + XGBoost + β from S3 (module-level cache → warm reuse)
  4. P(Failure|X) = β·P_RF + (1-β)·P_XGB  [Ecuación 2]
  5. UpdateItem: add p_failure, prediction_timestamp; set processed_by_prediction=True

Fault tolerance:
  If this Lambda crashes mid-batch, un-updated records keep processed_by_prediction=False
  and are retried in the next invocation — zero data loss.
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import boto3
import joblib
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── Constants (paper Section 5.3) ─────────────────────────────────────────────

FEATURE_COLS: List[str] = [
    "pressure", "flow", "vibration", "delta_pressure",
    "diameter", "material_enc", "length", "age",
]

MATERIAL_ENCODING: Dict[str, int] = {
    "PVC": 0, "HDPE": 0, "Steel": 0,
    "AC": 1,
    "DI": 2, "Ductile Iron": 2,
    "CI": 3, "Cast Iron": 3,
}

PREDICTION_BATCH_SIZE = int(os.getenv("PREDICTION_BATCH_SIZE", "100"))

# ── Module-level cache (survives warm invocations) ────────────────────────────

_model_cache: Dict[str, Dict[str, Any]] = {}
_config: Dict[str, Any] = {}


# ── Configuration ─────────────────────────────────────────────────────────────

def _load_config() -> None:
    global _config
    _config = {
        "table_name":    os.environ["DYNAMO_TABLE"],
        "model_bucket":  os.environ["MODEL_BUCKET"],
        "tenants":       ["tenant-A", "tenant-B"],
        "region":        os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    }


# ── Model loading (S3 + local /tmp cache) ─────────────────────────────────────

def _s3_key(tenant_id: str, filename: str) -> str:
    short = tenant_id.split("-")[1].lower()   # 'a' or 'b'
    return f"models/tenant-{short}/{filename}"


def _load_models(tenant_id: str) -> Dict[str, Any]:
    """
    Load rf_model.pkl, xgb_model.pkl, scaler.pkl, beta.json from S3.
    Result is cached in _model_cache for the lifetime of the Lambda container.
    """
    if tenant_id in _model_cache:
        return _model_cache[tenant_id]

    s3 = boto3.client("s3", region_name=_config["region"])
    bucket = _config["model_bucket"]
    tmp = tempfile.gettempdir()

    def _download(filename: str) -> str:
        local = os.path.join(tmp, f"{tenant_id}_{filename}")
        if not os.path.exists(local):
            s3.download_file(bucket, _s3_key(tenant_id, filename), local)
            logger.info(f"Downloaded {filename} for {tenant_id}")
        return local

    try:
        rf      = joblib.load(_download("rf_model.pkl"))
        xgb_mdl = joblib.load(_download("xgb_model.pkl"))
        scaler  = joblib.load(_download("scaler.pkl"))

        with open(_download("beta.json")) as f:
            beta_data = json.load(f)
        beta = float(beta_data["beta"])

        _model_cache[tenant_id] = {
            "rf": rf, "xgb": xgb_mdl, "scaler": scaler, "beta": beta
        }
        logger.info(f"[{tenant_id}] Models loaded — beta={beta:.2f}")
        return _model_cache[tenant_id]

    except Exception as e:
        logger.error(f"[{tenant_id}] Model load failed: {e}")
        raise


# ── DynamoDB helpers ──────────────────────────────────────────────────────────

def _query_unprocessed(tenant_id: str) -> List[Dict[str, Any]]:
    """
    T-05.01 — Query DynamoDB for anomalous, unprocessed records.
    Filter: isAnomaly=True AND (processed_by_prediction=False OR attribute missing)
    Uses Query (by PK=tenantId) + server-side FilterExpression → no full Scan.
    """
    dynamodb = boto3.client("dynamodb", region_name=_config["region"])
    table    = _config["table_name"]

    kwargs = {
        "TableName":                table,
        "KeyConditionExpression":   "tenantId = :tid",
        "FilterExpression": (
            "isAnomaly = :true AND "
            "(attribute_not_exists(processed_by_prediction) "
            "OR processed_by_prediction = :false)"
        ),
        "ExpressionAttributeValues": {
            ":tid":   {"S": tenant_id},
            ":true":  {"BOOL": True},
            ":false": {"BOOL": False},
        },
        "Limit": PREDICTION_BATCH_SIZE,
    }

    items: List[Dict[str, Any]] = []
    resp = dynamodb.query(**kwargs)
    items.extend(resp.get("Items", []))

    logger.info(f"[{tenant_id}] Unprocessed anomalies found: {len(items)}")
    return items


# ── Feature assembly ──────────────────────────────────────────────────────────

def _assemble_features(item: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    T-05.01 — Build 8-feature vector from a DynamoDB item.
    X = [pressure, flow, vibration, delta_pressure,
         diameter, material_enc, length, age]

    delta_pressure is set to 0.0 here because we have a single snapshot.
    For production, store delta_pressure in the anomaly Lambda (Kinesis
    records arrive in order, so delta is computable per-shard).
    """
    try:
        pressure  = float(item["pressure"]["N"])
        flow      = float(item["flow"]["N"])
        vibration = float(item["vibration"]["N"])

        meta_raw = item.get("metadata", {}).get("S", "{}")
        try:
            meta = json.loads(meta_raw)
        except (json.JSONDecodeError, TypeError):
            meta = {}

        material     = meta.get("material", "PVC")
        material_enc = MATERIAL_ENCODING.get(material, 0)
        diameter     = float(meta.get("diameter", 300.0))
        length       = float(meta.get("length",   250.0))
        age          = float(meta.get("age",       20.0))

        features = np.array(
            [pressure, flow, vibration, 0.0,
             diameter, material_enc, length, age],
            dtype=np.float32,
        ).reshape(1, -1)

        return features

    except Exception as e:
        logger.warning(f"Feature assembly failed for item {item.get('sortKey')}: {e}")
        return None


# ── Prediction ────────────────────────────────────────────────────────────────

def _predict(features: np.ndarray, models: Dict[str, Any]) -> float:
    """
    Ecuación 2: P(Failure|X) = β·P_RF + (1-β)·P_XGB
    """
    X_sc  = models["scaler"].transform(features)
    p_rf  = float(models["rf"].predict_proba(X_sc)[0, 1])
    p_xgb = float(models["xgb"].predict_proba(X_sc)[0, 1])
    beta  = models["beta"]
    return beta * p_rf + (1.0 - beta) * p_xgb


# ── DynamoDB persistence (T-05.02) ────────────────────────────────────────────

def _mark_processed(
    tenant_id: str,
    sort_key: str,
    p_failure: float,
    segment_id: str,
) -> None:
    """
    T-05.02 — UpdateItem: write p_failure, prediction_timestamp,
    and set processed_by_prediction=True on the original anomaly record.
    """
    dynamodb = boto3.client("dynamodb", region_name=_config["region"])
    table    = _config["table_name"]
    now_iso  = datetime.now(timezone.utc).isoformat()

    try:
        dynamodb.update_item(
            TableName=table,
            Key={
                "tenantId": {"S": tenant_id},
                "sortKey":  {"S": sort_key},
            },
            UpdateExpression=(
                "SET processed_by_prediction = :done, "
                "    #pf = :pf, "
                "    prediction_timestamp = :ts"
            ),
            ExpressionAttributeNames={"#pf": "p_failure"},
            ExpressionAttributeValues={
                ":done": {"BOOL": True},
                ":pf":   {"N":    str(round(p_failure, 6))},
                ":ts":   {"S":    now_iso},
            },
        )
    except Exception as e:
        logger.error(
            f"[{tenant_id}] UpdateItem failed for {sort_key}: {e}"
        )
        raise


# ── Main handler ──────────────────────────────────────────────────────────────

def lambda_handler(event: Dict, context: Any) -> Dict:
    """
    EventBridge invokes this Lambda on a schedule (rate 1 minute).
    Processes up to PREDICTION_BATCH_SIZE unprocessed anomalies per tenant.
    """
    _load_config()

    total_processed = 0
    total_errors    = 0
    results_by_tenant: Dict[str, Dict] = {}

    for tenant_id in _config["tenants"]:
        processed = 0
        errors    = 0

        # Load models (cached after first invocation)
        try:
            models = _load_models(tenant_id)
        except Exception as e:
            logger.error(f"[{tenant_id}] Skipping — models unavailable: {e}")
            results_by_tenant[tenant_id] = {"skipped": True, "reason": str(e)}
            continue

        # Query unprocessed anomalies
        try:
            items = _query_unprocessed(tenant_id)
        except Exception as e:
            logger.error(f"[{tenant_id}] DynamoDB query failed: {e}")
            results_by_tenant[tenant_id] = {"skipped": True, "reason": str(e)}
            continue

        for item in items:
            sort_key   = item.get("sortKey",   {}).get("S", "")
            segment_id = item.get("segmentId", {}).get("S", "unknown")

            # Assemble feature vector
            features = _assemble_features(item)
            if features is None:
                errors += 1
                continue

            # P(Failure|X) = β·P_RF + (1-β)·P_XGB
            try:
                p_failure = _predict(features, models)
            except Exception as e:
                logger.warning(f"Prediction error for {sort_key}: {e}")
                errors += 1
                continue

            # Persist result + mark processed
            try:
                _mark_processed(tenant_id, sort_key, p_failure, segment_id)
                processed += 1
                logger.info(
                    f"[{tenant_id}] {segment_id} → "
                    f"P(Failure)={p_failure:.4f}"
                )
            except Exception:
                errors += 1

        results_by_tenant[tenant_id] = {
            "processed": processed,
            "errors":    errors,
        }
        total_processed += processed
        total_errors    += errors

    logger.info(
        f"Batch complete — processed: {total_processed}, errors: {total_errors}"
    )

    return {
        "statusCode": 200,
        "body": json.dumps({
            "total_processed": total_processed,
            "total_errors":    total_errors,
            "by_tenant":       results_by_tenant,
        }),
    }
