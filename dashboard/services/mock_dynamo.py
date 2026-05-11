import json
import random
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional


def _now_iso(offset_seconds: int = 0) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=offset_seconds)).isoformat()


def check_connection() -> bool:
    return True


def _segment_prefix(tenant_id: str) -> str:
    return "SEG_A_" if tenant_id.lower().startswith("tenant-a") else "SEG_B_"


def _make_sensor_item(tenant_id: str, idx: int) -> Dict[str, Any]:
    prefix = _segment_prefix(tenant_id)
    seg_num = idx % (50 if prefix == "SEG_A_" else 40)
    segment_id = f"{prefix}{seg_num:03d}"
    ts = _now_iso(offset_seconds=idx)
    pressure = round(4.0 + random.uniform(-0.5, 0.5), 3)
    flow = round(0.8 + random.uniform(-0.2, 0.2), 3)
    vibration = round(2.5 + random.uniform(-0.8, 0.8), 3)
    is_anomaly = random.random() < 0.05
    anomaly_score = round(random.uniform(-1.5, 1.5), 3) if is_anomaly else round(random.uniform(-0.2, 0.2), 3)
    item = {
        "tenantId": {"S": tenant_id},
        "sortKey": {"S": f"{ts}#{segment_id}"},
        "timestamp": {"S": ts},
        "segmentId": {"S": segment_id},
        "pressure": {"N": str(pressure)},
        "flow": {"N": str(flow)},
        "vibration": {"N": str(vibration)},
        "isAnomaly": {"BOOL": is_anomaly},
        "anomalyScore": {"N": str(anomaly_score)},
        "metadata": {"S": json.dumps({"diameter": 300, "material": "Ductile Iron", "length": 250, "age": 20})},
        "processed_by_prediction": {"BOOL": random.random() < 0.5},
    }
    # occasionally include p_failure
    if random.random() < 0.2:
        p = round(random.random(), 3)
        item["p_failure"] = {"N": str(p)}
        item["prediction_timestamp"] = {"S": _now_iso(offset_seconds=idx - 30)}
    return item


def query_recent_sensor_records(tenant_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    return [_make_sensor_item(tenant_id, i) for i in range(min(limit, 200))]


def query_recent_anomalies(tenant_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    items = query_recent_sensor_records(tenant_id, limit * 4)
    anomalies = [it for it in items if it.get("isAnomaly", {}).get("BOOL", False)]
    return anomalies[:limit]


def query_latest_ranking(tenant_id: str) -> Optional[Dict[str, Any]]:
    # produce a ranking item with segments list
    prefix = _segment_prefix(tenant_id)
    segments = []
    for i in range(10):
        seg = f"{prefix}{i:03d}"
        segments.append({"segment_id": seg, "risk_propagated": round(random.random(), 3)})
    generated_at = _now_iso()
    return {
        "tenantId": {"S": tenant_id},
        "sortKey": {"S": f"twin_ranking_{generated_at}"},
        "generated_at": {"S": generated_at},
        "top_k": {"N": "5"},
        "segments": {"S": json.dumps(segments)},
        "alpha": {"N": "0.7"},
        "ttl_epoch": {"N": str(int(datetime.now(timezone.utc).timestamp()) + 7 * 24 * 3600)},
    }


def query_maintenance_alerts(tenant_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    items = query_recent_sensor_records(tenant_id, limit)
    alerts = [it for it in items if "p_failure" in it]
    return alerts[:limit]


def rebuild_cache():
    return None
