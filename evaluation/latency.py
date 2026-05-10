"""
QA1 — End-to-End Latency Measurement (US-07 / T-07.01)

Measures the time from sensor ingest (timestamp in DynamoDB) to twin update
(prediction_timestamp or twin ranking timestamp).

Calculates P50, P95, P99 latency percentiles.
Threshold: P95 < 2 seconds.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from evaluation.dynamo_queries import get_prediction_records

logger = logging.getLogger(__name__)

LATENCY_THRESHOLD_SEC = 2.0
TENANTS = ["tenant-A", "tenant-B"]


def compute_latencies(tenant_id: str, hours_back: int = 24) -> List[float]:
    records = get_prediction_records(tenant_id, hours_back=hours_back)
    latencies = []

    for r in records:
        ts_raw = r.get("timestamp", "")
        pred_ts_raw = r.get("prediction_timestamp", "")
        if not ts_raw or not pred_ts_raw:
            continue
        try:
            ingest = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            predict = datetime.fromisoformat(pred_ts_raw.replace("Z", "+00:00"))
            delta = (predict - ingest).total_seconds()
            if delta >= 0:
                latencies.append(delta)
        except (ValueError, TypeError):
            continue

    return latencies


def measure_latency(
    tenant_id: Optional[str] = None,
    hours_back: int = 24,
) -> Dict[str, Any]:
    t_start = time.perf_counter()

    tenants = [tenant_id] if tenant_id else TENANTS
    all_latencies: List[float] = []

    logger.info("=" * 60)
    logger.info("QA1 — End-to-End Latency Measurement")
    logger.info(f"  Hours back: {hours_back}")
    logger.info("=" * 60)

    for tid in tenants:
        latencies = compute_latencies(tid, hours_back=hours_back)
        all_latencies.extend(latencies)
        if latencies:
            logger.info(
                f"  [{tid}] {len(latencies)} records — "
                f"P50={np.median(latencies):.4f}s "
                f"P95={np.percentile(latencies, 95):.4f}s "
                f"P99={np.percentile(latencies, 99):.4f}s"
            )
        else:
            logger.warning(f"  [{tid}] No latency records found")

    if not all_latencies:
        result = {
            "p50_latency_sec": None,
            "p95_latency_sec": None,
            "p99_latency_sec": None,
            "records_measured": 0,
            "threshold_sec": LATENCY_THRESHOLD_SEC,
            "passed": False,
            "message": "No prediction records with timestamps found in DynamoDB",
        }
        duration = time.perf_counter() - t_start
        result["duration_seconds"] = round(duration, 3)
        result["timestamp"] = datetime.now(timezone.utc).isoformat()
        return result

    p50 = float(np.median(all_latencies))
    p95 = float(np.percentile(all_latencies, 95))
    p99 = float(np.percentile(all_latencies, 99))

    passed = p95 < LATENCY_THRESHOLD_SEC

    result = {
        "p50_latency_sec": round(p50, 4),
        "p95_latency_sec": round(p95, 4),
        "p99_latency_sec": round(p99, 4),
        "threshold_sec": LATENCY_THRESHOLD_SEC,
        "records_measured": len(all_latencies),
        "passed": passed,
    }

    duration = time.perf_counter() - t_start
    result["duration_seconds"] = round(duration, 3)
    result["timestamp"] = datetime.now(timezone.utc).isoformat()

    status = "PASS" if passed else "FAIL"
    logger.info(f"  QA1 result: {status}")
    logger.info(f"  P50 = {p50:.4f}s | P95 = {p95:.4f}s | P99 = {p99:.4f}s")
    logger.info(f"  Records measured: {len(all_latencies)}")
    logger.info(f"  Threshold: P95 < {LATENCY_THRESHOLD_SEC}s")

    return result
