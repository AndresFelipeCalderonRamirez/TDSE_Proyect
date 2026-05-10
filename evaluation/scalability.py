"""
QA3 — Scalability / Degradation Measurement (US-07 / T-07.03)

Measures P95 latency at baseline (1000 events/min) and under stress
(5000 events/min) using DynamoDB and synthetic load.

Degradation% = (P95_scaled - P95_base) / P95_base × 100
Threshold: degradation < 20%.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
import numpy as np

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from evaluation.dynamo_queries import TABLE_NAME, write_sensor_item

logger = logging.getLogger(__name__)

DEGRADATION_THRESHOLD_PCT = 20.0
TENANT_A = "tenant-A"
TENANT_B = "tenant-B"
SEGMENTS_A = 50
SEGMENTS_B = 40

BASELINE_EVENTS = 1000
STRESS_EVENTS = 5000


def _create_table(client) -> None:
    client.create_table(
        TableName=TABLE_NAME,
        KeySchema=[
            {"AttributeName": "tenantId", "KeyType": "HASH"},
            {"AttributeName": "sortKey", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "tenantId", "AttributeType": "S"},
            {"AttributeName": "sortKey", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )


def _simulate_load(
    client,
    n_events: int,
    label: str,
) -> float:
    writers = []
    for tenant_id, segments in [(TENANT_A, SEGMENTS_A), (TENANT_B, SEGMENTS_B)]:
        short = tenant_id.split("-")[1].upper()
        for i in range(n_events // 2):
            seg_idx = i % segments
            seg_id = f"SEG_{short}_{seg_idx:03d}"
            ts = (datetime.now(timezone.utc) + timedelta(microseconds=i)).isoformat()
            t0 = time.perf_counter()
            write_sensor_item(
                client,
                tenant_id,
                seg_id,
                ts,
                is_anomaly=(i % 50 == 0),
            )
            elapsed = (time.perf_counter() - t0) * 1000
            writers.append(elapsed)

    p95 = float(np.percentile(writers, 95)) if writers else 0.0
    logger.info(f"  [{label}] {len(writers)} writes, P95={p95:.2f}ms")
    return p95


def measure_scalability(
    baseline_events: int = BASELINE_EVENTS,
    stress_events: int = STRESS_EVENTS,
) -> Dict[str, Any]:
    try:
        from moto import mock_dynamodb
    except ImportError:
        logger.error("moto[dynamodb] required")
        return {
            "p95_base_ms": 0, "p95_scaled_ms": 0,
            "degradation_pct": 0, "threshold_pct": DEGRADATION_THRESHOLD_PCT,
            "passed": False,
            "message": "moto[dynamodb] not installed",
        }

    t_start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("QA3 — Scalability / Degradation Measurement")
    logger.info(f"  Baseline    : {baseline_events} events")
    logger.info(f"  Stress      : {stress_events} events")
    logger.info(f"  Threshold   : degradation < {DEGRADATION_THRESHOLD_PCT}%")
    logger.info("=" * 60)

    @mock_dynamodb
    def _run():
        client = boto3.client("dynamodb", region_name="us-east-1")
        _create_table(client)

        p95_base = _simulate_load(client, baseline_events, "BASELINE")
        p95_scaled = _simulate_load(client, stress_events, "STRESS")

        degradation = 0.0
        if p95_base > 0:
            degradation = ((p95_scaled - p95_base) / p95_base) * 100.0

        return {
            "p95_base_ms": round(p95_base, 2),
            "p95_scaled_ms": round(p95_scaled, 2),
            "degradation_pct": round(degradation, 2),
            "baseline_events": baseline_events,
            "stress_events": stress_events,
        }

    raw = _run()
    passed = raw["degradation_pct"] < DEGRADATION_THRESHOLD_PCT

    result: Dict[str, Any] = {
        **raw,
        "threshold_pct": DEGRADATION_THRESHOLD_PCT,
        "passed": passed,
        "duration_seconds": round(time.perf_counter() - t_start, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    status = "PASS" if passed else "FAIL"
    logger.info(f"  QA3 result: {status}")
    logger.info(f"  P95 base   = {raw['p95_base_ms']}ms")
    logger.info(f"  P95 stress = {raw['p95_scaled_ms']}ms")
    logger.info(f"  Degradation = {raw['degradation_pct']}% (threshold < {DEGRADATION_THRESHOLD_PCT}%)")

    return result
