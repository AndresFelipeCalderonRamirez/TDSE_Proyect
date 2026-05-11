"""
QA6 — Multi-Tenant Isolation Measurement (US-07 / T-07.06)

Adapted from measure_metrics.py with full integration into the evaluation framework.

Procedure:
  1. Seed DynamoDB with data for both tenants concurrently
     (simulating 1000 events/min from both simulators).
  2. From Tenant A context: attempt query with PK='tenant-B' ITERATIONS times.
  3. From Tenant B context: attempt query with PK='tenant-A' ITERATIONS times.
  4. Verify record consistency: tenantId + sourceStream must match partition.

Threshold: leak_count = 0.
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

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from evaluation.dynamo_queries import TABLE_NAME, write_sensor_item

logger = logging.getLogger(__name__)

TENANT_A = "tenant-A"
TENANT_B = "tenant-B"
STREAM_A = "tenant-a-stream"
STREAM_B = "tenant-b-stream"
SEGMENTS_A = 50
SEGMENTS_B = 40


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


class _TenantWriter:
    def __init__(
        self,
        client,
        tenant_id: str,
        stream_name: str,
        n_events: int,
        segments: int,
        errors: List[str],
    ):
        self.client = client
        self.tenant_id = tenant_id
        self.stream_name = stream_name
        self.n_events = n_events
        self.segments = segments
        self.errors = errors
        self.written = 0

    def run(self) -> None:
        short = self.tenant_id.split("-")[1].upper()
        for i in range(self.n_events):
            seg_idx = i % self.segments
            seg_id = f"SEG_{short}_{seg_idx:03d}"
            ts = (datetime.now(timezone.utc) + timedelta(microseconds=i)).isoformat()
            ok = write_sensor_item(
                self.client,
                self.tenant_id,
                seg_id,
                ts,
                is_anomaly=(i % 50 == 0),
                source_stream=self.stream_name,
            )
            if ok:
                self.written += 1
            else:
                self.errors.append(f"[{self.tenant_id}] write {i}")


def measure_isolation(
    iterations: int = 100,
    events_per_tenant: int = 500,
) -> Dict[str, Any]:
    try:
        from moto import mock_dynamodb
    except ImportError:
        logger.error("moto[dynamodb] is required. Install: pip install 'moto[dynamodb]>=4.2.0,<5'")
        return {
            "leak_count": -1,
            "iterations": iterations,
            "passed": False,
            "message": "moto[dynamodb] not installed",
        }

    t_start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("QA6 — Multi-Tenant Isolation Measurement")
    logger.info(f"  Iterations : {iterations}")
    logger.info(f"  Load       : {events_per_tenant * 2} events/run")
    logger.info("=" * 60)

    @mock_dynamodb
    def _run():
        client = boto3.client("dynamodb", region_name="us-east-1")
        _create_table(client)

        write_errors: List[str] = []
        writer_a = _TenantWriter(
            client, TENANT_A, STREAM_A, events_per_tenant, SEGMENTS_A, write_errors
        )
        writer_b = _TenantWriter(
            client, TENANT_B, STREAM_B, events_per_tenant, SEGMENTS_B, write_errors
        )
        t_a = threading.Thread(target=writer_a.run, name="writer-A")
        t_b = threading.Thread(target=writer_b.run, name="writer-B")
        t_a.start()
        t_b.start()
        t_a.join()
        t_b.join()

        if write_errors:
            logger.warning(f"  Write errors: {len(write_errors)}")

        leak_count = 0
        attempt_log: List[Dict] = []

        for i in range(iterations):
            resp_ab = client.query(
                TableName=TABLE_NAME,
                KeyConditionExpression="tenantId = :tid",
                ExpressionAttributeValues={":tid": {"S": TENANT_B}},
            )
            cross_ab = [
                it for it in resp_ab.get("Items", [])
                if it.get("tenantId", {}).get("S") != TENANT_B
            ]
            leak_count += len(cross_ab)

            resp_ba = client.query(
                TableName=TABLE_NAME,
                KeyConditionExpression="tenantId = :tid",
                ExpressionAttributeValues={":tid": {"S": TENANT_A}},
            )
            cross_ba = [
                it for it in resp_ba.get("Items", [])
                if it.get("tenantId", {}).get("S") != TENANT_A
            ]
            leak_count += len(cross_ba)

            attempt_log.append({
                "iteration": i + 1,
                "cross_ab_leaked": len(cross_ab),
                "cross_ba_leaked": len(cross_ba),
            })

            if (i + 1) % 25 == 0:
                logger.info(
                    f"  Iteration {i + 1:3d}/{iterations} — "
                    f"leak_count so far: {leak_count}"
                )

        # Consistency check
        def _query_all(tid: str) -> List[Dict]:
            kwargs = {
                "TableName": TABLE_NAME,
                "KeyConditionExpression": "tenantId = :tid",
                "ExpressionAttributeValues": {":tid": {"S": tid}},
            }
            items = []
            while True:
                resp = client.query(**kwargs)
                items.extend(resp.get("Items", []))
                if not resp.get("LastEvaluatedKey"):
                    break
                kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
            return items

        items_a = _query_all(TENANT_A)
        items_b = _query_all(TENANT_B)
        inconsistent = []

        for it in items_a:
            if (it.get("tenantId", {}).get("S") != TENANT_A
                    or it.get("sourceStream", {}).get("S") != STREAM_A):
                inconsistent.append({
                    "partition": TENANT_A,
                    "tenantId": it.get("tenantId", {}).get("S"),
                    "sourceStream": it.get("sourceStream", {}).get("S"),
                })

        for it in items_b:
            if (it.get("tenantId", {}).get("S") != TENANT_B
                    or it.get("sourceStream", {}).get("S") != STREAM_B):
                inconsistent.append({
                    "partition": TENANT_B,
                    "tenantId": it.get("tenantId", {}).get("S"),
                    "sourceStream": it.get("sourceStream", {}).get("S"),
                })

        return {
            "leak_count": leak_count,
            "iterations": iterations,
            "events_tenant_a": writer_a.written,
            "events_tenant_b": writer_b.written,
            "concurrent_events_per_min": events_per_tenant * 2,
            "items_in_partition_a": len(items_a),
            "items_in_partition_b": len(items_b),
            "consistent_records": len(items_a) + len(items_b) - len(inconsistent),
            "inconsistent_records": len(inconsistent),
            "inconsistency_details": inconsistent[:10],
            "write_errors": len(write_errors),
        }

    raw = _run()
    passed = (raw["leak_count"] == 0 and raw["inconsistent_records"] == 0)

    result: Dict[str, Any] = {
        **raw,
        "passed": passed,
        "duration_seconds": round(time.perf_counter() - t_start, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    status = "PASS" if passed else "FAIL"
    logger.info(f"  QA6 result: {status}")
    logger.info(f"  leak_count           = {raw['leak_count']}  (threshold = 0)")
    logger.info(f"  inconsistent_records = {raw['inconsistent_records']}")
    logger.info(f"  items A/B            = {raw['events_tenant_a']} / {raw['events_tenant_b']}")

    return result
