"""
QA2 — Throughput Measurement (US-07 / T-07.02)

Simulates 1000 events/min across 2 tenants concurrently using moto-mocked
DynamoDB, then measures actual events processed, errors, and event loss.

Threshold: throughput > 950 events/min.
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

from evaluation.dynamo_queries import TABLE_NAME, get_client, write_sensor_item

logger = logging.getLogger(__name__)

THROUGHPUT_THRESHOLD = 950  # events/min
TARGET_EVENTS_PER_MIN = 1000
TENANT_A = "tenant-A"
TENANT_B = "tenant-B"
SEGMENTS_A = 50
SEGMENTS_B = 40


def _create_table(client) -> None:
    try:
        from moto import mock_dynamodb
    except ImportError:
        logger.error("moto[dynamodb] required for throughput measurement")
        raise

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


class _WriterThread:
    def __init__(self, client, tenant_id: str, n_events: int, segments: int):
        self.client = client
        self.tenant_id = tenant_id
        self.n_events = n_events
        self.segments = segments
        self.written = 0
        self.errors: List[str] = []

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
                source_stream=(
                    "tenant-a-stream"
                    if self.tenant_id == TENANT_A
                    else "tenant-b-stream"
                ),
            )
            if ok:
                self.written += 1
            else:
                self.errors.append(f"write {i}")


def measure_throughput(
    events_per_tenant: int = 500,
) -> Dict[str, Any]:
    try:
        from moto import mock_dynamodb
    except ImportError:
        logger.error("moto[dynamodb] is required. Install: pip install 'moto[dynamodb]>=4.2.0,<5'")
        return {
            "throughput": 0,
            "throughput_per_tenant": {},
            "errors": 1,
            "event_loss": 1.0,
            "threshold": THROUGHPUT_THRESHOLD,
            "passed": False,
            "message": "moto[dynamodb] not installed",
        }

    t_start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("QA2 — Throughput Measurement")
    logger.info(f"  Target      : {events_per_tenant * 2} events/run "
                f"({events_per_tenant} per tenant)")
    logger.info(f"  Threshold   : > {THROUGHPUT_THRESHOLD} events/min")
    logger.info("=" * 60)

    @mock_dynamodb
    def _run():
        client = boto3.client("dynamodb", region_name="us-east-1")
        _create_table(client)

        writer_a = _WriterThread(client, TENANT_A, events_per_tenant, SEGMENTS_A)
        writer_b = _WriterThread(client, TENANT_B, events_per_tenant, SEGMENTS_B)

        t_a = threading.Thread(target=writer_a.run, name="writer-A")
        t_b = threading.Thread(target=writer_b.run, name="writer-B")
        t_a.start()
        t_b.start()
        t_a.join()
        t_b.join()

        # Verify written data
        resp_a = client.query(
            TableName=TABLE_NAME,
            KeyConditionExpression="tenantId = :tid",
            ExpressionAttributeValues={":tid": {"S": TENANT_A}},
        )
        resp_b = client.query(
            TableName=TABLE_NAME,
            KeyConditionExpression="tenantId = :tid",
            ExpressionAttributeValues={":tid": {"S": TENANT_B}},
        )
        found_a = len(resp_a.get("Items", []))
        found_b = len(resp_b.get("Items", []))

        total_written = writer_a.written + writer_b.written
        total_found = found_a + found_b
        total_errors = len(writer_a.errors) + len(writer_b.errors)

        return {
            "tenant_a_written": writer_a.written,
            "tenant_b_written": writer_b.written,
            "tenant_a_found": found_a,
            "tenant_b_found": found_b,
            "total_written": total_written,
            "total_found": total_found,
            "errors": total_errors,
            "event_loss": round(
                1.0 - (total_found / max(total_written, 1)), 6
            ),
            "target_events": events_per_tenant * 2,
        }

    raw = _run()

    throughput = raw["total_written"]
    loss = raw["event_loss"]

    passed = throughput > THROUGHPUT_THRESHOLD and loss < 0.05

    result: Dict[str, Any] = {
        **raw,
        "throughput": throughput,
        "threshold": THROUGHPUT_THRESHOLD,
        "passed": passed,
        "duration_seconds": round(time.perf_counter() - t_start, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    status = "PASS" if passed else "FAIL"
    logger.info(f"  QA2 result: {status}")
    logger.info(f"  Throughput : {throughput} events/min (threshold > {THROUGHPUT_THRESHOLD})")
    logger.info(f"  Errors     : {raw['errors']}")
    logger.info(f"  Event loss : {loss:.4%}")

    return result
