"""
QA-7  Multi-Tenant Isolation Tests  (EP-05)
============================================
Verifica cross-tenant data leaks = 0 (QA6 metric, paper Table 4 /
Tabla 2 multi-tenancy isolation mechanisms).

Capas testadas:
  1. Read isolation   — query tenant-A no devuelve datos de tenant-B
  2. Write isolation  — items escritos van a la particion correcta
  3. GetItem cross-tenant — siempre retorna vacio, nunca datos ajenos
  4. Topology isolation  — topology#config de cada tenant es exclusivo
  5. Ranking isolation   — twin_ranking_* por particion separada
  6. Concurrent writes   — carga concurrente no produce cross-leaks (QA6)
  7. Structural: queries — cada funcion Lambda siempre incluye tenantId
  8. T-07.03 guard       — anomaly Lambda rechaza payload con tenant spoofeado

Run:
    pytest tests/test_qa7_isolation.py -v

Requires:
    pip install "moto[dynamodb]>=4.2.0,<5"
"""

import importlib.util
import json
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List
from unittest import mock

import boto3
import numpy as np
import pytest
from moto import mock_dynamodb as mock_aws

ROOT = Path(__file__).parent.parent

# ── Helpers ────────────────────────────────────────────────────────────────────

TABLE_NAME = "water-twin-data"
REGION     = "us-east-1"


def _load_lambda(label: str, rel_path: str):
    """Import a Lambda module without sys.modules caching conflicts."""
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(label, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _create_table(client):
    """Create the water-twin DynamoDB table (real schema)."""
    client.create_table(
        TableName=TABLE_NAME,
        KeySchema=[
            {"AttributeName": "tenantId", "KeyType": "HASH"},
            {"AttributeName": "sortKey",  "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "tenantId", "AttributeType": "S"},
            {"AttributeName": "sortKey",  "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )


def _sensor_item(tenant_id: str, segment_id: str, ts: str) -> Dict:
    """Build a realistic DynamoDB sensor record."""
    return {
        "tenantId":                {"S": tenant_id},
        "sortKey":                 {"S": f"{ts}#{segment_id}"},
        "timestamp":               {"S": ts},
        "segmentId":               {"S": segment_id},
        "city":                    {"S": "Bogota" if tenant_id == "tenant-A" else "Medellin"},
        "pressure":                {"N": "4.5"},
        "flow":                    {"N": "0.8"},
        "vibration":               {"N": "2.5"},
        "isAnomaly":               {"BOOL": False},
        "anomalyScore":            {"N": "0.1"},
        "metadata":                {"S": "{}"},
        "processed_by_prediction": {"BOOL": False},
    }


def _query_tenant(client, tenant_id: str) -> List[Dict]:
    """Full query for a tenant, handling pagination."""
    kwargs = {
        "TableName":               TABLE_NAME,
        "KeyConditionExpression":  "tenantId = :tid",
        "ExpressionAttributeValues": {":tid": {"S": tenant_id}},
    }
    items = []
    while True:
        resp = client.query(**kwargs)
        items.extend(resp.get("Items", []))
        if not resp.get("LastEvaluatedKey"):
            break
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    return items


# ── 1. Read isolation ─────────────────────────────────────────────────────────

@mock_aws
def test_read_isolation_a_sees_only_own_data():
    """tenant-A query returns ZERO tenant-B records."""
    client = boto3.client("dynamodb", region_name=REGION)
    _create_table(client)

    ts = datetime.now(timezone.utc).isoformat()
    for seg in ["SEG_A_000", "SEG_A_001"]:
        client.put_item(TableName=TABLE_NAME, Item=_sensor_item("tenant-A", seg, ts))
    for seg in ["SEG_B_000", "SEG_B_001", "SEG_B_002"]:
        client.put_item(TableName=TABLE_NAME, Item=_sensor_item("tenant-B", seg, ts))

    items = _query_tenant(client, "tenant-A")

    assert len(items) == 2, f"Expected 2, got {len(items)}"
    for it in items:
        assert it["tenantId"]["S"] == "tenant-A", (
            f"Cross-tenant leak: tenantId={it['tenantId']['S']}"
        )


@mock_aws
def test_read_isolation_b_sees_only_own_data():
    """tenant-B query returns ZERO tenant-A records."""
    client = boto3.client("dynamodb", region_name=REGION)
    _create_table(client)

    ts = datetime.now(timezone.utc).isoformat()
    for seg in ["SEG_A_000", "SEG_A_001"]:
        client.put_item(TableName=TABLE_NAME, Item=_sensor_item("tenant-A", seg, ts))
    for seg in ["SEG_B_000"]:
        client.put_item(TableName=TABLE_NAME, Item=_sensor_item("tenant-B", seg, ts))

    items = _query_tenant(client, "tenant-B")

    assert len(items) == 1
    assert items[0]["tenantId"]["S"] == "tenant-B"
    assert items[0]["segmentId"]["S"] == "SEG_B_000"


# ── 2. Write isolation ────────────────────────────────────────────────────────

@mock_aws
def test_write_isolation_item_lands_in_correct_partition():
    """GetItem with wrong tenant key returns empty — never wrong data."""
    client = boto3.client("dynamodb", region_name=REGION)
    _create_table(client)

    ts       = datetime.now(timezone.utc).isoformat()
    sort_key = f"{ts}#SEG_A_000"

    client.put_item(TableName=TABLE_NAME, Item=_sensor_item("tenant-A", "SEG_A_000", ts))

    # Correct key → must exist
    ok = client.get_item(
        TableName=TABLE_NAME,
        Key={"tenantId": {"S": "tenant-A"}, "sortKey": {"S": sort_key}},
    )
    assert "Item" in ok, "Item not found with correct partition key"

    # Wrong tenant key → must be empty
    wrong = client.get_item(
        TableName=TABLE_NAME,
        Key={"tenantId": {"S": "tenant-B"}, "sortKey": {"S": sort_key}},
    )
    assert "Item" not in wrong, (
        "DynamoDB partition key isolation failed: "
        f"got item with wrong tenant key: {wrong.get('Item')}"
    )


# ── 3. GetItem cross-tenant variants ─────────────────────────────────────────

@mock_aws
@pytest.mark.parametrize("wrong_tenant", [
    "tenant-B", "tenant-C", "TENANT-A", "tenant-a", "unknown-tenant",
])
def test_get_item_cross_tenant_always_empty(wrong_tenant):
    """GetItem with any non-owning tenantId must return no data."""
    client = boto3.client("dynamodb", region_name=REGION)
    _create_table(client)

    ts       = datetime.now(timezone.utc).isoformat()
    sort_key = f"{ts}#SEG_A_007"
    client.put_item(TableName=TABLE_NAME, Item=_sensor_item("tenant-A", "SEG_A_007", ts))

    resp = client.get_item(
        TableName=TABLE_NAME,
        Key={"tenantId": {"S": wrong_tenant}, "sortKey": {"S": sort_key}},
    )
    assert "Item" not in resp, (
        f"Cross-tenant leak with tenantId='{wrong_tenant}': {resp.get('Item')}"
    )


# ── 4. Topology item isolation ────────────────────────────────────────────────

@mock_aws
def test_topology_config_scoped_to_partition():
    """
    topology#config for tenant-A contains only SEG_A_* nodes,
    and is not accessible via tenant-B partition key.
    """
    client = boto3.client("dynamodb", region_name=REGION)
    _create_table(client)

    for tenant_id, n in [("tenant-A", 50), ("tenant-B", 40)]:
        short = tenant_id.split("-")[1].upper()
        nodes = [f"SEG_{short}_{i:03d}" for i in range(n)]
        client.put_item(
            TableName=TABLE_NAME,
            Item={
                "tenantId":   {"S": tenant_id},
                "sortKey":    {"S": "topology#config"},
                "nodes":      {"S": json.dumps(nodes)},
                "edges":      {"S": "[]"},
                "n_segments": {"N": str(n)},
            },
        )

    # tenant-A reads its topology
    resp_a = client.get_item(
        TableName=TABLE_NAME,
        Key={"tenantId": {"S": "tenant-A"}, "sortKey": {"S": "topology#config"}},
    )
    assert "Item" in resp_a
    nodes_a = json.loads(resp_a["Item"]["nodes"]["S"])
    assert len(nodes_a) == 50
    assert all(n.startswith("SEG_A_") for n in nodes_a), (
        "tenant-A topology contains non-A segment IDs"
    )

    # tenant-B partition cannot reach tenant-A's topology item
    resp_cross = client.get_item(
        TableName=TABLE_NAME,
        Key={"tenantId": {"S": "tenant-B"}, "sortKey": {"S": "topology#config"}},
    )
    nodes_b = json.loads(resp_cross["Item"]["nodes"]["S"])
    assert all(n.startswith("SEG_B_") for n in nodes_b), (
        "Cross-tenant topology leak: tenant-B got tenant-A nodes"
    )


# ── 5. Ranking isolation ──────────────────────────────────────────────────────

@mock_aws
def test_ranking_items_isolated_by_partition():
    """twin_ranking_* items from tenant-A are not visible from tenant-B query."""
    client = boto3.client("dynamodb", region_name=REGION)
    _create_table(client)

    ts = datetime.now(timezone.utc).isoformat()
    for tenant_id in ("tenant-A", "tenant-B"):
        client.put_item(
            TableName=TABLE_NAME,
            Item={
                "tenantId":     {"S": tenant_id},
                "sortKey":      {"S": f"twin_ranking_{ts}"},
                "generated_at": {"S": ts},
                "top_k":        {"N": "5"},
                "segments":     {"S": "[]"},
                "alpha":        {"N": "0.7"},
            },
        )

    items_a = _query_tenant(client, "tenant-A")
    items_b = _query_tenant(client, "tenant-B")

    assert all(it["tenantId"]["S"] == "tenant-A" for it in items_a), (
        "tenant-A query returned tenant-B ranking items"
    )
    assert all(it["tenantId"]["S"] == "tenant-B" for it in items_b), (
        "tenant-B query returned tenant-A ranking items"
    )


# ── 6. Concurrent write isolation (QA6 target: leaks = 0) ────────────────────

@mock_aws
def test_concurrent_writes_zero_cross_tenant_leaks():
    """
    EP-05 objective: cross-tenant data leaks = 0 under concurrent load.
    20 records per tenant written concurrently; each partition must contain
    exactly its own 20 records and zero foreign records.
    """
    client = boto3.client("dynamodb", region_name=REGION)
    _create_table(client)

    errors: List[str] = []

    def write_records(tenant_id: str, count: int) -> None:
        short = tenant_id.split("-")[1].upper()
        for i in range(count):
            ts  = (datetime.now(timezone.utc) + timedelta(microseconds=i)).isoformat()
            seg = f"SEG_{short}_{i:03d}"
            try:
                client.put_item(
                    TableName=TABLE_NAME,
                    Item=_sensor_item(tenant_id, seg, ts),
                )
            except Exception as exc:
                errors.append(f"[{tenant_id}] write {i}: {exc}")

    N = 20
    t_a = threading.Thread(target=write_records, args=("tenant-A", N))
    t_b = threading.Thread(target=write_records, args=("tenant-B", N))
    t_a.start()
    t_b.start()
    t_a.join()
    t_b.join()

    assert not errors, f"Write errors during concurrent load: {errors}"

    items_a = _query_tenant(client, "tenant-A")
    items_b = _query_tenant(client, "tenant-B")

    # Count check
    assert len(items_a) == N, f"Expected {N} tenant-A records, got {len(items_a)}"
    assert len(items_b) == N, f"Expected {N} tenant-B records, got {len(items_b)}"

    # Cross-leak check
    cross_leaks = 0
    for it in items_a:
        if it["tenantId"]["S"] != "tenant-A":
            cross_leaks += 1
    for it in items_b:
        if it["tenantId"]["S"] != "tenant-B":
            cross_leaks += 1

    assert cross_leaks == 0, (
        f"Cross-tenant data leaks detected under concurrent load: {cross_leaks}"
    )


# ── 7. Structural: Lambda query always scopes to tenantId ─────────────────────

def test_failure_prediction_query_scopes_to_tenantId():
    """
    _query_unprocessed must always bind tenantId in KeyConditionExpression.
    Uses unittest.mock to capture the kwargs without hitting AWS.
    """
    fp = _load_lambda(
        "fp_lambda",
        "lambdas/failure_prediction/lambda_function.py",
    )
    fp._config = {"table_name": TABLE_NAME, "region": REGION}

    captured: List[Dict] = []
    fake_client = mock.MagicMock()
    fake_client.query.side_effect = lambda **kw: (captured.append(kw), {"Items": [], "Count": 0})[1]

    with mock.patch("boto3.client", return_value=fake_client):
        fp._query_unprocessed("tenant-A")

    assert captured, "No DynamoDB query was made"
    kce = captured[0].get("KeyConditionExpression", "")
    eav = captured[0].get("ExpressionAttributeValues", {})

    assert "tenantId" in kce, f"KeyConditionExpression missing tenantId: {kce!r}"
    assert ":tid" in eav,     "ExpressionAttributeValues missing :tid binding"
    assert eav[":tid"]["S"] == "tenant-A", (
        f":tid binding is '{eav[':tid']['S']}', expected 'tenant-A'"
    )


def test_digital_twin_load_risks_scopes_to_tenantId():
    """_load_segment_risks must always bind tenantId in KeyConditionExpression."""
    dt = _load_lambda(
        "dt_lambda",
        "lambdas/digital_twin/lambda_function.py",
    )
    dt._config = {"table_name": TABLE_NAME, "region": REGION}

    captured: List[Dict] = []
    fake_client = mock.MagicMock()
    fake_client.query.side_effect = lambda **kw: (captured.append(kw), {"Items": []})[1]

    with mock.patch("boto3.client", return_value=fake_client):
        dt._load_segment_risks("tenant-B")

    assert captured
    kce = captured[0].get("KeyConditionExpression", "")
    eav = captured[0].get("ExpressionAttributeValues", {})

    assert "tenantId" in kce
    assert ":tid" in eav
    assert eav[":tid"]["S"] == "tenant-B"


def test_digital_twin_load_topology_uses_correct_pk():
    """_load_topology must call get_item with the correct tenantId as PK."""
    dt = _load_lambda(
        "dt_lambda2",
        "lambdas/digital_twin/lambda_function.py",
    )
    dt._config = {"table_name": TABLE_NAME, "region": REGION}

    captured: List[Dict] = []
    fake_client = mock.MagicMock()
    fake_client.get_item.side_effect = lambda **kw: (captured.append(kw), {"Item": {}})[1]

    with mock.patch("boto3.client", return_value=fake_client):
        dt._load_topology("tenant-A")

    assert captured
    key = captured[0].get("Key", {})
    assert key.get("tenantId", {}).get("S") == "tenant-A", (
        f"Topology get_item used wrong PK: {key}"
    )


# ── 8. T-07.03: cross-tenant write guard in anomaly Lambda ────────────────────

def _make_kinesis_event(tenant_id: str, stream: str, extra_fields: Dict = None) -> Dict:
    """Build a realistic Kinesis-triggered Lambda event."""
    payload = {
        "tenant_id":  tenant_id,
        "segment_id": f"SEG_{tenant_id.split('-')[1].upper()}_000",
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "pressure":   4.5,
        "flow":       0.8,
        "vibration":  2.5,
        "city":       "Bogota",
        "metadata":   {},
    }
    if extra_fields:
        payload.update(extra_fields)
    return {
        "Records": [{
            "kinesis":        {"data": json.dumps(payload)},
            "eventSource":    "aws:kinesis",
            "eventSourceARN": (
                f"arn:aws:kinesis:us-east-1:123456789012:stream/{stream}"
            ),
        }]
    }


_TENANT_CONFIGS = {
    "tenant-A": {
        "stream_name": "tenant-a-stream", "segments": 50,
        "model_key": "k", "scaler_key": "k",
    },
    "tenant-B": {
        "stream_name": "tenant-b-stream", "segments": 40,
        "model_key": "k", "scaler_key": "k",
    },
}

_FAKE_FEATURES = np.array([[4.5, 0.8, 2.5, 0.0, 300.0, 0.0, 250.0, 20.0]])


def test_anomaly_lambda_rejects_spoofed_tenant_payload():
    """
    T-07.03: a payload with tenant_id='tenant-B' arriving on tenant-A's
    stream must be silently dropped — write_to_dynamodb must NOT be called.
    """
    an = _load_lambda("an_lambda", "lambdas/anomaly_detection/lambda_function.py")
    an._tenant_configs = _TENANT_CONFIGS

    written: List[str] = []
    event = _make_kinesis_event("tenant-B", "tenant-a-stream")  # spoofed!

    with (
        mock.patch.object(an, "load_config"),
        mock.patch.object(an, "write_to_dynamodb",
                          side_effect=lambda p, a, s: written.append(p["tenant_id"])),
        mock.patch.object(an, "prepare_features", return_value=_FAKE_FEATURES),
        mock.patch.object(an, "detect_anomaly",   return_value=(0, 0.0)),
    ):
        an.lambda_handler(event, None)

    assert len(written) == 0, (
        f"Cross-tenant write not blocked! tenant_id written: {written}"
    )


def test_anomaly_lambda_accepts_legitimate_tenant_payload():
    """
    T-07.03 positive path: tenant-A payload on tenant-A stream is processed
    and written to DynamoDB normally.
    """
    an = _load_lambda("an_lambda2", "lambdas/anomaly_detection/lambda_function.py")
    an._tenant_configs = _TENANT_CONFIGS

    written: List[str] = []
    event = _make_kinesis_event("tenant-A", "tenant-a-stream")  # legitimate

    with (
        mock.patch.object(an, "load_config"),
        mock.patch.object(an, "write_to_dynamodb",
                          side_effect=lambda p, a, s: written.append(p["tenant_id"])),
        mock.patch.object(an, "prepare_features", return_value=_FAKE_FEATURES),
        mock.patch.object(an, "detect_anomaly",   return_value=(0, 0.0)),
    ):
        an.lambda_handler(event, None)

    assert len(written) == 1, (
        f"Legitimate record was not processed. written={written}"
    )
    assert written[0] == "tenant-A"


def test_anomaly_lambda_accepts_non_kinesis_event():
    """
    T-07.03 edge case: non-Kinesis invocations (direct Lambda calls, tests)
    have no eventSourceARN and must NOT be blocked by the guard.
    """
    an = _load_lambda("an_lambda3", "lambdas/anomaly_detection/lambda_function.py")
    an._tenant_configs = _TENANT_CONFIGS

    written: List[str] = []
    # Non-Kinesis event: no 'kinesis' key, no eventSourceARN
    event = {
        "Records": [{
            "tenant_id":  "tenant-A",
            "segment_id": "SEG_A_000",
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "pressure":   4.5,
            "flow":       0.8,
            "vibration":  2.5,
            "city":       "Bogota",
            "metadata":   {},
        }]
    }

    with (
        mock.patch.object(an, "load_config"),
        mock.patch.object(an, "write_to_dynamodb",
                          side_effect=lambda p, a, s: written.append(p["tenant_id"])),
        mock.patch.object(an, "prepare_features", return_value=_FAKE_FEATURES),
        mock.patch.object(an, "detect_anomaly",   return_value=(0, 0.0)),
    ):
        an.lambda_handler(event, None)

    assert len(written) == 1, "Non-Kinesis event should not be blocked"


# ── 9. measure_isolation() integration (US-07 / T-07.01) ─────────────────────

def test_measure_isolation_leak_count_zero():
    """
    US-07 AC1: measure_isolation() must report leak_count = 0
    across 100 iterations of cross-tenant read attempts.
    Uses a small event set (20 per tenant) for pytest speed.
    """
    sys.path.insert(0, str(ROOT))
    from measure_metrics import measure_isolation  # noqa: E402

    result = measure_isolation(iterations=100, events_per_tenant=20, verbose=False)

    assert result["passed"] is True, (
        f"measure_isolation() reported FAIL: {result}"
    )
    assert result["leak_count"] == 0, (
        f"Cross-tenant leak detected! leak_count={result['leak_count']}"
    )
    assert result["inconsistent_records"] == 0, (
        f"Inconsistent records found: {result['inconsistent_records']}"
    )
    assert result["iterations"] == 100


def test_measure_isolation_concurrent_consistency():
    """
    US-07 AC2: under concurrent load, every record's tenantId must be
    consistent with its sourceStream (no cross-stream contamination).
    """
    sys.path.insert(0, str(ROOT))
    from measure_metrics import measure_isolation  # noqa: E402

    result = measure_isolation(iterations=10, events_per_tenant=50, verbose=False)

    assert result["inconsistent_records"] == 0, (
        f"Stream-tenantId mismatch under concurrent load: "
        f"{result['inconsistency_details']}"
    )
    assert result["events_tenant_a"] == 50
    assert result["events_tenant_b"] == 50


