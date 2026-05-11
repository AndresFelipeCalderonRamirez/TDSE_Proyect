"""
QA-6  Digital Twin Lambda — Integration Tests  (US-06, T-06.01 / T-06.02 / T-06.03)
=====================================================================================
Exercises the full lambda_handler() against a mocked DynamoDB (moto) to verify:
  - topology is loaded and BFS runs end-to-end
  - ranking is persisted with TTL
  - empty topology skips gracefully (no crash)
  - accumulated ranking/topology items are NOT loaded as p_failure risks

Run:
    pytest tests/test_qa6_handler.py -v
"""

import importlib.util
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
import pytest

try:
    from moto import mock_dynamodb
except ImportError:
    pytest.skip("moto[dynamodb] required", allow_module_level=True)

ROOT    = Path(__file__).parent.parent
TABLE   = "water-twin-data"
REGION  = "us-east-1"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_dt():
    """Load the digital twin Lambda module in isolation (avoids sys.modules cache)."""
    spec = importlib.util.spec_from_file_location(
        "dt_lambda_int",
        ROOT / "lambdas" / "digital_twin" / "lambda_function.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _create_table(client) -> None:
    client.create_table(
        TableName=TABLE,
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


def _seed_topology(client, tenant_id: str, n_nodes: int) -> list:
    """Write a simple linear topology item for one tenant, return node IDs."""
    short = tenant_id.split("-")[1].upper()
    nodes = [f"SEG_{short}_{i:03d}" for i in range(n_nodes)]
    edges = [[nodes[i], nodes[i + 1]] for i in range(n_nodes - 1)]
    client.put_item(
        TableName=TABLE,
        Item={
            "tenantId": {"S": tenant_id},
            "sortKey":  {"S": "topology#config"},
            "nodes":    {"S": json.dumps(nodes)},
            "edges":    {"S": json.dumps(edges)},
        },
    )
    return nodes


def _seed_prediction(client, tenant_id: str, seg_id: str, p_failure: float) -> None:
    """Write a processed prediction record (processed_by_prediction=True)."""
    ts = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
    client.put_item(
        TableName=TABLE,
        Item={
            "tenantId":                {"S": tenant_id},
            "sortKey":                 {"S": f"{ts}#{seg_id}"},
            "segmentId":               {"S": seg_id},
            "p_failure":               {"N": str(p_failure)},
            "processed_by_prediction": {"BOOL": True},
        },
    )


def _query_rankings(client, tenant_id: str) -> list:
    items = client.query(
        TableName=TABLE,
        KeyConditionExpression="tenantId = :t",
        ExpressionAttributeValues={":t": {"S": tenant_id}},
    )["Items"]
    return [
        it for it in items
        if it.get("sortKey", {}).get("S", "").startswith("twin_ranking_")
    ]


# ── Tests ─────────────────────────────────────────────────────────────────────

@mock_dynamodb
def test_handler_writes_ranking_after_full_flow(monkeypatch):
    """T-06.01 + T-06.02 + T-06.03: topology loaded, BFS propagated, ranking saved."""
    monkeypatch.setenv("DYNAMO_TABLE", TABLE)
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)

    dt = _load_dt()
    client = boto3.client("dynamodb", region_name=REGION)
    _create_table(client)
    nodes = _seed_topology(client, "tenant-A", 5)
    _seed_prediction(client, "tenant-A", nodes[0], 0.9)

    resp = dt.lambda_handler({}, None)
    assert resp["statusCode"] == 200

    rankings = _query_rankings(client, "tenant-A")
    assert len(rankings) == 1, "Expected exactly one ranking item"

    segs = json.loads(rankings[0]["segments"]["S"])
    assert len(segs) > 0
    assert segs[0]["risk_propagated"] > 0.0, "Top segment must have propagated risk > 0"


@mock_dynamodb
def test_handler_skips_tenant_with_empty_topology(monkeypatch):
    """Handler must not crash and must mark tenant as skipped when topology is absent."""
    monkeypatch.setenv("DYNAMO_TABLE", TABLE)
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)

    dt = _load_dt()
    client = boto3.client("dynamodb", region_name=REGION)
    _create_table(client)
    _seed_topology(client, "tenant-A", 3)
    # No topology for tenant-B

    resp = dt.lambda_handler({}, None)
    body = json.loads(resp["body"])
    assert resp["statusCode"] == 200
    assert body["tenant-B"].get("skipped") is True, "tenant-B should be reported as skipped"


@mock_dynamodb
def test_ranking_item_carries_ttl_epoch(monkeypatch):
    """Ranking items must include ttl_epoch > now so DynamoDB TTL can expire them."""
    monkeypatch.setenv("DYNAMO_TABLE", TABLE)
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)

    dt = _load_dt()
    client = boto3.client("dynamodb", region_name=REGION)
    _create_table(client)
    nodes = _seed_topology(client, "tenant-A", 3)
    _seed_prediction(client, "tenant-A", nodes[0], 0.7)

    dt.lambda_handler({}, None)

    rankings = _query_rankings(client, "tenant-A")
    assert rankings, "No ranking items found"
    for item in rankings:
        assert "ttl_epoch" in item, "ttl_epoch attribute missing from ranking item"
        ttl = int(item["ttl_epoch"]["N"])
        assert ttl > int(time.time()), "ttl_epoch must be in the future"


@mock_dynamodb
def test_accumulated_rankings_not_loaded_as_risk(monkeypatch):
    """
    After N invocations, accumulated twin_ranking_* items must not be picked up
    by _load_segment_risks as p_failure records (sortKey BETWEEN bound fix).
    """
    monkeypatch.setenv("DYNAMO_TABLE", TABLE)
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)

    dt = _load_dt()
    client = boto3.client("dynamodb", region_name=REGION)
    _create_table(client)
    nodes = _seed_topology(client, "tenant-A", 4)
    _seed_prediction(client, "tenant-A", nodes[0], 0.8)

    # First invocation writes a ranking item
    dt.lambda_handler({}, None)
    # Second invocation — ranking item from first run is in the table;
    # it must NOT be loaded as a p_failure source.
    resp = dt.lambda_handler({}, None)
    assert resp["statusCode"] == 200

    # The handler must still produce valid rankings (risk > 0)
    rankings = _query_rankings(client, "tenant-A")
    assert len(rankings) == 2  # two invocations → two ranking items
    for r in rankings:
        segs = json.loads(r["segments"]["S"])
        assert segs[0]["risk_propagated"] > 0.0
