"""
Dashboard API Lambda — serves data to the React dashboard via API Gateway REST.

Routes (via pathParameters.proxy):
  GET /sensor-records       → recent sensor records for a tenant
  GET /anomalies            → anomalous records only
  GET /ranking              → latest digital twin ranking
  GET /maintenance-alerts   → records with p_failure
  GET /topology             → network graph nodes + edges
  GET /stats                → summary counts
"""

import json
import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

TABLE_NAME = os.getenv("DYNAMO_TABLE", "water-twin-data")
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "GET,OPTIONS",
    "Content-Type": "application/json",
}


def _client():
    return boto3.client("dynamodb", region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"))


def _ok(body: Any) -> Dict:
    return {"statusCode": 200, "headers": CORS_HEADERS, "body": json.dumps(body)}


def _err(code: int, msg: str) -> Dict:
    return {"statusCode": code, "headers": CORS_HEADERS, "body": json.dumps({"error": msg})}


def _parse_item(item: Dict) -> Dict:
    result = {}
    for k, v in item.items():
        if "S" in v:
            result[k] = v["S"]
        elif "N" in v:
            raw = v["N"]
            result[k] = float(raw) if "." in raw else int(raw)
        elif "BOOL" in v:
            result[k] = v["BOOL"]
        elif "NULL" in v:
            result[k] = None
        else:
            result[k] = str(v)
    return result


def _paginated_query(kwargs: Dict, max_items: int = 500) -> List[Dict]:
    dynamo = _client()
    items: List[Dict] = []
    while len(items) < max_items:
        resp = dynamo.query(**kwargs)
        items.extend(resp.get("Items", []))
        if "LastEvaluatedKey" not in resp:
            break
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    return items[:max_items]


# ── Route handlers ────────────────────────────────────────────────────────────

def get_sensor_records(tenant_id: str, limit: int) -> Dict:
    items = _paginated_query({
        "TableName": TABLE_NAME,
        "KeyConditionExpression": "tenantId = :tid",
        "ExpressionAttributeValues": {":tid": {"S": tenant_id}},
        "ScanIndexForward": False,
        "Limit": limit,
    }, max_items=limit)
    return _ok([_parse_item(i) for i in items])


def get_anomalies(tenant_id: str, limit: int) -> Dict:
    items = _paginated_query({
        "TableName": TABLE_NAME,
        "KeyConditionExpression": "tenantId = :tid",
        "ExpressionAttributeValues": {
            ":tid": {"S": tenant_id},
            ":true": {"BOOL": True},
        },
        "FilterExpression": "isAnomaly = :true",
        "ScanIndexForward": False,
        "ProjectionExpression": "sortKey, #ts, segmentId, pressure, flow, vibration, anomalyScore, isAnomaly",
        "ExpressionAttributeNames": {"#ts": "timestamp"},
        "Limit": limit * 5,
    }, max_items=limit)
    return _ok([_parse_item(i) for i in items])


def get_ranking(tenant_id: str) -> Dict:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    items = _paginated_query({
        "TableName": TABLE_NAME,
        "KeyConditionExpression": "tenantId = :tid AND sortKey BETWEEN :start AND :end",
        "ExpressionAttributeValues": {
            ":tid": {"S": tenant_id},
            ":start": {"S": f"twin_ranking_{cutoff}"},
            ":end": {"S": "twin_ranking_~"},
        },
    })
    ranking_items = [i for i in items if i.get("sortKey", {}).get("S", "").startswith("twin_ranking_")]
    if not ranking_items:
        return _ok(None)
    ranking_items.sort(key=lambda x: x.get("generated_at", {}).get("S", ""), reverse=True)
    latest = _parse_item(ranking_items[0])
    if isinstance(latest.get("segments"), str):
        try:
            latest["segments"] = json.loads(latest["segments"])
        except (json.JSONDecodeError, TypeError):
            latest["segments"] = []
    return _ok(latest)


def get_maintenance_alerts(tenant_id: str, limit: int) -> Dict:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    items = _paginated_query({
        "TableName": TABLE_NAME,
        "KeyConditionExpression": "tenantId = :tid AND sortKey BETWEEN :cutoff AND :upper",
        "ExpressionAttributeValues": {
            ":tid": {"S": tenant_id},
            ":cutoff": {"S": cutoff},
            ":upper": {"S": "topology"},
            ":done": {"BOOL": True},
        },
        "FilterExpression": "attribute_exists(p_failure) AND processed_by_prediction = :done",
        "ProjectionExpression": "sortKey, #ts, segmentId, pressure, flow, vibration, p_failure, anomalyScore, prediction_timestamp",
        "ExpressionAttributeNames": {"#ts": "timestamp"},
        "Limit": limit * 5,
    }, max_items=limit)
    return _ok([_parse_item(i) for i in items])


def get_topology(tenant_id: str) -> Dict:
    dynamo = _client()
    try:
        resp = dynamo.get_item(
            TableName=TABLE_NAME,
            Key={"tenantId": {"S": tenant_id}, "sortKey": {"S": "topology#config"}},
        )
        item = resp.get("Item", {})
        if not item:
            return _ok({"nodes": [], "edges": []})
        nodes = json.loads(item.get("nodes", {}).get("S", "[]"))
        edges = json.loads(item.get("edges", {}).get("S", "[]"))
        return _ok({"nodes": nodes, "edges": edges})
    except (ClientError, json.JSONDecodeError) as e:
        logger.error(f"Topology error: {e}")
        return _ok({"nodes": [], "edges": []})


def get_stats(tenant_id: str) -> Dict:
    dynamo = _client()

    # Last 1h window — small enough to be fast, meaningful for live monitoring
    cutoff_1h = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    cutoff_24h = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()

    def _count_range(cutoff: str, filter_expr=None, extra_vals=None) -> int:
        vals = {":tid": {"S": tenant_id}, ":cutoff": {"S": cutoff}, ":upper": {"S": "topology"}}
        if extra_vals:
            vals.update(extra_vals)
        kwargs = {
            "TableName": TABLE_NAME,
            "KeyConditionExpression": "tenantId = :tid AND sortKey BETWEEN :cutoff AND :upper",
            "ExpressionAttributeValues": vals,
            "Select": "COUNT",
        }
        if filter_expr:
            kwargs["FilterExpression"] = filter_expr
        total = 0
        while True:
            resp = dynamo.query(**kwargs)
            total += resp.get("Count", 0)
            if "LastEvaluatedKey" not in resp:
                break
            kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
        return total

    sensor_records = _count_range(cutoff_24h)
    anomalies_detected = _count_range(
        cutoff_1h,
        filter_expr="isAnomaly = :true",
        extra_vals={":true": {"BOOL": True}},
    )

    return _ok({
        "sensor_records": sensor_records,
        "anomalies_detected": anomalies_detected,
    })


# ── Main handler ──────────────────────────────────────────────────────────────

def lambda_handler(event, context):
    method = event.get("httpMethod", "GET")
    if method == "OPTIONS":
        return {"statusCode": 200, "headers": CORS_HEADERS, "body": ""}

    path = event.get("path", "").strip("/")
    qs = event.get("queryStringParameters") or {}
    tenant_id = qs.get("tenant_id", "tenant-A")
    limit = int(qs.get("limit", "100"))

    logger.info(f"Request: {method} /{path} tenant={tenant_id}")

    routes = {
        "sensor-records":    lambda: get_sensor_records(tenant_id, limit),
        "anomalies":         lambda: get_anomalies(tenant_id, limit),
        "ranking":           lambda: get_ranking(tenant_id),
        "maintenance-alerts": lambda: get_maintenance_alerts(tenant_id, limit),
        "topology":          lambda: get_topology(tenant_id),
        "stats":             lambda: get_stats(tenant_id),
    }

    handler = routes.get(path)
    if not handler:
        return _err(404, f"Route '/{path}' not found")

    try:
        return handler()
    except Exception as e:
        logger.error(f"Handler error: {e}")
        return _err(500, str(e))
