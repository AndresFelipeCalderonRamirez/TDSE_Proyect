"""
DynamoDB Query Utilities — Reusable data access layer for evaluation metrics.

Provides paginated queries, item parsing, and tenant-specific helpers
shared across all QA measurement modules.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

TABLE_NAME = os.getenv("DYNAMO_TABLE", "water-twin-data")
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

TENANTS = ["tenant-A", "tenant-B"]


def get_client():
    return boto3.client("dynamodb", region_name=REGION)


def paginated_query(
    client,
    key_condition: str,
    expr_attr_values: Dict[str, Any],
    table: Optional[str] = None,
    filter_expr: Optional[str] = None,
    projection: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    kwargs = {
        "TableName": table or TABLE_NAME,
        "KeyConditionExpression": key_condition,
        "ExpressionAttributeValues": expr_attr_values,
    }
    if filter_expr:
        kwargs["FilterExpression"] = filter_expr
    if projection:
        kwargs["ProjectionExpression"] = projection
    if limit:
        kwargs["Limit"] = limit

    items = []
    try:
        while True:
            resp = client.query(**kwargs)
            items.extend(resp.get("Items", []))
            last = resp.get("LastEvaluatedKey")
            if not last:
                break
            kwargs["ExclusiveStartKey"] = last
    except ClientError as e:
        logger.error(f"DynamoDB query error: {e}")
        raise
    return items


def parse_sensor_item(item: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for key, value in item.items():
        if "S" in value:
            result[key] = value["S"]
        elif "N" in value:
            raw = value["N"]
            result[key] = float(raw) if "." in raw else int(raw)
        elif "BOOL" in value:
            result[key] = value["BOOL"]
    return result


def get_sensor_records(
    tenant_id: str,
    hours_back: int = 2,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    client = get_client()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat()
    items = paginated_query(
        client=client,
        key_condition="tenantId = :tid AND sortKey BETWEEN :cutoff AND :upper",
        expr_attr_values={
            ":tid": {"S": tenant_id},
            ":cutoff": {"S": cutoff},
            ":upper": {"S": "topology"},
        },
        limit=limit,
    )
    return [parse_sensor_item(it) for it in items]


def get_anomaly_records(
    tenant_id: str,
    hours_back: int = 24,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    client = get_client()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat()
    items = paginated_query(
        client=client,
        key_condition="tenantId = :tid AND sortKey BETWEEN :cutoff AND :upper",
        expr_attr_values={
            ":tid": {"S": tenant_id},
            ":cutoff": {"S": cutoff},
            ":upper": {"S": "topology"},
        },
        filter_expr="isAnomaly = :true",
        limit=limit,
    )
    return [parse_sensor_item(it) for it in items]


def get_prediction_records(
    tenant_id: str,
    hours_back: int = 24,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    client = get_client()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat()
    items = paginated_query(
        client=client,
        key_condition="tenantId = :tid AND sortKey BETWEEN :cutoff AND :upper",
        expr_attr_values={
            ":tid": {"S": tenant_id},
            ":cutoff": {"S": cutoff},
            ":upper": {"S": "topology"},
        },
        filter_expr=(
            "processed_by_prediction = :done AND attribute_exists(p_failure)"
        ),
        limit=limit,
    )
    return [parse_sensor_item(it) for it in items]


def get_latest_ranking(tenant_id: str) -> Optional[Dict[str, Any]]:
    client = get_client()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    items = paginated_query(
        client=client,
        key_condition="tenantId = :tid AND sortKey BETWEEN :start AND :end",
        expr_attr_values={
            ":tid": {"S": tenant_id},
            ":start": {"S": f"twin_ranking_{cutoff}"},
            ":end": {"S": "twin_ranking_"},
        },
    )
    ranking_items = [
        it
        for it in items
        if it.get("sortKey", {}).get("S", "").startswith("twin_ranking_")
    ]
    if not ranking_items:
        return None
    ranking_items.sort(
        key=lambda x: x.get("generated_at", {}).get("S", ""), reverse=True
    )
    latest = ranking_items[0]
    segs_raw = latest.get("segments", {}).get("S", "[]")
    try:
        latest["_parsed_segments"] = json.loads(segs_raw)
    except (json.JSONDecodeError, TypeError):
        latest["_parsed_segments"] = []
    return latest


def write_sensor_item(
    client,
    tenant_id: str,
    segment_id: str,
    timestamp: str,
    is_anomaly: bool = False,
    source_stream: Optional[str] = None,
) -> bool:
    try:
        item = {
            "tenantId": {"S": tenant_id},
            "sortKey": {"S": f"{timestamp}#{segment_id}"},
            "timestamp": {"S": timestamp},
            "segmentId": {"S": segment_id},
            "isAnomaly": {"BOOL": is_anomaly},
            "anomalyScore": {"N": "0.0"},
            "pressure": {"N": "4.5"},
            "flow": {"N": "0.8"},
            "vibration": {"N": "2.5"},
            "metadata": {"S": "{}"},
            "processed_by_prediction": {"BOOL": False},
        }
        if source_stream:
            item["sourceStream"] = {"S": source_stream}
        client.put_item(TableName=TABLE_NAME, Item=item)
        return True
    except Exception as exc:
        logger.error(f"Write failed: {exc}")
        return False
