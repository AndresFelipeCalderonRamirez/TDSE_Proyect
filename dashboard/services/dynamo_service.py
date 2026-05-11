import json
import os
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError
import streamlit as st

logger = logging.getLogger(__name__)


def _get_dynamo_client():
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    # Allow overriding the endpoint (useful for DynamoDB Local)
    endpoint = os.getenv("DYNAMO_ENDPOINT_URL")
    if endpoint:
        return boto3.client("dynamodb", region_name=region, endpoint_url=endpoint)
    return boto3.client("dynamodb", region_name=region)


def _get_table_name() -> str:
    return os.getenv("DYNAMO_TABLE", "water-twin-data")


def check_connection() -> bool:
    try:
        client = _get_dynamo_client()
        client.describe_table(TableName=_get_table_name())
        return True
    except (ClientError, EndpointConnectionError, Exception):
        return False


def _paginated_query(
    client,
    table_name: str,
    key_condition: str,
    expr_attr_values: Dict[str, Any],
    limit: Optional[int] = None,
    filter_expr: Optional[str] = None,
    projection: Optional[str] = None,
    expr_attr_names: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    kwargs = {
        "TableName": table_name,
        "KeyConditionExpression": key_condition,
        "ExpressionAttributeValues": expr_attr_values,
    }
    if limit is not None:
        kwargs["Limit"] = limit
    if filter_expr:
        kwargs["FilterExpression"] = filter_expr
    if projection:
        kwargs["ProjectionExpression"] = projection
    if expr_attr_names:
        kwargs["ExpressionAttributeNames"] = expr_attr_names

    items = []
    try:
        while True:
            resp = client.query(**kwargs)
            items.extend(resp.get("Items", []))
            last_key = resp.get("LastEvaluatedKey")
            if not last_key:
                break
            kwargs["ExclusiveStartKey"] = last_key
    except ClientError as e:
        logger.error(f"DynamoDB query error: {e}")
        raise
    return items


@st.cache_data(ttl=10, show_spinner="Loading sensor records...")
def query_recent_sensor_records(
    tenant_id: str,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    client = _get_dynamo_client()
    table = _get_table_name()
    return _paginated_query(
        client=client,
        table_name=table,
        key_condition="tenantId = :tid",
        expr_attr_values={":tid": {"S": tenant_id}},
        limit=limit,
    )


@st.cache_data(ttl=10, show_spinner="Loading anomaly data...")
def query_recent_anomalies(
    tenant_id: str,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    client = _get_dynamo_client()
    table = _get_table_name()
    return _paginated_query(
        client=client,
        table_name=table,
        key_condition="tenantId = :tid",
        expr_attr_values={":tid": {"S": tenant_id}, ":true": {"BOOL": True}},
        filter_expr="isAnomaly = :true",
        limit=limit,
        projection="sortKey, #ts, segmentId, pressure, flow, vibration, anomalyScore, isAnomaly, p_failure",
        expr_attr_names={"#ts": "timestamp"},
    )


@st.cache_data(ttl=10, show_spinner="Loading latest ranking...")
def query_latest_ranking(tenant_id: str) -> Optional[Dict[str, Any]]:
    client = _get_dynamo_client()
    table = _get_table_name()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    items = _paginated_query(
        client=client,
        table_name=table,
        key_condition="tenantId = :tid AND sortKey BETWEEN :start AND :end",
        expr_attr_values={
            ":tid": {"S": tenant_id},
            ":start": {"S": f"twin_ranking_{cutoff}"},
            ":end": {"S": "twin_ranking_~"},  # ~ (ASCII 126) sorts after all ISO timestamps
        },
    )
    ranking_items = [
        it for it in items
        if it.get("sortKey", {}).get("S", "").startswith("twin_ranking_")
    ]
    if not ranking_items:
        return None
    ranking_items.sort(
        key=lambda x: x.get("generated_at", {}).get("S", ""), reverse=True
    )
    latest = ranking_items[0]
    segments_raw = latest.get("segments", {}).get("S", "[]")
    try:
        latest["_parsed_segments"] = json.loads(segments_raw)
    except (json.JSONDecodeError, TypeError):
        latest["_parsed_segments"] = []
    return latest


@st.cache_data(ttl=10, show_spinner="Loading maintenance alerts...")
def query_maintenance_alerts(
    tenant_id: str,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    client = _get_dynamo_client()
    table = _get_table_name()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    items = _paginated_query(
        client=client,
        table_name=table,
        key_condition="tenantId = :tid AND sortKey BETWEEN :cutoff AND :upper",
        expr_attr_values={
            ":tid": {"S": tenant_id},
            ":cutoff": {"S": cutoff},
            ":upper": {"S": "topology"},
            ":done": {"BOOL": True},
        },
        filter_expr="attribute_exists(p_failure) AND processed_by_prediction = :done",
        limit=limit,
        projection="sortKey, #ts, segmentId, pressure, flow, vibration, p_failure, anomalyScore, prediction_timestamp",
        expr_attr_names={"#ts": "timestamp"},
    )
    return items


@st.cache_data(ttl=30, show_spinner="Loading topology...")
def query_topology(
    tenant_id: str,
) -> Tuple[List[str], List[List[str]]]:
    client = _get_dynamo_client()
    table = _get_table_name()
    try:
        resp = client.get_item(
            TableName=table,
            Key={
                "tenantId": {"S": tenant_id},
                "sortKey": {"S": "topology#config"},
            },
        )
        item = resp.get("Item", {})
        if not item:
            return [], []
        nodes = json.loads(item.get("nodes", {}).get("S", "[]"))
        edges = json.loads(item.get("edges", {}).get("S", "[]"))
        return nodes, edges
    except (ClientError, json.JSONDecodeError, Exception) as e:
        logger.error(f"Failed to load topology for {tenant_id}: {e}")
        return [], []


def rebuild_cache():
    st.cache_data.clear()
