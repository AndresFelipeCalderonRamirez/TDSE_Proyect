"""
WebSocket Broadcaster Lambda

Handles two event types:
  1. API Gateway WebSocket routes ($connect, $disconnect)
  2. DynamoDB Streams trigger → broadcasts new sensor records to all connected clients

Connections are stored in DynamoDB table water-twin-ws-connections.
"""

import json
import os
import logging
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

CONNECTIONS_TABLE = os.getenv("WS_CONNECTIONS_TABLE", "water-twin-ws-connections")
WS_ENDPOINT = os.getenv("WS_ENDPOINT", "")  # https://{api-id}.execute-api.{region}.amazonaws.com/{stage}
REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")


def _dynamo():
    return boto3.client("dynamodb", region_name=REGION)


def _apigw_management():
    return boto3.client("apigatewaymanagementapi", endpoint_url=WS_ENDPOINT, region_name=REGION)


# ── WebSocket route handlers ──────────────────────────────────────────────────

def handle_connect(connection_id: str, tenant_id: str) -> dict:
    _dynamo().put_item(
        TableName=CONNECTIONS_TABLE,
        Item={
            "connectionId": {"S": connection_id},
            "tenantId":     {"S": tenant_id or "tenant-A"},
        },
    )
    logger.info(f"Connected: {connection_id} tenant={tenant_id}")
    return {"statusCode": 200, "body": "Connected"}


def handle_disconnect(connection_id: str) -> dict:
    _dynamo().delete_item(
        TableName=CONNECTIONS_TABLE,
        Key={"connectionId": {"S": connection_id}},
    )
    logger.info(f"Disconnected: {connection_id}")
    return {"statusCode": 200, "body": "Disconnected"}


# ── DynamoDB Stream broadcaster ───────────────────────────────────────────────

def _parse_dynamo_item(item: dict) -> dict:
    result = {}
    for k, v in item.items():
        if "S" in v:
            result[k] = v["S"]
        elif "N" in v:
            raw = v["N"]
            result[k] = float(raw) if "." in raw else int(raw)
        elif "BOOL" in v:
            result[k] = v["BOOL"]
    return result


def _get_connections(tenant_id: str) -> list:
    dynamo = _dynamo()
    items = []
    kwargs = {
        "TableName": CONNECTIONS_TABLE,
        "FilterExpression": "tenantId = :tid",
        "ExpressionAttributeValues": {":tid": {"S": tenant_id}},
    }
    while True:
        resp = dynamo.scan(**kwargs)
        items.extend(resp.get("Items", []))
        if "LastEvaluatedKey" not in resp:
            break
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    return [i["connectionId"]["S"] for i in items]


def broadcast_record(record: dict):
    """Send a new sensor record to all WebSocket clients of the same tenant."""
    if not WS_ENDPOINT:
        logger.warning("WS_ENDPOINT not set — skipping broadcast")
        return

    tenant_id = record.get("tenantId", "tenant-A")
    connections = _get_connections(tenant_id)
    if not connections:
        return

    payload = json.dumps({"type": "sensor_record", "data": record}).encode()
    apigw = _apigw_management()
    stale = []

    for cid in connections:
        try:
            apigw.post_to_connection(ConnectionId=cid, Data=payload)
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("GoneException", "410"):
                stale.append(cid)
            else:
                logger.error(f"postToConnection {cid}: {e}")

    # Clean up stale connections
    if stale:
        dynamo = _dynamo()
        for cid in stale:
            dynamo.delete_item(
                TableName=CONNECTIONS_TABLE,
                Key={"connectionId": {"S": cid}},
            )
        logger.info(f"Removed {len(stale)} stale connections")


def handle_stream_event(event: dict):
    """Process DynamoDB Stream records and broadcast new sensor records."""
    for rec in event.get("Records", []):
        if rec.get("eventName") != "INSERT":
            continue
        new_image = rec.get("dynamodb", {}).get("NewImage", {})
        if not new_image:
            continue

        parsed = _parse_dynamo_item(new_image)

        # Only broadcast sensor records (exclude topology, rankings)
        sort_key = parsed.get("sortKey", "")
        if sort_key.startswith("topology") or sort_key.startswith("twin_ranking"):
            continue

        # Only broadcast anomalies to keep bandwidth low
        if not parsed.get("isAnomaly", False):
            continue

        broadcast_record(parsed)


# ── Main handler ──────────────────────────────────────────────────────────────

def lambda_handler(event, context):
    # API Gateway WebSocket event
    if "requestContext" in event:
        ctx = event["requestContext"]
        route = ctx.get("routeKey", "$default")
        connection_id = ctx.get("connectionId", "")
        qs = event.get("queryStringParameters") or {}
        tenant_id = qs.get("tenant_id", "tenant-A")

        if route == "$connect":
            return handle_connect(connection_id, tenant_id)
        elif route == "$disconnect":
            return handle_disconnect(connection_id)
        else:
            return {"statusCode": 200, "body": "OK"}

    # DynamoDB Stream event
    if "Records" in event and event["Records"] and "dynamodb" in event["Records"][0]:
        handle_stream_event(event)
        return {"statusCode": 200}

    return {"statusCode": 400, "body": "Unknown event type"}
