"""
Digital Twin Lambda — EP-04 / US-06
T-06.01 + T-06.02 + T-06.03

Trigger : EventBridge rate(1 minute)

Flow (per tenant):
  1. Load topology from DynamoDB (PK=tenantId, SK=topology#config)
     → NetworkX G=(V, E)  [T-06.01]
  2. Load latest p_failure per segment from DynamoDB
     → initial risk vector {segmentId: p_failure}
  3. BFS risk propagation (Ecuación 3, paper Section 5.4)  [T-06.02]
     Risk(ej) = max(Risk(ej), α·Risk(ei)),  α = 0.7 uniform
  4. Persist top-k ranking to DynamoDB  [T-06.03]
     PK=tenantId, SK=twin_ranking_{timestamp}

Stateless by design: graph is rebuilt every invocation from DynamoDB.
The DynamoDB state IS the digital twin state.
In production this would use AWS Neptune + streaming updates.
"""

import heapq
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import boto3
import networkx as nx

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ── Constants ─────────────────────────────────────────────────────────────────

ALPHA: float      = float(os.getenv("RISK_PROPAGATION_ALPHA", "0.7"))
TOP_K: int        = int(os.getenv("RANKING_TOP_K", "5"))
RISK_WINDOW_H: int = int(os.getenv("RISK_WINDOW_HOURS", "2"))
TENANTS: List[str] = ["tenant-A", "tenant-B"]

# ── Module-level config (warm reuse) ──────────────────────────────────────────

_config: Dict[str, str] = {}


def _load_config() -> None:
    global _config
    _config = {
        "table_name": os.environ["DYNAMO_TABLE"],
        "region":     os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    }


# ── T-06.01  Topology → NetworkX graph ───────────────────────────────────────

def _load_topology(tenant_id: str) -> nx.Graph:
    """
    Load pipe network topology stored in DynamoDB.

    Item schema:
      PK  = tenantId      (e.g. "tenant-A")
      SK  = topology#config
      nodes = JSON list of segment IDs
      edges = JSON list of [src, dst] pairs
    """
    dynamodb = boto3.client("dynamodb", region_name=_config["region"])
    resp = dynamodb.get_item(
        TableName=_config["table_name"],
        Key={
            "tenantId": {"S": tenant_id},
            "sortKey":  {"S": "topology#config"},
        },
    )
    item = resp.get("Item", {})
    if not item:
        logger.warning(f"[{tenant_id}] No topology in DynamoDB — empty graph")
        return nx.Graph()

    nodes: List[str]       = json.loads(item["nodes"]["S"])
    edges: List[List[str]] = json.loads(item["edges"]["S"])

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # AC2: log node / edge counts for CloudWatch verification
    logger.info(
        f"[{tenant_id}] Topology: {len(G.nodes())} nodes, {len(G.edges())} edges"
    )
    return G


# ── Risk loading ──────────────────────────────────────────────────────────────

def _load_segment_risks(tenant_id: str) -> Dict[str, float]:
    """
    Query DynamoDB for the most recent p_failure per segment
    within the last RISK_WINDOW_H hours.

    Sort key format: {ISO-timestamp}#{segmentId}
    ISO timestamps compare correctly as strings, so SK >= cutoff_ts
    returns only recent records.

    Non-sensor items (topology#*, twin_ranking_*) don't have p_failure
    and are excluded by FilterExpression.
    """
    dynamodb = boto3.client("dynamodb", region_name=_config["region"])
    table    = _config["table_name"]
    cutoff   = (
        datetime.now(timezone.utc) - timedelta(hours=RISK_WINDOW_H)
    ).isoformat()

    kwargs: Dict[str, Any] = {
        "TableName":               table,
        "KeyConditionExpression":  "tenantId = :tid AND sortKey >= :cutoff",
        "FilterExpression": (
            "processed_by_prediction = :done "
            "AND attribute_exists(p_failure)"
        ),
        "ExpressionAttributeValues": {
            ":tid":    {"S": tenant_id},
            ":cutoff": {"S": cutoff},
            ":done":   {"BOOL": True},
        },
    }

    risks: Dict[str, float] = {}
    while True:
        resp = dynamodb.query(**kwargs)
        for it in resp.get("Items", []):
            seg_id = it.get("segmentId", {}).get("S", "")
            p_fail = float(it.get("p_failure", {}).get("N", "0"))
            if seg_id:
                risks[seg_id] = max(risks.get(seg_id, 0.0), p_fail)
        last_key = resp.get("LastEvaluatedKey")
        if not last_key:
            break
        kwargs["ExclusiveStartKey"] = last_key

    logger.info(
        f"[{tenant_id}] Risk scores loaded: {len(risks)} segments with p_failure"
    )
    return risks


# ── T-06.02  BFS Risk Propagation (Ecuación 3) ───────────────────────────────

def _propagate_risk(
    G: nx.Graph,
    initial_risks: Dict[str, float],
    alpha: float = ALPHA,
) -> Dict[str, float]:
    """
    BFS risk propagation — Ecuación 3, paper Section 5.4.

        Risk(ej) = max(Risk(ej), α · Risk(ei))

    Uses a max-priority queue so the highest-risk node propagates first.
    The max operator guarantees:
      - Risk is monotone non-decreasing along any propagation path.
      - A node's risk is only ever raised, never lowered.
      - Convergence is guaranteed because each node is finalised at most once
        (lazy deletion removes stale queue entries).

    α = 0.7 → decays exponentially: 1.0 → 0.7 → 0.49 → 0.343 …
    Beyond ~10 hops the contribution drops below 0.03, bounding propagation.

    Time complexity: O((V + E) log V)
    """
    risk: Dict[str, float] = {
        node: initial_risks.get(node, 0.0) for node in G.nodes()
    }

    # Python heapq is min-heap; negate to get max-heap behaviour
    heap: List[Tuple[float, str]] = [
        (-r, node) for node, r in risk.items() if r > 0.0
    ]
    heapq.heapify(heap)

    while heap:
        neg_r, node = heapq.heappop(heap)
        curr_risk = -neg_r

        # Stale entry: node was already updated to a higher risk value
        if curr_risk < risk[node]:
            continue

        for neighbor in G.neighbors(node):
            propagated = alpha * curr_risk
            if propagated > risk[neighbor]:
                risk[neighbor] = propagated
                heapq.heappush(heap, (-propagated, neighbor))

    return risk


# ── T-06.03  Ranking persistence ──────────────────────────────────────────────

def _save_ranking(
    tenant_id: str,
    risk: Dict[str, float],
    top_k: int = TOP_K,
) -> None:
    """
    Persist top-k maintenance ranking to DynamoDB.

    Item schema (AC4):
      PK           = tenantId
      SK           = twin_ranking_{ISO-timestamp}
      generated_at = ISO timestamp
      top_k        = k
      segments     = JSON [{segment_id, risk_propagated}, ...] sorted desc
      alpha        = propagation coefficient used
    """
    dynamodb = boto3.client("dynamodb", region_name=_config["region"])
    table    = _config["table_name"]
    now_iso  = datetime.now(timezone.utc).isoformat()
    sort_key = f"twin_ranking_{now_iso}"

    ranking = sorted(risk.items(), key=lambda x: x[1], reverse=True)[:top_k]
    segments_json = json.dumps(
        [{"segment_id": sid, "risk_propagated": round(r, 6)} for sid, r in ranking]
    )

    dynamodb.put_item(
        TableName=table,
        Item={
            "tenantId":     {"S": tenant_id},
            "sortKey":      {"S": sort_key},
            "generated_at": {"S": now_iso},
            "top_k":        {"N": str(top_k)},
            "segments":     {"S": segments_json},
            "alpha":        {"N": str(ALPHA)},
        },
    )

    top_str = ", ".join(f"{sid}={r:.4f}" for sid, r in ranking)
    logger.info(f"[{tenant_id}] Ranking persisted ({sort_key}) — top-{top_k}: {top_str}")


# ── Lambda handler ─────────────────────────────────────────────────────────────

def lambda_handler(event: Dict, context: Any) -> Dict:
    """
    EventBridge invokes this Lambda every 60 seconds.
    Rebuilds graph from DynamoDB each invocation (stateless design).
    """
    _load_config()

    results: Dict[str, Any] = {}

    for tenant_id in TENANTS:
        try:
            # T-06.01: build NetworkX graph from DynamoDB topology
            G = _load_topology(tenant_id)
            if G.number_of_nodes() == 0:
                logger.warning(f"[{tenant_id}] Empty graph — skipping propagation")
                results[tenant_id] = {"skipped": True, "reason": "empty topology"}
                continue

            # Load initial risk vector from prediction Lambda output
            initial_risks = _load_segment_risks(tenant_id)

            # T-06.02: BFS risk propagation (Ecuación 3)
            propagated_risk = _propagate_risk(G, initial_risks, alpha=ALPHA)

            non_zero = sum(1 for r in propagated_risk.values() if r > 0.0)
            logger.info(
                f"[{tenant_id}] Propagation complete — "
                f"{non_zero}/{G.number_of_nodes()} nodes with risk > 0"
            )

            # T-06.03: persist top-k ranking
            _save_ranking(tenant_id, propagated_risk, top_k=TOP_K)

            results[tenant_id] = {
                "nodes":              G.number_of_nodes(),
                "edges":              G.number_of_edges(),
                "segments_with_risk": non_zero,
            }

        except Exception as e:
            logger.error(f"[{tenant_id}] Error: {e}", exc_info=True)
            results[tenant_id] = {"error": str(e)}

    return {
        "statusCode": 200,
        "body": json.dumps(results),
    }
