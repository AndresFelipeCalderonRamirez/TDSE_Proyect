import json
from datetime import datetime
from typing import Any, Dict, List, Optional


def parse_dynamo_record(item: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for key, value in item.items():
        if "S" in value:
            result[key] = value["S"]
        elif "N" in value:
            raw = value["N"]
            result[key] = float(raw) if "." in raw else int(raw)
        elif "BOOL" in value:
            result[key] = value["BOOL"]
        elif "NULL" in value:
            result[key] = None
        else:
            result[key] = str(value)
    return result


def parse_sensor_record(item: Dict[str, Any]) -> Dict[str, Any]:
    record = parse_dynamo_record(item)
    if "metadata" in record and isinstance(record["metadata"], str):
        try:
            record["metadata"] = json.loads(record["metadata"])
        except (json.JSONDecodeError, TypeError):
            record["metadata"] = {}
    return record


def parse_ranking_item(item: Dict[str, Any]) -> Dict[str, Any]:
    parsed = parse_dynamo_record(item)
    if "segments" in parsed and isinstance(parsed["segments"], str):
        try:
            parsed["segments"] = json.loads(parsed["segments"])
        except (json.JSONDecodeError, TypeError):
            parsed["segments"] = []
    return parsed


def get_latest_ranking(items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    ranking_items = [
        parse_ranking_item(it)
        for it in items
        if it.get("sortKey", {}).get("S", "").startswith("twin_ranking_")
    ]
    if not ranking_items:
        return None
    ranking_items.sort(key=lambda x: x.get("generated_at", ""), reverse=True)
    return ranking_items[0]


def get_top_segments(ranking_item: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
    segments = ranking_item.get("segments", [])
    segments.sort(key=lambda x: x.get("risk_propagated", 0), reverse=True)
    return segments[:top_k]


def iso_timestamp_key(ts: str) -> str:
    return ts.replace(":", "-").replace(".", "-")


def parse_timestamp(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
