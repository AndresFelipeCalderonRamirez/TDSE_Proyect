"""
CloudWatch Utilities — Lambda metric retrieval for QA7 Availability.

Reads CloudWatch Metrics for Lambda invocations, errors, and duration.
Falls back gracefully when CloudWatch is unavailable (mocked/local).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

LAMBDA_NAMES = [
    "water-twin-anomaly",
    "water-twin-prediction",
    "water-twin-digital-twin",
]

REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")


@dataclass
class LambdaMetrics:
    name: str
    invocations: int
    errors: int
    duration_avg_ms: float
    duration_p50_ms: float
    duration_p95_ms: float
    duration_p99_ms: float
    init_duration_avg_ms: Optional[float] = None
    throttles: int = 0
    availability: float = 1.0

    @property
    def availability_pct(self) -> float:
        if self.invocations == 0:
            return 1.0
        return (self.invocations - self.errors) / self.invocations


def query_lambda_metrics(
    function_name: str,
    period: int = 300,
    hours_back: int = 1,
) -> Optional[LambdaMetrics]:
    try:
        cw = boto3.client("cloudwatch", region_name=REGION)
    except Exception:
        logger.warning("CloudWatch client unavailable — using fallback metrics")
        return None

    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours_back)
    namespace = "AWS/Lambda"
    dimensions = [{"Name": "FunctionName", "Value": function_name}]

    def _get_stat(metric_name: str, stat: str) -> float:
        try:
            resp = cw.get_metric_statistics(
                Namespace=namespace,
                MetricName=metric_name,
                Dimensions=dimensions,
                StartTime=start,
                EndTime=now,
                Period=period,
                Statistics=[stat],
            )
            points = resp.get("Datapoints", [])
            if not points:
                return 0.0
            return sum(p[stat] for p in points) / len(points)
        except ClientError as e:
            logger.debug(f"CloudWatch {metric_name}/{stat}: {e}")
            return 0.0

    invocations = _get_stat("Invocations", "Sum")
    errors = _get_stat("Errors", "Sum")
    duration_avg = _get_stat("Duration", "Average")
    duration_p50 = _get_stat("Duration", "p50")
    duration_p95 = _get_stat("Duration", "p95")
    duration_p99 = _get_stat("Duration", "p99")
    throttles = _get_stat("Throttles", "Sum")

    init_duration = _get_stat("InitDuration", "Average")

    avail = 1.0
    if invocations > 0:
        avail = (invocations - errors) / invocations

    return LambdaMetrics(
        name=function_name,
        invocations=int(invocations),
        errors=int(errors),
        duration_avg_ms=duration_avg,
        duration_p50_ms=duration_p50,
        duration_p95_ms=duration_p95,
        duration_p99_ms=duration_p99,
        init_duration_avg_ms=init_duration if init_duration > 0 else None,
        throttles=int(throttles),
        availability=avail,
    )


def query_all_lambda_metrics(
    period: int = 300,
    hours_back: int = 1,
) -> Dict[str, LambdaMetrics]:
    results = {}
    for name in LAMBDA_NAMES:
        metrics = query_lambda_metrics(name, period=period, hours_back=hours_back)
        if metrics is not None:
            results[name] = metrics
    return results
