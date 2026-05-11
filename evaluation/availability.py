"""
QA7 — Availability Measurement (US-07 / T-07.07)

Computes Lambda availability from CloudWatch metrics or fallback estimation.

Availability = (TotalInvocations - Errors) / TotalInvocations

When CloudWatch is unavailable (local/dev), uses a synthetic approach:
  - Invokes each Lambda handler in-process with sample events
  - Measures success rate over N trials

Threshold: Availability > 0.99 (99%).
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from evaluation.cloudwatch_utils import query_all_lambda_metrics

logger = logging.getLogger(__name__)

AVAILABILITY_THRESHOLD = 0.99
ROOT = Path(__file__).parent.parent
LAMBDA_PATHS = {
    "water-twin-anomaly": ROOT / "lambdas" / "anomaly_detection" / "lambda_function.py",
    "water-twin-prediction": ROOT / "lambdas" / "failure_prediction" / "lambda_function.py",
    "water-twin-digital-twin": ROOT / "lambdas" / "digital_twin" / "lambda_function.py",
}


def _load_lambda_module(label: str, path: Path):
    spec = importlib.util.spec_from_file_location(label, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic_invoke(
    module,
    function_name: str,
    n_trials: int = 20,
) -> Dict[str, Any]:
    successes = 0
    errors = 0
    latencies_ms = []

    for i in range(n_trials):
        try:
            t0 = time.perf_counter()

            if function_name == "water-twin-anomaly":
                event = {"Records": []}
            elif function_name == "water-twin-prediction":
                event = {}
            elif function_name == "water-twin-digital-twin":
                event = {}
            else:
                event = {}

            result = module.lambda_handler(event, None)
            elapsed = (time.perf_counter() - t0) * 1000
            latencies_ms.append(elapsed)

            if isinstance(result, dict) and result.get("statusCode", 500) < 500:
                successes += 1
            else:
                errors += 1
        except Exception:
            errors += 1

    total = successes + errors
    availability = successes / max(total, 1)

    return {
        "successes": successes,
        "errors": errors,
        "total": total,
        "availability": round(availability, 6),
        "avg_latency_ms": round(sum(latencies_ms) / max(len(latencies_ms), 1), 2) if latencies_ms else 0,
    }


def measure_availability(
    use_cloudwatch: bool = True,
    n_trials: int = 20,
) -> Dict[str, Any]:
    t_start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("QA7 — Availability Measurement")
    logger.info(f"  Source      : {'CloudWatch' if use_cloudwatch else 'Synthetic invocation'}")
    logger.info(f"  Threshold   : > {AVAILABILITY_THRESHOLD} ({AVAILABILITY_THRESHOLD * 100:.1f}%)")
    logger.info("=" * 60)

    if use_cloudwatch:
        cw_metrics = query_all_lambda_metrics(period=300, hours_back=1)
        if cw_metrics:
            per_function = {}
            total_invocations = 0
            total_errors = 0
            passed = True

            for name, metrics in cw_metrics.items():
                avail = metrics.availability_pct
                per_function[name] = {
                    "invocations": metrics.invocations,
                    "errors": metrics.errors,
                    "availability": round(avail, 6),
                    "avg_duration_ms": round(metrics.duration_avg_ms, 2),
                    "p95_duration_ms": round(metrics.duration_p95_ms, 2),
                    "init_duration_avg_ms": (
                        round(metrics.init_duration_avg_ms, 2)
                        if metrics.init_duration_avg_ms else None
                    ),
                    "throttles": metrics.throttles,
                }
                total_invocations += metrics.invocations
                total_errors += metrics.errors
                fn_passed = avail >= AVAILABILITY_THRESHOLD
                if not fn_passed:
                    passed = False

                logger.info(
                    f"  [{name}] invocations={metrics.invocations} "
                    f"errors={metrics.errors} "
                    f"availability={avail:.4%} "
                    f"{'PASS' if fn_passed else 'FAIL'}"
                )

            overall_avail = (
                (total_invocations - total_errors) / max(total_invocations, 1)
            )
            passed = passed and overall_avail >= AVAILABILITY_THRESHOLD

            result = {
                "source": "CloudWatch",
                "per_function": per_function,
                "total_invocations": total_invocations,
                "total_errors": total_errors,
                "overall_availability": round(overall_avail, 6),
                "threshold": AVAILABILITY_THRESHOLD,
                "passed": passed,
            }
        else:
            logger.warning("CloudWatch returned no data — falling back to synthetic")
            return measure_availability(use_cloudwatch=False, n_trials=n_trials)
    else:
        per_function = {}
        passed = True

        for name, path in LAMBDA_PATHS.items():
            if not path.exists():
                logger.warning(f"  [{name}] Lambda file not found: {path}")
                per_function[name] = {"error": "File not found"}
                passed = False
                continue

            try:
                mod = _load_lambda_module(f"eval_{name}", path)
                # Patch environment for testing
                import os
                os.environ.setdefault("DYNAMO_TABLE", "water-twin-data")
                os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

                synth = _synthetic_invoke(mod, name, n_trials=n_trials)
                per_function[name] = synth
                fn_passed = synth["availability"] >= AVAILABILITY_THRESHOLD
                if not fn_passed:
                    passed = False

                logger.info(
                    f"  [{name}] successes={synth['successes']}/{synth['total']} "
                    f"availability={synth['availability']:.4%} "
                    f"avg_latency={synth['avg_latency_ms']}ms "
                    f"{'PASS' if fn_passed else 'FAIL'}"
                )
            except Exception as e:
                logger.error(f"  [{name}] Synthetic invoke failed: {e}")
                per_function[name] = {"error": str(e)}
                passed = False

        overall = [m for m in per_function.values() if "availability" in m]
        overall_avail = (
            sum(m["availability"] for m in overall) / max(len(overall), 1)
            if overall else 0.0
        )
        passed = passed and overall_avail >= AVAILABILITY_THRESHOLD

        result = {
            "source": "synthetic",
            "per_function": per_function,
            "overall_availability": round(overall_avail, 6),
            "threshold": AVAILABILITY_THRESHOLD,
            "passed": passed,
        }

    result["duration_seconds"] = round(time.perf_counter() - t_start, 3)
    result["timestamp"] = datetime.now(timezone.utc).isoformat()

    status = "PASS" if result["passed"] else "FAIL"
    logger.info(f"  QA7 result: {status}")
    logger.info(f"  Overall availability = {result['overall_availability']:.4%} "
                f"(threshold > {AVAILABILITY_THRESHOLD:.0%})")

    return result
