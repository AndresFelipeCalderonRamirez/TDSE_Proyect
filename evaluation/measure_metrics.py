#!/usr/bin/env python3
"""
measure_metrics.py  —  WaterTwinML Evaluation Framework  (EP-07)
===============================================================
Standalone CLI that runs QA1-QA7 metrics, consolidates results,
and generates evaluation_report.json.

Usage:
    # Run all metrics
    python evaluation/measure_metrics.py --all

    # Run individual metrics
    python evaluation/measure_metrics.py --latency
    python evaluation/measure_metrics.py --throughput
    python evaluation/measure_metrics.py --scalability
    python evaluation/measure_metrics.py --ml
    python evaluation/measure_metrics.py --isolation
    python evaluation/measure_metrics.py --availability

    # Custom parameters
    python evaluation/measure_metrics.py --isolation --iterations 50
    python evaluation/measure_metrics.py --ml --tenant tenant-A

Output:
    evaluation/evaluation_report.json
    Exit code 0 if ALL metrics pass; 1 if any fail.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure the project root is on sys.path so evaluation/ can be imported
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from evaluation.latency import measure_latency
from evaluation.throughput import measure_throughput
from evaluation.scalability import measure_scalability
from evaluation.ml_metrics import measure_if_recall, measure_ensemble_f1
from evaluation.isolation import measure_isolation
from evaluation.availability import measure_availability
from evaluation.report_generator import generate_summary_report, print_console_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("measure_metrics")


def run_latency(args) -> Dict[str, Any]:
    return measure_latency(tenant_id=getattr(args, "tenant", None), hours_back=24)


def run_throughput(args) -> Dict[str, Any]:
    return measure_throughput(events_per_tenant=getattr(args, "events_per_tenant", 500))


def run_scalability(args) -> Dict[str, Any]:
    return measure_scalability(
        baseline_events=getattr(args, "baseline_events", 1000),
        stress_events=getattr(args, "stress_events", 5000),
    )


def run_ml(args) -> Dict[str, Any]:
    tenant = getattr(args, "tenant", None)
    results = {}
    recall = measure_if_recall(tenant_id=tenant)
    f1 = measure_ensemble_f1(tenant_id=tenant)
    results["QA4_recall_if"] = recall
    results["QA5_f1_ensemble"] = f1
    return results


def run_isolation(args) -> Dict[str, Any]:
    return measure_isolation(
        iterations=getattr(args, "iterations", 100),
        events_per_tenant=getattr(args, "events_per_tenant", 500),
    )


def run_availability(args) -> Dict[str, Any]:
    return measure_availability(
        use_cloudwatch=not getattr(args, "synthetic", False),
        n_trials=getattr(args, "trials", 20),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WaterTwinML evaluation framework — QA1-QA7 metrics (EP-07)"
    )

    metric_group = parser.add_argument_group("Metrics")
    metric_group.add_argument("--all", action="store_true", help="Run all QA metrics")
    metric_group.add_argument("--latency", action="store_true", help="QA1: End-to-end latency")
    metric_group.add_argument("--throughput", action="store_true", help="QA2: Throughput")
    metric_group.add_argument("--scalability", action="store_true", help="QA3: Scalability")
    metric_group.add_argument("--ml", action="store_true", help="QA4 + QA5: ML metrics")
    metric_group.add_argument("--isolation", action="store_true", help="QA6: Multi-tenant isolation")
    metric_group.add_argument("--availability", action="store_true", help="QA7: Availability")

    param_group = parser.add_argument_group("Parameters")
    param_group.add_argument("--tenant", type=str, default=None,
                             choices=["tenant-A", "tenant-B"],
                             help="Tenant filter for ML metrics")
    param_group.add_argument("--iterations", type=int, default=100,
                             help="Cross-tenant read iterations (QA6, default: 100)")
    param_group.add_argument("--events-per-tenant", type=int, default=500,
                             dest="events_per_tenant",
                             help="Events per tenant for throughput/isolation tests")
    param_group.add_argument("--baseline-events", type=int, default=1000,
                             dest="baseline_events",
                             help="Baseline event count for scalability (QA3)")
    param_group.add_argument("--stress-events", type=int, default=5000,
                             dest="stress_events",
                             help="Stress event count for scalability (QA3)")
    param_group.add_argument("--synthetic", action="store_true",
                             help="Use synthetic invocation instead of CloudWatch (QA7)")
    param_group.add_argument("--trials", type=int, default=20,
                             help="Synthetic invocation trials (QA7, default: 20)")

    args = parser.parse_args()

    has_metric = any([
        args.all, args.latency, args.throughput, args.scalability,
        args.ml, args.isolation, args.availability,
    ])
    if not has_metric:
        parser.print_help()
        sys.exit(0)

    results: Dict[str, Dict[str, Any]] = {}

    if args.all:
        logger.info("=" * 60)
        logger.info("WaterTwinML — Full Evaluation (QA1-QA7)")
        logger.info("=" * 60)

        results["QA1_latency"] = run_latency(args)
        results["QA2_throughput"] = run_throughput(args)
        results["QA3_scalability"] = run_scalability(args)
        ml_results = run_ml(args)
        if isinstance(ml_results, dict):
            for k, v in ml_results.items():
                results[k] = v
        results["QA6_tenant_isolation"] = run_isolation(args)
        results["QA7_availability"] = run_availability(args)
    else:
        if args.latency:
            results["QA1_latency"] = run_latency(args)
        if args.throughput:
            results["QA2_throughput"] = run_throughput(args)
        if args.scalability:
            results["QA3_scalability"] = run_scalability(args)
        if args.ml:
            ml_results = run_ml(args)
            if isinstance(ml_results, dict):
                for k, v in ml_results.items():
                    results[k] = v
        if args.isolation:
            results["QA6_tenant_isolation"] = run_isolation(args)
        if args.availability:
            results["QA7_availability"] = run_availability(args)

    all_passed = print_console_summary(results)
    generate_summary_report(results)

    if all_passed:
        logger.info("All metrics PASSED.")
        sys.exit(0)
    else:
        logger.error("One or more metrics FAILED — see evaluation_report.json for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
