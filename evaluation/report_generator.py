"""
Report Generator — Consolidates QA1-QA7 results into evaluation_report.json.

Implements persistence layer for metric results following the existing
report format used by measure_metrics.py in the project root.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REPORT_PATH = Path(__file__).parent / "evaluation_report.json"

QA_INFO: Dict[str, Dict[str, str]] = {
    "QA1_latency": {
        "description": "End-to-end latency P95 < 2 segundos (ingest → twin update)",
        "table_ref": "Tabla 4 — Quality Attribute Scenarios, QA1",
    },
    "QA2_throughput": {
        "description": "Throughput > 950 eventos/min bajo carga concurrente de 2 tenants",
        "table_ref": "Tabla 4 — Quality Attribute Scenarios, QA2",
    },
    "QA3_scalability": {
        "description": "Degradación P95 < 20% al escalar de 1000 a 5000 eventos/min",
        "table_ref": "Tabla 4 — Quality Attribute Scenarios, QA3",
    },
    "QA4_recall_if": {
        "description": "Recall Isolation Forest > 0.80 contra ground truth",
        "table_ref": "Tabla 4 — Quality Attribute Scenarios, QA4",
    },
    "QA5_f1_ensemble": {
        "description": "F1-score ensemble RF+XGBoost > 0.75 en test split",
        "table_ref": "Tabla 4 — Quality Attribute Scenarios, QA5",
    },
    "QA6_tenant_isolation": {
        "description": "Cross-tenant leak_count = 0 bajo carga concurrente (EP-05)",
        "table_ref": "Tabla 4 / Tabla 2 — Multi-tenancy isolation mechanisms",
    },
    "QA7_availability": {
        "description": "Disponibilidad Lambda > 99% desde CloudWatch",
        "table_ref": "Tabla 4 — Quality Attribute Scenarios, QA7",
    },
}


def load_report() -> Dict[str, Any]:
    if REPORT_PATH.exists():
        try:
            with open(REPORT_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load existing report: {e}")
    return {
        "framework": "WaterTwinML",
        "paper": "TDSE — Sistema de Gemelo Digital para Redes Hídricas",
        "metrics": {},
    }


def save_report(report: Dict[str, Any]) -> None:
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Report saved → {REPORT_PATH}")


def record_metric(
    key: str,
    result: Dict[str, Any],
    description: Optional[str] = None,
    table_ref: Optional[str] = None,
) -> None:
    info = QA_INFO.get(key, {})
    report = load_report()
    report["metrics"][key] = {
        "description": description or info.get("description", ""),
        "table_ref": table_ref or info.get("table_ref", ""),
        "threshold": result.get("threshold", result.get("threshold_pct", "N/A")),
        "passed": result.get("passed", False),
        "result": result,
    }
    save_report(report)


def generate_summary_report(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    summary = {
        "framework": "WaterTwinML",
        "paper": "TDSE — Sistema de Gemelo Digital para Redes Hídricas",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {},
        "summary": {
            "total": len(results),
            "passed": 0,
            "failed": 0,
        },
    }

    for key, result in results.items():
        passed = result.get("passed", False)
        if passed:
            summary["summary"]["passed"] += 1
        else:
            summary["summary"]["failed"] += 1

        info = QA_INFO.get(key, {})
        summary["metrics"][key] = {
            "description": info.get("description", ""),
            "table_ref": info.get("table_ref", ""),
            "threshold": result.get("threshold", "N/A"),
            "passed": passed,
            "result": result,
        }

    save_report(summary)

    return summary


def print_console_summary(results: Dict[str, Dict[str, Any]]) -> None:
    print()
    print("=" * 60)
    print("  WaterTwinML — Evaluation Summary (QA1-QA7)")
    print("=" * 60)

    passed_count = 0
    failed_count = 0

    order = [
        "QA1_latency", "QA2_throughput", "QA3_scalability",
        "QA4_recall_if", "QA5_f1_ensemble",
        "QA6_tenant_isolation", "QA7_availability",
    ]

    for key in order:
        result = results.get(key)
        if result is None:
            print(f"  {key:25s}   SKIPPED  (not measured)")
            continue

        passed = result.get("passed", False)
        status = "PASS" if passed else "FAIL"
        duration = result.get("duration_seconds", 0)
        info = QA_INFO.get(key, {})
        label = info.get("description", key)[:55]

        if passed:
            passed_count += 1
        else:
            failed_count += 1

        status_color = "\033[92m" if passed else "\033[91m"
        reset = "\033[0m"

        print(f"  {status_color}{status:6s}{reset}  {label:55s}  ({duration:.1f}s)")

    print("=" * 60)
    total = passed_count + failed_count
    print(f"  Results: {passed_count} PASSED / {failed_count} FAILED / {total} TOTAL")
    print("=" * 60)
    print()

    return failed_count == 0
