"""
QA4 + QA5 — ML Model Evaluation (US-07 / T-07.04 + T-07.05)

QA4: Isolation Forest Recall
  - Compares ground truth (anomaly_ground_truth.csv) vs isAnomaly in DynamoDB
  - Computes: precision, recall, F1, ROC-AUC, MCC
  - Threshold: Recall > 0.80

QA5: Ensemble RF+XGBoost F1
  - Loads trained RF, XGBoost, scaler, beta from model artifacts
  - Evaluates on held-out test data (last 20% time-sorted)
  - Computes: F1, ROC-AUC, confusion matrix
  - Threshold: F1 > 0.75
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from evaluation.dynamo_queries import get_sensor_records

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent

RECALL_THRESHOLD = 0.80
F1_THRESHOLD = 0.75

FEATURE_COLS = [
    "pressure", "flow", "vibration", "delta_pressure",
    "diameter", "material_enc", "length", "age",
]

MATERIAL_ENCODING = {
    "PVC": 0, "HDPE": 0, "Steel": 0,
    "AC": 1,
    "DI": 2, "Ductile Iron": 2,
    "CI": 3, "Cast Iron": 3,
}

TENANTS = ["tenant-A", "tenant-B"]


# ── QA4: Isolation Forest Recall ────────────────────────────────────────────

def _load_ground_truth(tenant_id: str) -> Optional[List[Dict[str, Any]]]:
    gt_path = ROOT / "data" / "outputs" / f"anomaly_ground_truth_{tenant_id}.csv"
    if not gt_path.exists():
        gt_path = ROOT / "data" / "outputs" / "anomaly_ground_truth.csv"
    if not gt_path.exists():
        logger.warning(f"Ground truth CSV not found for {tenant_id}")
        return None

    records = []
    try:
        with open(gt_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
    except (OSError, csv.Error) as e:
        logger.error(f"Failed to read ground truth: {e}")
        return None
    return records


def _match_dynamodb_to_ground_truth(
    dynamo_records: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    gt_map = {}
    for gt in ground_truth:
        key = (gt.get("segmentId", ""), gt.get("timestamp", ""))
        gt_map[key] = gt.get("is_anomaly", "False").lower() in ("true", "1", "yes")

    y_true = []
    y_pred = []

    for dr in dynamo_records:
        seg_id = dr.get("segmentId", "")
        ts = dr.get("timestamp", "")
        key = (seg_id, ts)
        if key in gt_map:
            y_true.append(1 if gt_map[key] else 0)
            y_pred.append(1 if dr.get("isAnomaly", False) else 0)

    return np.array(y_true), np.array(y_pred)


def _compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics = {}

    if len(y_true) == 0:
        return {"error": "No matching records between ground truth and DynamoDB"}

    metrics["accuracy"] = round(float(accuracy_score(y_true, y_pred)), 4)
    metrics["precision"] = round(float(precision_score(y_true, y_pred, zero_division=0)), 4)
    metrics["recall"] = round(float(recall_score(y_true, y_pred, zero_division=0)), 4)
    metrics["f1"] = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
    metrics["mcc"] = round(float(matthews_corrcoef(y_true, y_pred)), 4)
    metrics["total_records"] = int(len(y_true))
    metrics["positive_class"] = int(np.sum(y_true))
    metrics["predicted_positive"] = int(np.sum(y_pred))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["confusion_matrix"] = {
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
    }

    if y_score is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 4)
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None

    return metrics


def measure_if_recall(tenant_id: Optional[str] = None) -> Dict[str, Any]:
    t_start = time.perf_counter()
    tenants = [tenant_id] if tenant_id else TENANTS

    logger.info("=" * 60)
    logger.info("QA4 — Isolation Forest Recall")
    logger.info(f"  Threshold: Recall > {RECALL_THRESHOLD}")
    logger.info("=" * 60)

    all_metrics = {}
    global_passed = True

    for tid in tenants:
        ground_truth = _load_ground_truth(tid)
        if ground_truth is None:
            logger.warning(f"  [{tid}] Ground truth unavailable")
            all_metrics[tid] = {"error": "Ground truth CSV not found"}
            global_passed = False
            continue

        dynamo_records = get_sensor_records(tid, hours_back=72)
        if not dynamo_records:
            logger.warning(f"  [{tid}] No DynamoDB records found")
            all_metrics[tid] = {"error": "No DynamoDB records"}
            global_passed = False
            continue

        y_true, y_pred = _match_dynamodb_to_ground_truth(dynamo_records, ground_truth)
        metrics = _compute_classification_metrics(y_true, y_pred)
        all_metrics[tid] = metrics

        recall = metrics.get("recall", 0)
        passed = recall >= RECALL_THRESHOLD
        if not passed:
            global_passed = False

        logger.info(
            f"  [{tid}] Recall={metrics.get('recall', 'N/A'):.4f} "
            f"Precision={metrics.get('precision', 'N/A'):.4f} "
            f"F1={metrics.get('f1', 'N/A'):.4f} "
            f"MCC={metrics.get('mcc', 'N/A'):.4f} "
            f"{'PASS' if passed else 'FAIL'}"
        )

    passed = global_passed and all(
        m.get("recall", 0) >= RECALL_THRESHOLD
        for m in all_metrics.values()
        if "recall" in m
    )

    result = {
        "per_tenant": all_metrics,
        "passed": passed,
        "threshold": RECALL_THRESHOLD,
        "duration_seconds": round(time.perf_counter() - t_start, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    status = "PASS" if passed else "FAIL"
    logger.info(f"  QA4 result: {status}")

    return result


# ── QA5: Ensemble F1 ────────────────────────────────────────────────────────

def _models_dir(tenant_id: str) -> Path:
    short = tenant_id.split("-")[1].lower()
    return ROOT / "models" / f"tenant-{short}"


def _artifacts_present(tenant_id: str) -> bool:
    d = _models_dir(tenant_id)
    return all(
        (d / f).exists()
        for f in ("rf_model.pkl", "xgb_model.pkl", "scaler.pkl", "beta.json")
    )


def _load_models(tenant_id: str):
    import joblib

    d = _models_dir(tenant_id)
    rf = joblib.load(d / "rf_model.pkl")
    xgb = joblib.load(d / "xgb_model.pkl")
    scaler = joblib.load(d / "scaler.pkl")
    with open(d / "beta.json") as f:
        beta = float(json.load(f)["beta"])
    return rf, xgb, scaler, beta


def _build_features(records: List[Dict]) -> np.ndarray:
    import pandas as pd

    df = pd.DataFrame(records)

    if "metadata" in df.columns:
        meta_df = df["metadata"].apply(
            lambda m: m if isinstance(m, dict) else {}
        ).apply(pd.Series)
        for col in ("diameter", "length", "age", "material"):
            if col not in df.columns and col in meta_df.columns:
                df[col] = meta_df[col]

    df["material_enc"] = (
        df.get("material", pd.Series(["PVC"] * len(df)))
        .map(MATERIAL_ENCODING)
        .fillna(0)
        .astype(int)
    )

    if "segment_id" in df.columns:
        df = df.sort_values(["segment_id", "timestamp"])
        df["delta_pressure"] = (
            df.groupby("segment_id")["pressure"]
            .transform(lambda x: x.diff().fillna(0.0))
        )
    else:
        df["delta_pressure"] = 0.0

    defaults = {
        "pressure": 0.0, "flow": 0.0, "vibration": 0.0,
        "delta_pressure": 0.0, "diameter": 150.0,
        "material_enc": 0, "length": 300.0, "age": 25.0,
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val

    return df[FEATURE_COLS].fillna(0.0).to_numpy(dtype=np.float32)


def _build_labels(records: List[Dict]) -> np.ndarray:
    import pandas as pd

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if "is_anomaly" not in df.columns:
        return np.zeros(len(df), dtype=np.int8)

    if "segment_id" not in df.columns:
        return df["is_anomaly"].to_numpy(dtype=np.int8)

    df = df.sort_values(["segment_id", "timestamp"]).reset_index(drop=True)
    WINDOW_STEPS = 288
    THRESHOLD = 0.10

    labels = np.zeros(len(df), dtype=np.int8)
    for _, grp in df.groupby("segment_id"):
        idx = grp.index.to_numpy()
        anom = grp["is_anomaly"].to_numpy(dtype=np.float32)
        n = len(anom)
        cumsum = np.concatenate(([0.0], np.cumsum(anom)))
        for i in range(n):
            end = min(i + WINDOW_STEPS, n)
            rate = (cumsum[end] - cumsum[i]) / max(end - i, 1)
            labels[idx[i]] = 1 if rate > THRESHOLD else 0

    return labels


def _data_path(tenant_id: str) -> Path:
    return ROOT / "data" / "outputs" / f"{tenant_id}_sensor_data.json"


def measure_ensemble_f1(tenant_id: Optional[str] = None) -> Dict[str, Any]:
    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

    t_start = time.perf_counter()
    tenants = [tenant_id] if tenant_id else TENANTS
    TEST_FRACTION = 0.20

    logger.info("=" * 60)
    logger.info("QA5 — Ensemble F1 (RF + XGBoost)")
    logger.info(f"  Threshold: F1 > {F1_THRESHOLD}")
    logger.info("=" * 60)

    all_metrics = {}
    global_passed = True

    for tid in tenants:
        if not _artifacts_present(tid):
            logger.warning(f"  [{tid}] Model artifacts not found in {_models_dir(tid)}")
            all_metrics[tid] = {"error": "Model artifacts not found"}
            global_passed = False
            continue

        data_path = _data_path(tid)
        if not data_path.exists():
            logger.warning(f"  [{tid}] Training data not found: {data_path}")
            all_metrics[tid] = {"error": "Training data not found"}
            global_passed = False
            continue

        with open(data_path, encoding="utf-8") as f:
            records = json.load(f)

        records.sort(key=lambda r: (r.get("segment_id", ""), r.get("timestamp", "")))
        split = int(len(records) * (1.0 - TEST_FRACTION))
        test_records = records[split:]

        if len(test_records) == 0:
            logger.warning(f"  [{tid}] Test split is empty")
            all_metrics[tid] = {"error": "Empty test split"}
            global_passed = False
            continue

        X_test = _build_features(test_records)
        y_test = _build_labels(test_records)

        rf, xgb_mdl, scaler, beta = _load_models(tid)
        X_sc = scaler.transform(X_test)
        p_rf = rf.predict_proba(X_sc)[:, 1]
        p_xgb = xgb_mdl.predict_proba(X_sc)[:, 1]
        p_ensemble = beta * p_rf + (1.0 - beta) * p_xgb
        y_pred = (p_ensemble >= 0.5).astype(int)

        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel().tolist()

        roc_auc = None
        if len(np.unique(y_test)) > 1:
            try:
                roc_auc = float(roc_auc_score(y_test, p_ensemble))
            except Exception:
                pass

        passed = f1 >= F1_THRESHOLD
        if not passed:
            global_passed = False

        all_metrics[tid] = {
            "f1": round(f1, 4),
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "beta": beta,
            "confusion_matrix": {
                "tn": int(cm[0]), "fp": int(cm[1]),
                "fn": int(cm[2]), "tp": int(cm[3]),
            },
            "test_records": len(test_records),
            "passed": passed,
        }

        logger.info(
            f"  [{tid}] F1={f1:.4f} ROC-AUC={roc_auc or 'N/A'} "
            f"beta={beta:.2f} test_records={len(test_records)} "
            f"{'PASS' if passed else 'FAIL'}"
        )

    passed = global_passed

    result = {
        "per_tenant": all_metrics,
        "passed": passed,
        "threshold": F1_THRESHOLD,
        "duration_seconds": round(time.perf_counter() - t_start, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    status = "PASS" if passed else "FAIL"
    logger.info(f"  QA5 result: {status}")

    return result
