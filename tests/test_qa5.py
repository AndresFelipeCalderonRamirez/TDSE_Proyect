"""
QA-5  Failure Prediction Quality Gate  (US-05, Acceptance Criterion 3)
=======================================================================
Asserts F1 >= 0.75 for the RF+XGBoost ensemble on the held-out 20 % test
split, for each tenant whose model artifacts and data are present.

Run:
    pytest tests/test_qa5.py -v

The test is automatically SKIPPED when:
  - model artifacts (models/tenant-X/*.pkl) do not exist yet
  - training data (data/outputs/tenant-X_sensor_data.json) is missing

Prerequisites (run once):
  1. python ml/battledim_adapter.py ...   (generates data/outputs/*.json)
  2. python ml/train_models.py --train-ensemble --tenant tenant-A
  3. python ml/train_models.py --train-ensemble --tenant tenant-B
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pytest

# Allow importing from project root
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Constants (must match train_models.py / paper Section 5.3) ────────────────

FEATURE_COLS: List[str] = [
    "pressure", "flow", "vibration", "delta_pressure",
    "diameter", "material_enc", "length", "age",
]

MATERIAL_ENCODING: Dict[str, int] = {
    "PVC": 0, "HDPE": 0, "Steel": 0,
    "AC": 1,
    "DI": 2, "Ductile Iron": 2,
    "CI": 3, "Cast Iron": 3,
}

TENANTS = ["tenant-A", "tenant-B"]
F1_THRESHOLD = 0.75
TEST_FRACTION = 0.20  # last 20 % of (time-sorted) records


# ── Helpers ────────────────────────────────────────────────────────────────────

def _models_dir(tenant_id: str) -> Path:
    short = tenant_id.split("-")[1].lower()
    return ROOT / "models" / f"tenant-{short}"


def _data_path(tenant_id: str) -> Path:
    return ROOT / "data" / "outputs" / f"{tenant_id}_sensor_data.json"


def _artifacts_present(tenant_id: str) -> bool:
    d = _models_dir(tenant_id)
    return all(
        (d / f).exists()
        for f in ("rf_model.pkl", "xgb_model.pkl", "scaler.pkl", "beta.json")
    )


def _load_models(tenant_id: str):
    import joblib

    d = _models_dir(tenant_id)
    rf     = joblib.load(d / "rf_model.pkl")
    xgb    = joblib.load(d / "xgb_model.pkl")
    scaler = joblib.load(d / "scaler.pkl")
    with open(d / "beta.json") as f:
        beta = float(json.load(f)["beta"])
    return rf, xgb, scaler, beta


def _build_features(records: List[Dict]) -> np.ndarray:
    """Build 8-feature matrix from JSON records (same logic as train_models.py)."""
    import pandas as pd

    df = pd.DataFrame(records)

    # Flatten metadata fields
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

    # delta_pressure per segment (time-sorted within segment)
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
    """
    Reproduce the future-failure label from train_models.py:
    label = 1 if next-24 h anomaly rate > 10 %, else 0.
    Uses time-sorted order within each segment.
    """
    import pandas as pd

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    if "is_anomaly" not in df.columns:
        return np.zeros(len(df), dtype=np.int8)

    if "segment_id" not in df.columns:
        return df["is_anomaly"].to_numpy(dtype=np.int8)

    df = df.sort_values(["segment_id", "timestamp"]).reset_index(drop=True)

    WINDOW_STEPS = 288   # 24 h at 5-min resolution
    THRESHOLD    = 0.10

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


def _predict(features: np.ndarray, rf, xgb, scaler, beta: float) -> np.ndarray:
    X_sc  = scaler.transform(features)
    p_rf  = rf.predict_proba(X_sc)[:, 1]
    p_xgb = xgb.predict_proba(X_sc)[:, 1]
    return beta * p_rf + (1.0 - beta) * p_xgb


# ── Fixtures ───────────────────────────────────────────────────────────────────

def pytest_generate_tests(metafunc):
    if "tenant_id" in metafunc.fixturenames:
        metafunc.parametrize("tenant_id", TENANTS)


# ── Test ───────────────────────────────────────────────────────────────────────

def test_qa5_f1_threshold(tenant_id: str):
    """P(Failure|X) ensemble achieves F1 >= 0.75 on held-out 20 % test split."""
    from sklearn.metrics import f1_score

    if not _artifacts_present(tenant_id):
        pytest.skip(
            f"Model artifacts missing for {tenant_id}. "
            f"Run: python ml/train_models.py --train-ensemble --tenant {tenant_id}"
        )

    data_path = _data_path(tenant_id)
    if not data_path.exists():
        pytest.skip(
            f"Training data missing: {data_path}. "
            f"Run battledim_adapter.py first."
        )

    with open(data_path, encoding="utf-8") as f:
        records = json.load(f)

    # Time-sorted test split (last 20 %)
    records.sort(key=lambda r: (r.get("segment_id", ""), r.get("timestamp", "")))
    split = int(len(records) * (1.0 - TEST_FRACTION))
    test_records = records[split:]

    assert len(test_records) > 0, "Test split is empty"

    X_test = _build_features(test_records)
    y_test = _build_labels(test_records)

    rf, xgb_mdl, scaler, beta = _load_models(tenant_id)
    p_failure = _predict(X_test, rf, xgb_mdl, scaler, beta)

    # Binary decision at 0.5 threshold (same as train_models.py evaluation)
    y_pred = (p_failure >= 0.5).astype(int)

    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(
        f"\n[{tenant_id}] test records={len(test_records):,} | "
        f"beta={beta:.2f} | F1={f1:.4f} (threshold={F1_THRESHOLD})"
    )

    assert f1 >= F1_THRESHOLD, (
        f"[{tenant_id}] F1={f1:.4f} is below required {F1_THRESHOLD}. "
        f"Retrain with more data or adjust beta grid."
    )
