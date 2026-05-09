"""
ML Training for Water Twin Prototype
US-04: RF+XGBoost ensemble with blending coefficient β (Ecuación 2)
P(Failure|X) = β·P_RF + (1-β)·P_XGB
"""

import json
import numpy as np
import pandas as pd
import argparse
import logging
import os
import boto3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import joblib
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Paper Section 5.3 constants ───────────────────────────────────────────────

# Material encoding: PVC=0, AC=1, DI=2, CI=3
MATERIAL_ENCODING: Dict[str, int] = {
    'PVC': 0, 'HDPE': 0, 'Steel': 0,
    'AC': 1,
    'DI': 2, 'Ductile Iron': 2,
    'CI': 3, 'Cast Iron': 3,
}

# X = [pressure, flow, vibration, Δpressure, diameter, material_enc, length, age]
FEATURE_COLS: List[str] = [
    'pressure', 'flow', 'vibration', 'delta_pressure',
    'diameter', 'material_enc', 'length', 'age',
]

# β search grid: 0.00, 0.05, …, 1.00  (21 values)
BETA_GRID: np.ndarray = np.round(np.arange(0.0, 1.05, 0.05), 2)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ModelMetrics:
    model_type: str
    tenant_id: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float] = None
    confusion_matrix: Optional[List] = None


# ── Main ML pipeline ──────────────────────────────────────────────────────────

class WaterTwinML:
    """RF+XGBoost ensemble failure prediction pipeline (US-04)."""

    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.tenant_short = tenant_id.split('-')[1].lower()   # 'a' or 'b'

        self.rf_params = {
            'n_estimators': 100,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1,
        }

        self.xgb_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'scale_pos_weight': 20,
            'random_state': 42,
            'eval_metric': 'logloss',
            'verbosity': 0,
        }

        self.isolation_forest_params = {
            'contamination': 0.15,
            'n_estimators': 100,
            'max_samples': 'auto',
            'random_state': 42,
        }

        # Artifacts go to models/tenant-{a,b}/
        self.models_dir = f'models/tenant-{self.tenant_short}'
        os.makedirs(self.models_dir, exist_ok=True)

        self.s3 = boto3.client(
            's3', region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        )
        self.model_bucket = os.getenv('MODEL_BUCKET')

        self.isolation_forest: Optional[IsolationForest] = None
        self.random_forest: Optional[RandomForestClassifier] = None
        self.xgboost_model: Optional[xgb.XGBClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.beta: Optional[float] = None

        self.metrics: Dict[str, ModelMetrics] = {}

    # ── T-04.01  Feature & label generation ───────────────────────────────────

    def _load_training_data(self, data_path: str) -> pd.DataFrame:
        """Load sensor data; prefers JSON (contains pipe metadata) over CSV."""
        if data_path.endswith('.json'):
            return self._load_from_json(data_path)

        json_path = data_path.replace('.csv', '.json')
        if os.path.exists(json_path):
            logger.info(f"Using JSON (has pipe metadata): {json_path}")
            return self._load_from_json(json_path)

        logger.warning(
            "No JSON found — loading CSV without pipe metadata; "
            "using defaults for diameter / material / length / age"
        )
        return pd.read_csv(data_path, encoding='utf-8-sig')

    def _load_from_json(self, json_path: str) -> pd.DataFrame:
        """Load JSON sensor data and flatten pipe metadata fields."""
        with open(json_path, 'r', encoding='utf-8') as f:
            records = json.load(f)

        rows = []
        for r in records:
            meta = r.get('metadata', {})
            rows.append({
                'tenant_id':  r['tenant_id'],
                'city':       r.get('city', ''),
                'segment_id': r['segment_id'],
                'timestamp':  r['timestamp'],
                'pressure':   float(r['pressure']),
                'flow':       float(r['flow']),
                'vibration':  float(r['vibration']),
                'is_anomaly': int(r.get('is_anomaly', 0)),
                'diameter':   float(meta.get('diameter', 300.0)),
                'material':   meta.get('material', 'PVC'),
                'length':     float(meta.get('length', 250.0)),
                'age':        float(meta.get('age', 20.0)),
            })

        df = pd.DataFrame(rows)
        logger.info(f"Loaded {len(df):,} records from {json_path}")
        return df

    def _build_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        T-04.01 — Build 8-feature matrix aligned with paper Section 5.3.
        X = [pressure, flow, vibration, Δpressure, diameter, material_enc, length, age]
        Material encoding: PVC=0, AC=1, DI=2, CI=3
        """
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['segment_id', 'timestamp'])

        # Δpressure per segment (first diff → NaN filled with 0)
        df['delta_pressure'] = (
            df.groupby('segment_id')['pressure']
              .transform(lambda x: x.diff().fillna(0))
        )

        # Material encoding
        if 'material' in df.columns:
            df['material_enc'] = (
                df['material'].map(MATERIAL_ENCODING).fillna(0).astype(int)
            )
        else:
            df['material_enc'] = 0

        # Ensure pipe-metadata columns exist with sensible defaults
        for col, default in [('diameter', 300.0), ('length', 250.0), ('age', 20.0)]:
            if col not in df.columns:
                df[col] = float(default)

        return df

    def _create_failure_labels(
        self, df: pd.DataFrame, window_hours: int = 24
    ) -> pd.Series:
        """
        T-04.01 — Label row i as future_failure=1 when the anomaly rate in the
        next `window_hours` for that segment exceeds 10 %.
        Uses prefix-sum + searchsorted for O(n log n) per segment.
        """
        df = df.copy()
        df['future_failure'] = 0
        window_ns = np.timedelta64(window_hours, 'h')

        for _seg_id, seg in df.groupby('segment_id'):
            seg = seg.sort_values('timestamp')
            orig_idx = seg.index

            ts = seg['timestamp'].values  # numpy datetime64
            anom = seg['is_anomaly'].fillna(0).values.astype(np.float32)
            n = len(seg)

            end_times = ts + window_ns
            # First position where ts[j] > ts[i] + window → exclusive upper bound
            j_ends = np.searchsorted(ts, end_times, side='right')

            prefix = np.concatenate([[0.0], np.cumsum(anom)])
            labels = np.zeros(n, dtype=np.int8)

            for i in range(n):
                j_s = i + 1          # strictly future
                j_e = int(j_ends[i])
                w = j_e - j_s
                if w > 0 and (prefix[j_e] - prefix[j_s]) / w > 0.1:
                    labels[i] = 1

            df.loc[orig_idx, 'future_failure'] = labels

        pos = int(df['future_failure'].sum())
        neg = len(df) - pos
        logger.info(
            f"[{self.tenant_id}] Failure labels — pos: {pos:,}, "
            f"neg: {neg:,}, ratio ≈ {neg / max(pos, 1):.1f}:1"
        )
        return df['future_failure']

    def _prepare_xy(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build X (8 features), y (failure label), stratified 80/20 split."""
        df_feat = self._build_feature_matrix(df)
        y = self._create_failure_labels(df_feat)
        X = df_feat[FEATURE_COLS].values

        return train_test_split(
            X, y.values, test_size=0.2, random_state=42, stratify=y
        )

    # ── T-04.02  Random Forest with SMOTE 5:1 ─────────────────────────────────

    def train_random_forest(self, df: pd.DataFrame) -> ModelMetrics:
        """
        T-04.02 — Train RandomForest with SMOTE 5:1.
        sampling_strategy=0.2 → n_minority/n_majority = 1/5 → 5:1 majority:minority.
        Saves rf_model.pkl + scaler.pkl to models/tenant-{a,b}/.
        """
        logger.info(f"[{self.tenant_id}] T-04.02: Training RF with SMOTE 5:1 ...")

        X_train, X_test, y_train, y_test = self._prepare_xy(df)

        smote = SMOTE(sampling_strategy=0.2, random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        logger.info(
            f"[{self.tenant_id}] After SMOTE → {len(X_res):,} samples "
            f"({int(y_res.sum()):,} positives)"
        )

        self.scaler = StandardScaler()
        X_res_sc = self.scaler.fit_transform(X_res)
        X_test_sc = self.scaler.transform(X_test)

        self.random_forest = RandomForestClassifier(**self.rf_params)
        self.random_forest.fit(X_res_sc, y_res)

        p_rf = self.random_forest.predict_proba(X_test_sc)[:, 1]
        metrics = self._build_metrics('RandomForest', y_test, p_rf)
        self.metrics['random_forest'] = metrics

        joblib.dump(self.random_forest, f'{self.models_dir}/rf_model.pkl')
        joblib.dump(self.scaler,        f'{self.models_dir}/scaler.pkl')
        logger.info(
            f"[{self.tenant_id}] RF — F1={metrics.f1_score:.3f}, "
            f"ROC-AUC={metrics.roc_auc:.3f}"
        )
        return metrics

    # ── T-04.03  XGBoost + β CV + full ensemble ────────────────────────────────

    def _find_optimal_beta(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Tuple[float, Dict]:
        """
        T-04.03 — 5-fold stratified CV on the training set to find β that
        maximises mean ROC-AUC for P(Failure|X) = β·P_RF + (1-β)·P_XGB.
        SMOTE is applied inside each fold to prevent data leakage.
        """
        logger.info(
            f"[{self.tenant_id}] Searching optimal β over {len(BETA_GRID)} values "
            "via 5-fold CV ..."
        )
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        smote = SMOTE(sampling_strategy=0.2, random_state=42)

        # β → list of per-fold ROC-AUC scores
        beta_scores: Dict[float, List[float]] = {float(b): [] for b in BETA_GRID}

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
            X_tr, y_tr   = X_train[tr_idx], y_train[tr_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]

            if y_val.sum() == 0:
                logger.warning(f"Fold {fold}: no positives in validation — skipped")
                continue

            X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)

            scaler_f = StandardScaler()
            X_tr_sc  = scaler_f.fit_transform(X_tr_res)
            X_val_sc = scaler_f.transform(X_val)

            rf_f = RandomForestClassifier(**self.rf_params)
            rf_f.fit(X_tr_sc, y_tr_res)
            p_rf = rf_f.predict_proba(X_val_sc)[:, 1]

            xgb_f = xgb.XGBClassifier(**self.xgb_params)
            xgb_f.fit(X_tr_sc, y_tr_res)
            p_xgb = xgb_f.predict_proba(X_val_sc)[:, 1]

            for b in BETA_GRID:
                blend = float(b) * p_rf + (1.0 - float(b)) * p_xgb
                try:
                    beta_scores[float(b)].append(roc_auc_score(y_val, blend))
                except Exception:
                    pass

            logger.info(f"  Fold {fold}/5 done")

        mean_auc = {
            b: float(np.mean(v)) for b, v in beta_scores.items() if v
        }
        std_auc = {
            b: float(np.std(v))  for b, v in beta_scores.items() if v
        }
        beta_opt = float(max(mean_auc, key=mean_auc.get))
        logger.info(
            f"[{self.tenant_id}] Optimal β={beta_opt:.2f}, "
            f"mean ROC-AUC={mean_auc[beta_opt]:.4f}"
        )

        cv_results = {
            'tenant_id':             self.tenant_id,
            'betas':                 [float(b) for b in BETA_GRID],
            'mean_roc_auc':          [mean_auc.get(float(b)) for b in BETA_GRID],
            'std_roc_auc':           [std_auc.get(float(b)) for b in BETA_GRID],
            'optimal_beta':          beta_opt,
            'optimal_mean_roc_auc':  mean_auc[beta_opt],
        }
        return beta_opt, cv_results

    def train_ensemble(self, df: pd.DataFrame) -> Dict:
        """
        T-04.03 — Full ensemble pipeline:
          (1) SMOTE 5:1 + RF
          (2) XGBoost (scale_pos_weight=20)
          (3) β optimal via 5-fold CV on ROC-AUC
          (4) Serialize rf_model.pkl, xgb_model.pkl, beta.json,
              beta_cv_results.json to models/tenant-{a,b}/
        Returns dict with β and per-model metrics evaluated on held-out test set.
        """
        logger.info(
            f"[{self.tenant_id}] T-04.03: Starting ensemble training "
            "(RF + XGBoost + β CV) ..."
        )

        df_feat = self._build_feature_matrix(df)
        y       = self._create_failure_labels(df_feat)
        X       = df_feat[FEATURE_COLS].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y.values, test_size=0.2, random_state=42, stratify=y
        )

        # ── (1) SMOTE 5:1 on training set (shared for both models)
        smote = SMOTE(sampling_strategy=0.2, random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        logger.info(
            f"[{self.tenant_id}] After SMOTE → {len(X_res):,} samples "
            f"({int(y_res.sum()):,} positives)"
        )

        self.scaler = StandardScaler()
        X_res_sc  = self.scaler.fit_transform(X_res)
        X_test_sc = self.scaler.transform(X_test)

        # ── (2) Train final RF
        self.random_forest = RandomForestClassifier(**self.rf_params)
        self.random_forest.fit(X_res_sc, y_res)
        logger.info(f"[{self.tenant_id}] RF trained on {len(X_res):,} samples")

        # ── (3) Train final XGBoost
        self.xgboost_model = xgb.XGBClassifier(**self.xgb_params)
        self.xgboost_model.fit(X_res_sc, y_res)
        logger.info(f"[{self.tenant_id}] XGBoost trained")

        # ── (4) Find β by 5-fold CV on the (non-resampled) training set
        self.beta, cv_results = self._find_optimal_beta(X_train, y_train)

        # ── (5) Evaluate ensemble on held-out test set
        p_rf  = self.random_forest.predict_proba(X_test_sc)[:, 1]
        p_xgb = self.xgboost_model.predict_proba(X_test_sc)[:, 1]
        p_ens = self.beta * p_rf + (1.0 - self.beta) * p_xgb  # Ecuación 2

        rf_metrics  = self._build_metrics('RandomForest', y_test, p_rf)
        xgb_metrics = self._build_metrics('XGBoost',      y_test, p_xgb)
        ens_metrics = self._build_metrics('Ensemble',     y_test, p_ens)

        self.metrics.update({
            'random_forest': rf_metrics,
            'xgboost':       xgb_metrics,
            'ensemble':      ens_metrics,
        })

        # ── (6) Serialize all artifacts
        self._save_ensemble_artifacts(cv_results)

        logger.info(
            f"[{self.tenant_id}] Ensemble → "
            f"F1={ens_metrics.f1_score:.3f}, "
            f"ROC-AUC={ens_metrics.roc_auc:.3f}, "
            f"β={self.beta:.2f}"
        )

        return {
            'beta':             self.beta,
            'rf_metrics':       asdict(rf_metrics),
            'xgb_metrics':      asdict(xgb_metrics),
            'ensemble_metrics': asdict(ens_metrics),
        }

    def _save_ensemble_artifacts(self, cv_results: Dict) -> None:
        """Serialize all ensemble artifacts to models/tenant-{a,b}/."""
        joblib.dump(self.random_forest, f'{self.models_dir}/rf_model.pkl')
        joblib.dump(self.xgboost_model, f'{self.models_dir}/xgb_model.pkl')
        joblib.dump(self.scaler,        f'{self.models_dir}/scaler.pkl')

        beta_meta = {
            'tenant_id':   self.tenant_id,
            'beta':        float(self.beta),
            'feature_cols': FEATURE_COLS,
            'trained_at':  datetime.utcnow().isoformat(),
        }
        with open(f'{self.models_dir}/beta.json', 'w') as f:
            json.dump(beta_meta, f, indent=2)

        with open(f'{self.models_dir}/beta_cv_results.json', 'w') as f:
            json.dump(cv_results, f, indent=2)

        logger.info(f"Artifacts saved to {self.models_dir}/")

    def predict_failure_probability(self, X: np.ndarray) -> np.ndarray:
        """
        Ecuación 2: P(Failure|X) = β·P_RF + (1-β)·P_XGB
        X must have exactly 8 columns matching FEATURE_COLS.
        """
        if None in (self.random_forest, self.xgboost_model, self.beta, self.scaler):
            raise RuntimeError("Models not trained. Run train_ensemble() first.")
        X_sc  = self.scaler.transform(X)
        p_rf  = self.random_forest.predict_proba(X_sc)[:, 1]
        p_xgb = self.xgboost_model.predict_proba(X_sc)[:, 1]
        return self.beta * p_rf + (1.0 - self.beta) * p_xgb

    # ── Isolation Forest (used by anomaly-detection Lambda) ───────────────────

    def train_isolation_forest(self, df: pd.DataFrame) -> ModelMetrics:
        """Train Isolation Forest for anomaly detection (legacy anomaly Lambda)."""
        logger.info(f"[{self.tenant_id}] Training Isolation Forest ...")

        df_feat = self._build_feature_matrix(df)
        if 'is_anomaly' not in df_feat.columns:
            raise ValueError("Column 'is_anomaly' is required for IF training")

        X = df_feat[FEATURE_COLS].values
        y = df_feat['is_anomaly'].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler_if   = StandardScaler()
        X_train_sc  = scaler_if.fit_transform(X_train)
        X_test_sc   = scaler_if.transform(X_test)

        self.isolation_forest = IsolationForest(**self.isolation_forest_params)
        self.isolation_forest.fit(X_train_sc)

        y_pred = (self.isolation_forest.predict(X_test_sc) == -1).astype(int)

        tp = ((y_test == 1) & (y_pred == 1)).sum()
        fp = ((y_test == 0) & (y_pred == 1)).sum()
        fn = ((y_test == 1) & (y_pred == 0)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        metrics = ModelMetrics(
            model_type='IsolationForest',
            tenant_id=self.tenant_id,
            accuracy=float((y_test == y_pred).mean()),
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=confusion_matrix(y_test, y_pred).tolist(),
        )

        # Save to ml/artifacts/ path expected by the anomaly Lambda
        if_dir = f'ml/artifacts/{self.tenant_short.upper()}'
        os.makedirs(if_dir, exist_ok=True)
        joblib.dump(self.isolation_forest, f'{if_dir}/isolation_forest.pkl')
        joblib.dump(scaler_if,             f'{if_dir}/scaler.pkl')

        self.metrics['isolation_forest'] = metrics
        logger.info(
            f"[{self.tenant_id}] IF — F1={f1:.3f}, Recall={recall:.3f}"
        )
        return metrics

    # ── S3 upload ─────────────────────────────────────────────────────────────

    def upload_models_to_s3(self) -> bool:
        """Upload ensemble artifacts from models/tenant-{a,b}/ to S3."""
        if not self.model_bucket:
            logger.warning("MODEL_BUCKET not set — skipping S3 upload")
            return False
        try:
            files = [
                'rf_model.pkl', 'xgb_model.pkl', 'scaler.pkl',
                'beta.json', 'beta_cv_results.json',
            ]
            for fname in files:
                local = f'{self.models_dir}/{fname}'
                if os.path.exists(local):
                    key = f'models/tenant-{self.tenant_short}/{fname}'
                    self.s3.upload_file(local, self.model_bucket, key)
                    logger.info(
                        f"Uploaded {fname} → s3://{self.model_bucket}/{key}"
                    )
            return True
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            return False

    # ── Data generation ───────────────────────────────────────────────────────

    def generate_training_data(self, hours: int = 72) -> str:
        """Generate training data via local simulator (returns JSON path)."""
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from simulator.local_simulator import LocalIoTSimulator, load_config

        logger.info(f"[{self.tenant_id}] Generating {hours}h training data ...")
        config    = load_config()
        simulator = LocalIoTSimulator(self.tenant_id, config[self.tenant_id])

        if not simulator.simulate(hours * 3600, export_ground_truth=True):
            raise RuntimeError("Simulator failed to generate training data")

        # Return JSON path — it includes pipe metadata required for 8-feature vector
        return f'data/outputs/{self.tenant_id}_sensor_data.json'

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_metrics(
        self,
        name: str,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float = 0.5,
    ) -> ModelMetrics:
        y_pred  = (y_proba >= threshold).astype(int)
        report  = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        cls1 = report.get('1', report.get(1, {}))
        try:
            auc = float(roc_auc_score(y_true, y_proba))
        except Exception:
            auc = 0.0
        return ModelMetrics(
            model_type=name,
            tenant_id=self.tenant_id,
            accuracy=float(report['accuracy']),
            precision=float(cls1.get('precision', 0)),
            recall=float(cls1.get('recall', 0)),
            f1_score=float(cls1.get('f1-score', 0)),
            roc_auc=auc,
            confusion_matrix=confusion_matrix(y_true, y_pred).tolist(),
        )


# ── CLI entry-point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Water Twin ML Training - US-04 RF+XGBoost Ensemble'
    )
    parser.add_argument(
        '--tenant', choices=['tenant-A', 'tenant-B', 'both'], default='both',
        help='Tenant to train (default: both)',
    )
    parser.add_argument(
        '--generate-data', action='store_true',
        help='Generate 72h training data from local simulator',
    )
    parser.add_argument(
        '--train-if', action='store_true',
        help='Train Isolation Forest (anomaly detection)',
    )
    parser.add_argument(
        '--train-rf', action='store_true',
        help='T-04.02: Train RandomForest with SMOTE 5:1',
    )
    parser.add_argument(
        '--train-ensemble', action='store_true',
        help='T-04.03: Train RF+XGBoost ensemble with beta CV -- saves all artifacts',
    )
    parser.add_argument(
        '--upload-models', action='store_true',
        help='Upload models to S3',
    )
    parser.add_argument(
        '--data-path', type=str,
        help='Path to training data (.json preferred, .csv accepted)',
    )
    args = parser.parse_args()

    tenants = ['tenant-A', 'tenant-B'] if args.tenant == 'both' else [args.tenant]

    for tenant_id in tenants:
        logger.info(f"\n{'='*55}")
        logger.info(f"Processing {tenant_id}")
        logger.info(f"{'='*55}")

        ml = WaterTwinML(tenant_id)

        # Resolve data path
        if args.generate_data:
            data_path = ml.generate_training_data(hours=72)
        elif args.data_path:
            data_path = args.data_path
        else:
            json_path = f'data/outputs/{tenant_id}_sensor_data.json'
            csv_path  = f'data/outputs/{tenant_id}_sensor_data.csv'
            data_path = json_path if os.path.exists(json_path) else csv_path

        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            continue

        df = ml._load_training_data(data_path)

        if args.train_if:
            ml.train_isolation_forest(df)

        if args.train_rf:
            ml.train_random_forest(df)

        if args.train_ensemble:
            results = ml.train_ensemble(df)
            logger.info(f"\n{tenant_id} — Ensemble Results:")
            logger.info(f"  β = {results['beta']:.2f}")
            for key in ('rf_metrics', 'xgb_metrics', 'ensemble_metrics'):
                m = results[key]
                logger.info(
                    f"  {m['model_type']:15s} "
                    f"F1={m['f1_score']:.3f}  "
                    f"ROC-AUC={m['roc_auc']:.3f}"
                )
            ens = results['ensemble_metrics']
            if ens['f1_score'] >= 0.75:
                logger.info(f"  ✓ QA5 PASSED: F1={ens['f1_score']:.3f} ≥ 0.75")
            else:
                logger.warning(
                    f"  ✗ QA5 NOT MET: F1={ens['f1_score']:.3f} < 0.75"
                )

        if args.upload_models:
            ml.upload_models_to_s3()

        # Final summary
        if ml.metrics:
            logger.info(f"\n{tenant_id} — Training Summary:")
            for name, m in ml.metrics.items():
                logger.info(
                    f"  {name:20s} "
                    f"F1={m.f1_score:.3f}  "
                    f"AUC={m.roc_auc or 0:.3f}"
                )


if __name__ == '__main__':
    main()
