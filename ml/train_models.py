"""
Machine Learning Training Script for Water Twin Prototype
Implements Isolation Forest for anomaly detection and Random Forest + XGBoost for failure prediction
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
from dataclasses import dataclass
import joblib
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, recall_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Data class for model performance metrics"""
    model_type: str
    tenant_id: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float] = None
    confusion_matrix: Optional[List] = None

class WaterTwinML:
    """
    Machine Learning pipeline for water twin anomaly detection and failure prediction
    """
    
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.tenant_short = tenant_id.split('-')[1]  # A or B
        
        # Model parameters
        self.isolation_forest_params = {
            'contamination': 0.15,  # Adjusted for 16% anomaly rate
            'n_estimators': 100,
            'max_samples': 'auto',
            'random_state': 42
        }
        
        self.random_forest_params = {
            'n_estimators': 100,
            'random_state': 42,
            'class_weight': 'balanced'
        }
        
        self.xgboost_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'random_state': 42
        }
        
        # Storage paths
        self.artifacts_dir = f'ml/artifacts/{self.tenant_short}'
        os.makedirs(self.artifacts_dir, exist_ok=True)
        
        # AWS S3 client
        self.s3 = boto3.client('s3', region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1'))
        self.model_bucket = os.getenv('MODEL_BUCKET')
        
        # Models storage
        self.isolation_forest = None
        self.random_forest = None
        self.xgboost_model = None
        self.scaler = None
        self.label_encoder = None
        
        # Performance metrics
        self.metrics = {}
    
    def _load_sensor_data(self, data_path: str) -> pd.DataFrame:
        """Load sensor data from CSV file"""
        try:
            df = pd.read_csv(data_path, encoding='utf-8-sig')
            logger.info(f"Loaded {len(df)} sensor readings from {data_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load data from {data_path}: {str(e)}")
            # Try alternative encodings
            try:
                df = pd.read_csv(data_path, encoding='latin1')
                logger.info(f"Loaded {len(df)} sensor readings from {data_path} (latin1)")
                return df
            except:
                raise
    
    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from raw sensor data"""
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Parse metadata (stored as string in CSV)
        if 'metadata' in df.columns:
            metadata_list = []
            for meta_str in df['metadata']:
                try:
                    metadata = json.loads(meta_str)
                    metadata_list.append(metadata)
                except:
                    metadata_list.append({})
            
            # Create features from metadata
            meta_df = pd.DataFrame(metadata_list)
            for col in ['diameter', 'age', 'length', 'elevation']:
                if col in meta_df.columns:
                    df[f'meta_{col}'] = meta_df[col]
            
            # Encode material
            if 'material' in meta_df.columns:
                df['meta_material'] = meta_df['material']
        
        # Create rolling statistics (for time series patterns)
        df = df.sort_values(['segment_id', 'timestamp'])
        
        # Rolling window features per segment
        window_sizes = [5, 10, 20]
        for window in window_sizes:
            for col in ['pressure', 'flow', 'vibration']:
                df[f'{col}_rolling_mean_{window}'] = df.groupby('segment_id')[col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                df[f'{col}_rolling_std_{window}'] = df.groupby('segment_id')[col].transform(
                    lambda x: x.rolling(window, min_periods=1).std()
                )
        
        # Rate of change features
        for col in ['pressure', 'flow', 'vibration']:
            df[f'{col}_rate'] = df.groupby('segment_id')[col].transform(
                lambda x: x.diff()
            )
        
        # Fill NaN values
        df = df.bfill().fillna(0)
        
        logger.info(f"Feature engineering completed. Total features: {len(df.columns)}")
        return df
    
    def _prepare_features(self, df: pd.DataFrame, target_column: str = 'is_anomaly') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare feature matrix and target vector"""
        # Select feature columns
        feature_columns = [
            'pressure', 'flow', 'vibration', 'hour', 'day_of_week', 'is_weekend'
        ]
        
        # Add metadata features if available
        meta_features = [col for col in df.columns if col.startswith('meta_')]
        feature_columns.extend(meta_features)
        
        # Add rolling features if available
        rolling_features = [col for col in df.columns if 'rolling' in col]
        feature_columns.extend(rolling_features)
        
        # Add rate features if available
        rate_features = [col for col in df.columns if '_rate' in col]
        feature_columns.extend(rate_features)
        
        # Handle categorical features
        X = df[feature_columns].copy()
        y = df[target_column]
        
        # Encode categorical variables
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col not in ['tenant_id', 'city', 'segment_id', 'timestamp']:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        logger.info(f"Prepared {X.shape[1]} features for {len(X)} samples")
        return X, y
    
    def _normalize_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Normalize features using StandardScaler"""
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return pd.DataFrame(X_train_scaled, columns=X_train.columns), pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    def train_isolation_forest(self, df: pd.DataFrame) -> ModelMetrics:
        """Train Isolation Forest for anomaly detection"""
        logger.info("Training Isolation Forest...")
        
        # Prepare features
        X, y = self._prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalize features
        X_train_scaled, X_test_scaled = self._normalize_features(X_train, X_test)
        
        # Train model
        self.isolation_forest = IsolationForest(**self.isolation_forest_params)
        self.isolation_forest.fit(X_train_scaled)
        
        # Predictions
        y_pred_train = self.isolation_forest.predict(X_train_scaled)
        y_pred_test = self.isolation_forest.predict(X_test_scaled)
        
        # Convert predictions: -1 (anomaly) -> 1, 1 (normal) -> 0
        y_pred_test_binary = (y_pred_test == -1).astype(int)
        
        # Calculate metrics
        recall = recall_score(y_test, y_pred_test_binary)
        
        # Calculate precision and F1 manually for anomaly detection
        tp = ((y_test == 1) & (y_pred_test_binary == 1)).sum()
        fp = ((y_test == 0) & (y_pred_test_binary == 1)).sum()
        fn = ((y_test == 1) & (y_pred_test_binary == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (y_test == y_pred_test_binary).mean()
        
        metrics = ModelMetrics(
            model_type='IsolationForest',
            tenant_id=self.tenant_id,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confusion_matrix=confusion_matrix(y_test, y_pred_test_binary).tolist()
        )
        
        logger.info(f"Isolation Forest trained - Recall: {recall:.3f}, Precision: {precision:.3f}, F1: {f1:.3f}")
        
        # Save model
        model_path = f'{self.artifacts_dir}/isolation_forest.pkl'
        scaler_path = f'{self.artifacts_dir}/scaler.pkl'
        
        joblib.dump(self.isolation_forest, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"Isolation Forest saved to {model_path}")
        
        self.metrics['isolation_forest'] = metrics
        return metrics
    
    def train_failure_prediction(self, df: pd.DataFrame) -> Tuple[ModelMetrics, ModelMetrics]:
        """Train Random Forest and XGBoost for failure prediction"""
        logger.info("Training failure prediction models...")
        
        # Create failure labels (next 24 hours)
        df = df.sort_values(['segment_id', 'timestamp'])
        
        # For each reading, check if there's a failure in next 24 hours
        df['future_failure'] = 0
        window_hours = 24
        
        for segment_id in df['segment_id'].unique():
            segment_data = df[df['segment_id'] == segment_id].copy()
            
            for i in range(len(segment_data)):
                current_time = segment_data.iloc[i]['timestamp']
                future_time = current_time + timedelta(hours=window_hours)
                
                # Check for anomalies in future window
                future_data = segment_data[
                    (segment_data['timestamp'] > current_time) & 
                    (segment_data['timestamp'] <= future_time)
                ]
                
                if len(future_data) > 0:
                    # High anomaly rate in future indicates potential failure
                    future_anomaly_rate = future_data['is_anomaly'].mean()
                    if future_anomaly_rate > 0.1:  # More than 10% anomalies
                        df.loc[segment_data.index[i], 'future_failure'] = 1
        
        # Prepare features
        X, y = self._prepare_features(df, target_column='future_failure')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42, sampling_strategy=5.0)  # 5:1 ratio
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        # Normalize features
        X_train_scaled, X_test_scaled = self._normalize_features(X_train_resampled, X_test)
        
        # Train Random Forest
        self.random_forest = RandomForestClassifier(**self.random_forest_params)
        self.random_forest.fit(X_train_scaled, y_train_resampled)
        
        # Train XGBoost
        self.xgboost_model = xgb.XGBClassifier(**self.xgboost_params)
        self.xgboost_model.fit(X_train_scaled, y_train_resampled)
        
        # Predictions and metrics
        rf_pred = self.random_forest.predict(X_test_scaled)
        rf_proba = self.random_forest.predict_proba(X_test_scaled)[:, 1]
        
        xgb_pred = self.xgboost_model.predict(X_test_scaled)
        xgb_proba = self.xgboost_model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        rf_metrics = self._calculate_classification_metrics(
            'RandomForest', y_test, rf_pred, rf_proba
        )
        
        xgb_metrics = self._calculate_classification_metrics(
            'XGBoost', y_test, xgb_pred, xgb_proba
        )
        
        # Save models
        rf_path = f'{self.artifacts_dir}/random_forest.pkl'
        xgb_path = f'{self.artifacts_dir}/xgboost.pkl'
        
        joblib.dump(self.random_forest, rf_path)
        joblib.dump(self.xgboost_model, xgb_path)
        
        logger.info(f"Random Forest saved to {rf_path}")
        logger.info(f"XGBoost saved to {xgb_path}")
        
        self.metrics['random_forest'] = rf_metrics
        self.metrics['xgboost'] = xgb_metrics
        
        return rf_metrics, xgb_metrics
    
    def _calculate_classification_metrics(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> ModelMetrics:
        """Calculate classification metrics"""
        report = classification_report(y_true, y_pred, output_dict=True)
        
        metrics = ModelMetrics(
            model_type=model_name,
            tenant_id=self.tenant_id,
            accuracy=report['accuracy'],
            precision=report['1']['precision'],
            recall=report['1']['recall'],
            f1_score=report['1']['f1-score'],
            roc_auc=roc_auc_score(y_true, y_proba),
            confusion_matrix=confusion_matrix(y_true, y_pred).tolist()
        )
        
        logger.info(f"{model_name} - Accuracy: {metrics.accuracy:.3f}, F1: {metrics.f1_score:.3f}, ROC-AUC: {metrics.roc_auc:.3f}")
        
        return metrics
    
    def upload_models_to_s3(self) -> bool:
        """Upload trained models to S3"""
        try:
            if not self.model_bucket:
                logger.warning("MODEL_BUCKET not configured, skipping S3 upload")
                return False
            
            # Upload models
            model_files = [
                'isolation_forest.pkl',
                'random_forest.pkl', 
                'xgboost.pkl',
                'scaler.pkl'
            ]
            
            for model_file in model_files:
                local_path = f'{self.artifacts_dir}/{model_file}'
                s3_key = f'models/{self.tenant_short}/{model_file}'
                
                if os.path.exists(local_path):
                    self.s3.upload_file(local_path, self.model_bucket, s3_key)
                    logger.info(f"Uploaded {model_file} to s3://{self.model_bucket}/{s3_key}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload models to S3: {str(e)}")
            return False
    
    def generate_training_data(self, hours: int = 72) -> str:
        """Generate training data using local simulator"""
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from simulator.local_simulator import LocalIoTSimulator, load_config
        
        logger.info(f"Generating {hours}h training data for {self.tenant_id}")
        
        config = load_config()
        simulator = LocalIoTSimulator(self.tenant_id, config[self.tenant_id])
        
        # Generate data
        duration_seconds = hours * 3600
        success = simulator.simulate(duration_seconds, export_ground_truth=True)
        
        if success:
            data_path = f'data/outputs/{self.tenant_id}_sensor_data.csv'
            logger.info(f"Training data generated: {data_path}")
            return data_path
        else:
            raise Exception("Failed to generate training data")
    
    def evaluate_models(self, test_data_path: str) -> Dict:
        """Evaluate trained models on test data"""
        logger.info(f"Evaluating models on {test_data_path}")
        
        df = self._load_sensor_data(test_data_path)
        df = self._feature_engineering(df)
        
        results = {}
        
        # Evaluate Isolation Forest
        if self.isolation_forest:
            X, y = self._prepare_features(df)
            X_scaled = self.scaler.transform(X)
            
            y_pred = self.isolation_forest.predict(X_scaled)
            y_pred_binary = (y_pred == -1).astype(int)
            
            recall = recall_score(y, y_pred_binary)
            results['isolation_forest'] = {'recall': recall}
        
        # Evaluate failure prediction models
        if self.random_forest and self.xgboost_model:
            # Create failure labels
            df = df.sort_values(['segment_id', 'timestamp'])
            df['future_failure'] = 0
            
            for segment_id in df['segment_id'].unique():
                segment_data = df[df['segment_id'] == segment_id].copy()
                
                for i in range(len(segment_data)):
                    current_time = segment_data.iloc[i]['timestamp']
                    future_time = current_time + timedelta(hours=24)
                    
                    future_data = segment_data[
                        (segment_data['timestamp'] > current_time) & 
                        (segment_data['timestamp'] <= future_time)
                    ]
                    
                    if len(future_data) > 0:
                        future_anomaly_rate = future_data['is_anomaly'].mean()
                        if future_anomaly_rate > 0.1:
                            df.loc[segment_data.index[i], 'future_failure'] = 1
            
            X, y = self._prepare_features(df, target_column='future_failure')
            X_scaled = self.scaler.transform(X)
            
            rf_pred = self.random_forest.predict(X_scaled)
            xgb_pred = self.xgboost_model.predict(X_scaled)
            
            rf_f1 = classification_report(y, rf_pred, output_dict=True)['1']['f1-score']
            xgb_f1 = classification_report(y, xgb_pred, output_dict=True)['1']['f1-score']
            
            results['random_forest'] = {'f1_score': rf_f1}
            results['xgboost'] = {'f1_score': xgb_f1}
        
        return results

def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description='Water Twin ML Training')
    parser.add_argument('--tenant', 
                       choices=['tenant-A', 'tenant-B', 'both'],
                       default='both',
                       help='Tenant to train models for (default: both)')
    parser.add_argument('--generate-data',
                       action='store_true',
                       help='Generate training data using simulator')
    parser.add_argument('--train-if',
                       action='store_true',
                       help='Train Isolation Forest models')
    parser.add_argument('--train-ensemble',
                       action='store_true',
                       help='Train Random Forest + XGBoost ensemble')
    parser.add_argument('--eval-if',
                       action='store_true',
                       help='Evaluate Isolation Forest models')
    parser.add_argument('--eval-ensemble',
                       action='store_true',
                       help='Evaluate ensemble models')
    parser.add_argument('--upload-models',
                       action='store_true',
                       help='Upload models to S3')
    parser.add_argument('--data-path',
                       type=str,
                       help='Path to training data CSV file')
    
    args = parser.parse_args()
    
    logger.info("Water Twin ML Training Pipeline")
    logger.info(f"Tenants: {args.tenant}")
    logger.info(f"Generate data: {args.generate_data}")
    logger.info(f"Train IF: {args.train_if}")
    logger.info(f"Train Ensemble: {args.train_ensemble}")
    logger.info(f"Upload to S3: {args.upload_models}")
    
    tenants = ['tenant-A', 'tenant-B'] if args.tenant == 'both' else [args.tenant]
    
    for tenant_id in tenants:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {tenant_id}")
        logger.info(f"{'='*50}")
        
        ml_pipeline = WaterTwinML(tenant_id)
        
        # Generate training data if requested
        if args.generate_data:
            data_path = ml_pipeline.generate_training_data(hours=72)
        elif args.data_path:
            data_path = args.data_path
        else:
            # Use existing data
            data_path = f'data/outputs/{tenant_id}_sensor_data.csv'
        
        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            continue
        
        # Load and prepare data
        df = ml_pipeline._load_sensor_data(data_path)
        df = ml_pipeline._feature_engineering(df)
        
        # Train models
        if args.train_if:
            ml_pipeline.train_isolation_forest(df)
        
        if args.train_ensemble:
            ml_pipeline.train_failure_prediction(df)
        
        # Upload models to S3
        if args.upload_models:
            ml_pipeline.upload_models_to_s3()
        
        # Evaluate models
        if args.eval_if or args.eval_ensemble:
            # Use separate test data if available
            test_path = f'data/outputs/{tenant_id}_sensor_data_test.csv'
            if os.path.exists(test_path):
                results = ml_pipeline.evaluate_models(test_path)
                logger.info(f"Evaluation results: {results}")
            else:
                logger.warning("Test data not found for evaluation")
        
        # Print summary
        logger.info(f"\n{tenant_id} Training Summary:")
        for model_name, metrics in ml_pipeline.metrics.items():
            logger.info(f"  {model_name}:")
            logger.info(f"    Accuracy: {metrics.accuracy:.3f}")
            logger.info(f"    Precision: {metrics.precision:.3f}")
            logger.info(f"    Recall: {metrics.recall:.3f}")
            logger.info(f"    F1-Score: {metrics.f1_score:.3f}")
            if metrics.roc_auc:
                logger.info(f"    ROC-AUC: {metrics.roc_auc:.3f}")

if __name__ == "__main__":
    main()
