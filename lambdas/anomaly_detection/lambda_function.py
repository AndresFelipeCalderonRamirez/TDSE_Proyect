"""
Anomaly Detection Lambda Function
Implements Isolation Forest for real-time anomaly detection
"""

import json
import boto3
import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global variables for model caching
_isolation_forest = None
_scaler = None
_model_bucket = None
_tenant_configs = {}

def load_config():
    """Load configuration from environment variables"""
    global _model_bucket, _tenant_configs
    
    _model_bucket = os.getenv('MODEL_BUCKET')
    
    # Tenant configurations
    _tenant_configs = {
        'tenant-A': {
            'model_key': 'models/a/isolation_forest.pkl',
            'scaler_key': 'models/a/scaler.pkl',
            'stream_name': os.getenv('TENANT_A_STREAM'),
            'segments': 50
        },
        'tenant-B': {
            'model_key': 'models/b/isolation_forest.pkl',
            'scaler_key': 'models/b/scaler.pkl',
            'stream_name': os.getenv('TENANT_B_STREAM'),
            'segments': 40
        }
    }

def load_model_from_s3(tenant_id: str):
    """Load Isolation Forest model and scaler from S3"""
    global _isolation_forest, _scaler
    
    try:
        config = _tenant_configs[tenant_id]
        s3 = boto3.client('s3')
        
        # Download model
        model_path = f'/tmp/isolation_forest_{tenant_id}.pkl'
        s3.download_file(_model_bucket, config['model_key'], model_path)
        
        # Download scaler
        scaler_path = f'/tmp/scaler_{tenant_id}.pkl'
        s3.download_file(_model_bucket, config['scaler_key'], scaler_path)
        
        # Load models
        _isolation_forest = joblib.load(model_path)
        _scaler = joblib.load(scaler_path)
        
        logger.info(f"Loaded model and scaler for {tenant_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model for {tenant_id}: {str(e)}")
        return False

def prepare_features(record: Dict) -> np.ndarray:
    """Prepare features from sensor reading"""
    try:
        # Extract basic features
        features = [
            record['pressure'],
            record['flow'],
            record['vibration']
        ]
        
        # Add time-based features
        timestamp = datetime.fromisoformat(record['timestamp'].replace('Z', '+00:00'))
        features.extend([
            timestamp.hour,
            timestamp.weekday(),
            1 if timestamp.weekday() >= 5 else 0  # weekend
        ])
        
        # Add metadata features if available
        metadata = record.get('metadata', {})
        features.extend([
            metadata.get('diameter', 300),
            metadata.get('age', 20),
            metadata.get('length', 250),
            metadata.get('elevation', 2000)
        ])
        
        # Add material encoding
        material = metadata.get('material', 'PVC')
        material_encoding = {
            'PVC': 0, 'HDPE': 1, 'Cast Iron': 2, 
            'Ductile Iron': 3, 'Steel': 4
        }
        features.append(material_encoding.get(material, 0))
        
        return np.array(features).reshape(1, -1)
        
    except Exception as e:
        logger.error(f"Feature preparation error: {str(e)}")
        return None

def detect_anomaly(features: np.ndarray) -> tuple:
    """Detect anomaly using Isolation Forest"""
    try:
        # Scale features
        features_scaled = _scaler.transform(features)
        
        # Predict
        prediction = _isolation_forest.predict(features_scaled)
        anomaly_score = _isolation_forest.decision_function(features_scaled)
        
        # Convert prediction: -1 (anomaly) -> 1, 1 (normal) -> 0
        is_anomaly = 1 if prediction[0] == -1 else 0
        
        return is_anomaly, float(anomaly_score[0])
        
    except Exception as e:
        logger.error(f"Anomaly detection error: {str(e)}")
        return 0, 0.0

def write_to_dynamodb(record: Dict, is_anomaly: int, anomaly_score: float):
    """Write results to DynamoDB"""
    try:
        dynamodb = boto3.client('dynamodb')
        table_name = os.getenv('DYNAMO_TABLE')
        
        # Create sort key (timestamp#segmentId)
        timestamp = record['timestamp']
        segment_id = record['segment_id']
        sort_key = f"{timestamp}#{segment_id}"
        
        # Prepare item
        item = {
            'tenantId': {'S': record['tenant_id']},
            'sortKey': {'S': sort_key},
            'timestamp': {'S': timestamp},
            'segmentId': {'S': segment_id},
            'city': {'S': record['city']},
            'pressure': {'N': str(record['pressure'])},
            'flow': {'N': str(record['flow'])},
            'vibration': {'N': str(record['vibration'])},
            'isAnomaly': {'BOOL': bool(is_anomaly)},
            'anomalyScore': {'N': str(anomaly_score)},
            'metadata': {'S': json.dumps(record.get('metadata', {}))}
        }
        
        # Write to DynamoDB
        dynamodb.put_item(
            TableName=table_name,
            Item=item
        )
        
        logger.info(f"Written to DynamoDB: {record['tenant_id']} - {segment_id}")
        
    except Exception as e:
        logger.error(f"DynamoDB write error: {str(e)}")

def invoke_prediction_lambda(record: Dict, is_anomaly: int, anomaly_score: float):
    """Invoke failure prediction Lambda asynchronously"""
    try:
        if is_anomaly:  # Only invoke for anomalies
            lambda_client = boto3.client('lambda')
            function_name = os.getenv('PREDICTION_LAMBDA')
            
            # Prepare payload
            payload = {
                'tenant_id': record['tenant_id'],
                'segment_id': record['segment_id'],
                'timestamp': record['timestamp'],
                'pressure': record['pressure'],
                'flow': record['flow'],
                'vibration': record['vibration'],
                'is_anomaly': is_anomaly,
                'anomaly_score': anomaly_score,
                'metadata': record.get('metadata', {})
            }
            
            # Invoke asynchronously
            lambda_client.invoke(
                FunctionName=function_name,
                InvocationType='Event',  # Asynchronous
                Payload=json.dumps(payload)
            )
            
            logger.info(f"Invoked prediction Lambda for {record['tenant_id']} - {record['segment_id']}")
        
    except Exception as e:
        logger.error(f"Prediction Lambda invocation error: {str(e)}")

def lambda_handler(event, context):
    """Main Lambda handler"""
    try:
        logger.info("Anomaly detection Lambda started")
        
        # Load configuration
        load_config()
        
        # Process each record
        processed_count = 0
        anomaly_count = 0
        
        for record in event.get('Records', []):
            try:
                # Parse Kinesis record
                if 'kinesis' in record:
                    payload = json.loads(record['kinesis']['data'])
                else:
                    payload = record
                
                tenant_id = payload['tenant_id']
                
                # Load model if not cached
                if tenant_id not in _tenant_configs:
                    logger.error(f"Unknown tenant: {tenant_id}")
                    continue
                
                # Prepare features
                features = prepare_features(payload)
                if features is None:
                    continue
                
                # Detect anomaly
                is_anomaly, anomaly_score = detect_anomaly(features)
                
                # Write to DynamoDB
                write_to_dynamodb(payload, is_anomaly, anomaly_score)
                
                # Invoke prediction Lambda for anomalies
                invoke_prediction_lambda(payload, is_anomaly, anomaly_score)
                
                processed_count += 1
                if is_anomaly:
                    anomaly_count += 1
                
                logger.info(f"Processed {tenant_id} - {payload['segment_id']}: "
                           f"anomaly={is_anomaly}, score={anomaly_score:.3f}")
                
            except Exception as e:
                logger.error(f"Record processing error: {str(e)}")
                continue
        
        logger.info(f"Lambda completed: {processed_count} records processed, "
                   f"{anomaly_count} anomalies detected")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'processed_records': processed_count,
                'anomalies_detected': anomaly_count
            })
        }
        
    except Exception as e:
        logger.error(f"Lambda handler error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
