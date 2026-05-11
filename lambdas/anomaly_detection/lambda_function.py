"""
Anomaly Detection Lambda Function
Z-score anomaly detection using pre-computed training statistics (numpy only,
no scipy/sklearn dependency). Stats are loaded from S3 as JSON.
"""

import base64
import json
import boto3
import math
import os
import time
from typing import Dict, List, Optional
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global cache: stats dict keyed by tenant_id
_stats_cache: Dict[str, dict] = {}
_model_bucket: Optional[str] = None
_tenant_configs: Dict[str, dict] = {}


def load_config():
    global _model_bucket, _tenant_configs
    _model_bucket = os.getenv('MODEL_BUCKET')
    _tenant_configs = {
        'tenant-A': {
            'stats_key': 'models/tenant-a/if_stats.json',
            'stream_name': os.getenv('TENANT_A_STREAM'),
        },
        'tenant-B': {
            'stats_key': 'models/tenant-b/if_stats.json',
            'stream_name': os.getenv('TENANT_B_STREAM'),
        },
    }


def load_stats_from_s3(tenant_id: str) -> bool:
    """Download and cache the z-score stats JSON for tenant_id."""
    global _stats_cache
    try:
        key = _tenant_configs[tenant_id]['stats_key']
        s3 = boto3.client('s3')
        resp = s3.get_object(Bucket=_model_bucket, Key=key)
        stats = json.loads(resp['Body'].read())
        _stats_cache[tenant_id] = {
            'mean':      [float(v) for v in stats['mean']],
            'std':       [float(v) for v in stats['std']],
            'threshold': float(stats.get('threshold', 3.0)),
        }
        logger.info(f"Loaded z-score stats for {tenant_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to load stats for {tenant_id}: {e}")
        return False


def prepare_features(record: Dict) -> Optional[List[float]]:
    """Build 8-feature list matching FEATURE_COLS from training.
    X = [pressure, flow, vibration, delta_pressure,
         diameter, material_enc, length, age]
    delta_pressure is 0 for single-record inference.
    """
    try:
        metadata = record.get('metadata', {})
        material_enc = {
            'PVC': 0, 'HDPE': 0, 'Steel': 0,
            'AC': 1,
            'DI': 2, 'Ductile Iron': 2,
            'CI': 3, 'Cast Iron': 3,
        }.get(metadata.get('material', 'PVC'), 0)

        return [
            float(record['pressure']),
            float(record['flow']),
            float(record['vibration']),
            0.0,                              # delta_pressure
            float(metadata.get('diameter', 300.0)),
            float(material_enc),
            float(metadata.get('length', 250.0)),
            float(metadata.get('age', 20.0)),
        ]
    except Exception as e:
        logger.error(f"Feature preparation error: {e}")
        return None


def detect_anomaly(features: List[float], tenant_id: str) -> tuple:
    """Z-score anomaly detection — pure Python, no numpy/scipy needed."""
    try:
        stats = _stats_cache[tenant_id]
        mean, std, threshold = stats['mean'], stats['std'], stats['threshold']
        max_z = max(
            abs((f - m) / (s + 1e-8))
            for f, m, s in zip(features, mean, std)
        )
        is_anomaly = 1 if max_z > threshold else 0
        # Negative score mimics sklearn decision_function convention
        anomaly_score = -(max_z / threshold)
        return is_anomaly, round(anomaly_score, 6)
    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        return 0, 0.0


def write_to_dynamodb(record: Dict, is_anomaly: int, anomaly_score: float):
    """Write results to DynamoDB.
    processed_by_prediction=False marks the record as pending for
    Lambda prediction (choreography pattern — US-05).
    """
    try:
        dynamodb = boto3.client('dynamodb')
        table_name = os.getenv('DYNAMO_TABLE')
        timestamp = record['timestamp']
        segment_id = record['segment_id']

        ttl_epoch = int(time.time()) + 7 * 24 * 3600  # expire after 7 days

        dynamodb.put_item(
            TableName=table_name,
            Item={
                'tenantId':               {'S': record['tenant_id']},
                'sortKey':                {'S': f"{timestamp}#{segment_id}"},
                'timestamp':              {'S': timestamp},
                'segmentId':              {'S': segment_id},
                'city':                   {'S': record['city']},
                'pressure':               {'N': str(record['pressure'])},
                'flow':                   {'N': str(record['flow'])},
                'vibration':              {'N': str(record['vibration'])},
                'isAnomaly':              {'BOOL': bool(is_anomaly)},
                'anomalyScore':           {'N': str(anomaly_score)},
                'metadata':               {'S': json.dumps(record.get('metadata', {}))},
                'processed_by_prediction': {'BOOL': False},
                'ttl_epoch':              {'N': str(ttl_epoch)},
            }
        )
        logger.info(f"Written to DynamoDB: {record['tenant_id']} - {segment_id}")
    except Exception as e:
        logger.error(f"DynamoDB write error: {e}")


def _get_expected_tenant(stream_arn: str) -> Optional[str]:
    """Resolve tenant_id from Kinesis stream ARN (EP-05 guard)."""
    if not stream_arn:
        return None
    stream_name = stream_arn.split('/')[-1]
    for tid, cfg in _tenant_configs.items():
        if cfg.get('stream_name') == stream_name:
            return tid
    return None


def lambda_handler(event, context):
    """Main Lambda handler"""
    try:
        logger.info("Anomaly detection Lambda started")
        load_config()

        processed_count = 0
        anomaly_count = 0

        for record in event.get('Records', []):
            try:
                if 'kinesis' in record:
                    payload = json.loads(base64.b64decode(record['kinesis']['data']))
                else:
                    payload = record

                tenant_id = payload['tenant_id']

                # T-07.03: cross-tenant write prevention
                if 'kinesis' in record:
                    expected = _get_expected_tenant(record.get('eventSourceARN', ''))
                    if expected and tenant_id != expected:
                        logger.warning(
                            f"Tenant mismatch: payload='{tenant_id}' "
                            f"on stream for '{expected}'. Dropped."
                        )
                        continue

                if tenant_id not in _tenant_configs:
                    logger.error(f"Unknown tenant: {tenant_id}")
                    continue

                # Load stats on first access or tenant switch
                if tenant_id not in _stats_cache:
                    if not load_stats_from_s3(tenant_id):
                        logger.error(f"Stats unavailable for {tenant_id}, skipping")
                        continue

                features = prepare_features(payload)
                if features is None:
                    continue

                is_anomaly, anomaly_score = detect_anomaly(features, tenant_id)
                write_to_dynamodb(payload, is_anomaly, anomaly_score)

                processed_count += 1
                if is_anomaly:
                    anomaly_count += 1

                logger.info(
                    f"Processed {tenant_id} - {payload['segment_id']}: "
                    f"anomaly={is_anomaly}, score={anomaly_score:.3f}"
                )

            except Exception as e:
                logger.error(f"Record processing error: {e}")
                continue

        logger.info(
            f"Lambda completed: {processed_count} records processed, "
            f"{anomaly_count} anomalies detected"
        )
        return {
            'statusCode': 200,
            'body': json.dumps({
                'processed_records': processed_count,
                'anomalies_detected': anomaly_count,
            }),
        }

    except Exception as e:
        logger.error(f"Lambda handler error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)}),
        }
