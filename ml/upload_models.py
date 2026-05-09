"""
Upload trained ML models to S3
Handles model versioning and metadata
"""

import os
import json
import boto3
import argparse
import logging
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelUploader:
    """
    Uploads trained ML models to S3 with metadata
    """
    
    def __init__(self):
        self.s3 = boto3.client('s3', region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1'))
        self.model_bucket = os.getenv('MODEL_BUCKET')
        
        if not self.model_bucket:
            raise ValueError("MODEL_BUCKET environment variable is required")
    
    def upload_model_files(self, tenant_id: str, artifacts_dir: str) -> bool:
        """Upload all model files for a tenant"""
        try:
            tenant_short = tenant_id.split('-')[1]  # A or B
            s3_prefix = f'models/{tenant_short}'
            
            # Model files to upload
            model_files = [
                'isolation_forest.pkl',
                'random_forest.pkl',
                'xgboost.pkl',
                'scaler.pkl'
            ]
            
            # Check if artifacts directory exists
            if not os.path.exists(artifacts_dir):
                logger.error(f"Artifacts directory not found: {artifacts_dir}")
                return False
            
            # Upload each model file
            uploaded_files = []
            for model_file in model_files:
                local_path = os.path.join(artifacts_dir, model_file)
                
                if os.path.exists(local_path):
                    s3_key = f'{s3_prefix}/{model_file}'
                    
                    # Upload file
                    self.s3.upload_file(local_path, self.model_bucket, s3_key)
                    logger.info(f"Uploaded {model_file} to s3://{self.model_bucket}/{s3_key}")
                    
                    uploaded_files.append({
                        'file': model_file,
                        's3_key': s3_key,
                        'size': os.path.getsize(local_path),
                        'last_modified': datetime.fromtimestamp(os.path.getmtime(local_path)).isoformat()
                    })
                else:
                    logger.warning(f"Model file not found: {local_path}")
            
            # Create and upload metadata
            metadata = {
                'tenant_id': tenant_id,
                'upload_timestamp': datetime.utcnow().isoformat(),
                'model_files': uploaded_files,
                'model_versions': {
                    'isolation_forest': '1.0.0',
                    'random_forest': '1.0.0',
                    'xgboost': '1.0.0'
                }
            }
            
            metadata_key = f'{s3_prefix}/metadata.json'
            self.s3.put_object(
                Bucket=self.model_bucket,
                Key=metadata_key,
                Body=json.dumps(metadata, indent=2),
                ContentType='application/json'
            )
            
            logger.info(f"Uploaded metadata to s3://{self.model_bucket}/{metadata_key}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload models for {tenant_id}: {str(e)}")
            return False
    
    def list_uploaded_models(self, tenant_id: str = None) -> Dict:
        """List uploaded models from S3"""
        try:
            if tenant_id:
                tenant_short = tenant_id.split('-')[1]
                prefix = f'models/{tenant_short}/'
            else:
                prefix = 'models/'
            
            response = self.s3.list_objects_v2(
                Bucket=self.model_bucket,
                Prefix=prefix
            )
            
            models = {}
            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    size = obj['Size']
                    last_modified = obj['LastModified'].isoformat()
                    
                    # Extract tenant from key
                    if 'models/' in key:
                        path_parts = key.split('/')
                        if len(path_parts) >= 2:
                            tenant = path_parts[1]
                            filename = path_parts[2] if len(path_parts) > 2 else ''
                            
                            if tenant not in models:
                                models[tenant] = {}
                            
                            models[tenant][filename] = {
                                's3_key': key,
                                'size': size,
                                'last_modified': last_modified
                            }
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}")
            return {}
    
    def download_model(self, tenant_id: str, model_name: str, local_path: str) -> bool:
        """Download a specific model from S3"""
        try:
            tenant_short = tenant_id.split('-')[1]
            s3_key = f'models/{tenant_short}/{model_name}'
            
            self.s3.download_file(self.model_bucket, s3_key, local_path)
            logger.info(f"Downloaded {model_name} to {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {str(e)}")
            return False

def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description='Upload ML models to S3')
    parser.add_argument('--tenant', 
                       choices=['tenant-A', 'tenant-B', 'both'],
                       default='both',
                       help='Tenant to upload models for (default: both)')
    parser.add_argument('--artifacts-dir',
                       type=str,
                       default='ml/artifacts',
                       help='Base artifacts directory (default: ml/artifacts)')
    parser.add_argument('--list',
                       action='store_true',
                       help='List uploaded models')
    parser.add_argument('--download',
                       type=str,
                       help='Download specific model (format: tenant-A:isolation_forest.pkl)')
    
    args = parser.parse_args()
    
    logger.info("Water Twin Model Uploader")
    
    uploader = ModelUploader()
    
    if args.list:
        models = uploader.list_uploaded_models()
        print("\nUploaded Models:")
        for tenant, files in models.items():
            print(f"\n{tenant}:")
            for filename, info in files.items():
                print(f"  {filename} ({info['size']} bytes, {info['last_modified']})")
        return
    
    if args.download:
        try:
            tenant_id, model_name = args.download.split(':')
            local_path = f"downloaded_{model_name}"
            
            success = uploader.download_model(tenant_id, model_name, local_path)
            if success:
                print(f"✅ Downloaded {model_name} to {local_path}")
            else:
                print(f"❌ Failed to download {model_name}")
        except ValueError:
            print("❌ Invalid download format. Use: tenant-A:isolation_forest.pkl")
        return
    
    tenants = ['tenant-A', 'tenant-B'] if args.tenant == 'both' else [args.tenant]
    
    for tenant_id in tenants:
        logger.info(f"\nProcessing {tenant_id}")
        
        tenant_short = tenant_id.split('-')[1]
        artifacts_dir = os.path.join(args.artifacts_dir, tenant_short)
        
        success = uploader.upload_model_files(tenant_id, artifacts_dir)
        
        if success:
            logger.info(f"✅ {tenant_id} models uploaded successfully")
        else:
            logger.error(f"❌ {tenant_id} upload failed")

if __name__ == "__main__":
    main()
