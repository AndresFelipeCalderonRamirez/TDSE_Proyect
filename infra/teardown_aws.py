"""
AWS Infrastructure Teardown Script
Removes all water twin prototype resources for lab reset
"""

import boto3
import time
import os
import logging
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WaterTwinTeardown:
    """
    Manages AWS infrastructure teardown for water twin prototype
    """
    
    def __init__(self):
        self.region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        
        # Initialize AWS clients
        self.kinesis = boto3.client('kinesis', region_name=self.region)
        self.dynamodb = boto3.client('dynamodb', region_name=self.region)
        self.s3 = boto3.client('s3', region_name=self.region)
        self.lambda_client = boto3.client('lambda', region_name=self.region)
        self.events = boto3.client('events', region_name=self.region)
        self.sts = boto3.client('sts', region_name=self.region)
        
        # Resource names
        self.account_id = self.sts.get_caller_identity()['Account']
        self.tenant_a_stream = 'tenant-a-stream'
        self.tenant_b_stream = 'tenant-b-stream'
        self.dynamo_table = 'water-twin-data'
        self.model_bucket = f'water-twin-{self.account_id}'
        
        # Lambda function names
        self.anomaly_lambda = 'water-twin-anomaly'
        self.prediction_lambda = 'water-twin-prediction'
        self.digital_twin_lambda = 'water-twin-digital-twin'
        
        # Layer names
        self.sklearn_layer_name = 'sklearn-layer'
        self.networkx_layer_name = 'networkx-layer'
        
        # EventBridge rule
        self.digital_twin_rule = 'water-twin-digital-twin-rule'
        
    def teardown_all(self):
        """Teardown all infrastructure components"""
        logger.info("Starting water twin infrastructure teardown...")
        start_time = time.time()
        
        try:
            # Step 1: Remove EventBridge rule
            self._remove_eventbridge_rule()
            
            # Step 2: Remove Lambda functions
            self._remove_lambda_functions()
            
            # Step 3: Remove Lambda layers
            self._remove_lambda_layers()
            
            # Step 4: Remove S3 bucket
            self._remove_s3_bucket()
            
            # Step 5: Remove DynamoDB table
            self._remove_dynamodb_table()
            
            # Step 6: Remove Kinesis streams
            self._remove_kinesis_streams()
            
            elapsed_time = time.time() - start_time
            logger.info(f"Infrastructure teardown completed in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Infrastructure teardown failed: {str(e)}")
            raise
    
    def _remove_eventbridge_rule(self):
        """Remove EventBridge rule"""
        logger.info("Removing EventBridge rule...")
        
        try:
            # Remove targets first
            self.events.remove_targets(
                Rule=self.digital_twin_rule,
                Ids=['1']
            )
            logger.info(f"Removed targets from rule {self.digital_twin_rule}")
            
            # Remove rule
            self.events.delete_rule(Name=self.digital_twin_rule)
            logger.info(f"Deleted EventBridge rule {self.digital_twin_rule}")
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                logger.info(f"EventBridge rule {self.digital_twin_rule} not found")
            else:
                raise
    
    def _remove_lambda_functions(self):
        """Remove Lambda functions"""
        logger.info("Removing Lambda functions...")
        
        functions = [
            self.anomaly_lambda,
            self.prediction_lambda,
            self.digital_twin_lambda
        ]
        
        for function_name in functions:
            try:
                self.lambda_client.delete_function(FunctionName=function_name)
                logger.info(f"Deleted Lambda function {function_name}")
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    logger.info(f"Lambda function {function_name} not found")
                else:
                    raise
    
    def _remove_lambda_layers(self):
        """Remove Lambda layers"""
        logger.info("Removing Lambda layers...")
        
        layers = [
            self.sklearn_layer_name,
            self.networkx_layer_name
        ]
        
        for layer_name in layers:
            try:
                # Get all versions
                versions = self.lambda_client.list_layer_versions(LayerName=layer_name)
                
                for version in versions['LayerVersions']:
                    self.lambda_client.delete_layer_version(
                        LayerName=layer_name,
                        VersionNumber=version['Version']
                    )
                    logger.info(f"Deleted layer {layer_name} version {version['Version']}")
                    
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    logger.info(f"Lambda layer {layer_name} not found")
                else:
                    raise
    
    def _remove_s3_bucket(self):
        """Remove S3 bucket and all contents"""
        logger.info("Removing S3 bucket...")
        
        try:
            # List and delete all objects
            paginator = self.s3.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(Bucket=self.model_bucket):
                if 'Contents' in page:
                    objects = [{'Key': obj['Key']} for obj in page['Contents']]
                    self.s3.delete_objects(Bucket=self.model_bucket, Delete={'Objects': objects})
            
            # Delete bucket
            self.s3.delete_bucket(Bucket=self.model_bucket)
            logger.info(f"Deleted S3 bucket {self.model_bucket}")
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchBucket':
                logger.info(f"S3 bucket {self.model_bucket} not found")
            else:
                raise
    
    def _remove_dynamodb_table(self):
        """Remove DynamoDB table"""
        logger.info("Removing DynamoDB table...")
        
        try:
            self.dynamodb.delete_table(TableName=self.dynamo_table)
            
            # Wait for table to be deleted
            self._wait_for_dynamodb_deletion()
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                logger.info(f"DynamoDB table {self.dynamo_table} not found")
            else:
                raise
    
    def _wait_for_dynamodb_deletion(self, timeout_seconds: int = 300):
        """Wait for DynamoDB table to be deleted"""
        logger.info(f"Waiting for table {self.dynamo_table} to be deleted...")
        
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            try:
                self.dynamodb.describe_table(TableName=self.dynamo_table)
                time.sleep(5)
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    logger.info(f"Table {self.dynamo_table} deleted successfully")
                    return
                else:
                    raise
        
        raise TimeoutError(f"Table {self.dynamo_table} was not deleted within {timeout_seconds} seconds")
    
    def _remove_kinesis_streams(self):
        """Remove Kinesis streams"""
        logger.info("Removing Kinesis streams...")
        
        streams = [self.tenant_a_stream, self.tenant_b_stream]
        
        for stream_name in streams:
            try:
                self.kinesis.delete_stream(StreamName=stream_name)
                
                # Wait for stream to be deleted
                self._wait_for_kinesis_deletion(stream_name)
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    logger.info(f"Kinesis stream {stream_name} not found")
                else:
                    raise
    
    def _wait_for_kinesis_deletion(self, stream_name: str, timeout_seconds: int = 300):
        """Wait for Kinesis stream to be deleted"""
        logger.info(f"Waiting for stream {stream_name} to be deleted...")
        
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            try:
                self.kinesis.describe_stream(StreamName=stream_name)
                time.sleep(5)
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    logger.info(f"Stream {stream_name} deleted successfully")
                    return
                else:
                    raise
        
        raise TimeoutError(f"Stream {stream_name} was not deleted within {timeout_seconds} seconds")

def main():
    """Main function to run infrastructure teardown"""
    try:
        teardown = WaterTwinTeardown()
        teardown.teardown_all()
        
        print("\n" + "="*50)
        print("✅ Water Twin Infrastructure Teardown Complete!")
        print("="*50)
        print("All resources have been removed from AWS.")
        print("You can now run setup_aws.py to recreate the infrastructure.")
        
    except Exception as e:
        logger.error(f"Teardown failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
