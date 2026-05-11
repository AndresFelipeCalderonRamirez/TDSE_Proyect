"""
AWS Infrastructure Setup Script
Creates all necessary resources for the water twin prototype
Idempotent: can be run multiple times without issues
"""

import boto3
import io
import json
import time
import os
import logging
import zipfile
from pathlib import Path
from botocore.exceptions import ClientError
from typing import Dict, List, Tuple
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WaterTwinInfrastructure:
    """
    Manages AWS infrastructure setup for water twin prototype
    """
    
    def __init__(self):
        self.region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        self.lab_role_arn = os.getenv('LAB_ROLE_ARN')
        
        # Initialize AWS clients
        self.kinesis = boto3.client('kinesis', region_name=self.region)
        self.dynamodb = boto3.client('dynamodb', region_name=self.region)
        self.s3 = boto3.client('s3', region_name=self.region)
        self.lambda_client = boto3.client('lambda', region_name=self.region)
        self.events = boto3.client('events', region_name=self.region)
        self.sts = boto3.client('sts', region_name=self.region)
        self.apigw = boto3.client('apigateway', region_name=self.region)
        
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
        self.dashboard_api_lambda = 'water-twin-dashboard-api'
        
        # Layer names
        self.sklearn_layer_name = 'sklearn-layer'
        self.networkx_layer_name = 'networkx-layer'
        
        # EventBridge rules
        self.digital_twin_rule  = 'water-twin-digital-twin-rule'
        self.prediction_rule    = 'water-twin-prediction-rule'
        
    def setup_all(self):
        """Setup all infrastructure components"""
        logger.info("Starting water twin infrastructure setup...")
        start_time = time.time()
        
        try:
            # Step 1: Create Kinesis streams
            self._create_kinesis_streams()
            
            # Step 2: Create DynamoDB table
            self._create_dynamodb_table()
            
            # Step 3: Create S3 bucket
            self._create_s3_bucket()
            
            # Step 4: Create Lambda layers
            self._create_lambda_layers()
            
            # Step 5: Create Lambda functions
            self._create_lambda_functions()
            
            # Step 6: Setup EventBridge rule
            self._create_eventbridge_rule()

            # Step 7: Kinesis → Anomaly Lambda triggers
            self._create_kinesis_triggers()

            # Step 8: Seed pipe-network topology into DynamoDB
            self._seed_topology()

            # Step 9: Deploy dashboard API Lambda + API Gateway
            api_url = self._create_dashboard_api()

            # Step 10: Update .env file
            self._update_env_file(api_url=api_url)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Infrastructure setup completed in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Infrastructure setup failed: {str(e)}")
            raise
    
    def _create_kinesis_streams(self):
        """Create Kinesis streams for both tenants"""
        logger.info("Creating Kinesis streams...")
        
        streams = [
            (self.tenant_a_stream, "Tenant A (Bogotá) sensor stream"),
            (self.tenant_b_stream, "Tenant B (Medellín) sensor stream")
        ]
        
        for stream_name, description in streams:
            try:
                self.kinesis.describe_stream(StreamName=stream_name)
                logger.info(f"Kinesis stream {stream_name} already exists")
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    logger.info(f"Creating Kinesis stream: {stream_name}")
                    self.kinesis.create_stream(
                        StreamName=stream_name,
                        ShardCount=1  # One shard per tenant for prototype
                    )
                    
                    # Wait for stream to become ACTIVE
                    self._wait_for_kinesis_stream(stream_name)
                else:
                    raise
    
    def _wait_for_kinesis_stream(self, stream_name: str, timeout_seconds: int = 300):
        """Wait for Kinesis stream to become ACTIVE"""
        logger.info(f"Waiting for stream {stream_name} to become ACTIVE...")
        
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            try:
                response = self.kinesis.describe_stream(StreamName=stream_name)
                status = response['StreamDescription']['StreamStatus']
                
                if status == 'ACTIVE':
                    logger.info(f"Stream {stream_name} is ACTIVE")
                    return
                elif status == 'CREATING':
                    time.sleep(5)
                else:
                    raise Exception(f"Unexpected stream status: {status}")
            except Exception as e:
                logger.error(f"Error checking stream status: {str(e)}")
                time.sleep(5)
        
        raise TimeoutError(f"Stream {stream_name} did not become ACTIVE within {timeout_seconds} seconds")
    
    def _create_dynamodb_table(self):
        """Create DynamoDB table with composite key"""
        logger.info("Creating DynamoDB table...")
        
        try:
            self.dynamodb.describe_table(TableName=self.dynamo_table)
            logger.info(f"DynamoDB table {self.dynamo_table} already exists")
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                logger.info(f"Creating DynamoDB table: {self.dynamo_table}")
                
                self.dynamodb.create_table(
                    TableName=self.dynamo_table,
                    KeySchema=[
                        {
                            'AttributeName': 'tenantId',
                            'KeyType': 'HASH'  # Partition key
                        },
                        {
                            'AttributeName': 'sortKey',
                            'KeyType': 'RANGE'  # Sort key (timestamp#segmentId)
                        }
                    ],
                    AttributeDefinitions=[
                        {
                            'AttributeName': 'tenantId',
                            'AttributeType': 'S'
                        },
                        {
                            'AttributeName': 'sortKey',
                            'AttributeType': 'S'
                        }
                    ],
                    BillingMode='PAY_PER_REQUEST',
                    Tags=[
                        {
                            'Key': 'Project',
                            'Value': 'water-twin'
                        },
                        {
                            'Key': 'Environment',
                            'Value': 'prototype'
                        }
                    ]
                )
                
                # Wait for table to become ACTIVE
                self._wait_for_dynamodb_table()
            else:
                raise

        # Enable TTL regardless of whether the table was just created or already
        # existed — idempotent call so re-runs are safe.
        self._ensure_ttl_enabled()

    def _ensure_ttl_enabled(self) -> None:
        """Enable DynamoDB TTL on the ttl_epoch attribute (idempotent)."""
        try:
            self.dynamodb.update_time_to_live(
                TableName=self.dynamo_table,
                TimeToLiveSpecification={
                    'Enabled': True,
                    'AttributeName': 'ttl_epoch',
                },
            )
            logger.info(f"TTL enabled on '{self.dynamo_table}' (attribute: ttl_epoch)")
        except ClientError as e:
            # ValidationException is raised when TTL is already enabled with the same attribute
            if e.response['Error']['Code'] == 'ValidationException':
                logger.info("TTL already active on 'ttl_epoch' — skipping")
            else:
                raise

    def _wait_for_dynamodb_table(self, timeout_seconds: int = 300):
        """Wait for DynamoDB table to become ACTIVE"""
        logger.info(f"Waiting for table {self.dynamo_table} to become ACTIVE...")
        
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            try:
                response = self.dynamodb.describe_table(TableName=self.dynamo_table)
                status = response['Table']['TableStatus']
                
                if status == 'ACTIVE':
                    logger.info(f"Table {self.dynamo_table} is ACTIVE")
                    return
                elif status in ['CREATING', 'UPDATING']:
                    time.sleep(5)
                else:
                    raise Exception(f"Unexpected table status: {status}")
            except Exception as e:
                logger.error(f"Error checking table status: {str(e)}")
                time.sleep(5)
        
        raise TimeoutError(f"Table {self.dynamo_table} did not become ACTIVE within {timeout_seconds} seconds")
    
    def _create_s3_bucket(self):
        """Create S3 bucket with required folders"""
        logger.info("Creating S3 bucket...")
        
        try:
            self.s3.head_bucket(Bucket=self.model_bucket)
            logger.info(f"S3 bucket {self.model_bucket} already exists")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.info(f"Creating S3 bucket: {self.model_bucket}")
                
                if self.region == 'us-east-1':
                    # us-east-1 doesn't need location constraint
                    self.s3.create_bucket(Bucket=self.model_bucket)
                else:
                    self.s3.create_bucket(
                        Bucket=self.model_bucket,
                        CreateBucketConfiguration={
                            'LocationConstraint': self.region
                        }
                    )
                
                # Wait for bucket to exist
                self._wait_for_s3_bucket()
            else:
                raise
        
        # Create required folders
        folders = ['models/', 'layers/']
        for folder in folders:
            self.s3.put_object(Bucket=self.model_bucket, Key=folder, Body=b'')
        
        logger.info(f"S3 bucket {self.model_bucket} ready with folders: {folders}")
    
    def _wait_for_s3_bucket(self, timeout_seconds: int = 60):
        """Wait for S3 bucket to become available"""
        logger.info(f"Waiting for bucket {self.model_bucket} to become available...")
        
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            try:
                self.s3.head_bucket(Bucket=self.model_bucket)
                logger.info(f"Bucket {self.model_bucket} is available")
                return
            except ClientError:
                time.sleep(2)
        
        raise TimeoutError(f"Bucket {self.model_bucket} did not become available within {timeout_seconds} seconds")
    
    def _create_lambda_layers(self):
        """Create Lambda layers for sklearn and networkx"""
        logger.info("Creating Lambda layers...")
        
        # Note: This is a placeholder - actual layer creation requires
        # building zip files with the required packages
        # The actual implementation will be in the build_*_layer.sh scripts
        
        layers = [
            (self.sklearn_layer_name, "scikit-learn, xgboost, joblib, numpy"),
            (self.networkx_layer_name, "networkx, joblib")
        ]
        
        for layer_name, description in layers:
            try:
                self.lambda_client.get_layer_version(LayerName=layer_name, VersionNumber=1)
                logger.info(f"Lambda layer {layer_name} already exists")
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    logger.info(f"Lambda layer {layer_name} will be created by build scripts")
                    # Placeholder - actual creation done by shell scripts
                else:
                    raise
    
    def _create_lambda_functions(self):
        """Create Lambda functions with placeholder handlers"""
        logger.info("Creating Lambda functions...")
        
        functions = [
            (self.anomaly_lambda, "anomaly_detection.lambda_handler"),
            (self.prediction_lambda, "failure_prediction.lambda_handler"),
            (self.digital_twin_lambda, "digital_twin.lambda_handler")
        ]
        
        # Placeholder lambda code — must be packaged as a real ZIP file
        placeholder_source = (
            "def lambda_handler(event, context):\n"
            "    return {'statusCode': 200, 'body': 'placeholder'}\n"
        )
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("lambda_function.py", placeholder_source)
        placeholder_code = buf.getvalue()

        for function_name, handler in functions:
            try:
                self.lambda_client.get_function(FunctionName=function_name)
                logger.info(f"Lambda function {function_name} already exists")
            except ClientError as e:
                if e.response['Error']['Code'] == 'ResourceNotFoundException':
                    logger.info(f"Creating Lambda function: {function_name}")
                    
                    self.lambda_client.create_function(
                        FunctionName=function_name,
                        Runtime='python3.12',
                        Role=self.lab_role_arn,
                        Handler=handler,
                        Code={
                            'ZipFile': placeholder_code
                        },
                        Description=f'Water Twin {function_name}',
                        Timeout=60,
                        MemorySize=512,
                        Environment={
                            'Variables': {
                                'DYNAMO_TABLE': self.dynamo_table,
                                'MODEL_BUCKET': self.model_bucket,
                                'TENANT_A_STREAM': self.tenant_a_stream,
                                'TENANT_B_STREAM': self.tenant_b_stream,
                            }
                        },
                        Tags={
                            'Project': 'water-twin',
                            'Environment': 'prototype'
                        }
                    )
                else:
                    raise
    
    def _create_kinesis_triggers(self):
        """Create Kinesis event source mappings for the Anomaly Lambda (idempotent)."""
        logger.info("Creating Kinesis → Anomaly Lambda event source mappings...")

        streams = [
            (self.tenant_a_stream, "tenant-A"),
            (self.tenant_b_stream, "tenant-B"),
        ]

        for stream_name, label in streams:
            # Get the stream ARN
            try:
                resp = self.kinesis.describe_stream(StreamName=stream_name)
                stream_arn = resp["StreamDescription"]["StreamARN"]
            except ClientError as e:
                logger.error(f"Cannot describe stream {stream_name}: {e}")
                raise

            # Check if mapping already exists
            existing = self.lambda_client.list_event_source_mappings(
                FunctionName=self.anomaly_lambda,
                EventSourceArn=stream_arn,
            )
            if existing.get("EventSourceMappings"):
                logger.info(f"Event source mapping for {stream_name} already exists")
                continue

            self.lambda_client.create_event_source_mapping(
                FunctionName=self.anomaly_lambda,
                EventSourceArn=stream_arn,
                StartingPosition="LATEST",
                BatchSize=100,
                BisectBatchOnFunctionError=True,
            )
            logger.info(f"Created Kinesis trigger: {stream_name} → {self.anomaly_lambda} ({label})")

    def _create_eventbridge_rule(self):
        """Create EventBridge rules for digital twin and failure prediction"""
        logger.info("Creating EventBridge rules...")

        self._ensure_eventbridge_rule(
            rule_name=self.digital_twin_rule,
            schedule='rate(1 minute)',
            description='Trigger digital twin updates every minute',
            target_function=self.digital_twin_lambda,
            target_id='1',
            statement_id='eventbridge-invoke',
        )
        self._ensure_eventbridge_rule(
            rule_name=self.prediction_rule,
            schedule='rate(1 minute)',
            description='Trigger failure prediction Lambda every minute (US-05 choreography)',
            target_function=self.prediction_lambda,
            target_id='1',
            statement_id='eventbridge-invoke',
        )

    def _ensure_eventbridge_rule(
        self,
        rule_name: str,
        schedule: str,
        description: str,
        target_function: str,
        target_id: str,
        statement_id: str,
    ) -> None:
        """Idempotent helper: create one EventBridge scheduled rule + Lambda target."""
        try:
            self.events.describe_rule(Name=rule_name)
            logger.info(f"EventBridge rule {rule_name} already exists")
            return
        except ClientError as e:
            if e.response['Error']['Code'] != 'ResourceNotFoundException':
                raise

        logger.info(f"Creating EventBridge rule: {rule_name}")

        self.events.put_rule(
            Name=rule_name,
            ScheduleExpression=schedule,
            State='ENABLED',
            Description=description,
        )

        lambda_arn = (
            f'arn:aws:lambda:{self.region}:{self.account_id}'
            f':function:{target_function}'
        )
        self.events.put_targets(
            Rule=rule_name,
            Targets=[
                {
                    'Id': target_id,
                    'Arn': lambda_arn,
                    'RetryPolicy': {'MaximumRetryAttempts': 0},
                }
            ],
        )

        self.lambda_client.add_permission(
            FunctionName=target_function,
            StatementId=statement_id,
            Action='lambda:InvokeFunction',
            Principal='events.amazonaws.com',
            SourceArn=(
                f'arn:aws:events:{self.region}:{self.account_id}:rule/{rule_name}'
            ),
        )
    
    # ── Topology seeding (EP-04 / T-06.01) ───────────────────────────────────

    def _seed_topology(self) -> None:
        """
        Write synthetic pipe-network topology for each tenant to DynamoDB.
        Item schema:
          PK  = tenantId
          SK  = topology#config
          nodes = JSON list of segment IDs (e.g. ["SEG_A_000", ...])
          edges = JSON list of [src, dst] pairs
        Idempotent: skips if the item already exists.
        """
        logger.info("Seeding pipe-network topology...")
        dynamodb = boto3.client('dynamodb', region_name=self.region)

        tenant_configs = [
            ("tenant-A", 50),
            ("tenant-B", 40),
        ]

        for tenant_id, n_segments in tenant_configs:
            # Idempotency check
            resp = dynamodb.get_item(
                TableName=self.dynamo_table,
                Key={
                    "tenantId": {"S": tenant_id},
                    "sortKey":  {"S": "topology#config"},
                },
            )
            if resp.get("Item"):
                logger.info(f"Topology for {tenant_id} already present — skipping")
                continue

            nodes, edges = self._build_synthetic_topology(n_segments, tenant_id)

            dynamodb.put_item(
                TableName=self.dynamo_table,
                Item={
                    "tenantId":      {"S": tenant_id},
                    "sortKey":       {"S": "topology#config"},
                    "nodes":         {"S": json.dumps(nodes)},
                    "edges":         {"S": json.dumps(edges)},
                    "n_segments":    {"N": str(n_segments)},
                    "topology_type": {"S": "synthetic"},
                },
            )
            logger.info(
                f"Seeded topology for {tenant_id}: "
                f"{len(nodes)} nodes, {len(edges)} edges"
            )

    @staticmethod
    def _build_synthetic_topology(
        n_segments: int,
        tenant_id: str,
    ):
        """
        Build a realistic water-distribution network with:
          - One main trunk (linear backbone)
          - Four branches distributed evenly along the trunk
          - Cross-links between adjacent branch ends (small redundancy loops)

        Segment IDs match the IoT simulator format: SEG_{A|B}_{NNN}

        Tenant A (50 nodes) → 10-node trunk + 4 × 10-node branches + 3 loops
        Tenant B (40 nodes) →  8-node trunk + 4 × 8-node branches  + 3 loops
        """
        short  = tenant_id.split("-")[1].upper()
        prefix = f"SEG_{short}_"
        nodes  = [f"{prefix}{i:03d}" for i in range(n_segments)]

        edges = []
        trunk_n      = max(5, n_segments // 5)   # 10 for A, 8 for B
        branch_count = 4
        branch_len   = (n_segments - trunk_n) // branch_count

        # Main trunk
        for i in range(trunk_n - 1):
            edges.append([nodes[i], nodes[i + 1]])

        # Branches: evenly distributed attachment points along the trunk
        for b in range(branch_count):
            attach = round((b + 1) * (trunk_n - 1) / (branch_count + 1))
            start  = trunk_n + b * branch_len
            end    = start + branch_len if b < branch_count - 1 else n_segments

            # Trunk → branch root
            edges.append([nodes[attach], nodes[start]])
            # Branch internal chain
            for i in range(start, end - 1):
                edges.append([nodes[i], nodes[i + 1]])

        # Cross-links: last node of branch b → first node of branch b+1
        for b in range(branch_count - 1):
            b_end        = trunk_n + (b + 1) * branch_len - 1
            b_next_start = trunk_n + (b + 1) * branch_len
            if b_end < n_segments and b_next_start < n_segments:
                edges.append([nodes[b_end], nodes[b_next_start]])

        return nodes, edges

    def _create_dashboard_api(self) -> str:
        """Deploy dashboard API Lambda and API Gateway REST endpoint. Returns the invoke URL."""
        logger.info("Creating dashboard API Lambda + API Gateway...")

        # ── 1. Package the Lambda code ────────────────────────────────────────
        src = Path(__file__).parent.parent / "lambdas" / "dashboard_api" / "lambda_function.py"
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(src, "lambda_function.py")
        zip_bytes = buf.getvalue()

        # ── 2. Create or update the Lambda ───────────────────────────────────
        try:
            self.lambda_client.get_function(FunctionName=self.dashboard_api_lambda)
            logger.info(f"Updating {self.dashboard_api_lambda} code...")
            self.lambda_client.update_function_code(
                FunctionName=self.dashboard_api_lambda,
                ZipFile=zip_bytes,
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                logger.info(f"Creating Lambda {self.dashboard_api_lambda}...")
                self.lambda_client.create_function(
                    FunctionName=self.dashboard_api_lambda,
                    Runtime="python3.12",
                    Role=self.lab_role_arn,
                    Handler="lambda_function.lambda_handler",
                    Code={"ZipFile": zip_bytes},
                    Timeout=30,
                    MemorySize=256,
                    Environment={
                        "Variables": {
                            "DYNAMO_TABLE": self.dynamo_table,
                        }
                    },
                )
            else:
                raise

        # Wait for Lambda to be active
        waiter = self.lambda_client.get_waiter("function_active_v2")
        waiter.wait(FunctionName=self.dashboard_api_lambda)

        lambda_arn = self.lambda_client.get_function(
            FunctionName=self.dashboard_api_lambda
        )["Configuration"]["FunctionArn"]

        # ── 3. Create or reuse API Gateway REST API ───────────────────────────
        api_name = "water-twin-dashboard-api"
        api_id = None
        for page in self.apigw.get_paginator("get_rest_apis").paginate():
            for item in page["items"]:
                if item["name"] == api_name:
                    api_id = item["id"]
                    logger.info(f"API Gateway '{api_name}' already exists: {api_id}")
                    break
            if api_id:
                break

        if not api_id:
            logger.info(f"Creating API Gateway '{api_name}'...")
            api_id = self.apigw.create_rest_api(
                name=api_name,
                description="Dashboard REST API for React frontend",
                endpointConfiguration={"types": ["REGIONAL"]},
            )["id"]

        # ── 4. Create /{proxy+} resource with ANY method → Lambda ────────────
        root_id = self.apigw.get_resources(restApiId=api_id)["items"][0]["id"]

        # Find or create {proxy+} resource
        resources = self.apigw.get_resources(restApiId=api_id)["items"]
        proxy_resource = next(
            (r for r in resources if r.get("pathPart") == "{proxy+}"), None
        )
        if not proxy_resource:
            proxy_resource = self.apigw.create_resource(
                restApiId=api_id,
                parentId=root_id,
                pathPart="{proxy+}",
            )

        proxy_id = proxy_resource["id"]
        lambda_uri = (
            f"arn:aws:apigateway:{self.region}:lambda:path/2015-03-31"
            f"/functions/{lambda_arn}/invocations"
        )

        for method in ("ANY", "OPTIONS"):
            try:
                self.apigw.get_method(
                    restApiId=api_id, resourceId=proxy_id, httpMethod=method
                )
            except ClientError:
                self.apigw.put_method(
                    restApiId=api_id,
                    resourceId=proxy_id,
                    httpMethod=method,
                    authorizationType="NONE",
                    requestParameters={"method.request.querystring.tenant_id": False},
                )
                self.apigw.put_integration(
                    restApiId=api_id,
                    resourceId=proxy_id,
                    httpMethod=method,
                    type="AWS_PROXY",
                    integrationHttpMethod="POST",
                    uri=lambda_uri,
                )

        # ── 5. Grant API Gateway permission to invoke Lambda ──────────────────
        source_arn = (
            f"arn:aws:execute-api:{self.region}:{self.account_id}:{api_id}/*/*/{{proxy+}}"
        )
        try:
            self.lambda_client.add_permission(
                FunctionName=self.dashboard_api_lambda,
                StatementId="apigw-invoke",
                Action="lambda:InvokeFunction",
                Principal="apigateway.amazonaws.com",
                SourceArn=source_arn,
            )
        except ClientError as e:
            if e.response["Error"]["Code"] != "ResourceConflictException":
                raise

        # ── 6. Deploy to 'prod' stage ─────────────────────────────────────────
        self.apigw.create_deployment(restApiId=api_id, stageName="prod")
        invoke_url = (
            f"https://{api_id}.execute-api.{self.region}.amazonaws.com/prod"
        )
        logger.info(f"Dashboard API deployed: {invoke_url}")
        return invoke_url

    def _update_env_file(self, api_url: str = ""):
        """Update .env file preserving existing credentials and layer ARNs."""
        logger.info("Updating .env file...")

        # Preserve values that must not be wiped on re-runs
        preserved_keys = [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "SKLEARN_LAYER_ARN",
            "NETWORKX_LAYER_ARN",
        ]
        preserved = {k: os.getenv(k, "") for k in preserved_keys}

        env_content = f"""# AWS Configuration (Auto-generated)
AWS_DEFAULT_REGION={self.region}
AWS_ACCESS_KEY_ID={preserved["AWS_ACCESS_KEY_ID"]}
AWS_SECRET_ACCESS_KEY={preserved["AWS_SECRET_ACCESS_KEY"]}
AWS_SESSION_TOKEN={preserved["AWS_SESSION_TOKEN"]}
LAB_ROLE_ARN={self.lab_role_arn}
ACCOUNT_ID={self.account_id}

# DynamoDB Configuration
DYNAMO_TABLE={self.dynamo_table}

# S3 Configuration
MODEL_BUCKET={self.model_bucket}

# Kinesis Streams
TENANT_A_STREAM={self.tenant_a_stream}
TENANT_B_STREAM={self.tenant_b_stream}

# Lambda Functions
ANOMALY_LAMBDA={self.anomaly_lambda}
PREDICTION_LAMBDA={self.prediction_lambda}
DIGITAL_TWIN_LAMBDA={self.digital_twin_lambda}
DASHBOARD_API_LAMBDA={self.dashboard_api_lambda}

# React Dashboard API
DASHBOARD_API_URL={api_url}

# Lambda Layers
SKLEARN_LAYER_ARN={preserved["SKLEARN_LAYER_ARN"]}
NETWORKX_LAYER_ARN={preserved["NETWORKX_LAYER_ARN"]}

# EventBridge Rules
DIGITAL_TWIN_RULE={self.digital_twin_rule}
PREDICTION_RULE={self.prediction_rule}

# Failure Prediction Parameters (US-05)
PREDICTION_BATCH_SIZE=100

# Model Training Parameters
ISOLATION_FOREST_CONTAMINATION=0.02
ISOLATION_FOREST_N_ESTIMATORS=100
ANOMALY_THRESHOLD=0.65

# Digital Twin Parameters (EP-04 / US-06)
RISK_PROPAGATION_ALPHA=0.7
RANKING_TOP_K=5
RISK_WINDOW_HOURS=2

# Simulation Parameters
ANOMALY_RATE=0.02
ANOMALY_DURATION_SECONDS=10
SIMULATION_FREQUENCY_HZ=1.0

# Tenant Configuration
TENANT_A_SEGMENTS=50
TENANT_B_SEGMENTS=40
TENANT_A_ALTITUDE=2600
TENANT_B_ALTITUDE=1500
"""

        env_path = Path(__file__).parent.parent / ".env"
        with open(env_path, "w") as f:
            f.write(env_content)

        logger.info(".env file updated successfully")

def main():
    """Main function to run infrastructure setup"""
    try:
        # Check for required environment variables
        if not os.getenv('LAB_ROLE_ARN'):
            raise ValueError("LAB_ROLE_ARN environment variable is required")
        
        # Setup infrastructure
        infra = WaterTwinInfrastructure()
        infra.setup_all()
        
        print("\n" + "="*50)
        print("✅ Water Twin Infrastructure Setup Complete!")
        print("="*50)
        print(f"Region: {infra.region}")
        print(f"Account ID: {infra.account_id}")
        print(f"Kinesis Streams: {infra.tenant_a_stream}, {infra.tenant_b_stream}")
        print(f"DynamoDB Table: {infra.dynamo_table}")
        print(f"S3 Bucket: {infra.model_bucket}")
        print(f"Lambda Functions: {infra.anomaly_lambda}, {infra.prediction_lambda}, {infra.digital_twin_lambda}")
        print(f"EventBridge Rules: {infra.digital_twin_rule}, {infra.prediction_rule}")
        print("\nNext steps:")
        print("1. Build Lambda layers using scripts in layers/ directory")
        print("2. Update .env file with layer ARNs")
        print("3. Deploy actual Lambda function code")
        print("4. Run IoT simulators")
        
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
