"""
Deploy real Lambda function code and attach layers.
Run after setup_aws.py and after layers are published.
"""

import boto3
import io
import os
import zipfile
import logging
from pathlib import Path
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

LAMBDAS = [
    {
        "name": os.getenv("ANOMALY_LAMBDA", "water-twin-anomaly"),
        "src_dir": "lambdas/anomaly_detection",
        "handler": "lambda_function.lambda_handler",
        "layers": [],  # pure Python + boto3 (built-in) — no layers needed
    },
    {
        "name": os.getenv("PREDICTION_LAMBDA", "water-twin-prediction"),
        "src_dir": "lambdas/failure_prediction",
        "handler": "lambda_function.lambda_handler",
        "layers": [
            os.getenv("COMBINED_ML_LAYER_ARN", ""),  # numpy+sklearn+scipy+xgboost
        ],
    },
    {
        "name": os.getenv("DIGITAL_TWIN_LAMBDA", "water-twin-digital-twin"),
        "src_dir": "lambdas/digital_twin",
        "handler": "lambda_function.lambda_handler",
        "layers": [
            os.getenv("NETWORKX_LAYER_ARN", ""),
        ],
    },
    {
        "name": os.getenv("DASHBOARD_API_LAMBDA", "water-twin-dashboard-api"),
        "src_dir": "lambdas/dashboard_api",
        "handler": "lambda_function.lambda_handler",
        "layers": [],
    },
]


def _zip_directory(src_dir: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for py_file in src_dir.glob("*.py"):
            zf.write(py_file, py_file.name)
    return buf.getvalue()


def deploy():
    client = boto3.client("lambda", region_name=REGION)
    project_root = Path(__file__).parent.parent

    for cfg in LAMBDAS:
        name = cfg["name"]
        src = project_root / cfg["src_dir"]

        if not src.exists():
            logger.warning(f"Source directory not found, skipping: {src}")
            continue

        logger.info(f"Zipping {src} ...")
        code_zip = _zip_directory(src)

        logger.info(f"Updating code for {name} ...")
        try:
            client.update_function_code(
                FunctionName=name,
                ZipFile=code_zip,
                Publish=True,
            )
            # Wait until Lambda finishes the code update before touching config
            waiter = client.get_waiter("function_updated")
            waiter.wait(FunctionName=name)
        except ClientError as e:
            logger.error(f"Failed to update code for {name}: {e}")
            raise

        # Attach layers (filter out empty strings)
        layers = [arn for arn in cfg["layers"] if arn]
        if layers:
            logger.info(f"Attaching layers to {name}: {layers}")
            try:
                client.update_function_configuration(
                    FunctionName=name,
                    Handler=cfg["handler"],
                    Layers=layers,
                )
            except ClientError as e:
                logger.error(f"Failed to update config for {name}: {e}")
                raise
        else:
            logger.info(f"No layers to attach for {name}")

        logger.info(f"✅ {name} deployed successfully")

    print("\n✅ All Lambda functions deployed!")


if __name__ == "__main__":
    deploy()
