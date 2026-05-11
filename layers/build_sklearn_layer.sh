#!/bin/bash

# Build Lambda Layer for scikit-learn, xgboost, joblib, numpy
# Compatible with Python 3.12 Lambda runtime

set -e

echo "Building scikit-learn Lambda Layer..."

# Create temporary directory
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Create directory structure for Lambda layer
mkdir -p python/lib/python3.12/site-packages

# Install packages for manylinux2014_x86_64 (Lambda architecture)
echo "Installing packages..."
pip install \
    --platform manylinux2014_x86_64 \
    --target=python/lib/python3.12/site-packages \
    --only-binary=:all: \
    --upgrade \
    scikit-learn==1.4.2 \
    xgboost==2.0.3 \
    joblib==1.3.2 \
    numpy==1.26.4

# Create zip file
echo "Creating layer zip..."
zip -r sklearn-layer.zip python/

# Get AWS account ID and region
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=${AWS_DEFAULT_REGION:-us-east-1}
BUCKET_NAME="water-twin-${ACCOUNT_ID}"

# Upload to S3
echo "Uploading to S3 bucket: ${BUCKET_NAME}"
aws s3 cp sklearn-layer.zip "s3://${BUCKET_NAME}/layers/sklearn-layer.zip"

# Publish Lambda layer
echo "Publishing Lambda layer..."
LAYER_VERSION=$(aws lambda publish-layer-version \
    --layer-name sklearn-layer \
    --description "scikit-learn, xgboost, joblib, numpy for Python 3.12" \
    --license-info "MIT" \
    --compatible-runtimes python3.12 \
    --content "fileb://sklearn-layer.zip" \
    --query Version \
    --output text)

LAYER_ARN="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:layer:sklearn-layer:${LAYER_VERSION}"

echo "✅ scikit-learn layer created successfully!"
echo "Layer ARN: ${LAYER_ARN}"
echo "Version: ${LAYER_VERSION}"

# Update .env file with layer ARN
if [ -f .env ]; then
    sed -i.bak "s/SKLEARN_LAYER_ARN=/SKLEARN_LAYER_ARN=${LAYER_ARN}/" .env
    echo "Updated .env file with sklearn layer ARN"
fi

# Cleanup
cd ..
rm -rf "$TEMP_DIR"

echo "Layer build completed!"
