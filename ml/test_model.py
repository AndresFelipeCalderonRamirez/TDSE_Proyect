"""
Simple test script for Isolation Forest model
Tests with small dataset to verify functionality
"""

import pandas as pd
import numpy as np
import joblib
import sys
import os
from sklearn.metrics import classification_report, confusion_matrix

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_isolation_forest():
    """Test Isolation Forest with existing data"""
    print("Testing Isolation Forest Model")
    print("="*50)
    
    # Load model
    model_path = 'ml/artifacts/A/isolation_forest.pkl'
    scaler_path = 'ml/artifacts/A/scaler.pkl'
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("Model and scaler loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Load test data
    data_path = 'data/outputs/tenant-A_sensor_data.csv'
    try:
        df = pd.read_csv(data_path, encoding='utf-8-sig')
        print(f"Loaded {len(df)} test records")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return
    
    # Prepare features (same as training)
    features = [
        'pressure', 'flow', 'vibration', 'hour', 'day_of_week', 'is_weekend'
    ]
    
    # Add rolling features if available
    rolling_features = [col for col in df.columns if 'rolling' in col]
    features.extend(rolling_features)
    
    # Add rate features if available  
    rate_features = [col for col in df.columns if '_rate' in col]
    features.extend(rate_features)
    
    # Add metadata features if available
    meta_features = [col for col in df.columns if col.startswith('meta_')]
    features.extend(meta_features)
    
    # Filter available features
    available_features = [f for f in features if f in df.columns]
    print(f"Using {len(available_features)} features: {available_features[:5]}...")
    
    # Add feature engineering (same as training)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Parse metadata if available
    if 'metadata' in df.columns:
        metadata_list = []
        for meta_str in df['metadata']:
            try:
                metadata = json.loads(meta_str)
                metadata_list.append(metadata)
            except:
                metadata_list.append({})
        
        meta_df = pd.DataFrame(metadata_list)
        for col in ['diameter', 'age', 'length', 'elevation']:
            if col in meta_df.columns:
                df[f'meta_{col}'] = meta_df[col]
        
        if 'material' in meta_df.columns:
            df['meta_material'] = meta_df['material']
    
    # Create rolling features (simplified for test)
    df = df.sort_values(['segment_id', 'timestamp'])
    
    for col in ['pressure', 'flow', 'vibration']:
        df[f'{col}_rolling_mean_5'] = df.groupby('segment_id')[col].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )
        df[f'{col}_rolling_mean_10'] = df.groupby('segment_id')[col].transform(
            lambda x: x.rolling(window=10, min_periods=1).mean()
        )
        df[f'{col}_rate'] = df.groupby('segment_id')[col].transform(
            lambda x: x.diff()
        )
    
    # Fill NaN values
    df = df.bfill().fillna(0)
    
    # Re-select available features after engineering
    available_features = [f for f in features if f in df.columns]
    print(f"Using {len(available_features)} features after engineering")
    
    # Prepare test data
    X = df[available_features].copy()
    y = df['is_anomaly']
    
    # Handle categorical features
    categorical_columns = X.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if col not in ['tenant_id', 'city', 'segment_id', 'timestamp']:
            le = {val: i for i, val in enumerate(X[col].unique())}
            X[col] = X[col].map(le)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Predict
    y_pred = model.predict(X_scaled)
    y_pred_binary = (y_pred == -1).astype(int)  # -1 = anomaly -> 1
    
    # Calculate metrics
    accuracy = (y == y_pred_binary).mean()
    
    # Detailed metrics
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy:.3f}")
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred_binary)
    print(f"Confusion Matrix:")
    print(f"True Negatives: {cm[0][0]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"False Negatives: {cm[1][0]}")
    print(f"True Positives: {cm[1][1]}")
    
    # Calculate recall, precision
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nDetailed Metrics:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    
    # Show some examples
    print(f"\nSample Predictions:")
    anomaly_indices = np.where(y_pred_binary == 1)[0][:5]
    normal_indices = np.where(y_pred_binary == 0)[0][:5]
    
    print("Anomalies detected:")
    for idx in anomaly_indices:
        print(f"  {df.iloc[idx]['segment_id']} - {df.iloc[idx]['timestamp']} - "
              f"Pressure: {df.iloc[idx]['pressure']:.2f}, "
              f"Flow: {df.iloc[idx]['flow']:.2f}, "
              f"Vibration: {df.iloc[idx]['vibration']:.2f}")
    
    print("Normal readings:")
    for idx in normal_indices:
        print(f"  {df.iloc[idx]['segment_id']} - {df.iloc[idx]['timestamp']} - "
              f"Pressure: {df.iloc[idx]['pressure']:.2f}, "
              f"Flow: {df.iloc[idx]['flow']:.2f}, "
              f"Vibration: {df.iloc[idx]['vibration']:.2f}")
    
    # Summary
    total_anomalies = y.sum()
    detected_anomalies = y_pred_binary.sum()
    
    print(f"\nSummary:")
    print(f"Total records: {len(df)}")
    print(f"Actual anomalies: {total_anomalies} ({total_anomalies/len(df)*100:.1f}%)")
    print(f"Detected anomalies: {detected_anomalies} ({detected_anomalies/len(df)*100:.1f}%)")
    print(f"Detection rate: {detected_anomalies/total_anomalies*100:.1f}%" if total_anomalies > 0 else "N/A")

if __name__ == "__main__":
    test_isolation_forest()
