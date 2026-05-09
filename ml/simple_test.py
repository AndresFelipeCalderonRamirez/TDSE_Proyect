"""
Simple test using the same data as training
"""

import pandas as pd
import numpy as np
import joblib
import sys
import os

def simple_test():
    """Test model with training data"""
    print("Simple Model Test")
    print("="*30)
    
    # Load model
    try:
        model = joblib.load('ml/artifacts/A/isolation_forest.pkl')
        scaler = joblib.load('ml/artifacts/A/scaler.pkl')
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Load the same data used for training
    try:
        df = pd.read_csv('data/outputs/tenant-A_sensor_data.csv', encoding='utf-8-sig')
        print(f"Loaded {len(df)} records")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return
    
    # Use basic features that should exist
    basic_features = ['pressure', 'flow', 'vibration']
    
    # Check if basic features exist
    if not all(feat in df.columns for feat in basic_features):
        print("Basic features not found in data")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Create simple test data
    X = df[basic_features].copy()
    y = df['is_anomaly']
    
    print(f"Using {len(basic_features)} basic features")
    
    # Try to predict (this will fail if model expects different features)
    try:
        # First, let's see what features the model expects
        if hasattr(model, 'n_features_in_'):
            print(f"Model expects {model.n_features_in_} features")
        
        # Try to fit a simple scaler on basic features
        from sklearn.preprocessing import StandardScaler
        simple_scaler = StandardScaler()
        X_scaled = simple_scaler.fit_transform(X)
        
        # Train a simple model on basic features
        from sklearn.ensemble import IsolationForest
        simple_model = IsolationForest(contamination=0.02, random_state=42)
        simple_model.fit(X_scaled)
        
        # Test the simple model
        y_pred = simple_model.predict(X_scaled)
        y_pred_binary = (y_pred == -1).astype(int)
        
        # Calculate metrics
        accuracy = (y == y_pred_binary).mean()
        
        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y, y_pred_binary)
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nSimple Model Results:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(f"F1-Score: {f1:.3f}")
        print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        # Show some predictions
        anomaly_count = y_pred_binary.sum()
        total_count = len(y_pred_binary)
        actual_anomalies = y.sum()
        
        print(f"\nSummary:")
        print(f"Total records: {total_count}")
        print(f"Actual anomalies: {actual_anomalies} ({actual_anomalies/total_count*100:.1f}%)")
        print(f"Detected anomalies: {anomaly_count} ({anomaly_count/total_count*100:.1f}%)")
        print(f"Detection rate: {anomaly_count/actual_anomalies*100:.1f}%" if actual_anomalies > 0 else "N/A")
        
        # Save simple model for testing
        joblib.dump(simple_model, 'ml/artifacts/A/simple_isolation_forest.pkl')
        joblib.dump(simple_scaler, 'ml/artifacts/A/simple_scaler.pkl')
        print(f"\nSimple model saved as 'simple_isolation_forest.pkl'")
        
    except Exception as e:
        print(f"Error in simple test: {e}")

if __name__ == "__main__":
    simple_test()
