"""
Tenant Configuration for IoT Simulation
Defines parameters for Bogotá and Medellín water networks
"""

TENANT_CONFIG = {
    "tenant-A": {
        "name": "Bogotá",
        "segments": 50,
        "altitude": 2600,  # meters above sea level
        "base_pressure": 4.5,  # bar (lower due to altitude)
        "base_flow": 0.8,  # m³/s
        "base_vibration": 2.5,  # mm/s
        "pressure_std": 0.15,
        "flow_std": 0.05,
        "vibration_std": 0.3,
        "stream_name": "tenant-a-stream"
    },
    "tenant-B": {
        "name": "Medellín",
        "segments": 40,
        "altitude": 1500,  # meters above sea level
        "base_pressure": 5.2,  # bar (higher due to lower altitude)
        "base_flow": 0.9,  # m³/s
        "base_vibration": 2.2,  # mm/s
        "pressure_std": 0.12,
        "flow_std": 0.04,
        "vibration_std": 0.25,
        "stream_name": "tenant-b-stream"
    }
}

# Materials available for pipe segments
PIPE_MATERIALS = ["PVC", "HDPE", "Cast Iron", "Ductile Iron", "Steel"]

# Anomaly injection parameters
ANOMALY_CONFIG = {
    "rate": 0.02,  # 2% of readings
    "duration_seconds": 10,  # consecutive seconds
    "pressure_drop_sigma": 3.0,  # 3σ drop
    "types": ["pressure_drop", "flow_anomaly", "vibration_spike"]
}

# Simulation parameters
SIMULATION_CONFIG = {
    "frequency_hz": 1.0,  # readings per second per segment
    "batch_size": 100,  # records per Kinesis batch
    "max_retries": 3,  # Kinesis publish retries
    "ground_truth_filename": "anomaly_ground_truth.csv"
}
