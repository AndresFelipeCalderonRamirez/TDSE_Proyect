"""
IoT Simulator for Tenant B (Medellín Water Network)
Generates synthetic sensor data for 40 pipe segments
"""

import json
import time
import numpy as np
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import os

class TenantBSimulator:
    """
    Simulates IoT sensor readings for Medellín water network
    40 pipe segments with pressure, flow, and vibration sensors
    """
    
    def __init__(self, num_segments: int = 40):
        self.num_segments = num_segments
        self.tenant_id = "tenant-B"
        self.city = "Medellín"
        self.altitude = 1500  # meters above sea level
        
        # Base parameters for Medellín (lower altitude = higher atmospheric pressure)
        self.base_pressure = 5.2  # bar (typical for Medellín)
        self.base_flow = 0.9  # m³/s
        self.base_vibration = 2.2  # mm/s
        
        # Sensor noise characteristics
        self.pressure_std = 0.12
        self.flow_std = 0.04
        self.vibration_std = 0.25
        
        # Anomaly injection parameters
        self.anomaly_rate = 0.02  # 2% of readings
        self.anomaly_duration = 10  # seconds
        self.pressure_drop_anomaly = 3 * self.pressure_std  # 3σ drop
        
        # Segment-specific variations (pipe diameter, material, age)
        self.segment_metadata = self._generate_segment_metadata()
        
        # Output file
        self.output_file = f"../../data/outputs/{self.tenant_id}_sensor_data.json"
        
    def _generate_segment_metadata(self) -> List[Dict]:
        """Generate realistic metadata for each pipe segment"""
        metadata = []
        materials = ["PVC", "HDPE", "Cast Iron", "Ductile Iron", "Steel"]
        
        for i in range(self.num_segments):
            segment = {
                "segment_id": f"SEG_B_{i:03d}",
                "diameter": np.random.uniform(150, 500),  # mm
                "material": random.choice(materials),
                "age": np.random.uniform(3, 35),  # years
                "length": np.random.uniform(75, 450),  # meters
                "elevation": np.random.uniform(1400, 1600)  # meters
            }
            metadata.append(segment)
        
        return metadata
    
    def _generate_normal_reading(self, segment_idx: int, timestamp: datetime) -> Dict:
        """Generate normal sensor reading for a segment"""
        segment = self.segment_metadata[segment_idx]
        
        # Adjust base values based on segment characteristics
        pressure_factor = 1.0 + (segment["elevation"] - self.altitude) / 10000
        flow_factor = segment["diameter"] / 300  # normalized to typical diameter
        vibration_factor = 1.0 + segment["age"] / 50  # older pipes vibrate more
        
        # Material-specific adjustments
        if segment["material"] == "Cast Iron":
            vibration_factor *= 1.3  # cast iron vibrates more
        elif segment["material"] == "PVC":
            vibration_factor *= 0.8  # PVC vibrates less
        
        pressure = np.random.normal(
            self.base_pressure * pressure_factor, 
            self.pressure_std
        )
        flow = np.random.normal(
            self.base_flow * flow_factor, 
            self.flow_std
        )
        vibration = np.random.normal(
            self.base_vibration * vibration_factor, 
            self.vibration_std
        )
        
        return {
            "tenant_id": self.tenant_id,
            "city": self.city,
            "segment_id": segment["segment_id"],
            "timestamp": timestamp.isoformat(),
            "pressure": max(0, pressure),  # ensure non-negative
            "flow": max(0, flow),
            "vibration": max(0, vibration),
            "is_anomaly": False,
            "metadata": segment
        }
    
    def _generate_anomaly_reading(self, normal_reading: Dict, anomaly_type: str = "pressure_drop") -> Dict:
        """Generate anomalous sensor reading"""
        reading = normal_reading.copy()
        
        if anomaly_type == "pressure_drop":
            # Simulate micro-burst: sudden pressure drop
            reading["pressure"] -= self.pressure_drop_anomaly
            reading["flow"] *= 0.6  # flow reduces with pressure drop
            reading["vibration"] *= 1.4  # vibration increases during burst
            reading["is_anomaly"] = True
        elif anomaly_type == "flow_anomaly":
            # Flow anomaly
            reading["flow"] *= 0.25  # significant flow reduction
            reading["pressure"] *= 1.15  # pressure builds up
            reading["is_anomaly"] = True
        elif anomaly_type == "vibration_spike":
            # Vibration anomaly (mechanical issue)
            reading["vibration"] *= 2.5
            reading["pressure"] *= 0.95
            reading["is_anomaly"] = True
        
        return reading
    
    def simulate_batch(self, duration_minutes: int = 5, frequency_hz: float = 1.0) -> List[Dict]:
        """
        Simulate sensor readings for a batch of time
        
        Args:
            duration_minutes: Duration of simulation in minutes
            frequency_hz: Sampling frequency in Hz (readings per second)
            
        Returns:
            List of sensor readings
        """
        readings = []
        start_time = datetime.now()
        total_readings = int(duration_minutes * 60 * frequency_hz)
        
        # Track active anomalies
        active_anomalies = {}  # segment_id -> (end_time, anomaly_type)
        
        for i in range(total_readings):
            timestamp = start_time + timedelta(seconds=i / frequency_hz)
            
            # Generate reading for each segment
            for segment_idx in range(self.num_segments):
                segment_id = self.segment_metadata[segment_idx]["segment_id"]
                
                # Check if we should start a new anomaly
                if segment_id not in active_anomalies and random.random() < self.anomaly_rate:
                    # Start new anomaly
                    anomaly_type = random.choice(["pressure_drop", "flow_anomaly", "vibration_spike"])
                    active_anomalies[segment_id] = (timestamp + timedelta(seconds=self.anomaly_duration), anomaly_type)
                
                # Check if anomaly is still active
                is_anomaly_active = segment_id in active_anomalies and timestamp < active_anomalies[segment_id][0]
                
                # Generate reading
                normal_reading = self._generate_normal_reading(segment_idx, timestamp)
                
                if is_anomaly_active:
                    # Apply anomaly
                    anomaly_type = active_anomalies[segment_id][1]
                    reading = self._generate_anomaly_reading(normal_reading, anomaly_type)
                else:
                    reading = normal_reading
                    # Clean up expired anomalies
                    if segment_id in active_anomalies and timestamp >= active_anomalies[segment_id][0]:
                        del active_anomalies[segment_id]
                
                readings.append(reading)
        
        return readings
    
    def save_to_file(self, readings: List[Dict]):
        """Save readings to JSON file"""
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        with open(self.output_file, 'w') as f:
            json.dump(readings, f, indent=2)
        
        print(f"Saved {len(readings)} readings to {self.output_file}")
    
    def run_simulation(self, duration_minutes: int = 5):
        """Run complete simulation and save results"""
        print(f"Starting simulation for {self.city} ({self.tenant_id})")
        print(f"Segments: {self.num_segments}")
        print(f"Duration: {duration_minutes} minutes")
        print(f"Anomaly rate: {self.anomaly_rate * 100:.1f}%")
        
        readings = self.simulate_batch(duration_minutes=duration_minutes)
        
        # Count anomalies
        anomaly_count = sum(1 for r in readings if r["is_anomaly"])
        print(f"Generated {len(readings)} total readings")
        print(f"Anomalies: {anomaly_count} ({anomaly_count/len(readings)*100:.1f}%)")
        
        self.save_to_file(readings)
        
        return readings

def main():
    """Main function to run the simulation"""
    simulator = TenantBSimulator(num_segments=40)
    
    # Run simulation for 5 minutes (adjust as needed)
    readings = simulator.run_simulation(duration_minutes=5)
    
    # Print sample readings
    print("\nSample readings:")
    for i, reading in enumerate(readings[:3]):
        print(f"Reading {i+1}:")
        print(f"  Segment: {reading['segment_id']}")
        print(f"  Time: {reading['timestamp']}")
        print(f"  Pressure: {reading['pressure']:.3f} bar")
        print(f"  Flow: {reading['flow']:.3f} m³/s")
        print(f"  Vibration: {reading['vibration']:.3f} mm/s")
        print(f"  Anomaly: {reading['is_anomaly']}")
        print()

if __name__ == "__main__":
    main()
