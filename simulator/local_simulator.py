"""
Local IoT Simulator for Water Twin Prototype
100% local testing without AWS dependencies
"""

import json
import time
import numpy as np
import argparse
import threading
import logging
import csv
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SensorReading:
    """Data class for sensor readings"""
    tenant_id: str
    city: str
    segment_id: str
    timestamp: str
    pressure: float
    flow: float
    vibration: float
    is_anomaly: bool
    metadata: Dict

class LocalIoTSimulator:
    """
    Local multi-tenant IoT simulator for water networks
    Generates sensor data and saves to local files
    """
    
    def __init__(self, tenant_id: str, config: Dict):
        self.tenant_id = tenant_id
        self.config = config
        self.city = config['name']
        self.num_segments = config['segments']
        self.altitude = config['altitude']
        
        # Sensor parameters
        self.base_pressure = config['base_pressure']
        self.base_flow = config['base_flow']
        self.base_vibration = config['base_vibration']
        self.pressure_std = config['pressure_std']
        self.flow_std = config['flow_std']
        self.vibration_std = config['vibration_std']
        
        # Anomaly parameters
        self.anomaly_rate = 0.02
        self.anomaly_duration_seconds = 10
        self.pressure_drop_anomaly = 3 * self.pressure_std
        
        # Simulation parameters
        self.frequency_hz = 1.0
        
        # Local storage
        self.readings_data = []
        self.ground_truth_data = []
        self.export_ground_truth = False
        
        # Segment metadata
        self.segment_metadata = self._generate_segment_metadata()
        
        # Active anomalies tracking
        self.active_anomalies = {}  # segment_id -> end_time, anomaly_type
        
    def _generate_segment_metadata(self) -> List[Dict]:
        """Generate realistic metadata for pipe segments"""
        metadata = []
        materials = ["PVC", "HDPE", "Cast Iron", "Ductile Iron", "Steel"]
        
        for i in range(self.num_segments):
            segment = {
                "segment_id": f"SEG_{self.tenant_id.split('-')[1]}_{i:03d}",
                "diameter": np.random.uniform(100, 600),  # mm
                "material": np.random.choice(materials),
                "age": np.random.uniform(5, 40),  # years
                "length": np.random.uniform(50, 500),  # meters
                "elevation": np.random.uniform(
                    self.altitude - 100, 
                    self.altitude + 100
                )  # meters
            }
            metadata.append(segment)
        
        return metadata
    
    def _generate_normal_reading(self, segment_idx: int, timestamp: datetime) -> SensorReading:
        """Generate normal sensor reading"""
        segment = self.segment_metadata[segment_idx]
        
        # Adjust base values based on segment characteristics
        pressure_factor = 1.0 + (segment["elevation"] - self.altitude) / 10000
        flow_factor = segment["diameter"] / 300  # normalized to typical diameter
        vibration_factor = 1.0 + segment["age"] / 50  # older pipes vibrate more
        
        # Material-specific adjustments
        if segment["material"] == "Cast Iron":
            vibration_factor *= 1.3
        elif segment["material"] == "PVC":
            vibration_factor *= 0.8
        
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
        
        return SensorReading(
            tenant_id=self.tenant_id,
            city=self.city,
            segment_id=segment["segment_id"],
            timestamp=timestamp.isoformat(),
            pressure=max(0, pressure),
            flow=max(0, flow),
            vibration=max(0, vibration),
            is_anomaly=False,
            metadata=segment
        )
    
    def _generate_anomaly_reading(self, normal_reading: SensorReading, anomaly_type: str = "pressure_drop") -> SensorReading:
        """Generate anomalous sensor reading"""
        reading = normal_reading
        
        if anomaly_type == "pressure_drop":
            # Simulate micro-burst: sudden pressure drop
            reading.pressure -= self.pressure_drop_anomaly
            reading.flow *= 0.7  # flow reduces with pressure drop
            reading.vibration *= 1.5  # vibration increases during burst
            reading.is_anomaly = True
        elif anomaly_type == "flow_anomaly":
            # Flow anomaly
            reading.flow *= 0.3  # significant flow reduction
            reading.pressure *= 1.2  # pressure builds up
            reading.is_anomaly = True
        elif anomaly_type == "vibration_spike":
            # Vibration anomaly (mechanical issue)
            reading.vibration *= 2.5
            reading.pressure *= 0.95
            reading.is_anomaly = True
        
        return reading
    
    def _should_start_anomaly(self, segment_id: str) -> bool:
        """Determine if we should start a new anomaly for this segment"""
        return (segment_id not in self.active_anomalies and 
                np.random.random() < self.anomaly_rate)
    
    def _is_anomaly_active(self, segment_id: str, current_time: datetime) -> Tuple[bool, Optional[str]]:
        """Check if anomaly is still active for this segment"""
        if segment_id in self.active_anomalies:
            end_time, anomaly_type = self.active_anomalies[segment_id]
            if current_time < end_time:
                return True, anomaly_type
            else:
                del self.active_anomalies[segment_id]
        return False, None
    
    def _generate_readings_batch(self, duration_seconds: int) -> List[SensorReading]:
        """Generate a batch of sensor readings"""
        readings = []
        start_time = datetime.now()
        total_readings = int(duration_seconds * self.frequency_hz * self.num_segments)
        
        for i in range(total_readings):
            # Calculate timestamp for this reading
            segment_idx = i % self.num_segments
            reading_offset = i // self.num_segments
            timestamp = start_time + timedelta(seconds=reading_offset / self.frequency_hz)
            
            segment_id = self.segment_metadata[segment_idx]["segment_id"]
            
            # Check if we should start a new anomaly
            if self._should_start_anomaly(segment_id):
                anomaly_type = np.random.choice(["pressure_drop", "flow_anomaly", "vibration_spike"])
                end_time = timestamp + timedelta(seconds=self.anomaly_duration_seconds)
                self.active_anomalies[segment_id] = (end_time, anomaly_type)
            
            # Check if anomaly is still active
            is_active, anomaly_type = self._is_anomaly_active(segment_id, timestamp)
            
            # Generate reading
            normal_reading = self._generate_normal_reading(segment_idx, timestamp)
            
            if is_active:
                reading = self._generate_anomaly_reading(normal_reading, anomaly_type)
            else:
                reading = normal_reading
            
            readings.append(reading)
            
            # Track ground truth
            self.ground_truth_data.append({
                'tenantId': reading.tenant_id,
                'timestamp': reading.timestamp,
                'segmentId': reading.segment_id,
                'is_anomaly': reading.is_anomaly
            })
        
        return readings
    
    def _save_to_local_files(self, readings: List[SensorReading]) -> bool:
        """Save readings to local files instead of Kinesis"""
        try:
            # Create data directory if it doesn't exist
            os.makedirs('data/outputs', exist_ok=True)
            
            # Save readings as JSON
            json_filename = f"data/outputs/{self.tenant_id}_sensor_data.json"
            readings_data = []
            
            for reading in readings:
                readings_data.append({
                    'tenant_id': reading.tenant_id,
                    'city': reading.city,
                    'segment_id': reading.segment_id,
                    'timestamp': reading.timestamp,
                    'pressure': reading.pressure,
                    'flow': reading.flow,
                    'vibration': reading.vibration,
                    'is_anomaly': reading.is_anomaly,
                    'metadata': reading.metadata
                })
            
            with open(json_filename, 'w', encoding='utf-8') as f:
                json.dump(readings_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(readings)} readings to {json_filename}")
            
            # Save as CSV for easier analysis
            csv_filename = f"data/outputs/{self.tenant_id}_sensor_data.csv"
            with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['tenant_id', 'city', 'segment_id', 'timestamp', 'pressure', 'flow', 'vibration', 'is_anomaly']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for reading in readings:
                    writer.writerow({
                        'tenant_id': reading.tenant_id,
                        'city': reading.city,
                        'segment_id': reading.segment_id,
                        'timestamp': reading.timestamp,
                        'pressure': reading.pressure,
                        'flow': reading.flow,
                        'vibration': reading.vibration,
                        'is_anomaly': reading.is_anomaly
                    })
            
            logger.info(f"Saved CSV to {csv_filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save local files: {str(e)}")
            return False
    
    def _export_ground_truth(self, filename: str = None) -> None:
        """Export ground truth data to CSV"""
        if not self.ground_truth_data:
            logger.warning("No ground truth data to export")
            return
        
        if filename is None:
            filename = f"data/outputs/anomaly_ground_truth_{self.tenant_id}.csv"
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['tenantId', 'timestamp', 'segmentId', 'is_anomaly']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in self.ground_truth_data:
                writer.writerow(row)
        
        logger.info(f"Exported {len(self.ground_truth_data)} ground truth records to {filename}")
    
    def simulate(self, duration_seconds: int = 300, export_ground_truth: bool = False) -> bool:
        """
        Run simulation for specified duration
        
        Args:
            duration_seconds: Duration in seconds
            export_ground_truth: Whether to export ground truth CSV
            
        Returns:
            Success status
        """
        logger.info(f"Starting LOCAL simulation for {self.city} ({self.tenant_id})")
        logger.info(f"Segments: {self.num_segments}")
        logger.info(f"Duration: {duration_seconds} seconds")
        logger.info(f"Frequency: {self.frequency_hz} Hz")
        logger.info(f"Anomaly rate: {self.anomaly_rate * 100:.1f}%")
        
        self.export_ground_truth = export_ground_truth
        
        # Generate readings
        readings = self._generate_readings_batch(duration_seconds)
        
        # Count anomalies
        anomaly_count = sum(1 for r in readings if r.is_anomaly)
        actual_anomaly_rate = anomaly_count / len(readings)
        
        logger.info(f"Generated {len(readings)} total readings")
        logger.info(f"Anomalies: {anomaly_count} ({actual_anomaly_rate * 100:.2f}%)")
        
        # Save to local files
        success = self._save_to_local_files(readings)
        
        # Export ground truth if requested
        if export_ground_truth:
            self._export_ground_truth()
        
        return success

def load_config() -> Dict:
    """Load tenant configuration"""
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from config import TENANT_CONFIG
    return TENANT_CONFIG

def run_tenant_simulation(tenant_id: str, duration: int, export_ground_truth: bool) -> bool:
    """Run simulation for a single tenant"""
    config = load_config()
    
    if tenant_id not in config:
        logger.error(f"Unknown tenant: {tenant_id}")
        return False
    
    simulator = LocalIoTSimulator(tenant_id, config[tenant_id])
    return simulator.simulate(duration, export_ground_truth)

def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description='Water Twin Local IoT Simulator')
    parser.add_argument('--tenant', 
                       choices=['tenant-A', 'tenant-B', 'both'],
                       default='both',
                       help='Tenant to simulate (default: both)')
    parser.add_argument('--duration', 
                       type=int,
                       default=300,
                       help='Simulation duration in seconds (default: 300)')
    parser.add_argument('--export-ground-truth',
                       action='store_true',
                       help='Export ground truth CSV file')
    
    args = parser.parse_args()
    
    logger.info("Water Twin LOCAL IoT Simulator")
    logger.info(f"Tenants: {args.tenant}")
    logger.info(f"Duration: {args.duration} seconds")
    logger.info(f"Export ground truth: {args.export_ground_truth}")
    
    start_time = time.time()
    
    if args.tenant == 'both':
        # Run both tenants concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(run_tenant_simulation, 'tenant-A', args.duration, args.export_ground_truth): 'tenant-A',
                executor.submit(run_tenant_simulation, 'tenant-B', args.duration, args.export_ground_truth): 'tenant-B'
            }
            
            success_count = 0
            for future in as_completed(futures):
                tenant = futures[future]
                try:
                    success = future.result()
                    if success:
                        success_count += 1
                        logger.info(f" {tenant} simulation completed successfully")
                    else:
                        logger.error(f" {tenant} simulation failed")
                except Exception as e:
                    logger.error(f" {tenant} simulation error: {str(e)}")
        
        logger.info(f"Simulation completed: {success_count}/2 tenants successful")
        
    else:
        # Run single tenant
        success = run_tenant_simulation(args.tenant, args.duration, args.export_ground_truth)
        if success:
            logger.info(f" {args.tenant} simulation completed successfully")
        else:
            logger.error(f" {args.tenant} simulation failed")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Total simulation time: {elapsed_time:.2f} seconds")
    
    # Show file locations
    logger.info("\n Generated files:")
    logger.info("data/outputs/tenant-A_sensor_data.json")
    logger.info("data/outputs/tenant-B_sensor_data.json")
    logger.info("data/outputs/tenant-A_sensor_data.csv")
    logger.info("data/outputs/tenant-B_sensor_data.csv")

if __name__ == "__main__":
    main()
