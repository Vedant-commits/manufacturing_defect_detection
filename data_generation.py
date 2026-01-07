# sensor_simulator.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

class ProductionLineSensorSimulator:
    """
    Simulates a production line with multiple sensors monitoring:
    - Temperature
    - Vibration
    - Pressure
    - Speed
    - Power consumption
    - Acoustic emissions
    """
    
    def __init__(self, config=None):
        self.config = config or self.get_default_config()
        self.current_time = datetime.now()
        
    def get_default_config(self):
        return {
            'sensors': {
                'temperature': {
                    'baseline': 65,  # Celsius
                    'normal_variation': 2,
                    'units': 'C',
                    'sampling_rate': 10  # Hz
                },
                'vibration': {
                    'baseline': 0.5,  # mm/s
                    'normal_variation': 0.1,
                    'units': 'mm/s',
                    'sampling_rate': 100  # Hz
                },
                'pressure': {
                    'baseline': 100,  # PSI
                    'normal_variation': 5,
                    'units': 'PSI',
                    'sampling_rate': 5  # Hz
                },
                'speed': {
                    'baseline': 1000,  # RPM
                    'normal_variation': 50,
                    'units': 'RPM',
                    'sampling_rate': 10  # Hz
                },
                'power': {
                    'baseline': 150,  # kW
                    'normal_variation': 10,
                    'units': 'kW',
                    'sampling_rate': 1  # Hz
                },
                'acoustic': {
                    'baseline': 70,  # dB
                    'normal_variation': 5,
                    'units': 'dB',
                    'sampling_rate': 50  # Hz
                }
            },
            'production_rate': 60,  # units per hour
            'shift_duration': 8,  # hours
        }
    
    def generate_normal_data(self, duration_minutes=60):
        """Generate normal operating sensor data"""
        data = []
        
        for minute in range(duration_minutes):
            for second in range(60):
                timestamp = self.current_time + timedelta(minutes=minute, seconds=second)
                
                # Generate sensor readings with normal variations
                reading = {
                    'timestamp': timestamp,
                    'machine_id': 'MACH_001',
                    'product_count': minute * (self.config['production_rate'] / 60),
                }
                
                # Add sensor readings with realistic patterns
                for sensor, params in self.config['sensors'].items():
                    # Add daily pattern (temperature higher during day)
                    hour_effect = 0
                    if sensor == 'temperature':
                        hour_effect = 3 * np.sin((timestamp.hour - 6) * np.pi / 12)
                    
                    # Add slight upward drift as machine runs
                    drift = minute * 0.01 if sensor in ['temperature', 'vibration'] else 0
                    
                    # Normal variation with some autocorrelation
                    noise = np.random.normal(0, params['normal_variation'])
                    
                    value = params['baseline'] + hour_effect + drift + noise
                    reading[f'{sensor}'] = round(value, 2)
                
                data.append(reading)
        
        return pd.DataFrame(data)
    
    def inject_defects(self, normal_data, defect_scenarios):
        """
        Inject various defect patterns into normal data
        
        Defect types:
        1. Bearing failure: Gradual increase in vibration
        2. Overheating: Temperature spike
        3. Pressure leak: Gradual pressure decrease
        4. Power surge: Sudden power spike
        5. Mechanical wear: Increasing vibration + acoustic
        """
        defective_data = normal_data.copy()
        defect_labels = pd.Series([0] * len(defective_data), index=defective_data.index)
        defect_types = pd.Series(['normal'] * len(defective_data), index=defective_data.index)
        
        for scenario in defect_scenarios:
            start_idx = scenario['start_idx']
            end_idx = scenario['end_idx']
            defect_type = scenario['type']
            
            if defect_type == 'bearing_failure':
                # Exponential increase in vibration
                duration = end_idx - start_idx
                for i in range(duration):
                    idx = start_idx + i
                    multiplier = 1 + (i / duration) ** 2 * 3  # Up to 4x normal
                    defective_data.loc[idx, 'vibration'] *= multiplier
                    defective_data.loc[idx, 'acoustic'] *= (1 + (i / duration) * 0.5)
                    defect_labels[idx] = 1
                    defect_types[idx] = 'bearing_failure'
            
            elif defect_type == 'overheating':
                # Sudden temperature increase
                defective_data.loc[start_idx:end_idx, 'temperature'] += np.random.uniform(15, 25)
                defective_data.loc[start_idx:end_idx, 'power'] *= 1.2
                defect_labels[start_idx:end_idx] = 1
                defect_types[start_idx:end_idx] = 'overheating'
            
            elif defect_type == 'pressure_leak':
                # Gradual pressure decrease
                duration = end_idx - start_idx
                for i in range(duration):
                    idx = start_idx + i
                    decrease = (i / duration) * 30  # Up to 30 PSI drop
                    defective_data.loc[idx, 'pressure'] -= decrease
                    defect_labels[idx] = 1
                    defect_types[idx] = 'pressure_leak'
            
            elif defect_type == 'power_surge':
                # Sudden power spikes
                surge_indices = np.random.choice(range(start_idx, end_idx), size=10)
                for idx in surge_indices:
                    defective_data.loc[idx, 'power'] *= np.random.uniform(1.5, 2.0)
                    defect_labels[idx] = 1
                    defect_types[idx] = 'power_surge'
            
            elif defect_type == 'mechanical_wear':
                # Gradual degradation across multiple sensors
                duration = end_idx - start_idx
                for i in range(duration):
                    idx = start_idx + i
                    wear_factor = 1 + (i / duration) * 0.5
                    defective_data.loc[idx, 'vibration'] *= wear_factor
                    defective_data.loc[idx, 'acoustic'] *= wear_factor
                    defective_data.loc[idx, 'power'] *= (1 + (i / duration) * 0.2)
                    defect_labels[idx] = 1
                    defect_types[idx] = 'mechanical_wear'
        
        defective_data['defect_label'] = defect_labels
        defective_data['defect_type'] = defect_types
        
        return defective_data

# Example usage
if __name__ == "__main__":
    simulator = ProductionLineSensorSimulator()
    
    # Generate 2 hours of normal data
    normal_data = simulator.generate_normal_data(duration_minutes=120)
    
    # Define defect scenarios
    defect_scenarios = [
        {'type': 'bearing_failure', 'start_idx': 3600, 'end_idx': 4200},
        {'type': 'overheating', 'start_idx': 5400, 'end_idx': 5700},
        {'type': 'pressure_leak', 'start_idx': 6600, 'end_idx': 7200}
    ]
    
    # Inject defects
    data_with_defects = simulator.inject_defects(normal_data, defect_scenarios)
    
    print(f"Generated {len(data_with_defects)} sensor readings")
    print(f"Defect rate: {data_with_defects['defect_label'].mean():.2%}")
    print(f"Defect types: {data_with_defects['defect_type'].value_counts().to_dict()}")
    
    # Save data
    data_with_defects.to_csv('sensor_data.csv', index=False)