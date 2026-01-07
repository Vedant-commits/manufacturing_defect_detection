# main_standalone.py - Complete working version in one file
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("MANUFACTURING DEFECT DETECTION SYSTEM")
print("="*60)

# Data Generation
print("\n[1] Generating sensor data...")

def generate_sensor_data(duration_minutes=120):
    """Generate synthetic sensor data with injected defects"""
    data = []
    current_time = datetime.now()
    
    for minute in range(duration_minutes):
        for second in range(60):
            timestamp = current_time + timedelta(minutes=minute, seconds=second)
            
            # Base sensor readings
            reading = {
                'timestamp': timestamp,
                'machine_id': 'MACH_001',
                'temperature': np.random.normal(65, 2),
                'vibration': np.random.normal(0.5, 0.1),
                'pressure': np.random.normal(100, 5),
                'speed': np.random.normal(1000, 50),
                'power': np.random.normal(150, 10),
                'acoustic': np.random.normal(70, 5),
                'defect_label': 0,
                'defect_type': 'normal'
            }
            
            # Inject defects at specific time ranges
            idx = minute * 60 + second
            
            # Overheating defect (minutes 20-25)
            if 1200 <= idx < 1500:
                reading['temperature'] += np.random.uniform(15, 25)
                reading['power'] *= 1.3
                reading['defect_label'] = 1
                reading['defect_type'] = 'overheating'
            
            # Bearing failure (minutes 50-55)
            elif 3000 <= idx < 3300:
                progress = (idx - 3000) / 300
                reading['vibration'] *= (1 + progress * 3)
                reading['acoustic'] += progress * 15
                reading['defect_label'] = 1
                reading['defect_type'] = 'bearing_failure'
            
            # Pressure leak (minutes 75-80)
            elif 4500 <= idx < 4800:
                progress = (idx - 4500) / 300
                reading['pressure'] -= progress * 30
                reading['defect_label'] = 1
                reading['defect_type'] = 'pressure_leak'
            
            # Power surge (minutes 100-105)
            elif 6000 <= idx < 6300:
                if np.random.random() < 0.3:  # Intermittent surges
                    reading['power'] *= np.random.uniform(1.5, 2.0)
                    reading['defect_label'] = 1
                    reading['defect_type'] = 'power_surge'
            
            data.append(reading)
    
    return pd.DataFrame(data)

# Generate data
sensor_data = generate_sensor_data(duration_minutes=120)
print(f"✓ Generated {len(sensor_data)} sensor readings")
print(f"✓ Injected {sensor_data['defect_label'].sum()} defective readings")

# Feature Engineering
print("\n[2] Engineering features...")

def engineer_features(df):
    """Add statistical features for better anomaly detection"""
    df = df.copy()
    
    # Rolling statistics (10-second window)
    for col in ['temperature', 'vibration', 'pressure', 'speed', 'power', 'acoustic']:
        df[f'{col}_rolling_mean'] = df[col].rolling(window=10, min_periods=1).mean()
        df[f'{col}_rolling_std'] = df[col].rolling(window=10, min_periods=1).std()
        df[f'{col}_diff'] = df[col].diff()
    
    # Fill NaN values
    df = df.fillna(method='ffill').fillna(0)
    
    return df

sensor_data_featured = engineer_features(sensor_data)
print(f"✓ Created {len(sensor_data_featured.columns)} features")

# Split data
train_size = int(0.6 * len(sensor_data_featured))
train_data = sensor_data_featured[:train_size]
test_data = sensor_data_featured[train_size:]

# Anomaly Detection Model
print("\n[3] Training anomaly detection model...")

# Prepare features
feature_cols = [col for col in sensor_data_featured.columns 
                if col not in ['timestamp', 'machine_id', 'defect_label', 'defect_type']]

# Train Isolation Forest
scaler = StandardScaler()
X_train = scaler.fit_transform(train_data[feature_cols])
X_test = scaler.transform(test_data[feature_cols])

# Train model
model = IsolationForest(
    contamination=0.1,
    random_state=42,
    n_estimators=100
)
model.fit(X_train)
print("✓ Isolation Forest model trained")

# Predictions
print("\n[4] Detecting anomalies...")
predictions = model.predict(X_test)
anomaly_scores = model.score_samples(X_test)

# Convert to binary (1 for anomaly, 0 for normal)
predictions = (predictions == -1).astype(int)

# Performance Metrics
print("\n[5] Evaluating performance...")
y_true = test_data['defect_label'].values
accuracy = accuracy_score(y_true, predictions)
precision = precision_score(y_true, predictions, zero_division=0)
recall = recall_score(y_true, predictions, zero_division=0)
f1 = f1_score(y_true, predictions, zero_division=0)

print(f"  Accuracy:  {accuracy:.3f}")
print(f"  Precision: {precision:.3f}")
print(f"  Recall:    {recall:.3f}")
print(f"  F1-Score:  {f1:.3f}")

# Alert Generation
print("\n[6] Generating alerts...")

def generate_alerts(data, predictions, scores):
    """Generate alerts for detected anomalies"""
    alerts = []
    anomaly_indices = np.where(predictions == 1)[0]
    
    for idx in anomaly_indices[:20]:  # Top 20 alerts
        row = data.iloc[idx]
        
        # Classify defect type based on sensor patterns
        defect_type = 'unknown'
        if row['temperature'] > 80:
            defect_type = 'overheating'
        elif row['vibration'] > 1.5:
            defect_type = 'bearing_failure'
        elif row['pressure'] < 70:
            defect_type = 'pressure_leak'
        elif row['power'] > 200:
            defect_type = 'power_surge'
        
        alert = {
            'timestamp': row['timestamp'],
            'machine_id': row['machine_id'],
            'detected_type': defect_type,
            'actual_type': row['defect_type'],
            'confidence': abs(scores[idx]),
            'sensor_values': {
                'temperature': row['temperature'],
                'vibration': row['vibration'],
                'pressure': row['pressure'],
                'power': row['power']
            }
        }
        alerts.append(alert)
    
    return alerts

alerts = generate_alerts(test_data, predictions, anomaly_scores)
print(f"✓ Generated {len(alerts)} alerts")

# Visualization
print("\n[7] Creating visualizations...")

fig = plt.figure(figsize=(16, 10))

# Plot 1: Sensor readings with anomalies
sensors = ['temperature', 'vibration', 'pressure', 'power']
for i, sensor in enumerate(sensors, 1):
    plt.subplot(3, 2, i)
    
    # Normal readings
    normal_mask = predictions == 0
    plt.plot(test_data.index[normal_mask], 
             test_data[sensor].values[normal_mask], 
             'b.', alpha=0.3, markersize=2, label='Normal')
    
    # Detected anomalies
    anomaly_mask = predictions == 1
    plt.scatter(test_data.index[anomaly_mask], 
               test_data[sensor].values[anomaly_mask], 
               c='red', s=20, alpha=0.7, label='Detected Anomaly')
    
    plt.title(f'{sensor.capitalize()} Monitoring')
    plt.xlabel('Time Index')
    plt.ylabel(sensor.capitalize())
    plt.legend()
    plt.grid(True, alpha=0.3)

# Plot 5: Anomaly scores over time
plt.subplot(3, 2, 5)
plt.plot(anomaly_scores, alpha=0.7)
plt.axhline(y=np.percentile(anomaly_scores, 10), color='r', linestyle='--', label='Threshold')
plt.title('Anomaly Scores Over Time')
plt.xlabel('Time Index')
plt.ylabel('Anomaly Score')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 6: Confusion Matrix
from sklearn.metrics import confusion_matrix
plt.subplot(3, 2, 6)
cm = confusion_matrix(y_true, predictions)
im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar(im)
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Normal', 'Defect'])
plt.yticks(tick_marks, ['Normal', 'Defect'])
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Add text annotations
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.suptitle('Manufacturing Defect Detection Dashboard', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('manufacturing_defect_dashboard.png', dpi=150, bbox_inches='tight')
print("✓ Dashboard saved to 'manufacturing_defect_dashboard.png'")

# Save results
print("\n[8] Saving results...")
results_df = test_data.copy()
results_df['predicted_anomaly'] = predictions
results_df['anomaly_score'] = anomaly_scores
results_df.to_csv('defect_detection_results.csv', index=False)
print("✓ Results saved to 'defect_detection_results.csv'")

# Summary Report
print("\n" + "="*60)
print("EXECUTIVE SUMMARY")
print("="*60)
print(f"Production Period: {test_data['timestamp'].min()} to {test_data['timestamp'].max()}")
print(f"Total Readings: {len(test_data):,}")
print(f"Actual Defects: {y_true.sum():,} ({y_true.mean():.1%})")
print(f"Detected Anomalies: {predictions.sum():,} ({predictions.mean():.1%})")
print(f"\nDefect Types Detected:")
for defect_type in test_data['defect_type'].unique():
    if defect_type != 'normal':
        count = (test_data['defect_type'] == defect_type).sum()
        print(f"  - {defect_type}: {count} occurrences")

print(f"\nBusiness Impact:")
print(f"  - Early Detection Rate: {recall:.1%}")
print(f"  - False Alarm Rate: {(1-precision):.1%}")
print(f"  - Estimated Downtime Prevented: {recall * 3.2:.1f} hours")
print(f"  - Estimated Cost Savings: ${recall * 45000:,.0f}")

print("\n✅ Manufacturing Defect Detection Analysis Complete!")