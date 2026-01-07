# anomaly_detection/models.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    """Base class for anomaly detection models"""
    
    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.model = None
        self.threshold = None
        
    def preprocess_data(self, df):
        """Extract and scale features"""
        feature_cols = ['temperature', 'vibration', 'pressure', 'speed', 'power', 'acoustic']
        
        # Add rolling statistics for temporal patterns
        for col in feature_cols:
            df[f'{col}_rolling_mean'] = df[col].rolling(window=10, min_periods=1).mean()
            df[f'{col}_rolling_std'] = df[col].rolling(window=10, min_periods=1).std()
            df[f'{col}_diff'] = df[col].diff()
        
        # Add rate of change
        for col in feature_cols:
            df[f'{col}_roc'] = df[col].pct_change()
        
        # Fill NaN values
        df = df.fillna(method='ffill').fillna(0)
        
        return df

class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest for anomaly detection"""
    
    def train(self, train_data):
        """Train Isolation Forest model"""
        print("Training Isolation Forest...")
        
        # Preprocess
        train_data = self.preprocess_data(train_data)
        
        # Select features
        feature_cols = [col for col in train_data.columns 
                       if col not in ['timestamp', 'machine_id', 'product_count', 
                                     'defect_label', 'defect_type']]
        
        X_train = train_data[feature_cols].values
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        self.model.fit(X_train_scaled)
        
        # Calculate threshold
        scores = self.model.score_samples(X_train_scaled)
        self.threshold = np.percentile(scores, self.contamination * 100)
        
        return self
    
    def predict(self, data):
        """Predict anomalies"""
        data = self.preprocess_data(data.copy())
        
        feature_cols = [col for col in data.columns 
                       if col not in ['timestamp', 'machine_id', 'product_count', 
                                     'defect_label', 'defect_type']]
        
        X = data[feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        # Get anomaly scores
        scores = self.model.score_samples(X_scaled)
        predictions = self.model.predict(X_scaled)
        
        # Convert to binary (0: normal, 1: anomaly)
        anomalies = (predictions == -1).astype(int)
        
        return anomalies, scores

class LSTMAutoencoder(AnomalyDetector):
    """LSTM Autoencoder for temporal anomaly detection"""
    
    def __init__(self, sequence_length=30, contamination=0.1):
        super().__init__(contamination)
        self.sequence_length = sequence_length
        
    def create_sequences(self, data, labels=None):
        """Create sequences for LSTM"""
        sequences = []
        sequence_labels = []
        
        for i in range(len(data) - self.sequence_length):
            sequences.append(data[i:i + self.sequence_length])
            if labels is not None:
                # Label sequence as anomaly if any point in sequence is anomaly
                sequence_labels.append(labels[i:i + self.sequence_length].max())
        
        return np.array(sequences), np.array(sequence_labels) if labels is not None else None
    
    def build_autoencoder(self, n_features):
        """Build LSTM autoencoder architecture"""
        model = Sequential([
            # Encoder
            LSTM(128, activation='relu', return_sequences=True, 
                 input_shape=(self.sequence_length, n_features)),
            LSTM(64, activation='relu', return_sequences=False),
            RepeatVector(self.sequence_length),
            # Decoder
            LSTM(64, activation='relu', return_sequences=True),
            LSTM(128, activation='relu', return_sequences=True),
            TimeDistributed(Dense(n_features))
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train(self, train_data):
        """Train LSTM Autoencoder"""
        print("Training LSTM Autoencoder...")
        
        # Preprocess
        train_data = self.preprocess_data(train_data)
        
        # Select features
        feature_cols = ['temperature', 'vibration', 'pressure', 'speed', 'power', 'acoustic']
        X_train = train_data[feature_cols].values
        
        # Scale data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create sequences
        X_sequences, _ = self.create_sequences(X_train_scaled)
        
        # Build and train model
        self.model = self.build_autoencoder(len(feature_cols))
        
        history = self.model.fit(
            X_sequences, X_sequences,
            epochs=50,
            batch_size=32,
            validation_split=0.1,
            verbose=0
        )
        
        # Calculate reconstruction error threshold
        X_pred = self.model.predict(X_sequences)
        mse = np.mean((X_sequences - X_pred) ** 2, axis=(1, 2))
        self.threshold = np.percentile(mse, 100 - self.contamination * 100)
        
        return self
    
    def predict(self, data):
        """Predict anomalies using reconstruction error"""
        data = self.preprocess_data(data.copy())
        
        feature_cols = ['temperature', 'vibration', 'pressure', 'speed', 'power', 'acoustic']
        X = data[feature_cols].values
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_sequences, _ = self.create_sequences(X_scaled)
        
        # Predict and calculate reconstruction error
        X_pred = self.model.predict(X_sequences)
        mse = np.mean((X_sequences - X_pred) ** 2, axis=(1, 2))
        
        # Detect anomalies
        anomalies = (mse > self.threshold).astype(int)
        
        # Pad to match original length
        anomalies = np.pad(anomalies, (self.sequence_length, 0), constant_values=0)
        scores = np.pad(mse, (self.sequence_length, 0), constant_values=0)
        
        return anomalies, scores

class StatisticalProcessControl(AnomalyDetector):
    """Statistical Process Control (SPC) for anomaly detection"""
    
    def __init__(self, n_sigma=3):
        super().__init__()
        self.n_sigma = n_sigma
        self.control_limits = {}
        
    def train(self, train_data):
        """Calculate control limits from training data"""
        print("Training Statistical Process Control...")
        
        feature_cols = ['temperature', 'vibration', 'pressure', 'speed', 'power', 'acoustic']
        
        for col in feature_cols:
            mean = train_data[col].mean()
            std = train_data[col].std()
            
            self.control_limits[col] = {
                'ucl': mean + self.n_sigma * std,  # Upper Control Limit
                'lcl': mean - self.n_sigma * std,  # Lower Control Limit
                'mean': mean
            }
        
        return self
    
    def predict(self, data):
        """Detect out-of-control points"""
        anomalies = np.zeros(len(data))
        scores = np.zeros(len(data))
        
        for col, limits in self.control_limits.items():
            # Check if values are outside control limits
            out_of_control = (data[col] > limits['ucl']) | (data[col] < limits['lcl'])
            anomalies = np.maximum(anomalies, out_of_control.astype(int))
            
            # Calculate z-score as anomaly score
            z_scores = np.abs((data[col] - limits['mean']) / 
                            (limits['ucl'] - limits['mean']) * self.n_sigma)
            scores = np.maximum(scores, z_scores)
        
        return anomalies.astype(int), scores

class EnsembleDetector:
    """Ensemble of multiple anomaly detection methods"""
    
    def __init__(self, voting='soft'):
        self.voting = voting
        self.detectors = {
            'isolation_forest': IsolationForestDetector(contamination=0.1),
            'lstm_autoencoder': LSTMAutoencoder(sequence_length=30, contamination=0.1),
            'spc': StatisticalProcessControl(n_sigma=3)
        }
        
    def train(self, train_data):
        """Train all detectors"""
        print("Training Ensemble Detector...")
        
        for name, detector in self.detectors.items():
            detector.train(train_data)
        
        return self
    
    def predict(self, data):
        """Combine predictions from all detectors"""
        all_predictions = []
        all_scores = []
        
        for name, detector in self.detectors.items():
            predictions, scores = detector.predict(data)
            all_predictions.append(predictions)
            all_scores.append(scores)
        
        # Ensemble voting
        if self.voting == 'hard':
            # Majority voting
            ensemble_predictions = np.round(np.mean(all_predictions, axis=0))
        else:
            # Soft voting using scores
            ensemble_scores = np.mean(all_scores, axis=0)
            threshold = np.percentile(ensemble_scores, 90)
            ensemble_predictions = (ensemble_scores > threshold).astype(int)
        
        return ensemble_predictions, np.mean(all_scores, axis=0)