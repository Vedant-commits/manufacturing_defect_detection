# real_time_pipeline/alert_system.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import deque
import smtplib
from email.mime.text import MIMEText
import json

class AlertSystem:
    """Real-time alert generation and management"""
    
    def __init__(self, config=None):
        self.config = config or self.get_default_config()
        self.alert_history = deque(maxlen=1000)
        self.active_alerts = {}
        
    def get_default_config(self):
        return {
            'alert_levels': {
                'low': {'threshold': 0.6, 'color': 'yellow', 'priority': 3},
                'medium': {'threshold': 0.75, 'color': 'orange', 'priority': 2},
                'high': {'threshold': 0.9, 'color': 'red', 'priority': 1},
                'critical': {'threshold': 0.95, 'color': 'darkred', 'priority': 0}
            },
            'cooldown_period': 300,  # seconds
            'escalation_time': 600,  # seconds
            'notification_channels': ['dashboard', 'email', 'slack']
        }
    
    def generate_alert(self, timestamp, machine_id, defect_type, confidence_score, 
                       sensor_readings, predicted_impact):
        """Generate alert based on anomaly detection"""
        
        # Determine alert level
        alert_level = self.determine_alert_level(confidence_score)
        
        # Create alert
        alert = {
            'alert_id': f"ALERT_{datetime.now().strftime('%Y%m%d%H%M%S')}_{np.random.randint(1000)}",
            'timestamp': timestamp,
            'machine_id': machine_id,
            'defect_type': defect_type,
            'confidence_score': confidence_score,
            'alert_level': alert_level,
            'priority': self.config['alert_levels'][alert_level]['priority'],
            'sensor_readings': sensor_readings,
            'predicted_impact': predicted_impact,
            'status': 'active',
            'acknowledged': False,
            'resolution_time': None
        }
        
        # Check for duplicate alerts (cooldown period)
        if not self.is_duplicate_alert(alert):
            self.active_alerts[alert['alert_id']] = alert
            self.alert_history.append(alert)
            
            # Send notifications
            self.send_notifications(alert)
            
            return alert
        
        return None
    
    def determine_alert_level(self, confidence_score):
        """Determine alert level based on confidence score"""
        for level, config in sorted(self.config['alert_levels'].items(), 
                                   key=lambda x: x[1]['threshold'], reverse=True):
            if confidence_score >= config['threshold']:
                return level
        return 'low'
    
    def is_duplicate_alert(self, alert):
        """Check if similar alert was recently generated"""
        current_time = datetime.now()
        cooldown = timedelta(seconds=self.config['cooldown_period'])
        
        for historical_alert in self.alert_history:
            if (historical_alert['machine_id'] == alert['machine_id'] and
                historical_alert['defect_type'] == alert['defect_type'] and
                current_time - historical_alert['timestamp'] < cooldown):
                return True
        
        return False
    
    def send_notifications(self, alert):
        """Send alert notifications through configured channels"""
        for channel in self.config['notification_channels']:
            if channel == 'dashboard':
                self.update_dashboard(alert)
            elif channel == 'email':
                self.send_email_alert(alert)
            elif channel == 'slack':
                self.send_slack_alert(alert)
    
    def update_dashboard(self, alert):
        """Update real-time dashboard with alert"""
        # In production, this would update a real dashboard
        print(f"[DASHBOARD] {alert['alert_level'].upper()} Alert: "
              f"{alert['defect_type']} on {alert['machine_id']}")
    
    def send_email_alert(self, alert):
        """Send email notification for high-priority alerts"""
        if alert['priority'] <= 1:  # Only for high and critical
            # In production, implement actual email sending
            print(f"[EMAIL] Sending alert to maintenance team: {alert['alert_id']}")
    
    def send_slack_alert(self, alert):
        """Send Slack notification"""
        # In production, integrate with Slack API
        print(f"[SLACK] Alert posted to #maintenance channel: {alert['defect_type']}")
    
    def acknowledge_alert(self, alert_id):
        """Mark alert as acknowledged"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id]['acknowledged'] = True
            self.active_alerts[alert_id]['acknowledged_time'] = datetime.now()
            return True
        return False
    
    def resolve_alert(self, alert_id, resolution_notes=""):
        """Mark alert as resolved"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert['status'] = 'resolved'
            alert['resolution_time'] = datetime.now()
            alert['resolution_notes'] = resolution_notes
            
            # Move to history
            del self.active_alerts[alert_id]
            
            return True
        return False
    
    def get_alert_statistics(self):
        """Get alert statistics for reporting"""
        stats = {
            'total_alerts': len(self.alert_history),
            'active_alerts': len(self.active_alerts),
            'alerts_by_level': {},
            'alerts_by_type': {},
            'avg_resolution_time': None
        }
        
        # Count by level and type
        for alert in self.alert_history:
            level = alert['alert_level']
            defect_type = alert['defect_type']
            
            stats['alerts_by_level'][level] = stats['alerts_by_level'].get(level, 0) + 1
            stats['alerts_by_type'][defect_type] = stats['alerts_by_type'].get(defect_type, 0) + 1
        
        # Calculate average resolution time
        resolution_times = []
        for alert in self.alert_history:
            if alert.get('resolution_time'):
                duration = (alert['resolution_time'] - alert['timestamp']).total_seconds()
                resolution_times.append(duration)
        
        if resolution_times:
            stats['avg_resolution_time'] = np.mean(resolution_times)
        
        return stats