# visualization/dashboard.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

class RealTimeDashboard:
    """Real-time monitoring dashboard using Streamlit"""
    
    def __init__(self):
        st.set_page_config(
            page_title="Manufacturing Defect Detection",
            page_icon="🏭",
            layout="wide"
        )
        
    def run(self, sensor_data, predictions, alerts):
        """Main dashboard loop"""
        
        # Header
        st.title("🏭 Manufacturing Defect Detection System")
        st.markdown("---")
        
        # Key Metrics Row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Production Rate", f"{sensor_data['product_count'].iloc[-1]:.0f} units/hr")
        
        with col2:
            defect_rate = predictions.mean() * 100
            st.metric("Defect Rate", f"{defect_rate:.2f}%", 
                     delta=f"{defect_rate - 5:.2f}%")
        
        with col3:
            active_alerts_count = len([a for a in alerts if a['status'] == 'active'])
            st.metric("Active Alerts", active_alerts_count)
        
        with col4:
            system_health = 100 - defect_rate
            st.metric("System Health", f"{system_health:.1f}%")
        
        with col5:
            st.metric("Uptime", "99.7%")
        
        st.markdown("---")
        
        # Main Content
        tab1, tab2, tab3, tab4 = st.tabs(["Real-time Monitoring", "Alerts", "Analytics", "Configuration"])
        
        with tab1:
            self.render_monitoring_tab(sensor_data, predictions)
        
        with tab2:
            self.render_alerts_tab(alerts)
        
        with tab3:
            self.render_analytics_tab(sensor_data, predictions)
        
        with tab4:
            self.render_configuration_tab()
    
    def render_monitoring_tab(self, sensor_data, predictions):
        """Render real-time monitoring visualizations"""
        
        # Sensor readings over time
        col1, col2 = st.columns(2)
        
        with col1:
            # Temperature and Pressure
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=sensor_data['timestamp'],
                y=sensor_data['temperature'],
                name='Temperature (°C)',
                line=dict(color='red')
            ))
            
            fig.add_trace(go.Scatter(
                x=sensor_data['timestamp'],
                y=sensor_data['pressure'],
                name='Pressure (PSI)',
                line=dict(color='blue'),
                yaxis='y2'
            ))
            
            # Highlight anomalies
            anomaly_indices = np.where(predictions == 1)[0]
            if len(anomaly_indices) > 0:
                fig.add_trace(go.Scatter(
                    x=sensor_data.iloc[anomaly_indices]['timestamp'],
                    y=sensor_data.iloc[anomaly_indices]['temperature'],
                    mode='markers',
                    name='Anomaly',
                    marker=dict(color='yellow', size=10, symbol='x')
                ))
            
            fig.update_layout(
                title="Temperature & Pressure Monitoring",
                xaxis_title="Time",
                yaxis=dict(title="Temperature (°C)", side='left'),
                yaxis2=dict(title="Pressure (PSI)", overlaying='y', side='right'),
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Vibration and Power
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=sensor_data['timestamp'],
                y=sensor_data['vibration'],
                name='Vibration (mm/s)',
                line=dict(color='green')
            ))
            
            fig.add_trace(go.Scatter(
                x=sensor_data['timestamp'],
                y=sensor_data['power'],
                name='Power (kW)',
                line=dict(color='purple'),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title="Vibration & Power Consumption",
                xaxis_title="Time",
                yaxis=dict(title="Vibration (mm/s)", side='left'),
                yaxis2=dict(title="Power (kW)", overlaying='y', side='right'),
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly Score Heatmap
        st.subheader("Anomaly Detection Heatmap")
        
        # Create heatmap data
        sensors = ['temperature', 'vibration', 'pressure', 'speed', 'power', 'acoustic']
        heatmap_data = []
        
        for i in range(0, len(sensor_data), 60):  # Sample every minute
            row = []
            for sensor in sensors:
                # Calculate z-score as anomaly indicator
                mean = sensor_data[sensor].mean()
                std = sensor_data[sensor].std()
                z_score = abs((sensor_data[sensor].iloc[i] - mean) / std)
                row.append(min(z_score, 3))  # Cap at 3
            heatmap_data.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=sensors,
            y=[f"{i} min" for i in range(len(heatmap_data))],
            colorscale='RdYlGn_r',
            colorbar=dict(title="Anomaly Score")
        ))
        
        fig.update_layout(
            title="Sensor Anomaly Heatmap (Last 2 Hours)",
            xaxis_title="Sensor",
            yaxis_title="Time",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_alerts_tab(self, alerts):
        """Render alerts management interface"""
        
        st.subheader("Active Alerts")
        
        # Filter active alerts
        active_alerts = [a for a in alerts if a['status'] == 'active']
        
        if active_alerts:
            for alert in active_alerts:
                with st.expander(f"🚨 {alert['alert_level'].upper()} - {alert['defect_type']}", 
                               expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Alert ID:** {alert['alert_id']}")
                        st.write(f"**Machine:** {alert['machine_id']}")
                        st.write(f"**Time:** {alert['timestamp'].strftime('%H:%M:%S')}")
                    
                    with col2:
                        st.write(f"**Confidence:** {alert['confidence_score']:.2%}")
                        st.write(f"**Priority:** {alert['priority']}")
                        st.write(f"**Impact:** {alert['predicted_impact']}")
                    
                    with col3:
                        if st.button(f"Acknowledge", key=f"ack_{alert['alert_id']}"):
                            alert['acknowledged'] = True
                            st.success("Alert acknowledged")
                        
                        if st.button(f"Resolve", key=f"res_{alert['alert_id']}"):
                            alert['status'] = 'resolved'
                            st.success("Alert resolved")
        else:
            st.info("No active alerts")
        
        # Alert History
        st.subheader("Alert History")
        
        if alerts:
            history_df = pd.DataFrame([
                {
                    'Time': a['timestamp'],
                    'Level': a['alert_level'],
                    'Type': a['defect_type'],
                    'Machine': a['machine_id'],
                    'Status': a['status']
                }
                for a in alerts
            ])
            
            st.dataframe(history_df, use_container_width=True)
        else:
            st.info("No alert history")
    
    def render_analytics_tab(self, sensor_data, predictions):
        """Render analytics and insights"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Defect Distribution
            defect_types = sensor_data[sensor_data['defect_label'] == 1]['defect_type'].value_counts()
            
            fig = px.pie(
                values=defect_types.values,
                names=defect_types.index,
                title="Defect Type Distribution"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Detection Performance
            performance_data = {
                'Metric': ['Precision', 'Recall', 'F1-Score', 'Accuracy'],
                'Score': [0.92, 0.88, 0.90, 0.94]  # Example values
            }
            
            fig = px.bar(
                performance_data,
                x='Metric',
                y='Score',
                title="Model Performance Metrics",
                color='Score',
                color_continuous_scale='viridis'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation Matrix
        st.subheader("Sensor Correlation Analysis")
        
        sensor_cols = ['temperature', 'vibration', 'pressure', 'speed', 'power', 'acoustic']
        correlation_matrix = sensor_data[sensor_cols].corr()
        
        fig = px.imshow(
            correlation_matrix,
            labels=dict(x="Sensor", y="Sensor", color="Correlation"),
            x=sensor_cols,
            y=sensor_cols,
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_configuration_tab(self):
        """Render system configuration interface"""
        
        st.subheader("System Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Alert Thresholds**")
            
            low_threshold = st.slider("Low Alert", 0.0, 1.0, 0.6)
            medium_threshold = st.slider("Medium Alert", 0.0, 1.0, 0.75)
            high_threshold = st.slider("High Alert", 0.0, 1.0, 0.9)
            critical_threshold = st.slider("Critical Alert", 0.0, 1.0, 0.95)
        
        with col2:
            st.write("**Notification Settings**")
            
            email_notifications = st.checkbox("Email Notifications", value=True)
            slack_notifications = st.checkbox("Slack Notifications", value=True)
            dashboard_notifications = st.checkbox("Dashboard Notifications", value=True)
            
            cooldown_period = st.number_input("Alert Cooldown (seconds)", value=300)
        
        if st.button("Save Configuration"):
            st.success("Configuration saved successfully")