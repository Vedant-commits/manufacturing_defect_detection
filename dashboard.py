# visualization/dashboard.py

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class RealTimeDashboard:
    """Real-time monitoring dashboard using Streamlit"""

    def __init__(self):
        st.set_page_config(
            page_title="Manufacturing Defect Detection",
            page_icon="🏭",
            layout="wide"
        )

        # Initialize configuration
        if "thresholds" not in st.session_state:
            st.session_state.thresholds = {
                "low": 0.6,
                "medium": 0.75,
                "high": 0.9,
                "critical": 0.95
            }

    def run(self, sensor_data: pd.DataFrame, predictions: np.ndarray, alerts: list):
        """Main dashboard loop"""

        st.title("🏭 Manufacturing Defect Detection System")
        st.markdown("---")

        # -------------------- KPIs --------------------
        col1, col2, col3, col4, col5 = st.columns(5)

        production_rate = sensor_data["product_count"].iloc[-1]

        defect_rate = predictions.mean() * 100
        historical_defect_rate = sensor_data["defect_label"].mean() * 100
        defect_delta = defect_rate - historical_defect_rate

        active_alerts = len([a for a in alerts if a["status"] == "active"])

        uptime = (sensor_data["machine_status"] == "running").mean() * 100
        system_health = 100 - defect_rate

        with col1:
            st.metric("Production Rate", f"{production_rate:.0f} units/hr")

        with col2:
            st.metric("Defect Rate", f"{defect_rate:.2f}%", delta=f"{defect_delta:.2f}%")

        with col3:
            st.metric("Active Alerts", active_alerts)

        with col4:
            st.metric("System Health", f"{system_health:.1f}%")

        with col5:
            st.metric("Uptime", f"{uptime:.2f}%")

        st.markdown("---")

        # -------------------- Tabs --------------------
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Real-time Monitoring", "Alerts", "Analytics", "Configuration"]
        )

        with tab1:
            self.render_monitoring_tab(sensor_data, predictions)

        with tab2:
            self.render_alerts_tab(alerts)

        with tab3:
            self.render_analytics_tab(sensor_data, predictions)

        with tab4:
            self.render_configuration_tab()

    # -------------------------------------------------
    def render_monitoring_tab(self, sensor_data, predictions):

        col1, col2 = st.columns(2)

        # Temperature & Pressure
        with col1:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=sensor_data["timestamp"],
                y=sensor_data["temperature"],
                name="Temperature (°C)"
            ))

            fig.add_trace(go.Scatter(
                x=sensor_data["timestamp"],
                y=sensor_data["pressure"],
                name="Pressure (PSI)",
                yaxis="y2"
            ))

            anomalies = np.where(predictions == 1)[0]
            if len(anomalies) > 0:
                fig.add_trace(go.Scatter(
                    x=sensor_data.iloc[anomalies]["timestamp"],
                    y=sensor_data.iloc[anomalies]["temperature"],
                    mode="markers",
                    name="Anomaly",
                    marker=dict(color="red", size=9, symbol="x")
                ))

            fig.update_layout(
                title="Temperature & Pressure",
                yaxis=dict(title="Temperature"),
                yaxis2=dict(title="Pressure", overlaying="y", side="right"),
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)

        # Vibration & Power
        with col2:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=sensor_data["timestamp"],
                y=sensor_data["vibration"],
                name="Vibration (mm/s)"
            ))

            fig.add_trace(go.Scatter(
                x=sensor_data["timestamp"],
                y=sensor_data["power"],
                name="Power (kW)",
                yaxis="y2"
            ))

            fig.update_layout(
                title="Vibration & Power",
                yaxis=dict(title="Vibration"),
                yaxis2=dict(title="Power", overlaying="y", side="right"),
                hovermode="x unified"
            )

            st.plotly_chart(fig, use_container_width=True)

        # ---------------- Heatmap ----------------
        st.subheader("Sensor Anomaly Heatmap")

        sensors = ["temperature", "vibration", "pressure", "speed", "power", "acoustic"]
        sampled = sensor_data.iloc[::60]

        heatmap = []
        for _, row in sampled.iterrows():
            scores = []
            for s in sensors:
                z = abs((row[s] - sensor_data[s].mean()) / sensor_data[s].std())
                scores.append(min(z, 3))
            heatmap.append(scores)

        fig = go.Figure(go.Heatmap(
            z=heatmap,
            x=sensors,
            y=sampled["timestamp"].dt.strftime("%H:%M"),
            colorscale="RdYlGn_r",
            colorbar=dict(title="Anomaly Score")
        ))

        fig.update_layout(title="Anomaly Heatmap (Sampled)")
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------
    def render_alerts_tab(self, alerts):

        active_alerts = [a for a in alerts if a["status"] == "active"]

        if active_alerts:
            for alert in active_alerts:
                with st.expander(f"🚨 {alert['alert_level'].upper()} - {alert['defect_type']}", True):
                    st.write(f"**Machine:** {alert['machine_id']}")
                    st.write(f"**Confidence:** {alert['confidence_score']:.2%}")
                    st.write(f"**Impact:** {alert['predicted_impact']}")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Acknowledge", key=f"ack_{alert['alert_id']}"):
                            alert["acknowledged"] = True
                    with col2:
                        if st.button("Resolve", key=f"res_{alert['alert_id']}"):
                            alert["status"] = "resolved"
        else:
            st.info("No active alerts")

        st.subheader("Alert History")
        if alerts:
            st.dataframe(pd.DataFrame(alerts), use_container_width=True)

    # -------------------------------------------------
    def render_analytics_tab(self, sensor_data, predictions):

        col1, col2 = st.columns(2)

        with col1:
            defect_counts = sensor_data[sensor_data["defect_label"] == 1]["defect_type"].value_counts()
            fig = px.pie(values=defect_counts.values, names=defect_counts.index)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            y_true = sensor_data["defect_label"]

            metrics = {
                "Precision": precision_score(y_true, predictions),
                "Recall": recall_score(y_true, predictions),
                "F1-Score": f1_score(y_true, predictions),
                "Accuracy": accuracy_score(y_true, predictions)
            }

            fig = px.bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                title="Model Performance"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Sensor Correlation Matrix")
        sensor_cols = ["temperature", "vibration", "pressure", "speed", "power", "acoustic"]
        corr = sensor_data[sensor_cols].corr()

        fig = px.imshow(corr, color_continuous_scale="RdBu", aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------
    def render_configuration_tab(self):

        st.subheader("Alert Threshold Configuration")

        for level in st.session_state.thresholds:
            st.session_state.thresholds[level] = st.slider(
                level.capitalize(),
                0.0, 1.0,
                st.session_state.thresholds[level]
            )

        st.success("Configuration stored in session")
