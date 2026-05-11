from typing import Any, Dict, List

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.utils.parsing import parse_sensor_record, parse_timestamp


def render_anomaly_chart(
    records: List[Dict[str, Any]],
    threshold: float = 0.65,
):
    if not records:
        st.info("No sensor records available for anomaly monitoring.")
        return

    parsed = [parse_sensor_record(r) for r in records]
    parsed.sort(key=lambda x: x.get("timestamp", ""))

    timestamps = []
    anomaly_scores = []
    is_anomaly = []
    segment_ids = []
    pressures = []
    flows = []
    vibrations = []

    for r in parsed:
        ts = r.get("timestamp", "")
        timestamps.append(ts)
        anomaly_scores.append(r.get("anomalyScore", 0))
        is_anomaly.append(r.get("isAnomaly", False))
        segment_ids.append(r.get("segmentId", ""))
        pressures.append(r.get("pressure", 0))
        flows.append(r.get("flow", 0))
        vibrations.append(r.get("vibration", 0))

    anomaly_count = sum(1 for a in is_anomaly if a)
    avg_score = sum(anomaly_scores) / max(len(anomaly_scores), 1)
    latest_ts = timestamps[-1] if timestamps else "N/A"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Anomalies", anomaly_count)
    with col2:
        st.metric("Avg Anomaly Score", f"{avg_score:.4f}")
    with col3:
        st.metric("Latest Record", latest_ts[:19] if len(latest_ts) > 19 else latest_ts)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Anomaly Score", "Sensor Readings", "Isolation Status"),
    )

    colors = ["#FF4444" if a else "#00CCFF" for a in is_anomaly]

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=anomaly_scores,
            mode="lines+markers",
            name="Anomaly Score",
            line=dict(color="#4488FF", width=1.5),
            marker=dict(color=colors, size=6),
            hovertemplate="<b>%{text}</b><br>Score: %{y:.4f}<br>TS: %{x}",
            text=segment_ids,
        ),
        row=1,
        col=1,
    )

    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="#FF8800",
        annotation_text=f"τ = {threshold}",
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=pressures,
            mode="lines",
            name="Pressure",
            line=dict(color="#00CCFF", width=1),
            hovertemplate="Pressure: %{y:.2f} bar<br>%{x}",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=flows,
            mode="lines",
            name="Flow",
            line=dict(color="#00FF66", width=1),
            hovertemplate="Flow: %{y:.2f} m³/s<br>%{x}",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=vibrations,
            mode="lines",
            name="Vibration",
            line=dict(color="#FF8800", width=1),
            hovertemplate="Vibration: %{y:.2f} mm/s<br>%{x}",
        ),
        row=2,
        col=1,
    )

    anomaly_flags = [1 if a else 0 for a in is_anomaly]
    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=anomaly_flags,
            mode="lines+markers",
            name="Is Anomaly",
            line=dict(color="#FF4444", width=2),
            marker=dict(
                color=["#FF4444" if a else "#333333" for a in is_anomaly],
                size=8,
                symbol="square",
            ),
            hovertemplate="Anomaly: %{y}<br>%{x}",
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        height=550,
        template="plotly_dark",
        hovermode="x unified",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    fig.update_xaxes(
        tickfont=dict(size=10, color="#888"),
        gridcolor="rgba(255,255,255,0.05)",
    )
    fig.update_yaxes(
        tickfont=dict(size=10, color="#888"),
        gridcolor="rgba(255,255,255,0.05)",
    )

    st.plotly_chart(fig, use_container_width=True)
