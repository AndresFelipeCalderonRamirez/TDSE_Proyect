from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from dashboard.utils.parsing import parse_sensor_record


def render_maintenance_table(records: List[Dict[str, Any]]):
    st.subheader("Maintenance Alerts")

    if not records:
        st.info("No maintenance alerts available. Prediction Lambda needs to process anomalies first.")
        return

    parsed = [parse_sensor_record(r) for r in records]
    parsed.sort(
        key=lambda x: (
            -x.get("p_failure", 0),
            x.get("timestamp", "") or "",
        )
    )

    rows = []
    for r in parsed:
        rows.append(
            {
                "Segment ID": r.get("segmentId", ""),
                "P(Failure)": round(r.get("p_failure", 0), 6),
                "Risk Score": round(r.get("anomalyScore", 0), 4),
                "Timestamp": (r.get("prediction_timestamp", "") or r.get("timestamp", ""))[:19],
                "Pressure": round(r.get("pressure", 0), 2),
                "Flow": round(r.get("flow", 0), 2),
                "Vibration": round(r.get("vibration", 0), 2),
                "Priority": _compute_priority(r.get("p_failure", 0)),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No records with p_failure data found.")
        return

    def _color_priority(val: str) -> str:
        if val == "CRITICAL":
            return "background-color: rgba(255, 68, 68, 0.25); color: #FF4444"
        elif val == "HIGH":
            return "background-color: rgba(255, 136, 0, 0.25); color: #FF8800"
        elif val == "MEDIUM":
            return "background-color: rgba(255, 204, 0, 0.2); color: #FFCC00"
        return "background-color: rgba(0, 204, 102, 0.2); color: #00CC66"

    def _color_pfailure(val: float) -> str:
        if val >= 0.7:
            return "color: #FF4444; font-weight: bold"
        elif val >= 0.4:
            return "color: #FF8800; font-weight: bold"
        elif val >= 0.2:
            return "color: #FFCC00"
        return "color: #00CC66"

    styled_df = (
        df.style
        .map(_color_priority, subset=["Priority"])
        .map(_color_pfailure, subset=["P(Failure)"])
        .set_properties(**{
            "background-color": "rgba(255,255,255,0.02)",
            "border-color": "rgba(255,255,255,0.08)",
            "font-size": "13px",
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "rgba(255,255,255,0.08)"),
                    ("color", "#ccc"),
                    ("font-weight", "600"),
                    ("font-size", "12px"),
                    ("text-transform", "uppercase"),
                    ("border-bottom", "1px solid rgba(255,255,255,0.15)"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("border-bottom", "1px solid rgba(255,255,255,0.05)"),
                ],
            },
            {
                "selector": "tr:hover",
                "props": [("background-color", "rgba(255,255,255,0.05)")],
            },
        ])
    )

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=min(400, 45 * len(df) + 50),
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        critical = sum(1 for r in rows if r["Priority"] == "CRITICAL")
        st.metric("Critical", critical)
    with col2:
        high = sum(1 for r in rows if r["Priority"] == "HIGH")
        st.metric("High", high)
    with col3:
        medium = sum(1 for r in rows if r["Priority"] == "MEDIUM")
        st.metric("Medium", medium)
    with col4:
        low = sum(1 for r in rows if r["Priority"] == "LOW")
        st.metric("Low", low)


def _compute_priority(p_failure: float) -> str:
    if p_failure >= 0.7:
        return "CRITICAL"
    elif p_failure >= 0.4:
        return "HIGH"
    elif p_failure >= 0.2:
        return "MEDIUM"
    return "LOW"
