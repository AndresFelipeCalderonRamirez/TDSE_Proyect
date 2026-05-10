import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent))

# Allow using a mock DynamoDB-backed service for dashboard testing.
USE_MOCK = os.getenv("DASHBOARD_USE_MOCK_DB", "false").lower() in ("1", "true", "yes")
if USE_MOCK:
    from dashboard.services.mock_dynamo import (
        check_connection,
        query_recent_sensor_records,
        query_recent_anomalies,
        query_latest_ranking,
        query_maintenance_alerts,
        rebuild_cache,
    )
else:
    from dashboard.services.dynamo_service import (
        check_connection,
        query_recent_sensor_records,
        query_recent_anomalies,
        query_latest_ranking,
        query_maintenance_alerts,
        rebuild_cache,
    )
from dashboard.services.metrics_service import compute_qa_metrics
from dashboard.components.anomaly_chart import render_anomaly_chart
from dashboard.components.risk_heatmap import render_risk_heatmap
from dashboard.components.maintenance_table import render_maintenance_table
from dashboard.components.qa_metrics import render_qa_metrics
from dashboard.utils.refresh import display_status_indicator


st.set_page_config(
    page_title="WaterTwinML — Operational Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* ── Dark NOC theme ────────────────────────────────────── */
    .stApp {
        background-color: #0D1117;
        color: #E6EDF3;
    }
    .stApp header,
    .stApp footer {
        background-color: #0D1117;
    }
    .css-1d391kg, .css-12oz5g7 {
        background-color: #0D1117;
    }
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0D1117;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .stSelectbox label {
        color: #aaa;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    /* Metrics cards */
    div[data-testid="stMetric"] {
        background-color: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label {
        color: #888;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    div[data-testid="stMetric"] div:first-child {
        color: #E6EDF3;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(255,255,255,0.03);
        border-radius: 8px;
        padding: 4px;
        gap: 2px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 20px;
        font-size: 14px;
        font-weight: 500;
        color: #888;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(68, 136, 255, 0.2);
        color: #4488FF;
        box-shadow: 0 0 12px rgba(68, 136, 255, 0.1);
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #ccc;
        background-color: rgba(255,255,255,0.05);
    }
    /* Dataframes */
    .stDataFrame {
        background-color: transparent;
    }
    /* Buttons */
    .stButton button {
        background-color: rgba(68, 136, 255, 0.15);
        border: 1px solid rgba(68, 136, 255, 0.3);
        color: #4488FF;
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton button:hover {
        background-color: rgba(68, 136, 255, 0.25);
        border-color: #4488FF;
    }
    /* Info/warning/error */
    .stAlert {
        background-color: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
    }
    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(255,255,255,0.03);
        border-radius: 6px;
        font-size: 13px;
        color: #aaa;
    }
    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💧 WaterTwinML — Operational Dashboard")
st.markdown(
    "<p style='color:#888;font-size:14px;margin-top:-12px;'>"
    "Digital Twin for Water Distribution Networks &nbsp;·&nbsp; "
    "Real-time monitoring · Anomaly detection · Risk propagation</p>",
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("Controls")

db_connected = check_connection()
display_status_indicator(db_connected)

tenant = st.sidebar.selectbox(
    "Tenant",
    options=["tenant-A", "tenant-B"],
    index=0,
    format_func=lambda x: "Tenant-A (Bogotá)" if x == "tenant-A" else "Tenant-B (Medellín)",
    key="tenant_selector",
)

refresh_sec = st.sidebar.slider(
    "Auto-refresh interval",
    min_value=5,
    max_value=60,
    value=15,
    step=5,
    help="Dashboard auto-refresh in seconds",
)

if st.sidebar.button("⟳ Refresh Now", use_container_width=True):
    rebuild_cache()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size:11px;color:#555;padding:8px;'>"
    "WaterTwinML v1.0<br>"
    "TDSE — Universidad<br>"
    "Stack: Streamlit + DynamoDB + Plotly"
    "</div>",
    unsafe_allow_html=True,
)

# ── Session state ────────────────────────────────────────────────────────────

if "last_tenant" not in st.session_state:
    st.session_state["last_tenant"] = tenant
if st.session_state["last_tenant"] != tenant:
    st.session_state["last_tenant"] = tenant
    rebuild_cache()
    st.rerun()

if "refresh_count" not in st.session_state:
    st.session_state["refresh_count"] = 0

import time
now = time.time()
last = st.session_state.get("_last_refresh_ts", 0)
if now - last >= refresh_sec:
    st.session_state["_last_refresh_ts"] = now
    st.session_state["refresh_count"] = st.session_state.get("refresh_count", 0) + 1

# ── Data loading ─────────────────────────────────────────────────────────────

with st.spinner("Loading data from DynamoDB..."):
    try:
        sensor_records = query_recent_sensor_records(tenant, limit=100)
        anomaly_records = query_recent_anomalies(tenant, limit=100)
        ranking_item = query_latest_ranking(tenant)
        maintenance_records = query_maintenance_alerts(tenant, limit=100)
        qa_metrics = compute_qa_metrics(tenant, sensor_records)
        data_loaded = True
    except Exception as e:
        st.error(f"Failed to load data from DynamoDB: {e}")
        sensor_records = []
        anomaly_records = []
        ranking_item = None
        maintenance_records = []
        qa_metrics = {}
        data_loaded = False

st.caption(
    f"Last refresh: {time.strftime('%Y-%m-%d %H:%M:%S')}  ·  "
    f"Auto-refresh every {refresh_sec}s  ·  "
    f"Refresh #{st.session_state.get('refresh_count', 0)}"
)

# ── Overview row ────────────────────────────────────────────────────────────

if data_loaded:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sensor Records", len(sensor_records))
    with col2:
        anomaly_count = sum(
            1 for r in sensor_records
            if r.get("isAnomaly", {}).get("BOOL", False)
        )
        st.metric("Anomalies Detected", anomaly_count)
    with col3:
        alert_count = len(maintenance_records)
        st.metric("Active Alerts", alert_count)
    with col4:
        has_ranking = "Yes" if ranking_item else "No"
        st.metric("Latest Ranking", has_ranking)

# ── Tabs ────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Anomaly Monitoring", "🌐 Network Risk", "🔔 Maintenance Alerts", "📋 QA Metrics"]
)

with tab1:
    if data_loaded:
        render_anomaly_chart(anomaly_records, threshold=0.65)
    else:
        st.warning("Anomaly data unavailable — check DynamoDB connection.")

with tab2:
    if data_loaded:
        render_risk_heatmap(ranking_item, tenant)
    else:
        st.warning("Risk data unavailable — check DynamoDB connection.")

with tab3:
    if data_loaded:
        render_maintenance_table(maintenance_records)
    else:
        st.warning("Maintenance data unavailable — check DynamoDB connection.")

with tab4:
    if data_loaded:
        render_qa_metrics(qa_metrics)
    else:
        st.warning("QA metrics unavailable — check DynamoDB connection.")

# ── Footer ──────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center;font-size:11px;color:#444;padding:8px 0;'>"
    "WaterTwinML — Cloud-Native Water Infrastructure Platform &nbsp;·&nbsp; "
    "AWS + Streamlit + DynamoDB &nbsp;·&nbsp; "
    "TDSE 2026"
    "</div>",
    unsafe_allow_html=True,
)
