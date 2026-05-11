import streamlit as st
from typing import Callable, Any


def setup_auto_refresh(refresh_sec: int = 15):
    st.markdown(
        f"""
        <meta http-equiv="refresh" content="{refresh_sec}">
        <script>
            setTimeout(function() {{
                window.location.reload();
            }}, {refresh_sec * 1000});
        </script>
        """,
        unsafe_allow_html=True,
    )


def handle_auto_refresh(placeholder_func: Callable[[], Any], refresh_sec: int = 15):
    import time
    last_refresh = st.session_state.get("last_refresh", 0)
    now = time.time()
    if now - last_refresh >= refresh_sec:
        st.session_state["last_refresh"] = now
        placeholder_func()


def display_status_indicator(connected: bool):
    color = "#00FF66" if connected else "#FF4444"
    label = "DynamoDB Connected" if connected else "DynamoDB Disconnected"
    st.sidebar.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:8px;padding:8px 12px;
                    border-radius:6px;background:rgba(255,255,255,0.05);
                    border-left:3px solid {color};">
            <div style="width:10px;height:10px;border-radius:50%;background:{color};
                        box-shadow:0 0 8px {color};"></div>
            <span style="font-size:13px;color:{color};font-weight:500;">{label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
