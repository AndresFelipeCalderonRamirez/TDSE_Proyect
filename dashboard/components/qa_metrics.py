from typing import Any, Dict

import streamlit as st


def render_qa_metrics(metrics: Dict[str, Dict[str, Any]]):
    st.subheader("QA Metrics Dashboard")

    if not metrics:
        st.info("No QA metrics available. Run `python measure_metrics.py --all` to generate.")
        return

    cols = st.columns(4)
    for i, (key, metric) in enumerate(metrics.items()):
        status = metric.get("status", "PASS")
        measured = metric.get("measured", "N/A")
        threshold = metric.get("threshold", "")
        label = metric.get("label", key)
        description = metric.get("description", "")
        unit = metric.get("unit", "")

        is_pass = status == "PASS"
        border_color = "#00FF66" if is_pass else "#FF4444"
        bg_color = "rgba(0,255,102,0.05)" if is_pass else "rgba(255,68,68,0.05)"
        icon = "✓" if is_pass else "✗"

        display_value = measured
        if isinstance(measured, float):
            if unit == "%":
                display_value = f"{measured:.2f}{unit}"
            elif key == "QA6_tenant_isolation":
                display_value = f"{int(measured)} leaks"
            else:
                display_value = f"{measured:.4f}"

        col_idx = i % 4
        with cols[col_idx]:
            st.markdown(
                f"""
                <div style="
                    background:{bg_color};
                    border:1px solid rgba(255,255,255,0.08);
                    border-left:3px solid {border_color};
                    border-radius:8px;
                    padding:14px 12px;
                    margin-bottom:12px;
                    height:160px;
                    display:flex;
                    flex-direction:column;
                    justify-content:space-between;
                ">
                    <div>
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <span style="font-size:12px;font-weight:600;color:#aaa;text-transform:uppercase;">
                                {label}
                            </span>
                            <span style="
                                font-size:14px;font-weight:700;
                                color:{border_color};
                                background:rgba(255,255,255,0.06);
                                padding:2px 10px;
                                border-radius:12px;
                            ">
                                {icon} {status}
                            </span>
                        </div>
                        <div style="font-size:24px;font-weight:700;color:#fff;margin:8px 0 4px;">
                            {display_value}
                        </div>
                    </div>
                    <div>
                        <div style="font-size:11px;color:#888;">
                            Threshold: {threshold}
                        </div>
                        <div style="font-size:10px;color:#666;margin-top:2px;">
                            {description}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div style="
            margin-top:16px;padding:12px;border-radius:8px;
            background:rgba(255,255,255,0.03);
            border:1px solid rgba(255,255,255,0.08);
            font-size:12px;color:#888;
        ">
        Legend: <span style="color:#00FF66;">✓ PASS</span> — Metric meets threshold &nbsp;|&nbsp;
        <span style="color:#FF4444;">✗ FAIL</span> — Metric below threshold &nbsp;|&nbsp;
        Metrics computed from <code>evaluation_report.json</code> and live DynamoDB data.
        <br>Run <code>python measure_metrics.py --all</code> to regenerate report.
        </div>
        """,
        unsafe_allow_html=True,
    )
