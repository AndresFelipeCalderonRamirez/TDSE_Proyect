from typing import Any, Dict, List, Optional

import plotly.graph_objects as go
import streamlit as st

from dashboard.utils.parsing import parse_ranking_item


def render_risk_heatmap(
    ranking_item: Optional[Dict[str, Any]],
    tenant_id: str,
):
    st.subheader("Network Risk Heatmap")

    if not ranking_item:
        st.info("No ranking data available yet. The Digital Twin Lambda needs to run first.")
        return

    parsed = parse_ranking_item(ranking_item)
    segments = parsed.get("_parsed_segments", parsed.get("segments", []))
    if not segments:
        st.info("No segments found in latest ranking.")
        return

    segments.sort(key=lambda x: x.get("risk_propagated", 0), reverse=True)
    top_k = min(5, len(segments))
    top_segments = segments[:top_k]
    alpha = parsed.get("alpha", 0.7)
    generated_at = parsed.get("generated_at", "N/A")

    cols = st.columns(top_k)
    for i, seg in enumerate(top_segments):
        seg_id = seg.get("segment_id", f"SEG_{i}")
        risk = seg.get("risk_propagated", 0)
        intensity = min(risk * 1.5, 1.0)

        r = int(255 * intensity)
        g = int(60 * (1 - intensity))
        b = int(60 * (1 - intensity))
        bg_color = f"rgb({r},{g},{b})"

        with cols[i]:
            st.markdown(
                f"""
                <div style="
                    background:{bg_color};
                    border-radius:10px;
                    padding:16px 8px;
                    text-align:center;
                    border:1px solid rgba(255,255,255,0.1);
                    box-shadow:0 4px 12px rgba(0,0,0,0.3);
                ">
                    <div style="font-size:12px;color:#aaa;margin-bottom:4px;">
                        {seg_id}
                    </div>
                    <div style="font-size:28px;font-weight:700;color:#fff;">
                        {risk:.4f}
                    </div>
                    <div style="font-size:11px;color:#88ff88;margin-top:4px;">
                        Risk Score
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        f"<div style='font-size:12px;color:#888;margin-top:8px;'>"
        f"Alpha: {alpha} | Generated: {generated_at[:19]} | "
        f"Showing top-{top_k} of {len(segments)} segments"
        f"</div>",
        unsafe_allow_html=True,
    )

    fig = go.Figure()

    seg_labels = [s.get("segment_id", f"S{i}") for s in top_segments]
    risk_values = [s.get("risk_propagated", 0) for s in top_segments]

    fig.add_trace(
        go.Bar(
            x=seg_labels,
            y=risk_values,
            marker=dict(
                color=risk_values,
                colorscale=[
                    [0, "#00CC66"],
                    [0.5, "#FFAA00"],
                    [1, "#FF3333"],
                ],
                cmin=0,
                cmax=max(risk_values) if risk_values else 1,
                showscale=True,
                colorbar=dict(
                    title="Risk",
                    thickness=15,
                    len=0.6,
                ),
            ),
            text=[f"{v:.4f}" for v in risk_values],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Risk: %{y:.6f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text="Top Critical Segments — Propagated Risk",
            font=dict(size=14, color="#ccc"),
        ),
        height=300,
        template="plotly_dark",
        hovermode="x",
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            tickfont=dict(size=11, color="#aaa"),
            gridcolor="rgba(255,255,255,0.05)",
        ),
        yaxis=dict(
            title="Risk Score",
            tickfont=dict(size=11, color="#aaa"),
            gridcolor="rgba(255,255,255,0.05)",
            range=[0, max(risk_values) * 1.15] if risk_values else [0, 1],
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Full ranking details"):
        st.json(top_segments)
