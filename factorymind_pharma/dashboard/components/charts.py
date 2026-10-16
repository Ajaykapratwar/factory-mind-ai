"""
dashboard/components/charts.py
Shared chart helpers — custom palette, layout tokens, reusable chart factory functions.
All charts use the token palette — never default Plotly blue-orange-green.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Optional

# Palette derived from design tokens (CSS vars can't be read at runtime in Python,
# so we mirror the hex values here)
CHART_COLORS = {
    "chart_1": "#2D7D7B",   # accent teal
    "chart_2": "#7AB5B4",   # muted teal
    "chart_3": "#C45B3C",   # terracotta
    "chart_4": "#6B5B8A",   # plum
    "chart_5": "#D4A843",   # warm amber
    "bg":      "#FDFBF7",
    "surface": "#F6F3EE",
    "border":  "#DDD9D1",
    "text":    "#1C1B1A",
    "muted":   "#6B6560",
}

COLOR_SEQ = [
    CHART_COLORS["chart_1"],
    CHART_COLORS["chart_3"],
    CHART_COLORS["chart_2"],
    CHART_COLORS["chart_4"],
    CHART_COLORS["chart_5"],
]


def make_chart_layout(
    title: str = "",
    height: int = 300,
    xaxis_title: str = "",
    yaxis_title: str = "",
    show_legend: bool = True,
    margin: Optional[dict] = None,
) -> dict:
    """Base Plotly layout dict matching our design tokens."""
    if margin is None:
        margin = dict(l=16, r=16, t=36 if title else 16, b=16)

    return dict(
        title=dict(text=title, font=dict(family="Inter, -apple-system, sans-serif",
                                          size=14, color=CHART_COLORS["text"])),
        height=height,
        paper_bgcolor=CHART_COLORS["bg"],
        plot_bgcolor=CHART_COLORS["bg"],
        margin=margin,
        font=dict(family="Inter, -apple-system, sans-serif", size=12, color=CHART_COLORS["text"]),
        showlegend=show_legend,
        legend=dict(
            font=dict(size=11, color=CHART_COLORS["muted"]),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        xaxis=dict(
            title=xaxis_title,
            gridcolor=CHART_COLORS["border"],
            gridwidth=1,
            zeroline=False,
            tickfont=dict(size=11, color=CHART_COLORS["muted"]),
        ),
        yaxis=dict(
            title=yaxis_title,
            gridcolor=CHART_COLORS["border"],
            gridwidth=1,
            zeroline=False,
            tickfont=dict(size=11, color=CHART_COLORS["muted"]),
        ),
        hovermode="x unified",
    )


def demand_vs_actual_chart(df: pd.DataFrame, height: int = 320) -> go.Figure:
    """
    Line chart: actual units_sold vs naive_pred vs hurdle/lgb point forecast over time.
    If pred_q10 / pred_q90 columns are present, draw a shaded uncertainty band.
    Aggregated or per-SKU.
    """
    fig = go.Figure()

    if "actual" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["actual"],
            name="Actual", mode="lines",
            line=dict(color=CHART_COLORS["chart_1"], width=2),
        ))

    # P10–P90 uncertainty band
    if "pred_q90" in df.columns and "pred_q10" in df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([df["date"], df["date"].iloc[::-1]]),
            y=pd.concat([df["pred_q90"], df["pred_q10"].iloc[::-1]]),
            fill="toself",
            fillcolor="rgba(99, 110, 250, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="P10–P90 band",
        ))

    if "naive_pred" in df.columns and df["naive_pred"].notna().any():
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["naive_pred"],
            name="Naive Seasonal", mode="lines",
            line=dict(color=CHART_COLORS["chart_3"], width=1.5, dash="dot"),
        ))

    # Prefer hurdle / full-grid column; fall back to legacy lgb_pred
    for col, label in [("pred_hurdle", "Hurdle Model"), ("pred_lgb", "LightGBM"),
                       ("lgb_pred", "LightGBM (legacy)")]:
        if col in df.columns and df[col].notna().any():
            fig.add_trace(go.Scatter(
                x=df["date"], y=df[col],
                name=label, mode="lines",
                line=dict(color=CHART_COLORS["chart_4"], width=1.5, dash="dash"),
            ))
            break

    layout = make_chart_layout(
        title="Demand — Actual vs Forecast",
        height=height,
        xaxis_title="",
        yaxis_title="Units Sold",
    )
    fig.update_layout(**layout)
    return fig



def inventory_trend_chart(df: pd.DataFrame, product_id: str, height: int = 280) -> go.Figure:
    """
    Area chart of closing stock trend for a specific product.
    """
    pdata = df[df["product_id"] == product_id].sort_values("date") if "product_id" in df.columns else df
    if pdata.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pdata["date"],
        y=pdata["closing_stock"] if "closing_stock" in pdata.columns else pdata.get("total_closing_stock", []),
        name="Closing Stock",
        mode="lines",
        fill="tozeroy",
        fillcolor=f"rgba(45, 125, 123, 0.12)",
        line=dict(color=CHART_COLORS["chart_1"], width=2),
    ))

    layout = make_chart_layout(
        title=f"Inventory Trend — {product_id}",
        height=height,
        yaxis_title="Stock Units",
    )
    fig.update_layout(**layout)
    return fig
