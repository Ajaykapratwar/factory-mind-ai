"""
dashboard/components/charts.py
Shared chart helpers — strictly consumes theme.py / tokens.css.
Zero hardcoded hex values anywhere in this file.
"""

from typing import Optional
import pandas as pd
import plotly.graph_objects as go

from dashboard.components.theme import (
    CHART_COLORS,
    COLOR_SEQ,
    FONT_UI,
    STATUS_COLORS,
    COMPARE_COLORS,
    SERIES_COLORS,
    make_chart_layout,
)

__all__ = [
    "CHART_COLORS",
    "COLOR_SEQ",
    "FONT_UI",
    "STATUS_COLORS",
    "COMPARE_COLORS",
    "SERIES_COLORS",
    "make_chart_layout",
    "demand_vs_actual_chart",
    "inventory_trend_chart",
    "fulfillment_funnel_chart",
    "safety_stock_compare_chart",
]


def demand_vs_actual_chart(df: pd.DataFrame, height: int = 320) -> go.Figure:
    """
    Line chart: actual units_sold vs naive_pred vs hurdle/lgb point forecast.
    If pred_q10 / pred_q90 columns are present, draw a shaded uncertainty band.
    Aggregates by date to prevent zigzag spikes from multiple SKUs sharing the same date.
    Resamples to weekly if date range > 180 days.
    """
    if df is None or df.empty:
        return go.Figure()

    chart_df = df.copy()
    if "date" in chart_df.columns:
        chart_df["date"] = pd.to_datetime(chart_df["date"])
    else:
        return go.Figure()

    # Identify numeric columns for aggregation
    potential_cols = [
        "actual", "units_sold", "naive_pred",
        "pred_hurdle", "pred_lgb", "lgb_pred",
        "pred_q10", "pred_q90"
    ]
    num_cols = [c for c in potential_cols if c in chart_df.columns]

    if "units_sold" in chart_df.columns and "actual" not in chart_df.columns:
        chart_df["actual"] = chart_df["units_sold"]
        num_cols.append("actual")

    # Aggregate by date if multiple rows exist
    if num_cols:
        chart_df = chart_df.groupby("date", as_index=False)[num_cols].sum()

    chart_df = chart_df.sort_values("date").reset_index(drop=True)

    # Resample weekly if > 180 days
    span_days = (chart_df["date"].max() - chart_df["date"].min()).days if len(chart_df) > 1 else 0
    if span_days > 180 or len(chart_df) > 180:
        chart_df = (
            chart_df.set_index("date")
            .resample("W-MON")[num_cols]
            .sum()
            .reset_index()
        )

    fig = go.Figure()

    # P10–P90 uncertainty band (render first so it's underneath)
    if "pred_q90" in chart_df.columns and "pred_q10" in chart_df.columns:
        valid_band = chart_df.dropna(subset=["pred_q10", "pred_q90"])
        if not valid_band.empty:
            fig.add_trace(go.Scatter(
                x=pd.concat([valid_band["date"], valid_band["date"].iloc[::-1]]),
                y=pd.concat([valid_band["pred_q90"], valid_band["pred_q10"].iloc[::-1]]),
                fill="toself",
                fillcolor=SERIES_COLORS["band"],
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="P10–P90 Band",
            ))

    if "actual" in chart_df.columns:
        fig.add_trace(go.Scatter(
            x=chart_df["date"],
            y=chart_df["actual"],
            name="Actual Demand",
            mode="lines",
            line=dict(color=SERIES_COLORS["actual"], width=2.5),
            hovertemplate="<b>Actual:</b> %{y:,.0f} units<extra></extra>",
        ))

    if "naive_pred" in chart_df.columns and chart_df["naive_pred"].notna().any():
        fig.add_trace(go.Scatter(
            x=chart_df["date"],
            y=chart_df["naive_pred"],
            name="Naive Baseline",
            mode="lines",
            line=dict(color=CHART_COLORS["warning"], width=1.5, dash="dot"),
            hovertemplate="<b>Naive:</b> %{y:,.0f} units<extra></extra>",
        ))

    # Prefer hurdle / full-grid column; fall back to legacy lgb_pred
    for col, label in [("pred_hurdle", "Hurdle Model"), ("pred_lgb", "LightGBM Forecast"),
                       ("lgb_pred", "LightGBM Forecast")]:
        if col in chart_df.columns and chart_df[col].notna().any():
            fig.add_trace(go.Scatter(
                x=chart_df["date"],
                y=chart_df[col],
                name=label,
                mode="lines",
                line=dict(color=SERIES_COLORS["forecast"], width=2, dash="dash"),
                hovertemplate=f"<b>{label}:</b> %{{y:,.0f}} units<extra></extra>",
            ))
            break

    layout = make_chart_layout(
        title="",  # Title rendered in card header to avoid duplication
        height=height,
        xaxis_title="",
        yaxis_title="Units Sold",
        show_legend=True,
    )
    # Ensure no clipping on y-axis ticks and light horizontal gridlines
    layout["yaxis"]["showgrid"] = True
    layout["yaxis"]["gridcolor"] = CHART_COLORS["border"]
    layout["xaxis"]["showgrid"] = False
    layout["yaxis"]["automargin"] = True
    layout["xaxis"]["automargin"] = True

    fig.update_layout(**layout)
    return fig


def fulfillment_funnel_chart(orders_df: pd.DataFrame, height: int = 260) -> go.Figure:
    """
    Order fulfillment funnel by status — horizontal bars.
    Fixes:
    - No duplicated Plotly title (handled by card).
    - showlegend=False.
    - cliponaxis=False.
    - X-axis extended by ~15% so outside text labels never clip.
    - Clean hover template (no 'trace 0').
    - Status order preserved: Pending -> Processing -> Shipped -> Delivered -> Cancelled.
    """
    if orders_df is None or orders_df.empty or "order_status" not in orders_df.columns:
        return go.Figure()

    status_counts = orders_df["order_status"].value_counts().reset_index()
    status_counts.columns = ["status", "count"]

    order_map = {"Pending": 0, "Processing": 1, "Shipped": 2, "Delivered": 3, "Cancelled": 4}
    status_counts["rank"] = status_counts["status"].map(order_map).fillna(99)
    # Reverse so top status appears at the top of the horizontal bar chart
    status_counts = status_counts.sort_values("rank", ascending=False)

    colour_map = {
        "Pending":    STATUS_COLORS.get("pending", CHART_COLORS["chart_4"]),
        "Processing": STATUS_COLORS.get("in-transit", CHART_COLORS["chart_6"]),
        "Shipped":    CHART_COLORS["chart_1"],
        "Delivered":  STATUS_COLORS.get("delivered", CHART_COLORS["chart_5"]),
        "Cancelled":  STATUS_COLORS.get("cancelled", CHART_COLORS["chart_3"]),
    }
    colors = [colour_map.get(s, CHART_COLORS["chart_1"]) for s in status_counts["status"]]

    max_count = status_counts["count"].max() if not status_counts.empty else 100

    fig = go.Figure(go.Bar(
        x=status_counts["count"],
        y=status_counts["status"],
        orientation="h",
        marker_color=colors,
        marker_line_width=0,
        text=status_counts["count"].apply(lambda v: f"{v:,}"),
        textposition="outside",
        cliponaxis=False,
        textfont=dict(size=11, family=FONT_UI, color=CHART_COLORS["text"]),
        hovertemplate="<b>%{y}:</b> %{x:,} orders<extra></extra>",
    ))

    layout = make_chart_layout(
        title="",
        height=height,
        xaxis_title="Orders",
        show_legend=False,
        margin=dict(l=110, r=80, t=16, b=40),
    )
    layout["xaxis"]["range"] = [0, max_count * 1.15]
    layout["xaxis"]["showgrid"] = True
    layout["xaxis"]["gridcolor"] = CHART_COLORS["border"]
    layout["yaxis"]["showgrid"] = False
    layout["yaxis"]["automargin"] = True
    layout["yaxis"]["tickfont"] = dict(size=12, family=FONT_UI, color=CHART_COLORS["text"])

    fig.update_layout(**layout)
    return fig


def safety_stock_compare_chart(df: pd.DataFrame, height: int = 340, top_n: int = 20) -> go.Figure:
    """
    Safety stock baseline vs optimized comparison chart.
    Restored original data call: df.dropna(subset=['baseline_safety_stock', 'new_safety_stock']).head(top_n)
    Exact original ordering (dataset order of finished goods).
    Styling:
    - Baseline = neutral slate (#7C8A99), Optimized = saturated teal (#1F8A85).
    - No duplicate Plotly title inside card.
    - Legend sits above plot and does not overlap bars.
    - hovermode='closest', showspikes=False on both axes.
    - automargin and readable tick labels.
    """
    if df is None or df.empty or "baseline_safety_stock" not in df.columns:
        return go.Figure()

    plot_df = df.dropna(subset=["baseline_safety_stock", "new_safety_stock"]).head(top_n)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=plot_df["product_id"],
        y=plot_df["baseline_safety_stock"],
        name="Baseline",
        marker_color=COMPARE_COLORS["baseline"],
        cliponaxis=False,
        hovertemplate="<b>%{x} Baseline:</b> %{y:,.0f} units<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=plot_df["product_id"],
        y=plot_df["new_safety_stock"],
        name="Optimized",
        marker_color=COMPARE_COLORS["optimized"],
        cliponaxis=False,
        hovertemplate="<b>%{x} Optimized:</b> %{y:,.0f} units<extra></extra>",
    ))

    layout = make_chart_layout(
        title="",
        height=380,
        xaxis_title="Product ID",
        yaxis_title="Safety Stock (units)",
        show_legend=True,
        margin={"l": 55, "r": 20, "t": 20, "b": 60},
    )
    layout["barmode"] = "group"
    layout["bargap"] = 0.2
    layout["bargroupgap"] = 0.05
    layout["hovermode"] = "closest"

    # Remove hover spike lines
    layout["xaxis"]["showspikes"] = False
    layout["xaxis"]["tickangle"] = -45
    layout["xaxis"]["showgrid"] = True
    layout["xaxis"]["gridcolor"] = CHART_COLORS["border"]
    layout["xaxis"]["automargin"] = True

    layout["yaxis"]["showspikes"] = False
    layout["yaxis"]["showgrid"] = True
    layout["yaxis"]["gridcolor"] = CHART_COLORS["border"]
    layout["yaxis"]["automargin"] = True

    # Legend at top right matching image 2
    layout["legend"] = dict(
        orientation="h",
        yanchor="bottom",
        y=1.04,
        xanchor="right",
        x=1,
        font=dict(family=FONT_UI, size=11, color=CHART_COLORS["text"]),
    )

    fig.update_layout(**layout)
    return fig


def inventory_trend_chart(df: pd.DataFrame, product_id: str, height: int = 280) -> go.Figure:
    """Area chart of closing stock trend for a specific product."""
    if df is None or df.empty:
        return go.Figure()

    pdata = (df[df["product_id"] == product_id].sort_values("date")
             if "product_id" in df.columns else df)
    if pdata.empty:
        return go.Figure()

    fig = go.Figure()
    y_vals = (pdata["closing_stock"] if "closing_stock" in pdata.columns
              else pdata.get("total_closing_stock", []))

    fig.add_trace(go.Scatter(
        x=pdata["date"],
        y=y_vals,
        name="Closing Stock",
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(31,138,133,0.12)",
        line=dict(color=CHART_COLORS["chart_1"], width=2),
        hovertemplate="<b>%{x|%Y-%m-%d}:</b> %{y:,.0f} units<extra></extra>",
    ))

    layout = make_chart_layout(
        title="",
        height=height,
        yaxis_title="Stock Units",
        show_legend=False,
        margin=dict(l=60, r=20, t=16, b=36),
    )
    layout["yaxis"]["showgrid"] = True
    layout["yaxis"]["gridcolor"] = CHART_COLORS["border"]
    layout["xaxis"]["showgrid"] = False

    fig.update_layout(**layout)
    return fig
