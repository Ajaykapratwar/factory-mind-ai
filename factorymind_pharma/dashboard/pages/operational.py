"""
dashboard/pages/operational.py — Step 1.6
Operational Planning Dashboard.

- Warehouse-level inventory table (opening/received/sold/closing, reorder point,
  safety stock, stockout flag highlighted).
- Order fulfillment funnel (ordered → delivered, by order_status).
- Product-level drill-down: select product_id → demand history + forecast + inventory trend.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import dash
from dash import html, dcc, callback, Input, Output
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from dashboard.data_loader import get_inventory_status, get_demand_vs_actual
from dashboard.components.charts import (
    demand_vs_actual_chart,
    inventory_trend_chart,
    CHART_COLORS,
    make_chart_layout,
)
import plotly.graph_objects as go

dash.register_page(__name__, path="/operational", name="Operational Planning")

_BASE = Path(__file__).parent.parent.parent
_CONFIG_PATH = _BASE / "configs" / "config.yaml"


def _load_config():
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _get_orders_df():
    cfg = _load_config()
    raw_dir = _BASE / cfg["data"]["raw_dir"]
    path = raw_dir / cfg["data"]["files"]["orders"]
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _get_inventory_raw():
    cfg = _load_config()
    raw_dir = _BASE / cfg["data"]["raw_dir"]
    path = raw_dir / cfg["data"]["files"]["inventory"]
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    return df


def _get_product_ids():
    try:
        inv = get_inventory_status()
        return sorted(inv["product_id"].unique().tolist())
    except Exception:
        return []


def _stockout_badge(flag):
    if flag == 1 or flag is True:
        return html.Span("STOCKOUT", className="badge badge-danger")
    return html.Span("OK", className="badge badge-success")


def _inventory_table(df: pd.DataFrame) -> html.Table:
    cols = ["product_id", "warehouse_id", "opening_stock", "received_quantity",
            "units_sold", "closing_stock", "safety_stock", "reorder_point", "stockout_flag"]
    rows = []
    for _, r in df.iterrows():
        cells = [html.Td(str(r.get(c, "—")), className="mono") for c in cols[:-1]]
        cells.append(html.Td(_stockout_badge(r.get("stockout_flag", 0))))
        rows.append(html.Tr(cells))

    return html.Table([
        html.Thead(html.Tr([
            html.Th(c.replace("_", " ").title()) for c in cols
        ])),
        html.Tbody(rows),
    ], className="data-table")


def _funnel_chart(orders_df: pd.DataFrame) -> go.Figure:
    """Order fulfillment funnel by status."""
    status_counts = orders_df["order_status"].value_counts().reset_index()
    status_counts.columns = ["status", "count"]

    # Sort by typical fulfillment order
    order_map = {"Pending": 0, "Processing": 1, "Shipped": 2, "Delivered": 3, "Cancelled": 4}
    status_counts["rank"] = status_counts["status"].map(order_map).fillna(99)
    status_counts = status_counts.sort_values("rank")

    colors = [
        CHART_COLORS["chart_4"],   # Pending
        CHART_COLORS["chart_5"],   # Processing
        CHART_COLORS["chart_2"],   # Shipped
        CHART_COLORS["chart_1"],   # Delivered
        CHART_COLORS["chart_3"],   # Cancelled
    ]

    fig = go.Figure(go.Bar(
        x=status_counts["count"],
        y=status_counts["status"],
        orientation="h",
        marker_color=colors[:len(status_counts)],
        text=status_counts["count"].apply(lambda v: f"{v:,}"),
        textposition="outside",
        textfont=dict(size=11, color=CHART_COLORS["muted"]),
    ))

    layout = make_chart_layout(
        title="Order Fulfillment by Status",
        height=240,
        xaxis_title="Orders",
    )
    layout["xaxis"]["showgrid"] = False
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def layout():
    product_ids = _get_product_ids()
    inv_df = get_inventory_status()
    orders_df = _get_orders_df()

    funnel_fig = _funnel_chart(orders_df)

    dropdown_opts = [{"label": pid, "value": pid} for pid in product_ids]
    default_pid = product_ids[0] if product_ids else None

    return html.Div([
        html.Div([
            html.Div([
                html.H1("Operational Planning"),
                html.P("Warehouse inventory · order fulfillment · product drill-down",
                       className="page-subtitle"),
            ], className="page-header"),

            # ── Inventory + Funnel ──────────────────────────────
            html.Div([
                # Inventory table (scrollable)
                html.Div([
                    html.Div([
                        html.Div("Warehouse Inventory Status", className="card-title"),
                        html.Div("Latest snapshot per product × warehouse", className="card-meta"),
                    ], className="card-header"),
                    html.Div([
                        _inventory_table(inv_df.head(40)),
                    ], style={"overflowX": "auto", "maxHeight": "360px", "overflowY": "auto"}),
                ], className="card"),

                # Funnel
                html.Div([
                    html.Div([
                        html.Div("Order Fulfillment Funnel", className="card-title"),
                        html.Div("All orders · all regions", className="card-meta"),
                    ], className="card-header"),
                    dcc.Graph(figure=funnel_fig, config={"displayModeBar": False},
                              id="ops-funnel-chart"),
                ], className="card"),

            ], className="section-row"),

            # ── Product drill-down ──────────────────────────────
            html.Div([
                html.Div("Product Drill-Down", className="card-title",
                         style={"marginBottom": "16px"}),

                html.Div([
                    html.Div("SELECT PRODUCT", className="filter-label"),
                    dcc.Dropdown(
                        id="ops-product-dropdown",
                        options=dropdown_opts,
                        value=default_pid,
                        clearable=False,
                        style={"width": "220px", "fontSize": "13px"},
                    ),
                ], className="filter-row"),

                html.Div(id="ops-drill-content"),

            ], className="card", style={"marginTop": "0"}),

        ], className="content"),
    ], className="page-wrapper")


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("ops-drill-content", "children"),
    Input("ops-product-dropdown", "value"),
)
def update_drill_down(product_id):
    if not product_id:
        return html.P("Select a product above.", className="card-meta")

    demand_df = get_demand_vs_actual(product_id=product_id)
    inv_raw = _get_inventory_raw()
    inv_product = inv_raw[inv_raw["product_id"] == product_id].sort_values("date")

    demand_fig = demand_vs_actual_chart(demand_df, height=280)

    # Inventory trend for this product
    inv_fig = go.Figure()
    if not inv_product.empty:
        inv_fig.add_trace(go.Scatter(
            x=inv_product["date"], y=inv_product["closing_stock"],
            name="Closing Stock", mode="lines",
            fill="tozeroy",
            fillcolor="rgba(45, 125, 123, 0.10)",
            line=dict(color=CHART_COLORS["chart_1"], width=2),
        ))
        # Reorder point line
        if "reorder_point" in inv_product.columns:
            inv_fig.add_trace(go.Scatter(
                x=inv_product["date"], y=inv_product["reorder_point"],
                name="Reorder Point", mode="lines",
                line=dict(color=CHART_COLORS["chart_3"], width=1.5, dash="dot"),
            ))
        if "safety_stock" in inv_product.columns:
            inv_fig.add_trace(go.Scatter(
                x=inv_product["date"], y=inv_product["safety_stock"],
                name="Safety Stock", mode="lines",
                line=dict(color=CHART_COLORS["chart_5"], width=1.5, dash="dash"),
            ))

    layout = make_chart_layout(
        title=f"Inventory Levels — {product_id}",
        height=280,
        yaxis_title="Stock Units",
    )
    inv_fig.update_layout(**layout)

    return html.Div([
        html.Div([
            html.Div([
                html.Div("Demand History & Forecast", className="card-title"),
            ], className="card-header"),
            dcc.Graph(figure=demand_fig, config={"displayModeBar": False}),
        ], className="card", style={"marginBottom": "16px"}),

        html.Div([
            html.Div([
                html.Div("Inventory Trend", className="card-title"),
                html.Div("Closing stock · reorder point · safety stock", className="card-meta"),
            ], className="card-header"),
            dcc.Graph(figure=inv_fig, config={"displayModeBar": False}),
        ], className="card"),
    ])
