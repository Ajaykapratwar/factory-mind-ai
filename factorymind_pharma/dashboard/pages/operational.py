"""
dashboard/pages/operational.py — Operational Planning Dashboard (redesigned v2).

- Warehouse-level inventory table with status badges and right-aligned numbers.
- Order fulfillment funnel with proper width, horizontal bars, no label clipping.
- Product-level drill-down: demand history + forecast + inventory trend.
"""

import sys, os
from pathlib import Path
_PKG_DIR = Path(__file__).resolve().parent.parent.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

import dash
from dash import html, dcc, callback, Input, Output
import pandas as pd
import numpy as np
import yaml
import plotly.graph_objects as go

from dashboard.data_loader import get_inventory_status, get_demand_vs_actual
from dashboard.components.charts import (
    demand_vs_actual_chart,
    fulfillment_funnel_chart,
    CHART_COLORS,
    FONT_UI,
    make_chart_layout,
)
from dashboard.components.header import render_page_header, PAGE_METADATA

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
        return html.Span([html.Span("▼ ", style={"fontSize": "0.65rem"}), "STOCKOUT"],
                         className="badge badge-danger",
                         title="Stockout detected")
    return html.Span([html.Span("✓ ", style={"fontSize": "0.65rem"}), "OK"],
                     className="badge badge-success",
                     title="Stock OK")


def _inventory_table(df: pd.DataFrame) -> html.Div:
    cols = [
        ("product_id", "Product ID", False),
        ("warehouse_id", "Warehouse", False),
        ("opening_stock", "Opening", True),
        ("received_quantity", "Received", True),
        ("units_sold", "Sold", True),
        ("closing_stock", "Closing", True),
        ("safety_stock", "Safety", True),
        ("reorder_point", "Reorder Pt", True),
    ]

    rows = []
    for _, r in df.iterrows():
        cells = []
        for field, _, is_num in cols:
            val = r.get(field, "—")
            if is_num and isinstance(val, (int, float)) and not np.isnan(val):
                val_str = f"{val:,.0f}"
                cells.append(html.Td(val_str, className="numeric"))
            else:
                val_str = str(val) if not (isinstance(val, float) and np.isnan(val)) else "—"
                cells.append(html.Td(val_str, style={"fontWeight": "600"} if field == "product_id" else {}))

        cells.append(html.Td(_stockout_badge(r.get("stockout_flag", 0))))

        row_style = {}
        if r.get("stockout_flag", 0) == 1:
            row_style = {"background": "var(--danger-pale)"}
        rows.append(html.Tr(cells, style=row_style))

    return html.Div([
        html.Table([
            html.Thead(html.Tr(
                [html.Th(title, className="numeric" if is_num else "") for _, title, is_num in cols] +
                [html.Th("Status")]
            )),
            html.Tbody(rows),
        ], className="data-table"),
    ], className="scroll-x scroll-y", style={"maxHeight": "360px"})


def layout():
    product_ids = _get_product_ids()
    inv_df = get_inventory_status()
    orders_df = _get_orders_df()
    funnel_fig = fulfillment_funnel_chart(orders_df, height=300)
    dropdown_opts = [{"label": pid, "value": pid} for pid in product_ids]
    default_pid = product_ids[0] if product_ids else None
    meta = PAGE_METADATA.get("/operational", {})

    return html.Div([
        # Shared page header (no duplicate H1)
        render_page_header(meta.get("subtitle", "Daily plant throughput, machine schedule & order fulfillment")),

        # Section 1: Inventory Table (Full width)
        html.Div([
            html.Div([
                html.Div([
                    html.H2("Warehouse Inventory Status", className="section-title"),
                    html.Div("Latest snapshot per product × warehouse (top 40 records)", className="section-subtitle"),
                ]),
            ], className="card-header-flex"),
            _inventory_table(inv_df.head(40)),
        ], className="card"),

        # Section 2: Order Fulfillment Funnel
        html.Div([
            html.Div([
                html.Div([
                    html.H2("Order Fulfillment Funnel", className="section-title"),
                    html.Div("All orders by fulfillment status stage", className="section-subtitle"),
                ]),
            ], className="card-header-flex"),
            dcc.Graph(
                figure=funnel_fig,
                config={"displayModeBar": False},
                id="ops-funnel-chart",
            ),
        ], className="card"),

        # Section 3: Product Drill-Down
        html.Div([
            html.Div([
                html.Div([
                    html.H2("Product Drill-Down", className="section-title"),
                    html.Div("Detailed demand history, point forecast, and inventory buffer telemetry", className="section-subtitle"),
                ]),
                html.Div([
                    html.Span("SKU Filter:", style={"fontSize": "0.78rem", "fontWeight": "600", "color": "var(--text-muted)", "marginRight": "8px"}),
                    dcc.Dropdown(
                        id="ops-product-dropdown",
                        options=dropdown_opts,
                        value=default_pid,
                        clearable=False,
                        style={"width": "200px"},
                    ),
                ], style={"display": "flex", "alignItems": "center"}),
            ], className="card-header-flex"),

            html.Div(id="ops-drill-content"),
        ], className="card"),

    ], className="page-content")


@callback(
    Output("ops-drill-content", "children"),
    Input("ops-product-dropdown", "value"),
)
def update_drill_down(product_id):
    if not product_id:
        return html.Div([
            html.Div("Select a product to view demand and inventory trends.", className="text-muted"),
        ], className="empty-state")

    demand_df = get_demand_vs_actual(product_id=product_id)
    inv_raw = _get_inventory_raw()
    inv_product = inv_raw[inv_raw["product_id"] == product_id].sort_values("date")

    demand_fig = demand_vs_actual_chart(demand_df, height=280)

    # Inventory trend
    inv_fig = go.Figure()
    if not inv_product.empty:
        inv_fig.add_trace(go.Scatter(
            x=inv_product["date"],
            y=inv_product["closing_stock"],
            name="Closing Stock",
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(31,138,133,0.10)",
            line=dict(color=CHART_COLORS["chart_1"], width=2),
            hovertemplate="<b>Closing Stock:</b> %{y:,.0f} units<extra></extra>",
        ))
        if "reorder_point" in inv_product.columns:
            inv_fig.add_trace(go.Scatter(
                x=inv_product["date"],
                y=inv_product["reorder_point"],
                name="Reorder Point",
                mode="lines",
                line=dict(color=CHART_COLORS["chart_3"], width=1.5, dash="dot"),
                hovertemplate="<b>Reorder Point:</b> %{y:,.0f} units<extra></extra>",
            ))
        if "safety_stock" in inv_product.columns:
            inv_fig.add_trace(go.Scatter(
                x=inv_product["date"],
                y=inv_product["safety_stock"],
                name="Safety Stock",
                mode="lines",
                line=dict(color=CHART_COLORS["chart_4"], width=1.5, dash="dash"),
                hovertemplate="<b>Safety Stock:</b> %{y:,.0f} units<extra></extra>",
            ))

    layout = make_chart_layout(
        title="",
        height=280,
        yaxis_title="Stock Units",
        show_legend=True,
    )
    layout["yaxis"]["showgrid"] = True
    layout["yaxis"]["gridcolor"] = CHART_COLORS["border"]
    layout["xaxis"]["showgrid"] = False
    layout["yaxis"]["automargin"] = True
    inv_fig.update_layout(**layout)

    return html.Div([
        html.Div([
            html.Div([
                html.H3("Demand History & Forecast — " + str(product_id), className="section-title"),
            ], className="card-header-flex"),
            dcc.Graph(figure=demand_fig, config={"displayModeBar": False}),
        ], style={"marginBottom": "20px"}),

        html.Div([
            html.Div([
                html.H3("Inventory Trend & Buffer Thresholds — " + str(product_id), className="section-title"),
            ], className="card-header-flex"),
            dcc.Graph(figure=inv_fig, config={"displayModeBar": False}),
        ]),
    ])
