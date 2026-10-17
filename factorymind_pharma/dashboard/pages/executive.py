"""
dashboard/pages/executive.py — Executive Summary Dashboard (redesigned v2).

KPI tiles: total revenue, total units sold, product-week WAPE,
           P90 quantile coverage, avg days-of-supply, stockout count.
NOTE: "1 − WAPE" at the product-day grain is NOT reported here.
"""

import sys, os
from pathlib import Path
_PKG_DIR = Path(__file__).resolve().parent.parent.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

import dash
from dash import html, dcc
import pandas as pd
import numpy as np

from dashboard.data_loader import (
    get_kpi_summary,
    get_demand_vs_actual,
    get_per_sku_mape_table,
)
from dashboard.components.charts import demand_vs_actual_chart
from dashboard.components.kpi import kpi_card
from dashboard.components.header import render_page_header, PAGE_METADATA

dash.register_page(__name__, path="/", name="Executive Summary")


def _fmt_num(v, prefix="", suffix="", decimals=0):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if abs(v) >= 1_000_000:
        return f"{prefix}{v/1_000_000:.1f}M{suffix}"
    if abs(v) >= 1_000:
        return f"{prefix}{v/1_000:.1f}K{suffix}"
    return f"{prefix}{v:,.{decimals}f}{suffix}"


def _mape_table(df: pd.DataFrame, title: str, top: bool = True) -> html.Table:
    """Top 5 (lowest MAPE) or Bottom 5 (highest MAPE) SKU table."""
    if df is None or df.empty:
        return html.Table([
            html.Tr([html.Td("No SKU data available", className="text-muted")])
        ], className="data-table")

    subset = df.head(5) if top else df.tail(5).iloc[::-1]
    rows = []
    for i, (_, r) in enumerate(subset.iterrows()):
        mape_val = r.get("mape_lgb")
        mape_str = f"{mape_val:.1f}%" if mape_val is not None else "—"
        badge_cls = "badge-success" if top else "badge-danger"
        status_icon = "▲ Best" if top else "▼ Needs Attn"

        rows.append(html.Tr([
            html.Td(f"#{i+1}", className="text-muted"),
            html.Td(r["product_id"], style={"fontWeight": "600"}),
            html.Td(html.Span(mape_str, className=f"badge {badge_cls}"), className="numeric"),
            html.Td(html.Span(status_icon, style={
                "fontSize": "0.75rem",
                "fontWeight": "600",
                "color": "var(--success)" if top else "var(--danger)",
            })),
        ]))

    return html.Table([
        html.Thead(html.Tr([
            html.Th("#"),
            html.Th("Product ID"),
            html.Th("MAPE (LGB)", className="numeric"),
            html.Th("Status"),
        ])),
        html.Tbody(rows),
    ], className="data-table")


def layout():
    kpis = get_kpi_summary()
    mape_df = get_per_sku_mape_table()
    demand_df = get_demand_vs_actual()

    rev_str = _fmt_num(kpis["total_revenue"], prefix="$")
    units_str = _fmt_num(kpis["total_units_sold"], prefix="")
    wape_val = kpis.get("forecast_wape_pct")
    wape_str = f"{wape_val:.1f}%" if wape_val is not None else "—"
    wape_grain = kpis.get("forecast_wape_grain") or "product-week"
    p90_val = kpis.get("p90_coverage_pct")
    p90_str = f"{p90_val:.1f}%" if p90_val is not None else "—"
    dos_str = f"{kpis['avg_days_of_supply']:.0f} days" if kpis.get("avg_days_of_supply") else "—"
    stockout_count = kpis.get("stockout_count", 0)
    stockout_str = f"{stockout_count:,}"

    fig_demand = demand_vs_actual_chart(demand_df, height=340)
    meta = PAGE_METADATA.get("/", {})

    return html.Div([
        # Page description (H1 is in top-bar)
        render_page_header(meta.get("subtitle", "Manufacturing control tower — all SKUs, all regions, all channels")),

        # KPI row
        html.Div([
            kpi_card(
                label="Total Revenue",
                raw_value=rev_str,
                subtext="All products · 2021–2022",
                primary=True,
                icon_name="dollar",
                icon_cls="teal",
            ),
            kpi_card(
                label="Units Sold",
                raw_value=units_str,
                subtext="Across 100 SKUs",
                icon_name="box",
                icon_cls="indigo",
            ),
            kpi_card(
                label="WAPE (Product-Week)",
                raw_value=wape_str,
                subtext=f"Grain: {wape_grain}",
                icon_name="chart",
                icon_cls="sky",
            ),
            kpi_card(
                label="Avg Days of Supply",
                raw_value=dos_str,
                subtext="Closing stock / avg daily sales",
                icon_name="activity",
                icon_cls="amber",
            ),
            kpi_card(
                label="Stockout Events",
                raw_value=stockout_str,
                subtext="Historical flag count",
                icon_name="alert",
                icon_cls="coral" if stockout_count > 0 else "teal",
            ),
        ], className="kpi-row"),

        # Demand Chart Card
        html.Div([
            html.Div([
                html.Div([
                    html.H2("Demand — Plan vs Actual", className="section-title"),
                    html.Div("Aggregate across all SKUs · Weekly resampled time series with P10–P90 uncertainty band", className="section-subtitle"),
                ]),
            ], className="card-header-flex"),
            dcc.Graph(
                figure=fig_demand,
                config={"displayModeBar": False},
                id="exec-demand-chart",
            ),
        ], className="card"),

        # Top / Bottom 5 SKU tables
        html.Div([
            html.Div([
                html.Div([
                    html.H2("Top 5 SKUs by Accuracy", className="section-title"),
                    html.Span("Lowest MAPE", className="badge badge-success"),
                ], className="card-header-flex"),
                _mape_table(mape_df, "Top 5", top=True),
            ], className="card"),

            html.Div([
                html.Div([
                    html.H2("Bottom 5 SKUs by Accuracy", className="section-title"),
                    html.Span("Needs Attention", className="badge badge-danger"),
                ], className="card-header-flex"),
                _mape_table(mape_df, "Bottom 5", top=False),
            ], className="card"),
        ], className="two-col-row"),

    ], className="page-content")
