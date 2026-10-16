"""
dashboard/pages/executive.py — Step 1.5
Executive Summary Dashboard.

KPI tiles: total revenue, total units sold, product-week WAPE (correct grain),
           P90 quantile coverage, avg days-of-supply, stockout count.
NOTE: "1 − WAPE" at the product-day grain is NOT reported; it conflates
      zero-demand days and gives a misleadingly high number. Use product-week WAPE.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import dash
from dash import html, dcc, callback, Input, Output
import pandas as pd
import numpy as np

from dashboard.data_loader import (
    get_kpi_summary,
    get_demand_vs_actual,
    get_per_sku_mape_table,
)
from dashboard.components.charts import demand_vs_actual_chart, CHART_COLORS

dash.register_page(__name__, path="/", name="Executive Summary")


# Helper to build KPI tiles

def _fmt_num(v, prefix="¤", suffix="", decimals=0):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if abs(v) >= 1_000_000:
        return f"{prefix}{v/1_000_000:.1f}M{suffix}"
    if abs(v) >= 1_000:
        return f"{prefix}{v/1_000:.1f}K{suffix}"
    return f"{prefix}{v:,.{decimals}f}{suffix}"


def _kpi_card(label, value, subtext="", primary=False):
    classes = "kpi-card primary" if primary else "kpi-card"
    return html.Div([
        html.Div(label, className="kpi-label"),
        html.Div(value, className="kpi-value"),
        html.Div(subtext, className="kpi-subtext") if subtext else None,
    ], className=classes)


def _mape_table(df: pd.DataFrame, title: str, top: bool = True) -> html.Div:
    """Top 5 (lowest MAPE) or Bottom 5 (highest MAPE) SKU table."""
    if df.empty:
        return html.Div("No data", className="card-meta")

    subset = df.head(5) if top else df.tail(5).iloc[::-1]
    rows = []
    for _, r in subset.iterrows():
        mape_val = r.get("mape_lgb")
        mape_str = f"{mape_val:.1f}%" if mape_val is not None else "—"
        badge_cls = "badge-success" if top else "badge-danger"
        rows.append(html.Tr([
            html.Td(r["product_id"], className="mono"),
            html.Td(html.Span(mape_str, className=f"badge {badge_cls}")),
        ]))

    return html.Table([
        html.Thead(html.Tr([
            html.Th("Product ID"),
            html.Th("MAPE (LGB)"),
        ])),
        html.Tbody(rows),
    ], className="data-table")


# Layout (rendered once on page load)

def layout():
    kpis = get_kpi_summary()
    mape_df = get_per_sku_mape_table()
    demand_df = get_demand_vs_actual()   # aggregate

    rev_str     = _fmt_num(kpis["total_revenue"], prefix="$")
    units_str   = _fmt_num(kpis["total_units_sold"], prefix="")
    wape_val    = kpis.get("forecast_wape_pct")
    wape_str    = f"{wape_val:.1f}%" if wape_val is not None else "—"
    wape_grain  = kpis.get("forecast_wape_grain") or "product-week"
    p90_val     = kpis.get("p90_coverage_pct")
    p90_str     = f"{p90_val:.1f}%" if p90_val is not None else "—"
    dos_str     = f"{kpis['avg_days_of_supply']:.0f} days" if kpis["avg_days_of_supply"] else "—"
    stockout_str = f"{kpis['stockout_count']:,}"

    fig_demand = demand_vs_actual_chart(demand_df, height=340)

    return html.Div([
        html.Div([
            html.Div([
                html.H1("Executive Summary"),
                html.P("Manufacturing control tower — all SKUs, all regions, all channels",
                       className="page-subtitle"),
            ], className="page-header"),

            # KPI row
            html.Div([
                _kpi_card("Total Revenue", rev_str,
                          subtext="All products · 2021–2022", primary=True),
                _kpi_card("Units Sold", units_str, "across 100 SKUs"),
                _kpi_card("WAPE (Product-Week)", wape_str,
                          f"Grain: {wape_grain} | SKU-level demand is highly intermittent (62% zero days)"),
                _kpi_card("P90 Coverage", p90_str,
                          "Empirical quantile coverage on full calendar grid (nominal 90%)"),
                _kpi_card("Avg Days of Supply", dos_str, "closing stock / avg daily sales"),
                _kpi_card("Stockout Events", stockout_str, "historical flag count"),
            ], className="kpi-grid"),

            # Plan-vs-Actual chart
            html.Div([
                html.Div([
                    html.Div("Demand — Plan vs Actual", className="card-title"),
                    html.Div("Aggregate across all SKUs", className="card-meta"),
                ], className="card-header"),
                dcc.Graph(figure=fig_demand, config={"displayModeBar": False},
                          id="exec-demand-chart"),
            ], className="card", style={"marginBottom": "20px"}),

            # Top/Bottom 5 SKU tables
            html.Div([
                html.Div([
                    html.Div([
                        html.Div("Top 5 SKUs by Accuracy", className="card-title"),
                        html.Div("Lowest MAPE — best performing", className="card-meta"),
                    ], className="card-header"),
                    _mape_table(mape_df, "Top 5", top=True),
                ], className="card"),

                html.Div([
                    html.Div([
                        html.Div("Bottom 5 SKUs by Accuracy", className="card-title"),
                        html.Div("Highest MAPE — needs attention", className="card-meta"),
                    ], className="card-header"),
                    _mape_table(mape_df, "Bottom 5", top=False),
                ], className="card"),
            ], className="section-row"),

        ], className="content"),
    ], className="page-wrapper")
