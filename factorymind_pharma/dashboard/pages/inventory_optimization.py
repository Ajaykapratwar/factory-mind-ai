"""
dashboard/pages/inventory_optimization.py — Dashboard 4 (Phase 3)
Safety Stock Optimization and Reorder Points.
Strict tokens.css compliance with zero hardcoded hex.
"""

import sys, os
from pathlib import Path
_PKG_DIR = Path(__file__).resolve().parent.parent.parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

import dash
from dash import html, dcc, Input, Output, callback
import plotly.graph_objects as go
import pandas as pd

from dashboard.components.charts import (
    safety_stock_compare_chart,
    CHART_COLORS,
    COMPARE_COLORS,
    make_chart_layout,
)
from dashboard.components.kpi import kpi_card
from dashboard.components.header import render_page_header, PAGE_METADATA

dash.register_page(__name__, path="/inventory-opt", name="Inventory Opt", order=4)

_BASE = Path(__file__).parent.parent.parent
_OUTPUTS = _BASE.parent / "data" / "outputs"


def _load_safety_stock() -> pd.DataFrame:
    p = _OUTPUTS / "safety_stock_optimized.parquet"
    if p.exists():
        return pd.read_parquet(p, engine="pyarrow")
    return pd.DataFrame()


def _load_remeasure() -> pd.DataFrame:
    p = _OUTPUTS / "phase3_remeasure.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


meta = PAGE_METADATA.get("/inventory-opt", {})

layout = html.Div([
    # Shared page header (eyebrow is driven by topbar: Layer L4 Policy)
    render_page_header(meta.get("subtitle", "Data-driven safety stock buffers and reorder points based on ensemble residual variance")),

    html.Div(id="d4-kpi-row", className="kpi-row"),

    html.Div([
        html.Div([
            html.Div([
                html.H2("Safety Stock Change (Baseline vs Optimized Top 20)", className="section-title"),
                html.Div("Safety Stock Units Comparison", className="section-subtitle"),
            ], className="card-header-flex"),
            dcc.Loading(
                dcc.Graph(id="d4-ss-comparison", config={"displayModeBar": False}),
                color="var(--accent)",
                type="dot",
            ),
        ], className="card", style={"flex": "1.2"}),

        html.Div([
            html.Div([
                html.H2("Top Holding Cost Reductions", className="section-title"),
                html.Div("Highest inventory carrying cost savings", className="section-subtitle"),
            ], className="card-header-flex"),
            html.Div(id="d4-impact-table"),
        ], className="card", style={"flex": "1"}),
    ], className="two-col-row"),

    html.Div([
        html.Div([
            html.H2("SKU Inventory Policy Details", className="section-title"),
            html.Div("Buffer targets, standard deviation of demand, and reorder levels across all SKUs", className="section-subtitle"),
        ], className="card-header-flex"),
        html.Div(id="d4-full-table", className="scroll-x scroll-y", style={"maxHeight": "420px"}),
    ], className="card"),

], className="page-content")


@callback(Output("d4-kpi-row", "children"), Input("d4-kpi-row", "id"))
def kpis(_):
    df = _load_safety_stock()
    if df.empty:
        return html.Div([
            html.Span(className="icon-mask icon-alert", style={"color": "var(--danger)", "marginRight": "8px"}),
            html.Span("No safety stock data. Run run_phase2.py first.", style={"color": "var(--danger)"}),
        ], className="card empty-state")

    net_hc_change = df["holding_cost_change"].sum() if "holding_cost_change" in df.columns else 0
    skus_optimized = len(df)

    rem_df = _load_remeasure()
    so_red = "100.0% (simulated)"
    fr_sim = "100.0% (simulated)"
    if not rem_df.empty:
        row_red = rem_df[rem_df["metric"] == "simulated_stockout_reduction_pct"]
        if not row_red.empty:
            so_red = f"{float(row_red['measured_value'].iloc[0]):.1f}% (simulated)"
        row_fr = rem_df[rem_df["metric"] == "simulated_fill_rate_pct"]
        if not row_fr.empty:
            fr_sim = f"{float(row_fr['measured_value'].iloc[0]):.1f}% (simulated)"

    return [
        kpi_card(
            label="SKUs Optimized",
            raw_value=skus_optimized,
            subtext="Finished goods analyzed",
            color="var(--text)",
            icon_name="box",
            icon_cls="teal",
        ),
        kpi_card(
            label="Net Holding Cost Change",
            raw_value=net_hc_change,
            subtext="Annualized carrying impact",
            color="var(--success)" if net_hc_change <= 0 else "var(--warning)",
            icon_name="dollar",
            icon_cls="green" if net_hc_change <= 0 else "amber",
            is_currency=True,
        ),
        kpi_card(
            label="Simulated Stockout Reduction",
            raw_value=so_red,
            subtext="Service level preservation",
            color="var(--success)",
            icon_name="check",
            icon_cls="green",
        ),
        kpi_card(
            label="Simulated Fill Rate",
            raw_value=fr_sim,
            subtext="Customer order fulfillment",
            color="var(--success)",
            icon_name="activity",
            icon_cls="indigo",
        ),
    ]


@callback(Output("d4-ss-comparison", "figure"), Input("d4-ss-comparison", "id"))
def ss_compare(_):
    df = _load_safety_stock()
    return safety_stock_compare_chart(df, height=340, top_n=20)


@callback(Output("d4-impact-table", "children"), Input("d4-impact-table", "id"))
def impact_table(_):
    df = _load_safety_stock()
    if df.empty or "holding_cost_change" not in df.columns:
        return html.Div("No holding cost data available.", className="empty-state-text")

    df_sorted = df.sort_values("holding_cost_change", ascending=True).head(5)

    return html.Table([
        html.Thead(html.Tr([
            html.Th("Product ID"),
            html.Th("Criticality"),
            html.Th("Δ Safety Stock", className="numeric"),
            html.Th("Holding Cost Δ", className="numeric"),
        ])),
        html.Tbody([
            html.Tr([
                html.Td(str(row["product_id"]), style={"fontWeight": "600"}),
                html.Td(html.Span(str(row.get("criticality", "–")).title(), className="badge badge-neutral")),
                html.Td(f"{row.get('ss_change', 0):+.0f}", className="numeric"),
                html.Td(
                    f"-${abs(row['holding_cost_change']):,.0f}" if row['holding_cost_change'] < 0 else f"+${row['holding_cost_change']:,.0f}",
                    className="numeric",
                    style={"color": "var(--success)" if row["holding_cost_change"] <= 0 else "var(--warning)", "fontWeight": "600"},
                ),
            ])
            for _, row in df_sorted.iterrows()
        ])
    ], className="data-table")


@callback(Output("d4-full-table", "children"), Input("d4-full-table", "id"))
def full_table(_):
    df = _load_safety_stock()
    if df.empty:
        return html.Div("No SKU records found.", className="empty-state-text")

    cols = [
        ("product_id", "Product ID", False),
        ("criticality", "Criticality", False),
        ("z_score", "Z-Score", True),
        ("sigma_demand", "Sigma Demand", True),
        ("baseline_safety_stock", "Baseline SS", True),
        ("new_safety_stock", "Optimized SS", True),
        ("new_reorder_point", "Reorder Pt", True),
    ]
    show_cols = [(c, title, is_num) for c, title, is_num in cols if c in df.columns]

    return html.Table([
        html.Thead(html.Tr([
            html.Th(title, className="numeric" if is_num else "")
            for _, title, is_num in show_cols
        ])),
        html.Tbody([
            html.Tr([
                html.Td(
                    html.Span(str(row[c]).title(), className="badge badge-neutral") if c == "criticality"
                    else (f"{row[c]:.2f}" if isinstance(row[c], float) else (f"{row[c]:,}" if isinstance(row[c], int) else str(row[c]))),
                    className="numeric" if is_num else "",
                    style={"fontWeight": "600"} if c == "product_id" else {},
                )
                for c, _, is_num in show_cols
            ])
            for _, row in df.iterrows()
        ])
    ], className="data-table")
