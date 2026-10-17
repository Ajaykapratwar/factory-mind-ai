"""
dashboard/pages/compliance_logistics.py — Dashboard 8/9 (Phase 3)
Compliance and Cold-Chain Gatings.
Strict tokens.css compliance with zero hardcoded hex.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import dash
from dash import html, dcc, Input, Output, callback
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

from dashboard.components.theme import CHART_COLORS, make_chart_layout
from dashboard.components.header import render_page_header, PAGE_METADATA
from dashboard.components.kpi import kpi_card

dash.register_page(__name__, path="/compliance", name="Compliance & Logistics", order=8)

_META = PAGE_METADATA["/compliance"]

_BASE    = Path(__file__).parent.parent.parent
_OUTPUTS = _BASE.parent / "data" / "outputs"


def _load_gated_orders() -> pd.DataFrame:
    p = _OUTPUTS / "gated_orders.parquet"
    if p.exists():
        return pd.read_parquet(p, engine="pyarrow")
    return pd.DataFrame()


layout = html.Div([
    render_page_header(_META["subtitle"]),

    html.Div(id="d8-kpi-row", className="kpi-row"),

    html.Div([
        html.Div([
            html.H2("Gate Funnel & Clearance", className="section-title"),
            dcc.Loading(
                dcc.Graph(id="d8-funnel", config={"displayModeBar": False}),
                color="var(--accent)",
                type="dot",
            ),
        ], className="card half-width"),

        html.Div([
            html.H2("Blocked Orders Detail (Top 50)", className="section-title"),
            html.Div(id="d8-blocked-table", style={"maxHeight": "400px", "overflowY": "auto"}),
        ], className="card half-width"),
    ], className="two-col-row"),

], className="page-content")


@callback(Output("d8-kpi-row", "children"), Input("d8-kpi-row", "id"))
def kpis(_):
    df = _load_gated_orders()
    if df.empty:
        return html.Div([
            html.Span(className="icon-mask icon-alert", style={"color": "var(--danger)", "marginRight": "8px"}),
            html.Span("No gating data found. Run run_phase2.py first.", style={"color": "var(--danger)"}),
        ], className="card empty-state")

    total = len(df)
    auto_app = int(df["auto_approved"].sum()) if "auto_approved" in df.columns else 0
    comp_block = int(df["compliance_blocked"].sum()) if "compliance_blocked" in df.columns else 0
    cold_block = int(df["coldchain_blocked"].sum()) if "coldchain_blocked" in df.columns else 0
    pct_auto = (auto_app / total * 100) if total > 0 else 0

    return [
        kpi_card("Total Processed", f"{total:,}", "All active supply chain orders", icon_name="layers", icon_cls="teal"),
        kpi_card("Auto-Approved", f"{auto_app:,}", f"{pct_auto:.1f}% cleared all gates", color="var(--success)", icon_name="check", icon_cls="green"),
        kpi_card("Compliance Blocked", f"{comp_block:,}", "Regulatory deviations flagged",
                 color="var(--danger)" if comp_block > 0 else "var(--text)",
                 icon_name="alert", icon_cls="coral" if comp_block > 0 else "teal"),
        kpi_card("Cold-Chain Blocked", f"{cold_block:,}", "Temperature excursions held",
                 color="var(--danger)" if cold_block > 0 else "var(--text)",
                 icon_name="trend", icon_cls="coral" if cold_block > 0 else "teal"),
    ]


@callback(Output("d8-funnel", "figure"), Input("d8-funnel", "id"))
def funnel_chart(_):
    df = _load_gated_orders()
    if df.empty or "auto_approved" not in df.columns:
        return go.Figure()

    total = len(df)
    comp_block = int(df["compliance_blocked"].sum())
    auto_app = int(df["auto_approved"].sum())

    fig = go.Figure(go.Funnel(
        y=["Total Orders", "Passed Compliance", "Auto-Approved (All Gates)"],
        x=[total, total - comp_block, auto_app],
        textinfo="value+percent initial",
        marker={"color": [CHART_COLORS["chart_1"], CHART_COLORS["chart_6"], CHART_COLORS["chart_5"]]},
        connector={"line": {"color": CHART_COLORS["border"], "width": 1}},
    ))

    layout = make_chart_layout(
        title="Order Clearance Gating Funnel",
        height=320,
        margin={"l": 150, "r": 20, "t": 40, "b": 20},
    )
    fig.update_layout(**layout)
    return fig


@callback(Output("d8-blocked-table", "children"), Input("d8-blocked-table", "id"))
def blocked_table(_):
    df = _load_gated_orders()
    if df.empty or "auto_approved" not in df.columns:
        return html.Div("No gating records available.", className="empty-state-text")

    blocked = df[~df["auto_approved"]].head(50)
    if blocked.empty:
        return html.Div([
            html.Span(className="icon-mask icon-check", style={"color": "var(--success)", "marginRight": "6px"}),
            html.Span("All orders cleared gates with zero blocks.", style={"color": "var(--success)", "fontWeight": "600"}),
        ], style={"padding": "16px"})

    order_col = next((c for c in ["order_id", "batch_id", "product_id"] if c in df.columns), df.columns[0])

    blocked_rows = []
    for _, row in blocked.iterrows():
        reasons = []
        if row.get("compliance_blocked"):
            reasons.append(html.Span(
                f"Compliance: {row.get('compliance_reason', 'Flagged')}",
                className="tier-chip block-tier",
                style={"marginRight": "6px", "marginBottom": "4px", "display": "inline-block"},
            ))
        if row.get("coldchain_blocked"):
            reasons.append(html.Span(
                f"Cold-Chain: {row.get('coldchain_reason', 'Flagged')}",
                className="tier-chip block-tier",
                style={"display": "inline-block"},
            ))

        blocked_rows.append(html.Tr([
            html.Td(str(row[order_col]), style={"fontWeight": "600"}),
            html.Td(reasons),
        ]))

    return html.Table([
        html.Thead(html.Tr([html.Th("Order / Batch ID"), html.Th("Gating Flags")])),
        html.Tbody(blocked_rows),
    ], className="data-table")
