"""
dashboard/pages/compliance_logistics.py — Dashboard 8/9 (Phase 3)
Compliance and Cold-Chain Gatings.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import dash
from dash import html, dcc, Input, Output, callback
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

dash.register_page(__name__, path="/compliance", name="Compliance & Logistics", order=8)

_BASE    = Path(__file__).parent.parent.parent
_OUTPUTS = _BASE.parent / "data" / "outputs"

def _load_gated_orders() -> pd.DataFrame:
    p = _OUTPUTS / "gated_orders.parquet"
    if p.exists():
        return pd.read_parquet(p, engine="pyarrow")
    return pd.DataFrame()

layout = html.Div([

    html.Div([
        html.H1("Compliance & Logistics Gating", className="page-title"),
        html.P("Phase 3 · Operational gates for Quality/Compliance and Cold-Chain excursions.",
               className="page-subtitle"),
    ], className="page-header"),

    html.Div(id="d8-kpi-row", className="kpi-row"),

    html.Div([
        html.Div([
            html.H2("Gate Funnel", className="section-title"),
            dcc.Graph(id="d8-funnel", config={"displayModeBar": False}),
        ], className="card half-width"),
        
        html.Div([
            html.H2("Blocked Orders Detail", className="section-title"),
            html.Div(id="d8-blocked-table", style={"maxHeight": "400px", "overflowY": "auto"}),
        ], className="card half-width"),
    ], className="two-col-row"),

], className="page-content")

@callback(Output("d8-kpi-row", "children"), Input("d8-kpi-row", "id"))
def kpis(_):
    df = _load_gated_orders()
    if df.empty:
        return html.P("No gating data found. Run run_phase2.py first.", style={"color": "var(--danger)"})

    total = len(df)
    auto_app = int(df["auto_approved"].sum()) if "auto_approved" in df.columns else 0
    comp_block = int(df["compliance_blocked"].sum()) if "compliance_blocked" in df.columns else 0
    cold_block = int(df["coldchain_blocked"].sum()) if "coldchain_blocked" in df.columns else 0
    pct_auto = (auto_app / total * 100) if total > 0 else 0
    
    kpis = [
        ("Total Processed", f"{total:,}", "All active supply chain orders", "var(--text)"),
        ("Auto-Approved", f"{auto_app:,}", f"{pct_auto:.1f}% cleared all gates", "var(--success)"),
        ("Compliance Blocked", f"{comp_block:,}", "Regulatory deviations flagged", "var(--danger)" if comp_block > 0 else "var(--text)"),
        ("Cold-Chain Blocked", f"{cold_block:,}", "Temperature excursions held", "var(--danger)" if cold_block > 0 else "var(--text)"),
    ]

    return [
        html.Div([
            html.Div(label, className="kpi-label"),
            html.Div(value, className="kpi-value", style={"color": color}),
            html.Div(subtext, className="kpi-subtext"),
        ], className="kpi-card")
        for label, value, subtext, color in kpis
    ]

@callback(Output("d8-funnel", "figure"), Input("d8-funnel", "id"))
def funnel_chart(_):
    df = _load_gated_orders()
    if df.empty or "auto_approved" not in df.columns:
        return go.Figure()
        
    total = len(df)
    comp_block = int(df["compliance_blocked"].sum())
    cold_block = int(df["coldchain_blocked"].sum())
    auto_app = int(df["auto_approved"].sum())
    
    fig = go.Figure(go.Funnel(
        y=["Total Orders", "Passed Compliance", "Passed Cold-Chain (Auto-Approved)"],
        x=[total, total - comp_block, auto_app],
        textinfo="value+percent initial",
        marker={"color": ["#2D7D7B", "#7AB5B4", "#3D7A4F"]}
    ))
    
    fig.update_layout(
        paper_bgcolor="#FDFBF7",
        plot_bgcolor="#FDFBF7",
        font={"family": "Inter, sans-serif", "color": "#1C1B1A"},
        margin={"l": 160, "r": 20, "t": 20, "b": 20},
    )
    return fig

@callback(Output("d8-blocked-table", "children"), Input("d8-blocked-table", "id"))
def blocked_table(_):
    df = _load_gated_orders()
    if df.empty or "auto_approved" not in df.columns:
        return html.P("No data.")
        
    blocked = df[~df["auto_approved"]].head(50)
    if blocked.empty:
        return html.P("All orders auto-approved!", style={"color": "var(--success)"})
        
    order_col = next((c for c in ["order_id", "batch_id", "product_id"] if c in df.columns), df.columns[0])
    
    # Combine reasons
    blocked_rows = []
    for _, row in blocked.iterrows():
        reasons = []
        if row.get("compliance_blocked"): reasons.append(f"Compliance: {row.get('compliance_reason', 'Flagged')}")
        if row.get("coldchain_blocked"): reasons.append(f"Cold-Chain: {row.get('coldchain_reason', 'Flagged')}")
        
        blocked_rows.append({
            "Order ID": str(row[order_col]),
            "Blocked Reason": " | ".join(reasons)
        })
        
    blocked_df = pd.DataFrame(blocked_rows)
    
    return html.Table([
        html.Thead(html.Tr([html.Th("Order ID"), html.Th("Blocked Reason")])),
        html.Tbody([
            html.Tr([
                html.Td(row["Order ID"], className="mono"),
                html.Td(row["Blocked Reason"], style={"color": "var(--danger)", "fontWeight": "500"}),
            ])
            for _, row in blocked_df.iterrows()
        ])
    ], className="data-table")
