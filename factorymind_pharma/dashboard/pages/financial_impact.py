"""
dashboard/pages/financial_impact.py — Dashboard 6 (Phase 4)
Financial Impact, P&L Waterfall, Unit Economics, and Modeled Risk.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import dash
from dash import html, dcc, Input, Output, callback
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from pathlib import Path

from factorymind_pharma.services.twin_svc.financial import FinancialIntelligence, COST_COLS

dash.register_page(__name__, path="/financial-impact", name="Financial Impact", order=6)

_BASE = Path(__file__).parent.parent.parent
_DATA = _BASE.parent / "data"

_fi = FinancialIntelligence(data_dir=_DATA)


def _load_data():
    fact_fin = _fi.build_fact_financial()
    pm = pd.read_csv(_DATA / "01_product_master.csv")
    modeled = _fi.compute_modeled_costs()
    return fact_fin, pm, modeled


layout = html.Div([
    html.Div([
        html.H1("Financial Impact", className="page-title"),
        html.P("Phase 4 · Full P&L decomposition, loss-making SKU diagnostics, modeled risk costs, and Profit-at-Risk.",
               className="page-subtitle"),
    ], className="page-header"),

    html.Div(id="d6-kpi-row", className="kpi-row"),

    html.Div([
        html.Div([
            html.H2("P&L Waterfall Decomposition", className="section-title"),
            dcc.Graph(id="d6-waterfall", config={"displayModeBar": False}),
        ], className="card half-width"),

        html.Div([
            html.H2("Margin by Product Category", className="section-title"),
            dcc.Graph(id="d6-category-margin", config={"displayModeBar": False}),
        ], className="card half-width"),
    ], className="two-col-row"),

    html.Div([
        html.Div([
            html.H2("Cost Breakdown Trend (Monthly)", className="section-title"),
            dcc.Graph(id="d6-cost-trend", config={"displayModeBar": False}),
        ], className="card half-width"),

        html.Div([
            html.H2("Loss-Making SKUs Drill-Down (Worst Profit)", className="section-title"),
            html.Div(id="d6-loss-table", style={"overflowX": "auto"}),
        ], className="card half-width"),
    ], className="two-col-row"),

    html.Div([
        html.H2("Modeled Operational Costs & Assumptions", className="section-title"),
        html.Div(id="d6-modeled-notes", className="card-body"),
    ], className="card"),

], className="page-content")


@callback(Output("d6-kpi-row", "children"), Input("d6-kpi-row", "id"))
def render_kpis(_):
    fact_fin, pm, modeled = _load_data()

    total_rev = fact_fin["revenue"].sum()
    total_prof = fact_fin["profit"].sum()
    loss_share = (fact_fin["profit"] < 0).mean() * 100.0

    cpq = modeled["cost_of_poor_quality"]["value"]
    dt_cost = modeled["downtime_cost"]["value"]

    kpis = [
        ("Total Revenue", f"${total_rev:,.0f}", "Gross sales across portfolio", "var(--text)"),
        ("Net Operating Profit", f"${total_prof:,.0f}", "Operating margin after costs", "var(--success)" if total_prof >= 0 else "var(--danger)"),
        ("Loss-Making Share (C6)", f"{loss_share:.1f}%", "Unprofitable SKU volume", "var(--warning)" if loss_share > 40 else "var(--success)"),
        ("Cost of Poor Quality", f"${cpq:,.0f}", "Scrap & defect loss", "#C45B3C"),
        ("Downtime Cost", f"${dt_cost:,.0f}", "Unplanned outage loss", "#6B5B8A"),
    ]

    return [
        html.Div([
            html.Div(label, className="kpi-label"),
            html.Div(value, className="kpi-value", style={"color": color}),
            html.Div(subtext, className="kpi-subtext"),
        ], className="kpi-card")
        for label, value, subtext, color in kpis
    ]


@callback(Output("d6-waterfall", "figure"), Input("d6-waterfall", "id"))
def render_waterfall(_):
    fact_fin, _, _ = _load_data()

    rev = float(fact_fin["revenue"].sum())
    costs = [float(fact_fin[c].sum()) for c in COST_COLS]
    profit = float(fact_fin["profit"].sum())

    labels = ["Revenue"] + [c.replace("_", " ").title() for c in COST_COLS] + ["Net Profit"]
    measures = ["relative"] + ["relative"] * len(COST_COLS) + ["total"]
    values = [rev] + [-c for c in costs] + [profit]

    fig = go.Figure(go.Waterfall(
        name="P&L",
        orientation="v",
        measure=measures,
        x=labels,
        textposition="outside",
        text=[f"{v:,.0f}" for v in values],
        y=values,
        connector={"line": {"color": "var(--border)"}},
        decreasing={"marker": {"color": "var(--chart-5)"}},
        increasing={"marker": {"color": "var(--chart-1)"}},
        totals={"marker": {"color": "var(--chart-2)"}},
    ))

    fig.update_layout(
        paper_bgcolor="var(--surface)",
        plot_bgcolor="var(--surface)",
        font={"family": "Inter, sans-serif", "color": "var(--text)"},
        margin={"l": 40, "r": 20, "t": 20, "b": 60},
        xaxis_tickangle=-30,
    )
    return fig


@callback(Output("d6-category-margin", "figure"), Input("d6-category-margin", "id"))
def render_category_margin(_):
    fact_fin, pm, _ = _load_data()
    merged = fact_fin.merge(pm[["product_id", "category"]], on="product_id", how="left")

    cat_fin = merged.groupby("category", as_index=False).agg({"revenue": "sum", "profit": "sum"})
    cat_fin["margin_pct"] = (cat_fin["profit"] / np.maximum(cat_fin["revenue"], 1.0)) * 100.0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cat_fin["category"],
        y=cat_fin["margin_pct"],
        text=[f"{m:.1f}%" for m in cat_fin["margin_pct"]],
        textposition="auto",
        marker_color=["var(--chart-1)" if m >= 0 else "var(--danger)" for m in cat_fin["margin_pct"]],
    ))

    fig.update_layout(
        paper_bgcolor="var(--surface)",
        plot_bgcolor="var(--surface)",
        font={"family": "Inter, sans-serif", "color": "var(--text)"},
        yaxis_title="Margin %",
        xaxis_title="Category",
        margin={"l": 40, "r": 20, "t": 20, "b": 40},
    )
    return fig


@callback(Output("d6-cost-trend", "figure"), Input("d6-cost-trend", "id"))
def render_cost_trend(_):
    fact_fin, _, _ = _load_data()
    df = fact_fin.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    monthly = df.groupby("month", as_index=False)[COST_COLS].sum()

    colors = [
        "var(--chart-1)",
        "var(--chart-2)",
        "var(--chart-3)",
        "var(--chart-4)",
        "var(--chart-5)",
        "var(--text-muted)",
    ]

    fig = go.Figure()
    for col, color in zip(COST_COLS, colors):
        fig.add_trace(go.Scatter(
            x=monthly["month"],
            y=monthly[col],
            mode="lines",
            name=col.replace("_", " ").title(),
            line={"color": color, "width": 2},
        ))

    fig.update_layout(
        paper_bgcolor="var(--surface)",
        plot_bgcolor="var(--surface)",
        font={"family": "Inter, sans-serif", "color": "var(--text)"},
        xaxis_title="Month",
        yaxis_title="Cost",
        margin={"l": 40, "r": 20, "t": 20, "b": 40},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


@callback(Output("d6-loss-table", "children"), Input("d6-loss-table", "id"))
def render_loss_table(_):
    fact_fin, pm, _ = _load_data()
    sku_fin = fact_fin.groupby("product_id", as_index=False).agg({"revenue": "sum", "profit": "sum"})
    sku_fin["margin_pct"] = (sku_fin["profit"] / np.maximum(sku_fin["revenue"], 1.0)) * 100.0
    sku_fin = sku_fin.merge(pm[["product_id", "product_name", "category"]], on="product_id", how="left")

    worst = sku_fin.sort_values("profit", ascending=True).head(8)

    return html.Table([
        html.Thead(html.Tr([
            html.Th("Product ID"),
            html.Th("Name"),
            html.Th("Category"),
            html.Th("Revenue"),
            html.Th("Profit"),
            html.Th("Margin %"),
        ])),
        html.Tbody([
            html.Tr([
                html.Td(str(row["product_id"])),
                html.Td(str(row.get("product_name", "–"))),
                html.Td(str(row.get("category", "–"))),
                html.Td(f"{row['revenue']:,.0f}"),
                html.Td(f"{row['profit']:,.0f}", style={"color": "var(--danger)"}),
                html.Td(f"{row['margin_pct']:.1f}%", style={"color": "var(--danger)"}),
            ])
            for _, row in worst.iterrows()
        ]),
    ], className="data-table")


@callback(Output("d6-modeled-notes", "children"), Input("d6-modeled-notes", "id"))
def render_modeled_notes(_):
    _, _, modeled = _load_data()
    cpq = modeled["cost_of_poor_quality"]
    dt = modeled["downtime_cost"]
    supp = modeled["supplier_risk_cost"]

    return html.Div([
        html.P([
            html.Strong("Cost of Poor Quality (modeled): "),
            f"{cpq['value']:,.0f}. Assumes: {cpq['assumptions']}. Derived from {cpq['components']['fail_batches']} FAIL batches scrapped and {cpq['components']['retest_batches']} RETEST batches delayed.",
        ]),
        html.P([
            html.Strong("Downtime Cost (modeled): "),
            f"{dt['value']:,.0f}. Assumes: {dt['assumptions']}. Includes direct maintenance cost ({dt['components']['direct_maintenance_cost']:,.0f}) + lost throughput margin ({dt['components']['lost_margin']:,.0f}) across {dt['components']['downtime_hours']:,.1f} downtime hours.",
        ]),
        html.P([
            html.Strong("Supplier Risk Cost (modeled): "),
            f"{supp['value']:,.0f}. Assumes: {supp['assumptions']}. Accounts for {supp['components']['expected_late_days']:,.1f} expected late-delivery days.",
        ]),
        html.P(
            "Note: All figures above carry the 'modeled' designation per Spec §5.3. Currency figures are presented unlabelled per Spec §3.4.",
            style={"fontSize": "12px", "color": "var(--text-muted)", "marginTop": "8px"},
        ),
    ])
