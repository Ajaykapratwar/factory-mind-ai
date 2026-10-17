"""
dashboard/pages/financial_impact.py — Dashboard 6 (Phase 4)
Financial Impact, P&L Waterfall, Unit Economics, and Modeled Risk.
Redesigned v2: uses shared theme, no hardcoded hex.
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
from dashboard.components.theme import CHART_COLORS, COLOR_SEQ, make_chart_layout
from dashboard.components.header import render_page_header, PAGE_METADATA
from dashboard.components.kpi import kpi_card

dash.register_page(__name__, path="/financial-impact", name="Financial Impact", order=6)

_META = PAGE_METADATA["/financial-impact"]

_BASE = Path(__file__).parent.parent.parent
_DATA = _BASE.parent / "data"

_fi = FinancialIntelligence(data_dir=_DATA)


def _load_data():
    fact_fin = _fi.build_fact_financial()
    pm = pd.read_csv(_DATA / "01_product_master.csv")
    modeled = _fi.compute_modeled_costs()
    return fact_fin, pm, modeled


# ── Layout ────────────────────────────────────────────────────────────────

layout = html.Div([
    render_page_header(_META["subtitle"]),

    html.Div(id="d6-kpi-row", className="kpi-row"),

    html.Div([
        html.Div([
            html.Div([
                html.Div("P&L Waterfall Decomposition", className="card-title"),
                html.Div("Revenue → costs → net profit", className="card-meta"),
            ], className="card-header"),
            dcc.Graph(id="d6-waterfall", config={"displayModeBar": False}),
        ], className="card half-width"),

        html.Div([
            html.Div([
                html.Div("Margin by Product Category", className="card-title"),
                html.Div("Net margin % by category", className="card-meta"),
            ], className="card-header"),
            dcc.Graph(id="d6-category-margin", config={"displayModeBar": False}),
        ], className="card half-width"),
    ], className="two-col-row"),

    html.Div([
        html.Div([
            html.Div([
                html.Div("Cost Breakdown Trend (Monthly)", className="card-title"),
                html.Div("All cost components over time", className="card-meta"),
            ], className="card-header"),
            dcc.Graph(id="d6-cost-trend", config={"displayModeBar": False}),
        ], className="card half-width"),

        html.Div([
            html.Div([
                html.Div("Loss-Making SKUs", className="card-title"),
                html.Span("Worst Profit", className="badge badge-danger"),
            ], className="card-header", style={"alignItems": "center"}),
            html.Div(id="d6-loss-table", className="scroll-x"),
        ], className="card half-width"),
    ], className="two-col-row"),

    html.Div([
        html.Div([
            html.Div("Modeled Operational Costs & Assumptions", className="card-title"),
            html.Div("All figures carry the 'modeled' designation per Spec §5.3", className="card-meta"),
        ], className="card-header"),
        html.Div(id="d6-modeled-notes"),
    ], className="card"),

], className="page-content")


@callback(Output("d6-kpi-row", "children"), Input("d6-kpi-row", "id"))
def render_kpis(_):
    fact_fin, pm, modeled = _load_data()

    total_rev  = fact_fin["revenue"].sum()
    total_prof = fact_fin["profit"].sum()
    loss_share = (fact_fin["profit"] < 0).mean() * 100.0
    cpq        = modeled["cost_of_poor_quality"]["value"]
    dt_cost    = modeled["downtime_cost"]["value"]

    return [
        kpi_card("Total Revenue", total_rev, "Gross sales across portfolio", is_currency=True, primary=True, icon_name="trend", icon_cls="indigo"),
        kpi_card("Net Operating Profit", total_prof, "Operating margin after costs", is_currency=True,
                 color="var(--success)" if total_prof >= 0 else "var(--danger)",
                 icon_name="chart", icon_cls="green" if total_prof >= 0 else "coral"),
        kpi_card("Loss-Making Share", f"{loss_share:.1f}%", "Unprofitable SKU volume",
                 color="var(--warning)" if loss_share > 40 else "var(--text)",
                 icon_name="alert", icon_cls="amber"),
        kpi_card("Cost of Poor Quality", cpq, "Scrap & defect loss (modeled)", is_currency=True,
                 color="var(--danger)", icon_name="alert", icon_cls="coral"),
        kpi_card("Downtime Cost", dt_cost, "Unplanned outage loss (modeled)", is_currency=True,
                 color="var(--warning)", icon_name="trend", icon_cls="indigo"),
    ]


@callback(Output("d6-waterfall", "figure"), Input("d6-waterfall", "id"))
def render_waterfall(_):
    fact_fin, _, _ = _load_data()
    rev    = float(fact_fin["revenue"].sum())
    costs  = [float(fact_fin[c].sum()) for c in COST_COLS]
    profit = float(fact_fin["profit"].sum())

    labels   = ["Revenue"] + [c.replace("_", " ").title() for c in COST_COLS] + ["Net Profit"]
    measures = ["relative"] + ["relative"] * len(COST_COLS) + ["total"]
    values   = [rev] + [-c for c in costs] + [profit]

    fig = go.Figure(go.Waterfall(
        name="P&L",
        orientation="v",
        measure=measures,
        x=labels,
        textposition="outside",
        text=[f"${v:,.0f}" for v in values],
        y=values,
        connector={"line": {"color": CHART_COLORS["border"]}},
        decreasing={"marker": {"color": CHART_COLORS["chart_3"]}},
        increasing={"marker": {"color": CHART_COLORS["chart_5"]}},
        totals={"marker": {"color": CHART_COLORS["chart_1"]}},
    ))

    layout = make_chart_layout(height=340, xaxis_title="", yaxis_title="Value ($)")
    layout["margin"] = dict(l=40, r=20, t=20, b=60)
    layout["xaxis"]["tickangle"] = -30
    fig.update_layout(**layout)
    return fig


@callback(Output("d6-category-margin", "figure"), Input("d6-category-margin", "id"))
def render_category_margin(_):
    fact_fin, pm, _ = _load_data()
    merged = fact_fin.merge(pm[["product_id", "category"]], on="product_id", how="left")
    cat_fin = merged.groupby("category", as_index=False).agg(
        {"revenue": "sum", "profit": "sum"}
    )
    cat_fin["margin_pct"] = (
        cat_fin["profit"] / np.maximum(cat_fin["revenue"], 1.0)
    ) * 100.0

    colors = [
        CHART_COLORS["chart_5"] if m >= 0 else CHART_COLORS["chart_3"]
        for m in cat_fin["margin_pct"]
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cat_fin["category"],
        y=cat_fin["margin_pct"],
        text=[f"{m:.1f}%" for m in cat_fin["margin_pct"]],
        textposition="auto",
        marker_color=colors,
        marker_line_width=0,
    ))

    layout = make_chart_layout(
        height=320, xaxis_title="Category", yaxis_title="Margin %"
    )
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


@callback(Output("d6-cost-trend", "figure"), Input("d6-cost-trend", "id"))
def render_cost_trend(_):
    fact_fin, _, _ = _load_data()
    df = fact_fin.copy()
    df["date"]  = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month", as_index=False)[COST_COLS].sum()

    fig = go.Figure()
    for col, colour in zip(COST_COLS, COLOR_SEQ):
        fig.add_trace(go.Scatter(
            x=monthly["month"],
            y=monthly[col],
            mode="lines",
            name=col.replace("_", " ").title(),
            line={"color": colour, "width": 2},
        ))

    layout = make_chart_layout(
        height=320, xaxis_title="Month", yaxis_title="Cost ($)"
    )
    fig.update_layout(**layout)
    return fig


@callback(Output("d6-loss-table", "children"), Input("d6-loss-table", "id"))
def render_loss_table(_):
    fact_fin, pm, _ = _load_data()
    sku_fin = fact_fin.groupby("product_id", as_index=False).agg(
        {"revenue": "sum", "profit": "sum"}
    )
    sku_fin["margin_pct"] = (
        sku_fin["profit"] / np.maximum(sku_fin["revenue"], 1.0)
    ) * 100.0
    sku_fin = sku_fin.merge(
        pm[["product_id", "product_name", "category"]], on="product_id", how="left"
    )
    worst = sku_fin.sort_values("profit", ascending=True).head(8)

    rows = []
    for _, row in worst.iterrows():
        is_loss = row["profit"] < 0
        rows.append(html.Tr([
            html.Td(str(row["product_id"]), style={"fontWeight": "600"}),
            html.Td(str(row.get("product_name", "–"))),
            html.Td(str(row.get("category", "–"))),
            html.Td(f"${row['revenue']:,.0f}", style={"textAlign": "right"}),
            html.Td([
                f"${row['profit']:,.0f} ",
                html.Span("LOSS", className="badge badge-danger") if is_loss else "",
            ], style={"textAlign": "right",
                      "color": "var(--danger)" if is_loss else "var(--success)"}),
            html.Td(f"{row['margin_pct']:.1f}%",
                    style={"textAlign": "right",
                           "color": "var(--danger)" if is_loss else "var(--success)"}),
        ], style={"background": "var(--danger-pale)"} if is_loss else {}))

    return html.Table([
        html.Thead(html.Tr([
            html.Th("Product ID"), html.Th("Name"), html.Th("Category"),
            html.Th("Revenue"), html.Th("Profit"), html.Th("Margin %"),
        ])),
        html.Tbody(rows),
    ], className="data-table")


@callback(Output("d6-modeled-notes", "children"), Input("d6-modeled-notes", "id"))
def render_modeled_notes(_):
    _, _, modeled = _load_data()
    cpq  = modeled["cost_of_poor_quality"]
    dt   = modeled["downtime_cost"]
    supp = modeled["supplier_risk_cost"]

    def _note(label, value, assumptions, extra=""):
        return html.Div([
            html.Div([
                html.Span(label, style={"fontWeight": "700", "fontSize": "0.8rem"}),
                html.Span(f"  ${value:,.0f}", style={
                    "fontWeight": "600",
                    "color": "var(--chart_3)",
                }),
                html.Span(" [MODELED]", className="badge badge-warning",
                          style={"marginLeft": "6px"}),
            ], style={"marginBottom": "4px"}),
            html.Div(f"Assumptions: {assumptions}",
                     style={"fontSize": "0.75rem", "color": "var(--text-muted)", "lineHeight": "1.5"}),
            html.Div(extra, style={"fontSize": "0.75rem", "color": "var(--text-muted)"}),
        ], style={"marginBottom": "14px", "paddingBottom": "14px",
                  "borderBottom": "1px solid var(--border)"})

    return html.Div([
        _note("Cost of Poor Quality", cpq["value"], cpq["assumptions"],
              f"From {cpq['components']['fail_batches']} FAIL batches + "
              f"{cpq['components']['retest_batches']} RETEST batches."),
        _note("Downtime Cost", dt["value"], dt["assumptions"],
              f"Direct maintenance: ${dt['components']['direct_maintenance_cost']:,.0f} + "
              f"Lost margin: ${dt['components']['lost_margin']:,.0f} "
              f"({dt['components']['downtime_hours']:,.1f} downtime hours)."),
        _note("Supplier Risk Cost", supp["value"], supp["assumptions"],
              f"{supp['components']['expected_late_days']:,.1f} expected late-delivery days."),
        html.Div("Currency figures are unlabelled per Spec §3.4.",
                 style={"fontSize": "0.72rem", "color": "var(--text-xmuted)", "fontStyle": "italic"}),
    ])
