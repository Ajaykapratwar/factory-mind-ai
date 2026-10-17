"""
dashboard/pages/supplier_qc.py — Dashboard 7 (Phase 2, §3.4)
Supplier Risk & QC Anomaly monitoring.
Redesigned v2: uses shared theme, no hardcoded hex.
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

dash.register_page(__name__, path="/supplier-qc", name="Supplier & QC", order=7)

_META = PAGE_METADATA["/supplier-qc"]

_BASE    = Path(__file__).parent.parent.parent
_OUTPUTS = _BASE.parent / "data" / "outputs"


def _load_supplier_risk() -> pd.DataFrame:
    p = _OUTPUTS / "supplier_risk_scores.parquet"
    if p.exists():
        df = pd.read_parquet(p, engine="pyarrow")
        if not df.empty and "supplier_id" in df.columns:
            if len(df) > df["supplier_id"].nunique():
                supp_name_col = "supplier_name" if "supplier_name" in df.columns else "supplier_id"
                agg_df = df.groupby("supplier_id").agg({
                    supp_name_col: "first",
                    "delivery_unreliability_pct": "mean",
                    "rejection_rate_pct_clean": "mean",
                    "risk_score": "mean",
                }).reset_index()
                agg_df["risk_tier"] = pd.cut(
                    agg_df["risk_score"],
                    bins=[-1, 33, 66, 101],
                    labels=["Low", "Medium", "High"],
                ).astype(str)
                return agg_df
        return df
    return pd.DataFrame()


def _load_qc_scores() -> pd.DataFrame:
    p = _OUTPUTS / "qc_anomaly_scores.parquet"
    if p.exists():
        df = pd.read_parquet(p, engine="pyarrow")
        if "batch_date" in df.columns:
            df["batch_date"] = pd.to_datetime(df["batch_date"])
        return df
    return pd.DataFrame()


# ── Layout ────────────────────────────────────────────────────────────────

layout = html.Div([

    render_page_header(_META["subtitle"]),

    # KPI row
    html.Div(id="d7-kpi-row", className="kpi-row"),

    # Main dual panels
    html.Div([
        # Supplier Risk
        html.Div([
            html.Div([
                html.Div("Supplier Risk Profile", className="card-title"),
                html.Div("Delivery reliability vs rejection rate · dot = supplier", className="card-meta"),
            ], className="card-header"),
            dcc.Graph(id="d7-supplier-chart", config={"displayModeBar": False}),
            html.Div(style={"marginTop": "20px"}),
            html.Div([
                html.Div("High Risk Suppliers", className="card-title"),
            ], className="card-header"),
            html.Div(id="d7-supplier-table"),
        ], className="card half-width"),

        # QC Anomalies
        html.Div([
            html.Div([
                html.Div("QC Anomaly Rates (Isolation Forest)", className="card-title"),
                html.Div("Negative scores = more anomalous", className="card-meta"),
            ], className="card-header"),
            dcc.Graph(id="d7-qc-chart", config={"displayModeBar": False}),
            html.Div(style={"marginTop": "20px"}),
            html.Div([
                html.Div("Recent Anomalous Batches", className="card-title"),
            ], className="card-header"),
            html.Div(id="d7-qc-table"),
        ], className="card half-width"),
    ], className="two-col-row"),

], className="page-content")


@callback(Output("d7-kpi-row", "children"), Input("d7-kpi-row", "id"))
def kpi_row(_):
    supp_df = _load_supplier_risk()
    qc_df   = _load_qc_scores()

    n_supp = len(supp_df)
    high_risk_supp = (
        len(supp_df[supp_df.get("risk_tier", pd.Series()) == "High"])
        if not supp_df.empty else 0
    )
    n_batches   = len(qc_df)
    n_anomalies = (int(qc_df["anomaly_flag"].sum())
                   if not qc_df.empty and "anomaly_flag" in qc_df.columns else 0)

    return [
        kpi_card("Total Suppliers", f"{n_supp:,}",
                 f"{high_risk_supp} high-risk monitored", icon_name="layers", icon_cls="indigo"),
        kpi_card("High Risk Suppliers", f"{high_risk_supp:,}",
                 "Requires escalation — HIGH RISK" if high_risk_supp > 0 else "All suppliers compliant",
                 color="var(--danger)" if high_risk_supp > 0 else "var(--success)",
                 icon_name="alert", icon_cls="coral" if high_risk_supp > 0 else "green"),
        kpi_card("Total QC Batches", f"{n_batches:,}",
                 "Production lots tested", icon_name="chart", icon_cls="sky"),
        kpi_card("Detected Anomalies", f"{n_anomalies:,}",
                 "Flagged by Isolation Forest — INVESTIGATE" if n_anomalies > 0 else "No anomalies detected",
                 color="var(--warning)" if n_anomalies > 0 else "var(--text)",
                 icon_name="trend", icon_cls="amber" if n_anomalies > 0 else "green"),
    ]


@callback(Output("d7-supplier-chart", "figure"), Input("d7-supplier-chart", "id"))
def supplier_scatter(_):
    df = _load_supplier_risk()
    if df.empty:
        return go.Figure()

    supp_id_col = next(
        (c for c in ["supplier_name", "supplier_id", "id"] if c in df.columns),
        df.columns[0],
    )

    if ("rejection_rate_pct_clean" not in df.columns or
            "delivery_unreliability_pct" not in df.columns):
        return go.Figure()

    fig = go.Figure()
    tier_colours = {
        "High":   CHART_COLORS["chart_3"],
        "Medium": CHART_COLORS["chart_4"],
        "Low":    CHART_COLORS["chart_5"],
    }
    for tier, colour in tier_colours.items():
        subset = df[df["risk_tier"] == tier]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset["delivery_unreliability_pct"],
            y=subset["rejection_rate_pct_clean"],
            mode="markers+text",
            name=f"{tier} Risk",
            text=subset["supplier_id"],
            textposition="top center",
            textfont=dict(family="Inter, sans-serif", size=10,
                          color=CHART_COLORS["muted"]),
            marker=dict(
                size=14,
                color=colour,
                opacity=0.88,
                line=dict(width=1.5, color=CHART_COLORS["text"]),
            ),
            hovertemplate=(
                f"<b>%{{customdata[0]}}</b> ({tier} Risk)<br>"
                "Delivery Unreliability: %{x:.1f}%<br>"
                "Rejection Rate: %{y:.2f}%<br>"
                "Risk Score: %{customdata[1]:.1f}/100<extra></extra>"
            ),
            customdata=subset[[supp_id_col, "risk_score"]].values,
        ))

    layout = make_chart_layout(
        height=300,
        xaxis_title="Delivery Unreliability %",
        yaxis_title="Rejection Rate %",
    )
    fig.update_layout(**layout)
    return fig


@callback(Output("d7-supplier-table", "children"), Input("d7-supplier-table", "id"))
def supplier_table(_):
    df = _load_supplier_risk()
    if df.empty:
        return html.Div([
            html.Div("◈", className="empty-state-icon"),
            html.Div("No supplier data", className="empty-state-title"),
        ], className="empty-state")

    supp_id_col = next(
        (c for c in ["supplier_id", "supplier_name", "id"] if c in df.columns),
        df.columns[0],
    )
    df_show = df.sort_values("risk_score", ascending=False).head(5)

    rows = []
    for _, row in df_show.iterrows():
        tier = str(row.get("risk_tier", "–"))
        tier_badge = (
            html.Span([html.Span("⬥ "), tier], className="tier-chip tier-block")
            if tier == "High" else
            html.Span([html.Span("◆ "), tier], className="tier-chip tier-review")
            if tier == "Medium" else
            html.Span([html.Span("◇ "), tier], className="tier-chip tier-auto")
        )
        rows.append(html.Tr([
            html.Td(str(row[supp_id_col]), style={"fontWeight": "600"}),
            html.Td(f"{row['risk_score']:.1f}", style={"textAlign": "right", "fontWeight": "700"}),
            html.Td(tier_badge),
        ]))

    return html.Table([
        html.Thead(html.Tr([html.Th("Supplier"), html.Th("Risk Score"), html.Th("Tier")])),
        html.Tbody(rows),
    ], className="data-table")


@callback(Output("d7-qc-chart", "figure"), Input("d7-qc-chart", "id"))
def qc_chart(_):
    df = _load_qc_scores()
    if df.empty or "anomaly_score" not in df.columns:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["anomaly_score"],
        nbinsx=30,
        marker_color=CHART_COLORS["chart_1"],
        marker_line_width=0,
        name="Batches",
    ))
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color=CHART_COLORS["chart_3"],
        annotation_text="Anomaly Threshold",
        annotation_position="top left",
        annotation_font=dict(family="Inter, sans-serif",
                             color=CHART_COLORS["chart_3"], size=11),
    )

    layout = make_chart_layout(
        height=280,
        xaxis_title="Anomaly Score (more negative = more anomalous)",
        yaxis_title="Count",
    )
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


@callback(Output("d7-qc-table", "children"), Input("d7-qc-table", "id"))
def qc_table(_):
    df = _load_qc_scores()
    if df.empty or "anomaly_flag" not in df.columns:
        return html.P("No data.", className="text-muted")

    anomalies = (df[df["anomaly_flag"] == 1]
                 .sort_values("anomaly_score", ascending=True)
                 .head(5))
    if anomalies.empty:
        return html.Div([
            html.Span("✓ ", style={"color": "var(--success)", "fontWeight": "700"}),
            html.Span("No recent anomalies detected.", style={"color": "var(--success)"}),
        ], style={"fontSize": "0.82rem", "padding": "8px 0"})

    batch_col = next(
        (c for c in ["batch_id", "order_id"] if c in df.columns),
        df.columns[0],
    )
    cols = [batch_col, "assay_pct", "dissolution_pct", "impurity_level", "anomaly_score"]
    show_cols = [c for c in cols if c in anomalies.columns]

    return html.Table([
        html.Thead(html.Tr([html.Th(c.replace("_", " ").title()) for c in show_cols])),
        html.Tbody([
            html.Tr([
                html.Td(f"{row[c]:.2f}" if isinstance(row[c], float) else str(row[c]),
                        style={"textAlign": "right"} if isinstance(row[c], (int, float)) else {})
                for c in show_cols
            ], style={"background": "var(--danger-pale)"})
            for _, row in anomalies.iterrows()
        ]),
    ], className="data-table")
