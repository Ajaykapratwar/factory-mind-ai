"""
dashboard/pages/supplier_qc.py — Dashboard 7 (Phase 2, §3.4)
Supplier Risk & QC Anomaly monitoring.
Reuses dashboard/assets/tokens.css unchanged.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import dash
from dash import html, dcc, Input, Output, callback
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

dash.register_page(__name__, path="/supplier-qc", name="Supplier & QC", order=7)

_BASE    = Path(__file__).parent.parent.parent
_OUTPUTS = _BASE.parent / "data" / "outputs"

def _load_supplier_risk() -> pd.DataFrame:
    p = _OUTPUTS / "supplier_risk_scores.parquet"
    if p.exists():
        df = pd.read_parquet(p, engine="pyarrow")
        if not df.empty and "supplier_id" in df.columns:
            # Aggregate per supplier if multiple records exist per supplier
            if len(df) > df["supplier_id"].nunique():
                supp_name_col = "supplier_name" if "supplier_name" in df.columns else "supplier_id"
                agg_df = df.groupby("supplier_id").agg({
                    supp_name_col: "first",
                    "delivery_unreliability_pct": "mean",
                    "rejection_rate_pct_clean": "mean",
                    "risk_score": "mean",
                }).reset_index()
                # Recompute tier cleanly
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

layout = html.Div([

    html.Div([
        html.H1("Supplier Risk & QC Quality", className="page-title"),
        html.P("Monitor supplier reliability and production quality anomalies · Phase 2",
               className="page-subtitle"),
    ], className="page-header"),

    # KPI row
    html.Div(id="d7-kpi-row", className="kpi-row"),

    # Main dual panels
    html.Div([
        # Supplier Risk
        html.Div([
            html.H2("Supplier Risk Profile", className="section-title"),
            dcc.Graph(id="d7-supplier-chart", config={"displayModeBar": False}),
            html.H3("High Risk Suppliers", style={"marginTop": "24px", "fontSize": "0.95rem"}),
            html.Div(id="d7-supplier-table"),
        ], className="card half-width"),

        # QC Anomalies
        html.Div([
            html.H2("QC Anomaly Rates (Isolation Forest)", className="section-title"),
            dcc.Graph(id="d7-qc-chart", config={"displayModeBar": False}),
            html.H3("Recent Anomalous Batches", style={"marginTop": "24px", "fontSize": "0.95rem"}),
            html.Div(id="d7-qc-table"),
        ], className="card half-width"),
    ], className="two-col-row"),

], className="page-content")


@callback(Output("d7-kpi-row", "children"), Input("d7-kpi-row", "id"))
def kpi_row(_):
    supp_df = _load_supplier_risk()
    qc_df   = _load_qc_scores()

    n_supp = len(supp_df)
    high_risk_supp = len(supp_df[supp_df.get("risk_tier", "") == "High"]) if not supp_df.empty else 0

    n_batches = len(qc_df)
    n_anomalies = int(qc_df["anomaly_flag"].sum()) if not qc_df.empty and "anomaly_flag" in qc_df.columns else 0

    kpis = [
        ("Total Suppliers", f"{n_supp:,}", f"{high_risk_supp} high-risk monitored", "var(--text)"),
        ("High Risk Suppliers", f"{high_risk_supp:,}", "Requires escalation" if high_risk_supp > 0 else "All compliant", "var(--danger)" if high_risk_supp > 0 else "var(--success)"),
        ("Total QC Batches", f"{n_batches:,}", "Production lots tested", "var(--text)"),
        ("Detected Anomalies", f"{n_anomalies:,}", "Flagged by Isolation Forest", "var(--warning)" if n_anomalies > 0 else "var(--success)"),
    ]

    return [
        html.Div([
            html.Div(label, className="kpi-label"),
            html.Div(value, className="kpi-value", style={"color": color}),
            html.Div(subtext, className="kpi-subtext"),
        ], className="kpi-card")
        for label, value, subtext, color in kpis
    ]


@callback(Output("d7-supplier-chart", "figure"), Input("d7-supplier-chart", "id"))
def supplier_scatter(_):
    df = _load_supplier_risk()
    if df.empty:
        return go.Figure()

    supp_id_col = next((c for c in ["supplier_name", "supplier_id", "id"] if c in df.columns), df.columns[0])
    
    if "rejection_rate_pct_clean" not in df.columns or "delivery_unreliability_pct" not in df.columns:
        return go.Figure()

    fig = go.Figure()
    
    for tier, color in [("High", "#8B2E2E"), ("Medium", "#9A6B1A"), ("Low", "#3D7A4F")]:
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
            textfont=dict(family="Inter, sans-serif", size=10, color="#6B6560"),
            marker=dict(
                size=14,
                color=color,
                opacity=0.88,
                line=dict(width=1.5, color="#1C1B1A")
            ),
            hovertemplate=(
                "<b>%{customdata[0]} (%{text})</b><br>"
                "Risk Tier: " + tier + "<br>"
                "Delivery Unreliability: %{x:.1f}%<br>"
                "Rejection Rate: %{y:.2f}%<br>"
                "Risk Score: %{customdata[1]:.1f}/100<extra></extra>"
            ),
            customdata=subset[[supp_id_col, "risk_score"]].values
        ))
    
    fig.update_layout(
        paper_bgcolor="#FDFBF7",
        plot_bgcolor="#FDFBF7",
        font={"family": "Inter, sans-serif", "color": "#1C1B1A"},
        xaxis=dict(
            title="Delivery Unreliability %",
            gridcolor="#DDD9D1",
            zerolinecolor="#DDD9D1",
        ),
        yaxis=dict(
            title="Rejection Rate %",
            gridcolor="#DDD9D1",
            zerolinecolor="#DDD9D1",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(family="Inter, sans-serif", size=11),
        ),
        margin={"l": 50, "r": 20, "t": 30, "b": 45},
    )
    return fig


@callback(Output("d7-supplier-table", "children"), Input("d7-supplier-table", "id"))
def supplier_table(_):
    df = _load_supplier_risk()
    if df.empty:
        return html.P("No data.")

    supp_id_col = next((c for c in ["supplier_id", "supplier_name", "id"] if c in df.columns), df.columns[0])
    df_show = df.sort_values("risk_score", ascending=False).head(5)
    
    return html.Table([
        html.Thead(html.Tr([html.Th("Supplier"), html.Th("Risk Score"), html.Th("Tier")])),
        html.Tbody([
            html.Tr([
                html.Td(str(row[supp_id_col])),
                html.Td(f"{row['risk_score']:.1f}"),
                html.Td(str(row.get("risk_tier", "–")), style={"color": "var(--danger)" if row.get("risk_tier") == "High" else "inherit"}),
            ])
            for _, row in df_show.iterrows()
        ])
    ], className="data-table")


@callback(Output("d7-qc-chart", "figure"), Input("d7-qc-chart", "id"))
def qc_chart(_):
    df = _load_qc_scores()
    if df.empty or "anomaly_score" not in df.columns:
        return go.Figure()

    fig = go.Figure()
    
    # Histogram of anomaly scores
    fig.add_trace(go.Histogram(
        x=df["anomaly_score"],
        nbinsx=30,
        marker_color="#2D7D7B",
        name="Batches",
    ))
    
    # The IsolationForest threshold is 0.0 generally, or where predict==-1
    fig.add_vline(x=0, line_dash="dash", line_color="#8B2E2E",
                  annotation_text="Anomaly Threshold", annotation_position="top left",
                  annotation_font=dict(family="Inter, sans-serif", color="#8B2E2E", size=11))
                  
    fig.update_layout(
        paper_bgcolor="#FDFBF7",
        plot_bgcolor="#FDFBF7",
        font={"family": "Inter, sans-serif", "color": "#1C1B1A"},
        xaxis=dict(
            title="Anomaly Score (More negative = more anomalous)",
            gridcolor="#DDD9D1",
        ),
        yaxis=dict(
            title="Count",
            gridcolor="#DDD9D1",
        ),
        showlegend=False,
        margin={"l": 50, "r": 20, "t": 20, "b": 40},
    )
    return fig


@callback(Output("d7-qc-table", "children"), Input("d7-qc-table", "id"))
def qc_table(_):
    df = _load_qc_scores()
    if df.empty or "anomaly_flag" not in df.columns:
        return html.P("No data.")

    anomalies = df[df["anomaly_flag"] == 1].sort_values("anomaly_score", ascending=True).head(5)
    if anomalies.empty:
        return html.P("No recent anomalies detected.", style={"color": "var(--success)"})
        
    batch_col = next((c for c in ["batch_id", "order_id"] if c in df.columns), df.columns[0])
    
    cols = [batch_col, "assay_pct", "dissolution_pct", "impurity_level", "anomaly_score"]
    show_cols = [c for c in cols if c in anomalies.columns]
    
    return html.Table([
        html.Thead(html.Tr([html.Th(c.replace("_", " ").title()) for c in show_cols])),
        html.Tbody([
            html.Tr([
                html.Td(f"{row[c]:.2f}" if isinstance(row[c], float) else str(row[c]))
                for c in show_cols
            ])
            for _, row in anomalies.iterrows()
        ])
    ], className="data-table")
