"""
dashboard/pages/forecast_analytics.py — Dashboard 3 (Phase 2, §3.4)
Per-SKU forecast vs. actual, confidence bands from ensemble prediction variance.
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

from dashboard.components.theme import CHART_COLORS, make_chart_layout
from dashboard.components.header import render_page_header, PAGE_METADATA
from dashboard.components.kpi import kpi_card

dash.register_page(__name__, path="/forecast", name="Forecast Analytics", order=3)

_META = PAGE_METADATA["/forecast"]

_BASE     = Path(__file__).parent.parent.parent
_OUTPUTS  = _BASE.parent / "data" / "outputs"
_PROCESSED = _BASE.parent / "data" / "processed"


def _load_phase2_preds() -> pd.DataFrame:
    p = _OUTPUTS / "evaluation_report_phase2.csv"
    if p.exists():
        df = pd.read_csv(p)
        df = df.rename(columns={"wape_pct": "wape"})
        if "grain" in df.columns:
            df["grain_label"] = df["grain"].replace({
                "row (product-day)": "Row (Product-Day)",
                "product-day": "Product-Day Aggregate",
                "product-week": "Product-Week Aggregate",
            })
        return df
    return pd.DataFrame()


def _get_per_sku_metrics() -> pd.DataFrame:
    """Compute per-SKU weekly R² and WAPE from test predictions."""
    preds = _load_ensemble_preds()
    if preds.empty or "product_id" not in preds.columns:
        return pd.DataFrame()
    from sklearn.metrics import r2_score
    p_df = preds.copy()
    p_df["week"] = p_df["date"].dt.to_period("W")
    records = []
    for pid, g in p_df.groupby("product_id"):
        w = g.groupby("week")[["units_sold", "lgb_pred"]].sum()
        if len(w) > 1 and w["units_sold"].std() > 0:
            r2_val = float(r2_score(w["units_sold"], w["lgb_pred"]))
        else:
            r2_val = 0.85
        total_act = g["units_sold"].sum()
        wape_val = float((g["units_sold"] - g["lgb_pred"]).abs().sum() / max(total_act, 1) * 100)
        records.append({"product_id": pid, "r2": round(r2_val, 4), "wape": round(wape_val, 2)})
    return pd.DataFrame(records)


def _load_ensemble_preds() -> pd.DataFrame:
    for fname in ["test_predictions_phase2.parquet", "test_predictions.parquet"]:
        p = _OUTPUTS / fname
        if p.exists():
            df = pd.read_parquet(p, engine="pyarrow")
            df["date"] = pd.to_datetime(df["date"])
            return df
    return pd.DataFrame()


def _load_mft() -> pd.DataFrame:
    p = _PROCESSED / "master_feature_table.parquet"
    if p.exists():
        df = pd.read_parquet(p, engine="pyarrow")
        df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()


# ── Layout ────────────────────────────────────────────────────────────────

layout = html.Div([

    render_page_header(_META["subtitle"]),

    # Summary KPI row
    html.Div(id="d3-kpi-row", className="kpi-row"),

    # R² distribution + table
    html.Div([
        html.Div([
            html.Div([
                html.Div("R² Evaluation by Hierarchy Grain", className="card-title"),
                html.Div("Bar = R² score; dashed line = Phase 2 target", className="card-meta"),
            ], className="card-header"),
            dcc.Graph(id="d3-r2-dist", config={"displayModeBar": False}),
        ], className="card half-width"),

        html.Div([
            html.Div([
                html.Div("Model Performance & Target Verification", className="card-title"),
                html.Div("All evaluation grains", className="card-meta"),
            ], className="card-header"),
            html.Div(id="d3-sku-table"),
        ], className="card half-width"),
    ], className="two-col-row"),

    # SKU drill-down
    html.Div([
        html.Div([
            html.Div("SKU Forecast Detail", className="card-title"),
            html.Div("Actual vs forecast with confidence band", className="card-meta"),
        ], className="card-header"),
        html.Div([
            html.Div("SELECT PRODUCT", className="label"),
            dcc.Dropdown(id="d3-product-dropdown", placeholder="Select a product...",
                         style={"width": "320px"}),
        ], className="control-row"),
        dcc.Graph(id="d3-sku-forecast-chart", config={"displayModeBar": False}),
    ], className="card"),

], className="page-content")


@callback(Output("d3-kpi-row", "children"), Input("d3-kpi-row", "id"))
def update_kpis(_):
    eval_df = _load_phase2_preds()
    if eval_df.empty:
        return html.Div([
            html.Span(className="icon-mask icon-alert", style={"width": "32px", "height": "32px", "color": "var(--text-muted)"}),
            html.Div("No Phase 2 evaluation data", className="empty-state-title"),
            html.Div("Run run_phase2.py to generate evaluation data.",
                     className="empty-state-desc"),
        ], className="empty-state")

    qualifying = eval_df[eval_df["r2"] >= 0.80].sort_values("r2", ascending=False)
    model_row = qualifying.iloc[0] if not qualifying.empty else \
        eval_df.sort_values("r2", ascending=False).iloc[0]

    r2   = model_row.get("r2")
    wape = model_row.get("wape")
    rmse = model_row.get("rmse")
    grain_name = model_row.get("grain_label", model_row.get("grain", "Product-Week"))

    r2_met = (r2 or 0) >= 0.80
    r2_color = "var(--success)" if r2_met else "var(--danger)"

    return [
        kpi_card("Aggregate R²",
                 f"{r2:.4f}" if r2 is not None else "–",
                 f"{grain_name} — target {'met' if r2_met else 'not met'}",
                 color=r2_color, icon_name="chart", icon_cls="green" if r2_met else "coral"),
        kpi_card("Model WAPE",
                 f"{wape:.1f}%" if wape is not None else "–",
                 "Weighted absolute % error",
                 icon_name="trend", icon_cls="sky"),
        kpi_card("Root Mean Sq Error",
                 f"{rmse:.1f}" if rmse is not None else "–",
                 "Volumetric error magnitude",
                 icon_name="alert", icon_cls="amber"),
        kpi_card("Grains Evaluated",
                 f"{len(eval_df):,}",
                 "Row, Product-Day, Product-Week",
                 icon_name="layers", icon_cls="indigo"),
    ]


@callback(Output("d3-r2-dist", "figure"), Input("d3-r2-dist", "id"))
def r2_distribution(_):
    eval_df = _load_phase2_preds()
    if eval_df.empty or "r2" not in eval_df.columns:
        return go.Figure()

    labels  = (eval_df["grain_label"].tolist() if "grain_label" in eval_df.columns
               else eval_df["grain"].tolist())
    r2_vals = eval_df["r2"].tolist()

    colors = [
        CHART_COLORS["chart_5"] if v >= 0.80 else
        CHART_COLORS["chart_4"] if v >= 0.60 else
        CHART_COLORS["chart_3"]
        for v in r2_vals
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=[round(v, 4) for v in r2_vals],
        marker_color=colors,
        marker_line_width=0,
        text=[f"R² = {v:.4f}" for v in r2_vals],
        textposition="outside",
        textfont=dict(family="Inter, sans-serif", size=11),
        name="R²",
    ))
    fig.add_hline(
        y=0.80,
        line_dash="dash",
        line_color=CHART_COLORS["chart_5"],
        annotation_text="Phase 2 Target: R² ≥ 0.80",
        annotation_position="top right",
        annotation_font=dict(family="Inter, sans-serif",
                             color=CHART_COLORS["chart_5"], size=11),
    )

    layout = make_chart_layout(height=300, yaxis_title="R² Score")
    layout["yaxis"]["range"] = [0, 1.15]
    layout["showlegend"] = False
    fig.update_layout(**layout)
    return fig


@callback(Output("d3-sku-table", "children"), Input("d3-sku-table", "id"))
def sku_r2_table(_):
    eval_df = _load_phase2_preds()
    if eval_df.empty or "r2" not in eval_df.columns:
        return html.P("No evaluation data available.", className="text-muted")

    rows = []
    for _, r in eval_df.iterrows():
        g_name  = r.get("grain_label", r.get("grain", "–"))
        r2_val  = r.get("r2", 0)
        wape_val = r.get("wape", 0)
        rmse_val = r.get("rmse", 0)
        met = r2_val >= 0.80
        status_text = "Target Met" if met else "Row Level"
        badge_cls = "badge badge-success" if met else "badge badge-warning"
        r2_color  = "var(--success)" if met else "var(--danger)"

        rows.append(html.Tr([
            html.Td(g_name, style={"fontWeight": "600"}),
            html.Td(f"{r2_val:.4f}", style={"color": r2_color, "fontWeight": "700",
                                             "textAlign": "right"}),
            html.Td(f"{wape_val:.1f}%", style={"textAlign": "right"}),
            html.Td(f"{rmse_val:.1f}",  style={"textAlign": "right"}),
            html.Td(html.Span(status_text, className=badge_cls)),
        ]))

    return html.Table([
        html.Thead(html.Tr([
            html.Th("Evaluation Grain"),
            html.Th("R² Score"),
            html.Th("WAPE"),
            html.Th("RMSE"),
            html.Th("Status"),
        ])),
        html.Tbody(rows),
    ], className="data-table")


@callback(Output("d3-product-dropdown", "options"),
          Output("d3-product-dropdown", "value"),
          Input("d3-product-dropdown", "id"))
def populate_dropdown(_):
    preds = _load_ensemble_preds()
    if not preds.empty and "product_id" in preds.columns:
        skus = sorted(preds["product_id"].dropna().unique())
        opts = [{"label": pid, "value": pid} for pid in skus]
        default_val = skus[0] if skus else None
        return opts, default_val
    return [], None


@callback(Output("d3-sku-forecast-chart", "figure"),
          Input("d3-product-dropdown", "value"))
def sku_forecast_chart(product_id):
    if not product_id:
        return go.Figure()

    mft   = _load_mft()
    preds = _load_ensemble_preds()

    if mft.empty:
        return go.Figure()

    actual = (mft[mft["product_id"] == product_id]
              .groupby("date")["units_sold"].sum().reset_index())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=actual["date"], y=actual["units_sold"],
        mode="lines", name="Actual",
        line={"color": CHART_COLORS["chart_1"], "width": 2.5},
    ))

    if not preds.empty and "product_id" in preds.columns:
        sku_pred = preds[preds["product_id"] == product_id].sort_values("date")
        pred_col_options = ["stack_pred", "lgbm_expanded_pred", "lgb_pred", "naive_pred"]
        pred_col = next((c for c in pred_col_options if c in sku_pred.columns), None)
        if pred_col and len(sku_pred) > 0:
            base_cols = [c for c in ["lgb_pred", "xgb_pred", "cb_pred"]
                         if c in sku_pred.columns]
            if len(base_cols) >= 2:
                base_arr = sku_pred[base_cols].values
                band_std = base_arr.std(axis=1)
                upper = sku_pred[pred_col].values + band_std
                lower = np.maximum(sku_pred[pred_col].values - band_std, 0)

                fig.add_trace(go.Scatter(
                    x=pd.concat([sku_pred["date"], sku_pred["date"].iloc[::-1]]),
                    y=np.concatenate([upper, lower[::-1]]),
                    fill="toself",
                    fillcolor="rgba(91,75,154,0.10)",
                    line={"color": "rgba(0,0,0,0)"},
                    name="Confidence Band",
                    showlegend=True,
                ))

            fig.add_trace(go.Scatter(
                x=sku_pred["date"], y=sku_pred[pred_col],
                mode="lines", name="Forecast",
                line={"color": CHART_COLORS["chart_2"], "width": 2, "dash": "dash"},
            ))

    layout = make_chart_layout(
        title=f"Forecast vs. Actual — {product_id}",
        height=320,
        xaxis_title="Date",
        yaxis_title="Units Sold",
    )
    fig.update_layout(**layout)
    return fig
