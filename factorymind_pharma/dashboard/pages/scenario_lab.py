"""
dashboard/pages/scenario_lab.py — Dashboard 10 (Phase 4)
Digital Twin Scenario Lab: Monte Carlo shock builder, fan charts,
ranked mitigations, and assumptions drawer per Spec §5.6.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import dash
from dash import html, dcc, Input, Output, State, callback
import plotly.graph_objects as go
import yaml
from pathlib import Path

from factorymind_pharma.services.twin_svc import DigitalTwinSimulator
from factorymind_pharma.contracts.scenario import (
    Scenario,
    MachineOutage,
    SupplierDelay,
    DemandShock,
    ColdChainEvent,
    QualityShift,
)

dash.register_page(__name__, path="/scenario-lab", name="Scenario Lab", order=10)

_BASE = Path(__file__).parent.parent.parent
_CONFIG_PATH = _BASE / "configs" / "twin.yaml"
_sim = DigitalTwinSimulator()


def _load_assumptions():
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


layout = html.Div([
    html.Div([
        html.H1("Scenario Lab (Digital Twin)", className="page-title"),
        html.P("Phase 4 · Stochastic Monte Carlo 'what-if' simulation with Common Random Numbers and ranked mitigations.",
               className="page-subtitle"),
    ], className="page-header"),

    html.Div([
        # Left: Scenario Builder Form
        html.Div([
            html.H2("Scenario Builder", className="section-title"),
            html.Label("Horizon (days)", className="control-label"),
            dcc.Dropdown(
                id="d10-horizon",
                options=[{"label": "30 Days", "value": 30}, {"label": "60 Days", "value": 60}, {"label": "90 Days", "value": 90}],
                value=30,
                clearable=False,
                className="dropdown",
            ),

            html.Label("Primary Shock Type", className="control-label", style={"marginTop": "12px"}),
            dcc.Dropdown(
                id="d10-shock-type",
                options=[
                    {"label": "Machine Outage (48h)", "value": "machine_outage"},
                    {"label": "Supplier Lead Time Delay (+7d)", "value": "supplier_delay"},
                    {"label": "Demand Shock (+25% surge)", "value": "demand_shock"},
                    {"label": "Cold-Chain Temperature Excursion", "value": "cold_chain_event"},
                    {"label": "QC Retest Rate Spike (2x)", "value": "quality_shift"},
                ],
                value="supplier_delay",
                clearable=False,
                className="dropdown",
            ),

            html.Label("Target Identifier", className="control-label", style={"marginTop": "12px"}),
            dcc.Input(
                id="d10-target-id",
                type="text",
                value="S004",
                placeholder="e.g. S004, M001, P0001",
                className="text-input",
                style={"width": "100%", "padding": "8px", "borderRadius": "4px", "border": "1px solid var(--border)"},
            ),

            html.Label("Number of Monte Carlo Paths", className="control-label", style={"marginTop": "12px"}),
            dcc.Slider(
                id="d10-paths",
                min=100,
                max=1000,
                step=100,
                value=500,
                marks={100: "100", 500: "500", 1000: "1000"},
            ),

            html.Button(
                "Run Scenario Simulation",
                id="d10-run-btn",
                n_clicks=0,
                className="btn btn-primary",
                style={"marginTop": "20px", "width": "100%", "padding": "10px", "cursor": "pointer"},
            ),

            html.Div(id="d10-run-status", style={"marginTop": "10px", "fontSize": "12px", "color": "var(--text-muted)"}),
        ], className="card one-third-width"),

        # Center: Simulation Fan Charts & KPIs
        html.Div([
            html.H2("Simulation Results (Baseline vs Shocked)", className="section-title"),
            html.Div(id="d10-metrics-strip", className="kpi-row", style={"marginBottom": "16px"}),
            dcc.Graph(id="d10-fan-chart", config={"displayModeBar": False}),
        ], className="card two-thirds-width"),
    ], className="two-col-row"),

    # Row 2: Ranked Mitigations
    html.Div([
        html.H2("Ranked Mitigations (Evaluated via Re-Simulation)", className="section-title"),
        html.Div(id="d10-mitigation-table", style={"overflowX": "auto"}),
    ], className="card"),

    # Row 3: Assumptions Drawer (tagged assumed | fitted per §4.2)
    html.Div([
        html.H2("Model Assumptions & Topology (Audited)", className="section-title"),
        html.Div(id="d10-assumptions-drawer", className="card-body"),
    ], className="card"),

], className="page-content")


@callback(
    [
        Output("d10-metrics-strip", "children"),
        Output("d10-fan-chart", "figure"),
        Output("d10-mitigation-table", "children"),
        Output("d10-run-status", "children"),
    ],
    [Input("d10-run-btn", "n_clicks")],
    [
        State("d10-horizon", "value"),
        State("d10-shock-type", "value"),
        State("d10-target-id", "value"),
        State("d10-paths", "value"),
    ],
)
def run_simulation(n_clicks, horizon, shock_type, target_id, n_paths):
    # Default scenario
    shocks = []
    target_id = target_id.strip() if target_id else "S004"

    if shock_type == "machine_outage":
        shocks.append(MachineOutage(machine_id=target_id, start_day=5, duration_h=48.0))
    elif shock_type == "supplier_delay":
        shocks.append(SupplierDelay(supplier_id=target_id, extra_days=7, start_day=3))
    elif shock_type == "demand_shock":
        shocks.append(DemandShock(product_id=target_id if target_id.startswith("P") else "P0001", pct_change=0.25, start_day=2, duration_days=14))
    elif shock_type == "cold_chain_event":
        shocks.append(ColdChainEvent(start_day=4, duration_days=10))
    elif shock_type == "quality_shift":
        shocks.append(QualityShift(product_id=target_id if target_id.startswith("P") else "P0001", retest_rate_multiplier=2.0))

    sc = Scenario(
        id="SC_ACTIVE",
        name=f"Interactive Shock: {shock_type}",
        horizon_days=horizon,
        n_paths=n_paths,
        seed=42,
        shocks=shocks,
    )

    res = _sim.run_scenario(sc)

    # 1. Metrics Strip
    metrics = [
        ("Profit Delta vs Baseline", f"${res.profit_delta:+,.0f}", "Scenario financial divergence", "var(--danger)" if res.profit_delta < 0 else "var(--success)"),
        ("Profit-at-Risk (PaR P5)", f"${res.profit_at_risk_p5:,.0f}", "95% confidence downside", "var(--warning)"),
        ("Fill Rate (Baseline → Shock)", f"{res.baseline_metrics.fill_rate_p50*100:.1f}% → {res.shocked_metrics.fill_rate_p50*100:.1f}%", "Order fulfillment impact", "var(--text)"),
        ("Stockout Risk Change", f"{res.p_stockout_delta*100:+.2f}%", "Probability shift", "var(--danger)" if res.p_stockout_delta > 0 else "var(--success)"),
    ]
    metrics_strip = [
        html.Div([
            html.Div(lbl, className="kpi-label"),
            html.Div(val, className="kpi-value", style={"color": col}),
            html.Div(sub, className="kpi-subtext"),
        ], className="kpi-card")
        for lbl, val, col, sub in metrics
    ]

    # 2. Fan Chart: Fill Rate & P(Stockout)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Baseline",
        x=["Fill Rate (P5)", "Fill Rate (Median)", "Fill Rate (P95)", "Stockout Risk"],
        y=[
            res.baseline_metrics.fill_rate_p5 * 100,
            res.baseline_metrics.fill_rate_p50 * 100,
            res.baseline_metrics.fill_rate_p95 * 100,
            res.baseline_metrics.p_stockout_mean * 100,
        ],
        marker_color="var(--chart-1)",
    ))
    fig.add_trace(go.Bar(
        name="Shocked Scenario",
        x=["Fill Rate (P5)", "Fill Rate (Median)", "Fill Rate (P95)", "Stockout Risk"],
        y=[
            res.shocked_metrics.fill_rate_p5 * 100,
            res.shocked_metrics.fill_rate_p50 * 100,
            res.shocked_metrics.fill_rate_p95 * 100,
            res.shocked_metrics.p_stockout_mean * 100,
        ],
        marker_color="var(--chart-5)",
    ))

    fig.update_layout(
        barmode="group",
        paper_bgcolor="var(--surface)",
        plot_bgcolor="var(--surface)",
        font={"family": "Inter, sans-serif", "color": "var(--text)"},
        yaxis_title="Percentage (%)",
        margin={"l": 40, "r": 20, "t": 20, "b": 40},
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # 3. Ranked Mitigations Table
    mit_rows = []
    for rank, m in enumerate(res.ranked_mitigations, start=1):
        mit_rows.append(html.Tr([
            html.Td(str(rank)),
            html.Td(m.name),
            html.Td(m.description),
            html.Td(f"{m.assumed_cost:,.0f} (assumed)"),
            html.Td(f"{m.expected_profit_saved:+,.0f}", style={"color": "var(--success)" if m.expected_profit_saved > 0 else "var(--text)"}),
            html.Td(f"{m.p_stockout_delta*100:+.2f}%", style={"color": "var(--success)" if m.p_stockout_delta < 0 else "var(--text)"}),
        ]))

    mit_table = html.Table([
        html.Thead(html.Tr([
            html.Th("Rank"),
            html.Th("Mitigation Action"),
            html.Th("Operational Mechanism"),
            html.Th("Cost"),
            html.Th("Expected Profit Saved"),
            html.Th("Stockout Risk Δ"),
        ])),
        html.Tbody(mit_rows),
    ], className="data-table")

    status_msg = f"Completed {n_paths} paths over {horizon} days using Common Random Numbers (seed=42)."
    return metrics_strip, fig, mit_table, status_msg


@callback(Output("d10-assumptions-drawer", "children"), Input("d10-assumptions-drawer", "id"))
def render_assumptions(_):
    cfg = _load_assumptions()
    params = cfg.get("parameters", {})
    top = cfg.get("topology", {})

    items = [
        ("Machine Topology", top.get("machine_line_mapping_rule"), top.get("machine_line_mapping_source", "assumed")),
        ("Warehouse Network", top.get("warehouses", {}).get("network_structure"), top.get("warehouses", {}).get("network_source", "assumed")),
        ("QC Retest Delay", f"{params.get('retest_delay_days', {}).get('value')} days", params.get("retest_delay_days", {}).get("source", "assumed")),
        ("Holding Cost Proxy", f"{params.get('holding_cost_per_unit_day', {}).get('value')} / unit-day", params.get("holding_cost_per_unit_day", {}).get("source", "assumed")),
        ("Transfer Lead Time", f"{params.get('transfer_lead_time_days', {}).get('value')} days", params.get("transfer_lead_time_days", {}).get("source", "assumed")),
        ("Transfer Cost", f"{params.get('transfer_cost_per_unit', {}).get('value')} / unit", params.get("transfer_cost_per_unit", {}).get("source", "assumed")),
        ("Expedite Cost", f"{params.get('raw_material_expedite_cost', {}).get('value')} / shipment", params.get("raw_material_expedite_cost", {}).get("source", "assumed")),
        ("Maintenance Cost", f"{params.get('maintenance_pull_forward_cost', {}).get('value')} / event", params.get("maintenance_pull_forward_cost", {}).get("source", "assumed")),
        ("Cold-Chain Excursion Rate", f"{params.get('cold_chain_excursion_rate', {}).get('value')*100:.2f}%", params.get("cold_chain_excursion_rate", {}).get("source", "fitted")),
    ]

    return html.Table([
        html.Thead(html.Tr([html.Th("Parameter"), html.Th("Specification Value"), html.Th("Source Provenance Tag")])),
        html.Tbody([
            html.Tr([
                html.Td(p_name),
                html.Td(p_val),
                html.Td(
                    html.Span(
                        f"[{p_src.upper()}]",
                        style={
                            "padding": "2px 8px",
                            "borderRadius": "4px",
                            "fontSize": "11px",
                            "fontWeight": "600",
                            "backgroundColor": "var(--surface-sunken)" if p_src == "fitted" else "var(--border)",
                            "color": "var(--primary)" if p_src == "fitted" else "var(--text-muted)",
                        },
                    )
                ),
            ])
            for p_name, p_val, p_src in items
        ]),
    ], className="data-table")
