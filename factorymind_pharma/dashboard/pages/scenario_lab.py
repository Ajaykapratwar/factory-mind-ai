"""
dashboard/pages/scenario_lab.py — Dashboard 10 (Phase 4)
Digital Twin Scenario Lab: Monte Carlo shock builder, fan charts,
ranked mitigations, and assumptions drawer per Spec §5.6.
Strict tokens.css compliance with zero hardcoded hex.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import dash
from dash import html, dcc, Input, Output, State, callback
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
from dashboard.components.header import render_page_header, PAGE_METADATA
from dashboard.components.kpi import kpi_card

dash.register_page(__name__, path="/scenario-lab", name="Scenario Lab", order=10)

_META = PAGE_METADATA["/scenario-lab"]

_BASE = Path(__file__).parent.parent.parent
_CONFIG_PATH = _BASE / "configs" / "twin.yaml"
_sim = DigitalTwinSimulator()


def _load_assumptions():
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


layout = html.Div([
    render_page_header(_META["subtitle"]),

    # Clear 3-Panel Layout (§5.6)
    # Panel 1: Scenario Builder | Panel 2: Simulation Results
    html.Div([
        # Panel 1: Scenario Builder Form
        html.Div([
            html.Div([
                html.Span(className="icon-mask icon-sliders", style={"color": "var(--accent)", "marginRight": "8px"}),
                html.H2("1. Scenario Builder", className="section-title", style={"margin": "0"}),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "16px"}),

            html.Label("Horizon (days)", className="control-label"),
            dcc.Dropdown(
                id="d10-horizon",
                options=[{"label": "30 Days", "value": 30}, {"label": "60 Days", "value": 60}, {"label": "90 Days", "value": 90}],
                value=30,
                clearable=False,
                className="dropdown",
            ),

            html.Label("Primary Shock Type", className="control-label", style={"marginTop": "14px"}),
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

            html.Label("Target Identifier", className="control-label", style={"marginTop": "14px"}),
            dcc.Input(
                id="d10-target-id",
                type="text",
                value="S004",
                placeholder="e.g. S004, M001, P0001",
                className="text-input",
            ),

            html.Label("Number of Monte Carlo Paths", className="control-label", style={"marginTop": "14px"}),
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
                style={"marginTop": "22px", "width": "100%"},
            ),

            html.Div(id="d10-run-status", style={"marginTop": "12px", "fontSize": "0.78rem", "color": "var(--text-muted)"}),
        ], className="card one-third-width"),

        # Panel 2: Simulation Results
        html.Div([
            html.Div([
                html.Span(className="icon-mask icon-chart", style={"color": "var(--accent)", "marginRight": "8px"}),
                html.H2("2. Simulation Results (Baseline vs Shocked)", className="section-title", style={"margin": "0"}),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "16px"}),

            dcc.Loading(
                html.Div(id="d10-metrics-strip", className="kpi-row"),
                color="var(--accent)",
                type="dot",
            ),
        ], className="card two-thirds-width"),
    ], className="two-col-row"),

    # Panel 3: Ranked Mitigations
    html.Div([
        html.Div([
            html.Span(className="icon-mask icon-shield", style={"color": "var(--success)", "marginRight": "8px"}),
            html.H2("3. Ranked Mitigations (Evaluated via Re-Simulation)", className="section-title", style={"margin": "0"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "16px"}),

        dcc.Loading(
            html.Div(id="d10-mitigation-table", style={"overflowX": "auto"}),
            color="var(--accent)",
            type="dot",
        ),
    ], className="card"),

    # Row 4: Assumptions Drawer (tagged assumed | fitted per §4.2)
    html.Div([
        html.H2("Model Assumptions & Topology (Audited)", className="section-title"),
        html.Div(id="d10-assumptions-drawer", className="card-body"),
    ], className="card"),

], className="page-content")


@callback(
    [
        Output("d10-metrics-strip", "children"),
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
    metrics_strip = [
        kpi_card("Profit Delta vs Baseline", f"${res.profit_delta:+,.0f}", "Scenario financial divergence",
                 color="var(--danger)" if res.profit_delta < 0 else "var(--success)",
                 icon_name="chart", icon_cls="coral" if res.profit_delta < 0 else "green"),
        kpi_card("Profit-at-Risk (PaR P5)", f"${res.profit_at_risk_p5:,.0f}", "95% confidence downside",
                 color="var(--warning)", icon_name="alert", icon_cls="amber"),
        kpi_card("Stockout Risk Change", f"{res.p_stockout_delta*100:+.2f}%", "Probability shift",
                 color="var(--danger)" if res.p_stockout_delta > 0 else "var(--success)",
                 icon_name="trend", icon_cls="coral" if res.p_stockout_delta > 0 else "green"),
    ]

    mit_rows = []
    for rank, m in enumerate(res.ranked_mitigations, start=1):
        mit_rows.append(html.Tr([
            html.Td(str(rank), style={"fontWeight": "700"}),
            html.Td(m.name, style={"fontWeight": "600"}),
            html.Td(m.description),
            html.Td(f"${m.assumed_cost:,.0f} (assumed)", style={"textAlign": "right"}),
            html.Td(
                f"${m.expected_profit_saved:+,.0f}",
                style={"color": "var(--success)" if m.expected_profit_saved > 0 else "var(--text)", "fontWeight": "600", "textAlign": "right"},
            ),
            html.Td(
                f"{m.p_stockout_delta*100:+.2f}%",
                style={"color": "var(--success)" if m.p_stockout_delta < 0 else "var(--text)", "fontWeight": "600", "textAlign": "right"},
            ),
        ]))

    mit_table = html.Table([
        html.Thead(html.Tr([
            html.Th("Rank"),
            html.Th("Mitigation Action"),
            html.Th("Operational Mechanism"),
            html.Th("Assumed Cost"),
            html.Th("Expected Profit Saved"),
            html.Th("Stockout Risk Δ"),
        ])),
        html.Tbody(mit_rows),
    ], className="data-table")

    status_msg = f"Completed {n_paths} paths over {horizon} days using Common Random Numbers (seed=42)."
    return metrics_strip, mit_table, status_msg


@callback(Output("d10-assumptions-drawer", "children"), Input("d10-assumptions-drawer", "id"))
def render_assumptions(_):
    cfg = _load_assumptions()
    params = cfg.get("parameters", {})
    top = cfg.get("topology", {})

    items = [
        ("Machine Topology", top.get("machine_line_mapping_rule"), top.get("machine_line_mapping_source", "assumed")),
        ("Warehouse Network", top.get("warehouses", {}).get("network_structure"), top.get("warehouses", {}).get("network_source", "assumed")),
        ("QC Retest Delay", f"{params.get('retest_delay_days', {}).get('value')} days", params.get("retest_delay_days", {}).get("source", "assumed")),
        ("Holding Cost Proxy", f"${params.get('holding_cost_per_unit_day', {}).get('value')} / unit-day", params.get("holding_cost_per_unit_day", {}).get("source", "assumed")),
        ("Transfer Lead Time", f"{params.get('transfer_lead_time_days', {}).get('value')} days", params.get("transfer_lead_time_days", {}).get("source", "assumed")),
        ("Transfer Cost", f"${params.get('transfer_cost_per_unit', {}).get('value')} / unit", params.get("transfer_cost_per_unit", {}).get("source", "assumed")),
        ("Expedite Cost", f"${params.get('raw_material_expedite_cost', {}).get('value')} / shipment", params.get("raw_material_expedite_cost", {}).get("source", "assumed")),
        ("Maintenance Cost", f"${params.get('maintenance_pull_forward_cost', {}).get('value')} / event", params.get("maintenance_pull_forward_cost", {}).get("source", "assumed")),
        ("Cold-Chain Excursion Rate", f"{params.get('cold_chain_excursion_rate', {}).get('value')*100:.2f}%", params.get("cold_chain_excursion_rate", {}).get("source", "fitted")),
    ]

    return html.Table([
        html.Thead(html.Tr([html.Th("Parameter"), html.Th("Specification Value"), html.Th("Source Provenance Tag")])),
        html.Tbody([
            html.Tr([
                html.Td(p_name, style={"fontWeight": "500"}),
                html.Td(p_val, className="mono"),
                html.Td(
                    html.Span(
                        f"[{p_src.upper()}]",
                        className="badge-tag",
                        style={
                            "backgroundColor": "var(--primary-pale)" if p_src == "fitted" else "var(--surface-alt)",
                            "color": "var(--primary)" if p_src == "fitted" else "var(--text-muted)",
                            "border": "1px solid var(--border)",
                            "borderRadius": "var(--radius-sm)",
                            "padding": "2px 8px",
                            "fontSize": "0.75rem",
                            "fontWeight": "600",
                        },
                    )
                ),
            ])
            for p_name, p_val, p_src in items
        ]),
    ], className="data-table")
