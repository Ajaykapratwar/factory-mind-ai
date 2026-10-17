"""
dashboard/components/header.py
Shared header configuration and page metadata.

Rules:
- Layer tag mapping:
  - Executive and Operational: Overview (L1)
  - Forecast, Machine Health, Supplier & QC: L3
  - Inventory Opt and Compliance: L4
  - Scenario Lab: L2
  - Control Tower: L5
- Page body does NOT duplicate the H1 (which lives in the route-driven top bar).
- Page body displays the clean one-line subtitle description.
"""

from dash import html

PAGE_METADATA = {
    "/": {
        "title": "Executive Summary",
        "subtitle": "Manufacturing control tower — all SKUs, all regions, all channels",
        "layer": "Overview (L1)",
        "layer_var": "--layer-l1",
    },
    "/operational": {
        "title": "Operational Planning",
        "subtitle": "Daily plant throughput, machine schedule & order fulfillment",
        "layer": "Overview (L1)",
        "layer_var": "--layer-l1",
    },
    "/forecast": {
        "title": "Forecast Analytics",
        "subtitle": "Latest Phase 2 Ensemble Forecast vs. Actual Demand · Target R² ≥ 0.80",
        "layer": "Layer L3 Forecast",
        "layer_var": "--layer-l3",
    },
    "/machine-health": {
        "title": "Machine Health & Predictive Maintenance",
        "subtitle": "Real-time telemetry, failure probability & maintenance schedule for all plant equipment",
        "layer": "Layer L3 Sensing",
        "layer_var": "--layer-l3",
    },
    "/supplier-qc": {
        "title": "Supplier & QC Analytics",
        "subtitle": "Vendor scorecard, raw material compliance and batch release anomaly detection",
        "layer": "Layer L3 Quality",
        "layer_var": "--layer-l3",
    },
    "/inventory-opt": {
        "title": "Safety Stock & Inventory Optimization",
        "subtitle": "Data-driven safety stock buffers and reorder points based on ensemble residual variance",
        "layer": "Layer L4 Policy",
        "layer_var": "--layer-l4",
    },
    "/compliance": {
        "title": "Compliance & Logistics Gating",
        "subtitle": "Operational gates for Quality/Compliance deviations and Cold-Chain temperature excursions",
        "layer": "Layer L4 Policy",
        "layer_var": "--layer-l4",
    },
    "/financial-impact": {
        "title": "Financial Impact & P&L Analysis",
        "subtitle": "Unit economics, portfolio gross margins, cost breakdown, and SKU profitability",
        "layer": "Layer L3 Financial",
        "layer_var": "--layer-l3",
    },
    "/scenario-lab": {
        "title": "Scenario Lab (Digital Twin)",
        "subtitle": "Stochastic Monte Carlo 'what-if' simulation with Common Random Numbers and ranked mitigations",
        "layer": "Layer L2 Twin",
        "layer_var": "--layer-l2",
    },
    "/control-tower": {
        "title": "L5 Control Tower & Orchestration",
        "subtitle": "Autonomous Decision Governance, Operational Approval Queue, and Closed-Loop Model Lifecycle",
        "layer": "Layer L5 Tower",
        "layer_var": "--layer-l5",
    },
}


def render_page_header(subtitle: str = "") -> html.Div:
    """Renders the standard page description without a redundant H1."""
    return html.Div([
        html.P(subtitle, className="page-subtitle"),
    ], className="page-header")
