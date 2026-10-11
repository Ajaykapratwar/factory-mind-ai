"""
dashboard/app.py — Main Dash multi-page application entry point.
Both dashboards (Executive + Operational) share the same nav bar and design tokens.
"""

import sys, os
from pathlib import Path

# Ensure both workspace root and package are in sys.path
_PKG_DIR = Path(__file__).resolve().parent.parent
_ROOT_DIR = _PKG_DIR.parent
for _p in [str(_ROOT_DIR), str(_PKG_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import dash
from dash import html, dcc
import yaml

# Load config
_BASE = Path(__file__).parent.parent
_CONFIG_PATH = _BASE / "configs" / "config.yaml"

def _cfg():
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)

cfg = _cfg()

# App init — multi-page
app = dash.Dash(
    __name__,
    use_pages=True,
    pages_folder=str(Path(__file__).parent / "pages"),
    assets_folder=str(Path(__file__).parent / "assets"),
    title=cfg["dashboard"]["title"],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # for production WSGI deployment

# Nav bar

def _nav_link(label, href):
    return dcc.Link(label, href=href, className="nav-link")

from factorymind_pharma.dashboard.components.copilot_panel import render_copilot_panel

app.layout = html.Div([
    # Navigation bar
    html.Nav([
        html.Div([
            html.Span("⬡", style={"marginRight": "6px", "fontSize": "1.1rem"}),
            "FactoryMind Pharma",
        ], className="nav-logo"),

        html.Div([
            _nav_link("Executive Summary", "/"),
            _nav_link("Operational", "/operational"),
            _nav_link("Forecast", "/forecast"),
            _nav_link("Machine Health", "/machine-health"),
            _nav_link("Supplier & QC", "/supplier-qc"),
            _nav_link("Inventory Opt", "/inventory-opt"),
            _nav_link("Compliance", "/compliance"),
            _nav_link("Financial Impact", "/financial-impact"),
            _nav_link("Scenario Lab", "/scenario-lab"),
            _nav_link("Control Tower", "/control-tower"),
        ], className="nav-links", style={"display": "flex", "flexWrap": "wrap", "gap": "10px"}),

        html.Div("Enterprise L1–L5", className="nav-timestamp"),
    ], className="nav-bar"),

    # Page content rendered here
    dash.page_container,

    # AI Copilot collapsible panel live on all pages (§9.1)
    render_copilot_panel(),
])

if __name__ == "__main__":
    port = cfg["dashboard"].get("port", 8050)
    debug = cfg["dashboard"].get("debug", False)
    print(f"\n  FactoryMind Pharma Dashboard (Phase 2 & 3)")
    print(f"  Running on http://localhost:{port}")
    app.run(debug=debug, port=port)
