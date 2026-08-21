"""
dashboard/app.py — Main Dash multi-page application entry point.
Both dashboards (Executive + Operational) share the same nav bar and design tokens.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import dash
from dash import html, dcc
import yaml
from pathlib import Path

# Load config
_BASE = Path(__file__).parent.parent
_CONFIG_PATH = _BASE / "configs" / "config.yaml"

def _cfg():
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)

cfg = _cfg()

# ---------------------------------------------------------------------------
# App init — multi-page
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Nav bar
# ---------------------------------------------------------------------------

def _nav_link(label, href):
    return dcc.Link(label, href=href, className="nav-link")

app.layout = html.Div([
    # Navigation bar
    html.Nav([
        html.Div([
            html.Span("⬡", style={"marginRight": "6px", "fontSize": "1.1rem"}),
            "FactoryMind Pharma",
        ], className="nav-logo"),

        html.Div([
            _nav_link("Executive Summary", "/"),
            _nav_link("Operational Planning", "/operational"),
        ], className="nav-links"),

        html.Div("Phase 1 · Baseline", className="nav-timestamp"),
    ], className="nav-bar"),

    # Page content rendered here
    dash.page_container,
])

if __name__ == "__main__":
    port = cfg["dashboard"].get("port", 8050)
    debug = cfg["dashboard"].get("debug", False)
    print(f"\n  FactoryMind Pharma Dashboard")
    print(f"  Running on http://localhost:{port}")
    print(f"  Executive:   http://localhost:{port}/")
    print(f"  Operational: http://localhost:{port}/operational\n")
    app.run(debug=debug, port=port)
