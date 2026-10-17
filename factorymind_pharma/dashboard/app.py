"""
dashboard/app.py — Main Dash multi-page application entry point.
Redesigned v2: fixed left sidebar (240 px) with deep ink-teal background,
section groups, top-bar with breadcrumb + status chips.
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

# Register the Plotly theme before any page loads
from dashboard.components.theme import apply_template
apply_template()

# App init — multi-page
app = dash.Dash(
    __name__,
    use_pages=True,
    pages_folder=str(Path(__file__).parent / "pages"),
    assets_folder=str(Path(__file__).parent / "assets"),
    title="FactoryMind AI",
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # for production WSGI deployment


from dashboard.components.header import PAGE_METADATA


# ── Sidebar helpers ────────────────────────────────────────────────────────

def _layer_dot(color_var: str) -> html.Span:
    return html.Span(className="layer-dot", style={"background": f"var({color_var})"})


def _nav_item(label: str, href: str, icon_cls: str, layer_var: str = "--layer-l1") -> dcc.Link:
    return dcc.Link(
        [
            html.Span(className=f"icon-mask {icon_cls}"),
            html.Span(label, className="nav-label"),
            _layer_dot(layer_var),
        ],
        href=href,
        className="nav-link",
    )


def _section_label(text: str) -> html.Div:
    return html.Div(text, className="sidebar-section-label")


def _build_sidebar() -> html.Div:
    return html.Div([
        # Logo
        html.Div([
            html.Span("FactoryMind AI", className="sidebar-logo-text"),
            html.Span("Pharma · Enterprise L1–L5", className="sidebar-logo-sub"),
        ], className="sidebar-logo"),

        # Nav
        html.Nav([
            _section_label("Overview"),
            _nav_item("Executive Summary", "/", "icon-dashboard", "--layer-l1"),

            _section_label("Plan"),
            _nav_item("Forecast Analytics", "/forecast", "icon-chart", "--layer-l3"),
            _nav_item("Inventory Opt", "/inventory-opt", "icon-box", "--layer-l4"),

            _section_label("Operate"),
            _nav_item("Operational", "/operational", "icon-activity", "--layer-l1"),
            _nav_item("Machine Health", "/machine-health", "icon-cpu", "--layer-l3"),

            _section_label("Risk & Quality"),
            _nav_item("Supplier & QC", "/supplier-qc", "icon-shield", "--layer-l3"),
            _nav_item("Compliance", "/compliance", "icon-compliance", "--layer-l4"),

            _section_label("Money"),
            _nav_item("Financial Impact", "/financial-impact", "icon-dollar", "--layer-l3"),

            _section_label("Control"),
            _nav_item("Scenario Lab", "/scenario-lab", "icon-sliders", "--layer-l2"),
            _nav_item("Control Tower", "/control-tower", "icon-tower", "--layer-l5"),
        ], className="sidebar-nav"),

        # Footer
        html.Div([
            html.Div("FactoryMind AI v2", className="sidebar-footer-text", style={"fontWeight": "600", "color": "var(--sidebar-text)"}),
            html.Div("Data through 2022-12", className="sidebar-footer-text"),
            html.Div("Pharma L1–L5 · 100 SKUs", className="sidebar-footer-text"),
        ], className="sidebar-footer"),
    ], className="sidebar")


def _build_topbar() -> html.Div:
    return html.Div(
        id="topbar-content",
        className="top-bar",
    )


from dashboard.components.copilot_panel import render_copilot_panel

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    _build_sidebar(),

    # Main content (offset by sidebar width via CSS)
    html.Div([
        _build_topbar(),

        # Page content rendered here
        dash.page_container,
    ], className="main-content"),

    # AI Copilot collapsible panel (§9.1)
    render_copilot_panel(),
], className="app-shell")


@app.callback(
    dash.Output("topbar-content", "children"),
    dash.Input("url", "pathname"),
)
def _update_topbar(pathname: str):
    # Normalize trailing slash
    path = pathname or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    meta = PAGE_METADATA.get(path, {
        "title": "FactoryMind AI",
        "subtitle": "",
        "layer": "Overview (L1)",
        "layer_var": "--layer-l1",
    })
    return html.Div([
        html.Span(meta["title"], className="top-bar-title"),
        html.Div([
            html.Span(className="layer-dot", style={"background": f"var({meta['layer_var']})"}),
            html.Span(meta["layer"]),
        ], className="top-bar-layer-chip"),
    ], className="top-bar-title-row")



if __name__ == "__main__":
    port = cfg["dashboard"].get("port", 8050)
    debug = cfg["dashboard"].get("debug", False)
    print(f"\n  FactoryMind AI (Pharma) Dashboard v2")
    print(f"  Running on http://localhost:{port}")
    app.run(debug=debug, port=port)
