"""
dashboard/components/theme.py
Reads tokens.css and builds a shared Plotly template ("factorymind")
so every chart shares one source of truth with the CSS.

Global rules:
- FONT: single token --font-ui: "Segoe UI", Inter, system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif
- STATUS_COLORS: fixed semantic colours (delivered/AUTO=green, in-transit/pending=sky, delayed/REVIEW=amber, cancelled/BLOCK=coral)
- COMPARE_COLORS: baseline = neutral slate (#7C8A99), subject/optimised = saturated teal (#1F8A85)
- SERIES_COLORS: actual = dark ink-teal solid, forecast = coral dashed, band = pale tint
- Light horizontal gridlines only; no vertical gridlines on time series.
- automargin=True on both axes; legend top-left horizontal outside plot.
"""

import re
import plotly.io as pio
import plotly.graph_objects as go
from pathlib import Path

# ─── Parse tokens.css to extract colour hex values ───────────────────────
_CSS_PATH = Path(__file__).parent.parent / "assets" / "tokens.css"


def _parse_tokens() -> dict:
    """Parse :root { --var: value; } from tokens.css."""
    tokens: dict = {}
    try:
        text = _CSS_PATH.read_text(encoding="utf-8")
        for match in re.finditer(r"--([\w-]+)\s*:\s*([^;]+);", text):
            name, value = match.group(1), match.group(2).strip()
            # Keep hex colour values
            if re.match(r"^#[0-9A-Fa-f]{3,8}$", value):
                tokens[name] = value
    except Exception:
        pass
    return tokens


_TOKENS = _parse_tokens()


def tok(name: str, fallback: str = "#1F8A85") -> str:
    """Get a hex colour by CSS variable name."""
    return _TOKENS.get(name, fallback)


FONT_UI = '"Segoe UI", Inter, system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif'

# ─── Colour constants (shared with all chart modules) ───────────────────
CHART_COLORS = {
    # Series palette
    "chart_1": tok("chart-1", "#1F8A85"),
    "chart_2": tok("chart-2", "#5B4B9A"),
    "chart_3": tok("chart-3", "#E2604A"),
    "chart_4": tok("chart-4", "#E5A93C"),
    "chart_5": tok("chart-5", "#3F8F5B"),
    "chart_6": tok("chart-6", "#3B82A0"),
    # Surfaces
    "bg":      tok("bg",      "#F4F1EA"),
    "surface": tok("surface", "#FFFEFB"),
    "border":  tok("border",  "#D8D3CA"),
    # Text
    "text":    tok("text",    "#1C1B1A"),
    "muted":   tok("text-muted", "#6B6560"),
    # Semantic
    "success": tok("success", "#2E7D4F"),
    "warning": tok("warning", "#B07D14"),
    "danger":  tok("danger",  "#C0392B"),
    "info":    tok("info",    "#3B82A0"),
    # Layer colours
    "layer_l1": tok("layer-l1", "#6B8E8C"),
    "layer_l2": tok("layer-l2", "#1F8A85"),
    "layer_l3": tok("layer-l3", "#5B4B9A"),
    "layer_l4": tok("layer-l4", "#E2604A"),
    "layer_l5": tok("layer-l5", "#12343B"),
}

STATUS_COLORS = {
    "delivered":  tok("status-delivered", "#2E7D4F"),
    "auto":       tok("status-auto", "#2E7D4F"),
    "in-transit": tok("status-intransit", "#3B82A0"),
    "pending":    tok("status-pending", "#3B82A0"),
    "delayed":    tok("status-delayed", "#B07D14"),
    "review":     tok("status-review", "#B07D14"),
    "cancelled":  tok("status-cancelled", "#C0392B"),
    "block":      tok("status-block", "#C0392B"),
}

COMPARE_COLORS = {
    "baseline":  tok("compare-baseline", "#7C8A99"),  # neutral slate
    "optimized": tok("compare-optimized", "#1F8A85"), # saturated teal
}

SERIES_COLORS = {
    "actual":   tok("series-actual", "#12343B"),   # dark ink-teal solid
    "forecast": tok("series-forecast", "#E2604A"), # coral dashed
    "band":     "rgba(226, 96, 74, 0.12)",         # pale tint
}

COLOR_SEQ = [
    CHART_COLORS["chart_1"],
    CHART_COLORS["chart_2"],
    CHART_COLORS["chart_3"],
    CHART_COLORS["chart_4"],
    CHART_COLORS["chart_5"],
    CHART_COLORS["chart_6"],
]


# ─── Chart layout factory ────────────────────────────────────────────────

def make_chart_layout(
    title: str = "",
    height: int = 300,
    xaxis_title: str = "",
    yaxis_title: str = "",
    show_legend: bool = True,
    margin: dict | None = None,
    bg: str | None = None,
) -> dict:
    """Return a Plotly layout dict matching our design tokens."""
    if margin is None:
        margin = dict(l=50, r=20, t=16 if not title else 36, b=40)
    _bg = bg or CHART_COLORS["surface"]

    layout_dict = dict(
        height=height,
        paper_bgcolor=_bg,
        plot_bgcolor=_bg,
        margin=margin,
        font=dict(
            family=FONT_UI,
            size=11,
            color=CHART_COLORS["text"],
        ),
        showlegend=show_legend,
        legend=dict(
            font=dict(family=FONT_UI, size=11, color=CHART_COLORS["muted"]),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(family=FONT_UI, size=11, color=CHART_COLORS["muted"])) if xaxis_title else None,
            showgrid=False,  # No vertical gridlines
            zeroline=False,
            automargin=True,
            tickfont=dict(family=FONT_UI, size=10, color=CHART_COLORS["muted"]),
            showline=False,
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=dict(family=FONT_UI, size=11, color=CHART_COLORS["muted"])) if yaxis_title else None,
            showgrid=True,   # Light horizontal gridlines only
            gridcolor=CHART_COLORS["border"],
            gridwidth=1,
            zeroline=False,
            automargin=True,
            tickfont=dict(family=FONT_UI, size=10, color=CHART_COLORS["muted"]),
            showline=False,
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=CHART_COLORS["surface"],
            bordercolor=CHART_COLORS["border"],
            font=dict(family=FONT_UI, size=12, color=CHART_COLORS["text"]),
            namelength=-1,
        ),
    )

    if title:
        layout_dict["title"] = dict(
            text=title,
            font=dict(family=FONT_UI, size=12, color=CHART_COLORS["text"]),
            x=0,
            xanchor="left",
            pad=dict(l=4),
        )

    return layout_dict


# ─── Register Plotly template ────────────────────────────────────────────

def apply_template() -> None:
    """Register the 'factorymind' Plotly theme. Call once at app startup."""
    _bg = CHART_COLORS["surface"]
    _border = CHART_COLORS["border"]
    _text = CHART_COLORS["text"]
    _muted = CHART_COLORS["muted"]

    layout = go.Layout(
        paper_bgcolor=_bg,
        plot_bgcolor=_bg,
        colorway=COLOR_SEQ,
        font=dict(family=FONT_UI, size=11, color=_text),
        xaxis=dict(
            showgrid=False,  # No vertical gridlines
            zeroline=False,
            automargin=True,
            tickfont=dict(family=FONT_UI, size=10, color=_muted),
            showline=False,
        ),
        yaxis=dict(
            showgrid=True,   # Light horizontal gridlines only
            gridcolor=_border,
            gridwidth=1,
            zeroline=False,
            automargin=True,
            tickfont=dict(family=FONT_UI, size=10, color=_muted),
            showline=False,
        ),
        showlegend=True,
        legend=dict(
            font=dict(family=FONT_UI, size=11, color=_muted),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=_bg,
            bordercolor=_border,
            font=dict(family=FONT_UI, size=12, color=_text),
            namelength=-1,
        ),
        margin=dict(l=50, r=20, t=16, b=40),
    )
    pio.templates["factorymind"] = go.layout.Template(layout=layout)
    pio.templates.default = "factorymind"
