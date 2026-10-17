"""
dashboard/components/kpi.py
Shared KPI card builder and formatting helpers across all dashboard pages.

Rules:
- One shared helper for formatting numbers.
- Negative money formatted as -$4.2M (not $-4.2M).
- Compact K/M with exact value in HTML title tooltip.
- Tags like "(simulated)" shown as a small chip under the value, NEVER inside it.
- Icons render cleanly with SVG masks.
- font-variant-numeric: tabular-nums for all values.
"""

from typing import Optional, Tuple
import math
import numpy as np
from dash import html


def format_value_and_chip(raw_val: any, prefix: str = "", suffix: str = "", is_currency: bool = False) -> Tuple[str, str, Optional[str]]:
    """
    Format numeric value into (compact_display, full_tooltip, chip_tag).
    Example:
        -4198217 -> ("-$4.2M", "-$4,198,217", None)
        "100.0% (simulated)" -> ("100.0%", "100.0% (simulated)", "simulated")
    """
    if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
        return ("—", "No data", None)

    # If already a string containing (tag)
    chip_tag = None
    if isinstance(raw_val, str):
        val_str = raw_val.strip()
        if "(" in val_str and val_str.endswith(")"):
            base, tag = val_str[:-1].split("(", 1)
            chip_tag = tag.strip()
            val_str = base.strip()

        # Check for currency with misplaced negative sign like "$-4,198,217"
        if val_str.startswith("$-"):
            val_str = "-$" + val_str[2:]
        elif val_str.startswith("$+"):
            val_str = "+$" + val_str[2:]

        return (val_str, str(raw_val), chip_tag)

    # Numeric handling
    try:
        val = float(raw_val)
    except (ValueError, TypeError):
        return (str(raw_val), str(raw_val), None)

    sign = "-" if val < 0 else ("+" if prefix.startswith("+") else "")
    abs_val = abs(val)
    curr = "$" if is_currency or prefix.endswith("$") else prefix.lstrip("+-")

    full_tooltip = f"{sign}{curr}{abs_val:,.0f}{suffix}"

    if abs_val >= 1_000_000_000:
        compact = f"{sign}{curr}{abs_val / 1_000_000_000:.1f}B{suffix}"
    elif abs_val >= 1_000_000:
        compact = f"{sign}{curr}{abs_val / 1_000_000:.1f}M{suffix}"
    elif abs_val >= 10_000:
        compact = f"{sign}{curr}{abs_val / 1_000:.1f}K{suffix}"
    elif abs_val >= 100:
        compact = f"{sign}{curr}{abs_val:,.0f}{suffix}"
    elif isinstance(raw_val, float):
        compact = f"{sign}{curr}{abs_val:.2f}{suffix}"
    else:
        compact = f"{sign}{curr}{abs_val:,.0f}{suffix}"

    return (compact, full_tooltip, None)


def kpi_card(
    label: str,
    raw_value: any,
    subtext: str = "",
    color: str = "var(--text)",
    icon_name: str = "chart",
    icon_cls: str = "teal",
    is_currency: bool = False,
    prefix: str = "",
    suffix: str = "",
    primary: bool = False,
) -> html.Div:
    """Renders a standardized, rich KPI card."""
    compact, tooltip, chip = format_value_and_chip(raw_value, prefix=prefix, suffix=suffix, is_currency=is_currency)

    value_children = [
        html.Span(compact, title=tooltip, className="kpi-value-text", style={"color": color}),
    ]
    if chip:
        value_children.append(
            html.Div(chip, className="kpi-chip simulated")
        )

    icon_chip = html.Div(
        html.Span(className=f"icon-mask icon-{icon_name}"),
        className=f"kpi-icon-chip {icon_cls}",
    )

    card_class = "kpi-card primary" if primary else "kpi-card"

    return html.Div([
        html.Div([
            html.Div(label, className="kpi-label"),
            icon_chip,
        ], className="kpi-header-row"),
        html.Div(value_children, className="kpi-value"),
        html.Div(subtext, className="kpi-subtext") if subtext else None,
    ], className=card_class)
