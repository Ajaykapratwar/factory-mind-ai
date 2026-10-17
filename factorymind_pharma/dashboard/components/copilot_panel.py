"""
FactoryMind AI (Pharma) - AI Copilot Dashboard Panel Component (§9.1)
Collapsible right-hand panel embedded across all dashboard pages.
Features:
  - Strictly consumes CSS variables from tokens.css (zero hardcoded hex).
  - One-line "based on: tables, as-of date" source strip.
  - Expandable tool execution trace.
  - Suggested-question chips tailored to operational context.
  - Zero ungrounded numbers with code-level validation badge.
"""
from __future__ import annotations
from typing import List, Optional
import os
import dash
from dash import html, dcc, callback, Input, Output, State, ALL
from factorymind_pharma.services.tower_svc.copilot.client import CopilotClient

copilot_client = CopilotClient()

SUGGESTED_CHIPS = [
    "What is the overall executive KPI summary?",
    "What is the 30-day demand forecast for product P0020?",
    "Why did batch B100001 require a retest?",
    "What is the failure risk and sensor telemetry for machine M001?",
    "Is product P0020 profitable or loss-making?",
]


def render_copilot_panel() -> html.Div:
    """Renders the collapsible Copilot drawer and toggle button."""
    return html.Div([
        # Floating Trigger Button
        html.Button(
            [
                html.Span(className="icon-mask icon-spark", style={"marginRight": "8px"}),
                "AI Copilot",
            ],
            id="copilot-toggle-btn",
            style={
                "position": "fixed",
                "bottom": "24px",
                "right": "24px",
                "zIndex": "1000",
                "backgroundColor": "var(--accent)",
                "color": "var(--surface)",
                "border": "1px solid var(--border)",
                "borderRadius": "var(--radius-md)",
                "padding": "10px 18px",
                "fontFamily": "var(--font-body)",
                "fontSize": "0.9rem",
                "fontWeight": "600",
                "boxShadow": "var(--shadow-md)",
                "cursor": "pointer",
                "display": "flex",
                "alignItems": "center",
            },
        ),

        # Collapsible Drawer / Sidebar
        html.Div(
            id="copilot-drawer",
            style={
                "position": "fixed",
                "top": "0",
                "right": "0",
                "width": "420px",
                "maxWidth": "90vw",
                "height": "100vh",
                "backgroundColor": "var(--surface)",
                "borderLeft": "1px solid var(--border)",
                "boxShadow": "var(--shadow-md)",
                "zIndex": "1050",
                "display": "none",
                "flexDirection": "column",
                "fontFamily": "var(--font-body)",
            },
            children=[
                # Drawer Header
                html.Div([
                    html.Div([
                        html.Span(className="icon-mask icon-spark", style={"color": "var(--accent)", "marginRight": "8px"}),
                        html.Span("FactoryMind AI Copilot", style={"fontWeight": "600", "fontSize": "1.05rem", "color": "var(--text)"}),
                    ], style={"display": "flex", "alignItems": "center"}),
                    html.Div([
                        html.Span(
                            "Copilot Ready" if bool(os.environ.get("GROQ_API_KEY", "").strip()) else "No Groq Key",
                            className="copilot-chip-header " + ("ready" if bool(os.environ.get("GROQ_API_KEY", "").strip()) else "no-key"),
                            style={"marginRight": "10px"},
                        ),
                        html.Button(
                            "✕",
                            id="copilot-close-btn",
                            style={
                                "background": "none",
                                "border": "none",
                                "fontSize": "1.1rem",
                                "color": "var(--text-muted)",
                                "cursor": "pointer",
                                "padding": "4px",
                            },
                        ),
                    ], style={"display": "flex", "alignItems": "center"}),
                ], style={
                    "padding": "16px 20px",
                    "borderBottom": "1px solid var(--border)",
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "backgroundColor": "var(--surface-alt)",
                }),

                # Subtitle & Decision Support Badge
                html.Div([
                    html.Span(
                        "Operational Decision Support • Zero-Ungrounded-Numbers Gate",
                        style={"fontSize": "0.75rem", "color": "var(--text-muted)"}
                    ),
                ], style={"padding": "8px 20px", "borderBottom": "1px solid var(--border)", "backgroundColor": "var(--surface)"}),

                # Suggested Question Chips
                html.Div([
                    html.Div("Suggested Questions:", style={"fontSize": "0.75rem", "fontWeight": "600", "color": "var(--text-muted)", "marginBottom": "6px"}),
                    html.Div([
                        html.Button(
                            chip,
                            id={"type": "copilot-chip", "index": i},
                            style={
                                "fontSize": "0.75rem",
                                "backgroundColor": "var(--accent-pale)",
                                "color": "var(--accent-dark)",
                                "border": "1px solid var(--border)",
                                "borderRadius": "var(--radius-sm)",
                                "padding": "4px 8px",
                                "margin": "3px 4px 3px 0",
                                "cursor": "pointer",
                                "textAlign": "left",
                                "display": "inline-block",
                            }
                        ) for i, chip in enumerate(SUGGESTED_CHIPS)
                    ], style={"display": "flex", "flexWrap": "wrap"}),
                ], style={"padding": "12px 20px", "backgroundColor": "var(--surface-alt)", "borderBottom": "1px solid var(--border)"}),

                # Conversation Display Area
                html.Div(
                    id="copilot-conversation",
                    style={
                        "flex": "1",
                        "overflowY": "auto",
                        "padding": "16px 20px",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "14px",
                    },
                    children=[
                        html.Div([
                            html.P(
                                "Hello! I am your pharma manufacturing operational copilot. Ask me about executive KPIs, SKU forecasts, batch QC deviations, equipment sensor health, P&L financials, or digital twin scenarios.",
                                style={"fontSize": "0.85rem", "color": "var(--text-muted)", "lineHeight": "1.4"}
                            ),
                        ], style={"backgroundColor": "var(--surface-alt)", "padding": "12px", "borderRadius": "var(--radius-sm)", "border": "1px solid var(--border)"}),
                    ],
                ),

                # Input Form
                html.Div([
                    dcc.Textarea(
                        id="copilot-input",
                        placeholder="Ask an operational question (e.g. 'What is the forecast for P0020?')...",
                        style={
                            "width": "100%",
                            "height": "70px",
                            "backgroundColor": "var(--surface)",
                            "color": "var(--text)",
                            "border": "1px solid var(--border)",
                            "borderRadius": "var(--radius-sm)",
                            "padding": "8px",
                            "fontSize": "0.85rem",
                            "fontFamily": "var(--font-body)",
                            "resize": "none",
                            "boxSizing": "border-box",
                        },
                    ),
                    html.Div([
                        html.Button(
                            "Submit Query",
                            id="copilot-submit-btn",
                            style={
                                "backgroundColor": "var(--accent)",
                                "color": "var(--surface)",
                                "border": "none",
                                "borderRadius": "var(--radius-sm)",
                                "padding": "8px 16px",
                                "fontSize": "0.82rem",
                                "fontWeight": "600",
                                "cursor": "pointer",
                                "width": "100%",
                                "marginTop": "8px",
                            },
                        ),
                    ]),
                ], style={
                    "padding": "16px 20px",
                    "borderTop": "1px solid var(--border)",
                    "backgroundColor": "var(--surface-alt)",
                }),
            ]
        ),
    ])


@callback(
    Output("copilot-drawer", "style"),
    [Input("copilot-toggle-btn", "n_clicks"), Input("copilot-close-btn", "n_clicks")],
    State("copilot-drawer", "style"),
    prevent_initial_call=True,
)
def toggle_copilot_drawer(open_clicks, close_clicks, current_style):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_style

    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    style = current_style.copy() if current_style else {}

    if trigger_id == "copilot-toggle-btn":
        is_hidden = style.get("display") == "none"
        style["display"] = "flex" if is_hidden else "none"
    elif trigger_id == "copilot-close-btn":
        style["display"] = "none"

    return style


@callback(
    [Output("copilot-conversation", "children"), Output("copilot-input", "value")],
    [Input("copilot-submit-btn", "n_clicks"), Input({"type": "copilot-chip", "index": ALL}, "n_clicks")],
    [State("copilot-input", "value"), State("copilot-conversation", "children")],
    prevent_initial_call=True,
)
def handle_copilot_query(submit_clicks, chip_clicks, input_val, current_children):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_children, ""

    prompt = ""
    prop_id = ctx.triggered[0]["prop_id"]

    if "copilot-submit-btn" in prop_id:
        if not input_val or not input_val.strip():
            return current_children, ""
        prompt = input_val.strip()
    elif "copilot-chip" in prop_id:
        # Extract index
        import json
        chip_dict = json.loads(prop_id.split(".")[0])
        idx = chip_dict.get("index", 0)
        if idx < len(SUGGESTED_CHIPS):
            prompt = SUGGESTED_CHIPS[idx]

    if not prompt:
        return current_children, ""

    # Execute Copilot Query
    resp = copilot_client.ask(prompt)

    # Render User Message
    user_bubble = html.Div([
        html.Div("You", style={"fontSize": "0.75rem", "fontWeight": "600", "color": "var(--text-muted)", "marginBottom": "2px"}),
        html.Div(prompt, style={"fontSize": "0.85rem", "color": "var(--text)"}),
    ], style={
        "backgroundColor": "var(--surface-alt)",
        "padding": "10px 14px",
        "borderRadius": "var(--radius-sm)",
        "border": "1px solid var(--border)",
        "alignSelf": "flex-end",
        "maxWidth": "85%",
    })

    # Render Copilot Response Bubble
    grounding_status = "100% Numerically Grounded" if resp.grounding.is_grounded else "Grounding Check Failed"
    grounding_color = "var(--success)" if resp.grounding.is_grounded else "var(--danger)"

    trace_details = []
    for t in resp.trace:
        trace_details.append(html.Div([
            html.Span(f"Tool: {t.get('tool')}", style={"fontWeight": "600"}),
            html.Br(),
            html.Span(f"Source: {t.get('provenance', {}).get('source_table')} | As-of: {t.get('provenance', {}).get('as_of')}"),
        ], style={"fontSize": "0.72rem", "padding": "4px 0"}))

    ai_bubble = html.Div([
        # Sender line
        html.Div([
            html.Span(className="icon-mask icon-spark", style={"color": "var(--accent)", "marginRight": "4px"}),
            html.Span("FactoryMind AI", style={"fontSize": "0.75rem", "fontWeight": "600", "color": "var(--accent)"}),
            html.Span(f" • {resp.model_used}", style={"fontSize": "0.7rem", "color": "var(--text-xmuted)"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}),

        # Answer text
        html.Div(resp.answer, style={"fontSize": "0.85rem", "color": "var(--text)", "lineHeight": "1.45"}),

        # Source strip
        html.Div(
            resp.provenance_strip,
            style={
                "fontSize": "0.72rem",
                "color": "var(--text-muted)",
                "backgroundColor": "var(--surface-alt)",
                "padding": "4px 8px",
                "borderRadius": "var(--radius-sm)",
                "marginTop": "8px",
                "border": "1px solid var(--border)",
            }
        ),

        # Grounding status badge
        html.Div([
            html.Span("✓ " if resp.grounding.is_grounded else "⚠ ", style={"fontWeight": "bold"}),
            html.Span(grounding_status),
        ], style={"fontSize": "0.7rem", "color": grounding_color, "marginTop": "6px", "fontWeight": "500"}),

        # Collapsible trace
        html.Details([
            html.Summary("View Tool Execution Trace", style={"fontSize": "0.72rem", "color": "var(--text-muted)", "cursor": "pointer", "marginTop": "6px"}),
            html.Div(trace_details if trace_details else "Direct response without external tools.", style={"padding": "6px 8px", "backgroundColor": "var(--surface)", "border": "1px solid var(--border)", "marginTop": "4px", "borderRadius": "var(--radius-sm)"}),
        ], style={"marginTop": "4px"}),
    ], style={
        "backgroundColor": "var(--surface)",
        "padding": "12px 16px",
        "borderRadius": "var(--radius-sm)",
        "border": "1px solid var(--border)",
        "boxShadow": "var(--shadow-xs)",
        "alignSelf": "flex-start",
        "maxWidth": "95%",
    })

    new_children = list(current_children) + [user_bubble, ai_bubble]
    return new_children, ""
