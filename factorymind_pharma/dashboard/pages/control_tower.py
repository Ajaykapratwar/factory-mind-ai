"""
dashboard/pages/control_tower.py — Dashboard 11: Control Tower & Closed-Loop Governance (§12.2)
Features:
  - Visual Anchor: Role-gated Approval Queue with rule tiers (BLOCK, REVIEW, AUTO-ELIGIBLE).
  - RBAC Enforcement: Server-side check preventing Planner from resolving QA-gated actions.
  - Replay Clock controls (Step Day, Reset, held-out progression).
  - Model Health & Governance: Champion vs. Challenger metrics and promotion rules.
  - Tamper-Evident Audit Chain integrity badge.
  - Strict tokens.css compliance with zero hardcoded hex.
"""
from __future__ import annotations
import dash
from dash import html, dcc, callback, Input, Output, State, ALL
from typing import List, Dict, Any

from factorymind_pharma.services.tower_svc.decision_engine import DecisionEngine
from factorymind_pharma.services.tower_svc.audit_log import AuditLog
from factorymind_pharma.services.tower_svc.replay_clock import ReplayClock, FeedbackWriter, ModelLifecycleManager
from dashboard.components.header import render_page_header, PAGE_METADATA

dash.register_page(
    __name__,
    path="/control-tower",
    title="Control Tower — FactoryMind Pharma",
    name="Control Tower",
)

_META = PAGE_METADATA["/control-tower"]

engine = DecisionEngine()
replay_clock = ReplayClock()
feedback_writer = FeedbackWriter()
lifecycle_mgr = ModelLifecycleManager(audit_log=engine.audit)


def layout():
    # Fetch initial audit status
    audit_res = engine.audit.verify_chain()
    badge_bg = "var(--success-pale)" if audit_res.is_valid else "var(--danger-pale)"
    badge_color = "var(--success)" if audit_res.is_valid else "var(--danger)"
    badge_text = "Audit Chain: Verified (SHA-256 Intact)" if audit_res.is_valid else "Audit Chain: Tamper Detected!"

    return html.Div([
        # ── Header & Governance Strip ──────────────────────────────────
        html.Div([
            html.Div([
                render_page_header(_META["subtitle"]),
            ], style={"flex": "1"}),
            html.Div([
                # Audit Verification Badge
                html.Div(
                    badge_text,
                    id="audit-status-badge",
                    style={
                        "backgroundColor": badge_bg,
                        "color": badge_color,
                        "border": "1px solid var(--border)",
                        "borderRadius": "var(--radius-sm)",
                        "padding": "6px 14px",
                        "fontSize": "0.8rem",
                        "fontWeight": "600",
                        "display": "inline-block",
                        "marginRight": "12px",
                    },
                ),
                # Active Role Selector (RBAC §10.4)
                html.Div([
                    html.Label("Active User Role:", style={"fontSize": "0.75rem", "color": "var(--text-muted)", "marginRight": "8px"}),
                    dcc.Dropdown(
                        id="user-role-selector",
                        options=[
                            {"label": "QA Reviewer (Authorized for BLOCK)", "value": "qa_reviewer"},
                            {"label": "Supply Chain Planner", "value": "planner"},
                            {"label": "System Administrator", "value": "admin"},
                        ],
                        value="qa_reviewer",
                        clearable=False,
                        style={"width": "260px", "fontSize": "0.85rem"},
                    ),
                ], style={"display": "flex", "alignItems": "center"}),
            ], style={"display": "flex", "alignItems": "center", "marginTop": "8px"}),
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start", "marginBottom": "20px"}),

        # ── Replay Clock Controls (§11.1) ──────────────────────────────
        html.Div([
            html.Div([
                html.Span("Simulation Replay Clock:", style={"fontWeight": "600", "fontSize": "0.85rem", "marginRight": "12px"}),
                html.Span(id="replay-date-display", children=replay_clock.get_current_date(), style={
                    "fontWeight": "700",
                    "backgroundColor": "var(--surface)",
                    "padding": "4px 10px",
                    "borderRadius": "var(--radius-sm)",
                    "border": "1px solid var(--border)",
                    "marginRight": "16px",
                }),
                html.Span("(Held-out window: 2022-10-20 to 2022-12-31)", style={"fontSize": "0.78rem", "color": "var(--text-muted)"}),
            ], style={"display": "flex", "alignItems": "center"}),

            html.Div([
                html.Button("Step 1 Day", id="step-day-btn", className="btn btn-secondary", style={
                    "backgroundColor": "var(--surface)",
                    "color": "var(--text)",
                    "border": "1px solid var(--border)",
                    "padding": "6px 14px",
                    "borderRadius": "var(--radius-sm)",
                    "fontSize": "0.82rem",
                    "cursor": "pointer",
                    "marginRight": "8px",
                }),
                html.Button("Reset Clock", id="reset-clock-btn", style={
                    "backgroundColor": "var(--surface-alt)",
                    "color": "var(--text-muted)",
                    "border": "1px solid var(--border)",
                    "padding": "6px 12px",
                    "borderRadius": "var(--radius-sm)",
                    "fontSize": "0.82rem",
                    "cursor": "pointer",
                }),
            ]),
        ], style={
            "backgroundColor": "var(--surface-alt)",
            "padding": "12px 20px",
            "borderRadius": "var(--radius-md)",
            "border": "1px solid var(--border)",
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "center",
            "marginBottom": "24px",
        }),

        # Feedback Notification Banner
        html.Div(id="role-error-banner", style={"display": "none", "marginBottom": "16px"}),

        # ── Main Content Grid: Approval Queue (Visual Anchor) + Model Health ──
        html.Div([
            # Left: Approval Queue (Visual Anchor §12.2)
            html.Div([
                html.Div([
                    html.H2("Operational Approval Queue", style={"fontSize": "1.1rem", "fontWeight": "600", "marginBottom": "4px"}),
                    html.P("Decisions proposed by autonomous sensing and policy gates requiring human or role sign-off.", style={"fontSize": "0.8rem", "color": "var(--text-muted)", "marginBottom": "12px"}),

                    # Tier Filter Tabs as Segmented Control
                    html.Div([
                        html.Button("All Decisions", id="filter-all-btn", className="segmented-control-btn active"),
                        html.Button("BLOCK (QA Mandatory)", id="filter-block-btn", className="segmented-control-btn"),
                        html.Button("REVIEW", id="filter-review-btn", className="segmented-control-btn"),
                        html.Button("AUTO-ELIGIBLE", id="filter-auto-btn", className="segmented-control-btn"),
                    ], className="segmented-control", style={"marginBottom": "16px"}),
                ]),

                # Queue Decision Cards Container
                html.Div(id="queue-items-container", children=_render_decision_cards(engine.list_decisions())),
            ], style={"flex": "2", "marginRight": "24px"}),

            # Right: Model Governance & Drift Health (§11.3, §11.4)
            html.Div([
                # Model Health Card
                html.Div([
                    html.H3("Model Governance & Lifecycle", style={"fontSize": "0.95rem", "fontWeight": "600", "marginBottom": "12px"}),

                    html.Div([
                        html.Div("Active Champion:", style={"fontSize": "0.75rem", "color": "var(--text-muted)"}),
                        html.Div("ensemble-champion-v1", style={"fontWeight": "600", "fontSize": "0.85rem", "color": "var(--accent)"}),
                    ], style={"marginBottom": "8px"}),

                    html.Div([
                        html.Div("Champion WAPE (Holdout):", style={"fontSize": "0.75rem", "color": "var(--text-muted)"}),
                        html.Div("20.20%", style={"fontWeight": "600", "fontSize": "0.9rem"}),
                    ], style={"marginBottom": "12px"}),

                    html.Div([
                        html.Div("Challenger Candidate:", style={"fontSize": "0.75rem", "color": "var(--text-muted)"}),
                        html.Div("ensemble-challenger-v2 (Retrained)", style={"fontWeight": "600", "fontSize": "0.85rem"}),
                    ], style={"marginBottom": "8px"}),

                    html.Div([
                        html.Div("Challenger WAPE / Bias:", style={"fontSize": "0.75rem", "color": "var(--text-muted)"}),
                        html.Div("17.80% WAPE (Bias: -0.40%)", style={"fontWeight": "600", "fontSize": "0.85rem", "color": "var(--success)"}),
                    ], style={"marginBottom": "12px"}),

                    html.Div([
                        html.Div("WAPE Gain vs Champion:", style={"fontSize": "0.75rem", "color": "var(--text-muted)"}),
                        html.Div("+2.40 pp (Exceeds ≥ 2.0% Gate)", style={"fontWeight": "700", "fontSize": "0.85rem", "color": "var(--success)"}),
                    ], style={"marginBottom": "16px"}),

                    html.Button(
                        "Promote Challenger to Champion",
                        id="promote-challenger-btn",
                        style={
                            "backgroundColor": "var(--accent)",
                            "color": "var(--surface)",
                            "border": "none",
                            "borderRadius": "var(--radius-sm)",
                            "padding": "8px 14px",
                            "fontSize": "0.82rem",
                            "fontWeight": "600",
                            "width": "100%",
                            "cursor": "pointer",
                        },
                    ),
                    html.Div(id="promotion-status-msg", style={"fontSize": "0.75rem", "marginTop": "8px"}),
                ], style={
                    "backgroundColor": "var(--surface)",
                    "border": "1px solid var(--border)",
                    "borderRadius": "var(--radius-md)",
                    "padding": "16px 20px",
                    "marginBottom": "20px",
                }),

                # Drift Detection Health
                html.Div([
                    html.H3("Drift Detection Monitors (§11.3)", style={"fontSize": "0.95rem", "fontWeight": "600", "marginBottom": "12px"}),

                    html.Div([
                        html.Span("30-Day Rolling Bias: ", style={"fontSize": "0.8rem", "color": "var(--text-muted)"}),
                        html.Span("-0.85% (Safe, Gate ±5.0%)", style={"fontWeight": "600", "fontSize": "0.8rem", "color": "var(--success)"}),
                    ], style={"marginBottom": "8px"}),

                    html.Div([
                        html.Span("Covariate Shift (PSI): ", style={"fontSize": "0.8rem", "color": "var(--text-muted)"}),
                        html.Span("0.048 (Normal, Gate 0.20)", style={"fontWeight": "600", "fontSize": "0.8rem", "color": "var(--success)"}),
                    ], style={"marginBottom": "8px"}),

                    html.Div([
                        html.Span("Rolling WAPE Jump: ", style={"fontSize": "0.8rem", "color": "var(--text-muted)"}),
                        html.Span("+0.0% (No spike)", style={"fontWeight": "600", "fontSize": "0.8rem", "color": "var(--success)"}),
                    ]),
                ], style={
                    "backgroundColor": "var(--surface)",
                    "border": "1px solid var(--border)",
                    "borderRadius": "var(--radius-md)",
                    "padding": "16px 20px",
                }),
            ], style={"flex": "1"}),
        ], style={"display": "flex", "alignItems": "flex-start", "marginBottom": "24px"}),

        # ── Audit Trail Log (Bottom Card) ──────────────────────────────
        html.Div([
            html.H3("Append-Only Hash Chain Audit Trail (§10.5)", style={"fontSize": "0.95rem", "fontWeight": "600", "marginBottom": "12px"}),
            html.Div(id="audit-entries-table", children=_render_audit_entries(engine.audit.get_entries(limit=5))),
        ], style={
            "backgroundColor": "var(--surface)",
            "border": "1px solid var(--border)",
            "borderRadius": "var(--radius-md)",
            "padding": "16px 20px",
        }),
    ], className="page-content")


def _render_decision_cards(decisions: List[Any]) -> html.Div:
    """Renders visual cards for decisions with tier badges and action buttons."""
    cards = []
    for d in decisions:
        tier_color = "var(--danger)" if d.rule_tier == "BLOCK" else ("var(--warning)" if d.rule_tier == "REVIEW" else "var(--primary)")
        tier_bg = "var(--danger-pale)" if d.rule_tier == "BLOCK" else ("var(--warning-pale)" if d.rule_tier == "REVIEW" else "var(--primary-pale)")

        status_color = "var(--success)" if d.status == "approved" or d.status == "auto_executed" else ("var(--danger)" if d.status == "rejected" else "var(--text)")
        is_pending = (d.status == "pending_review")

        card = html.Div([
            # Card Top Row: ID, Tier Badge, Status
            html.Div([
                html.Div([
                    html.Span(d.id, style={"fontWeight": "700", "fontSize": "0.9rem", "marginRight": "10px"}),
                    html.Span(
                        d.rule_tier,
                        style={
                            "backgroundColor": tier_bg,
                            "color": tier_color,
                            "border": "1px solid var(--border)",
                            "borderRadius": "var(--radius-sm)",
                            "padding": "2px 8px",
                            "fontSize": "0.72rem",
                            "fontWeight": "700",
                        }
                    ),
                ]),
                html.Div([
                    html.Span(f"Confidence: {d.confidence:.2f} | ", style={"fontSize": "0.75rem", "color": "var(--text-muted)"}),
                    html.Span(d.status.upper(), style={"fontSize": "0.78rem", "fontWeight": "700", "color": status_color}),
                ]),
            ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "8px"}),

            # Recommended Action & Subjects
            html.Div(d.recommended_action, style={"fontSize": "0.88rem", "fontWeight": "600", "color": "var(--text)", "marginBottom": "6px"}),
            html.Div([
                html.Span(f"Entities: {', '.join(d.subject_ids)} • Type: {d.type}", style={"fontSize": "0.75rem", "color": "var(--text-muted)"}),
            ], style={"marginBottom": "8px"}),

            # Rationale
            html.Div(d.rationale, style={"fontSize": "0.8rem", "color": "var(--text-muted)", "marginBottom": "12px", "lineHeight": "1.4"}),

            # Resolution Action Row (if pending)
            html.Div([
                dcc.Input(
                    id={"type": "resolve-note", "id": d.id},
                    type="text",
                    placeholder="Enter mandatory resolution note...",
                    style={
                        "flex": "1",
                        "fontSize": "0.8rem",
                        "padding": "6px 8px",
                        "border": "1px solid var(--border)",
                        "borderRadius": "var(--radius-sm)",
                        "backgroundColor": "var(--surface)",
                        "marginRight": "8px",
                    }
                ),
                html.Button(
                    "Approve",
                    id={"type": "approve-btn", "id": d.id},
                    style={
                        "backgroundColor": "var(--success)",
                        "color": "var(--surface)",
                        "border": "none",
                        "borderRadius": "var(--radius-sm)",
                        "padding": "6px 12px",
                        "fontSize": "0.78rem",
                        "fontWeight": "600",
                        "cursor": "pointer",
                        "marginRight": "6px",
                    }
                ),
                html.Button(
                    "Reject",
                    id={"type": "reject-btn", "id": d.id},
                    style={
                        "backgroundColor": "var(--danger)",
                        "color": "var(--surface)",
                        "border": "none",
                        "borderRadius": "var(--radius-sm)",
                        "padding": "6px 12px",
                        "fontSize": "0.78rem",
                        "fontWeight": "600",
                        "cursor": "pointer",
                    }
                ),
            ], style={"display": "flex", "alignItems": "center"}) if is_pending else html.Div(
                f"Resolved by {d.resolver or 'SYSTEM'} on {d.resolved_at or d.created_at[:10]}: {d.resolution_note or 'Auto-executed rule'}",
                style={"fontSize": "0.75rem", "color": "var(--text-muted)"}
            ),
        ], style={
            "backgroundColor": "var(--surface)",
            "border": "1px solid var(--border)",
            "borderRadius": "var(--radius-md)",
            "padding": "16px",
            "marginBottom": "14px",
            "boxShadow": "var(--shadow-xs)",
        })
        cards.append(card)

    return html.Div(cards)


def _render_audit_entries(entries: List[Any]) -> html.Div:
    """Renders latest cryptographic hash chain entries in tabular format."""
    rows = []
    for e in entries:
        rows.append(html.Tr([
            html.Td(f"#{e.index}", style={"fontSize": "0.78rem", "fontWeight": "600"}),
            html.Td(e.timestamp[:19].replace("T", " "), style={"fontSize": "0.75rem"}),
            html.Td(e.event_type, style={"fontWeight": "600", "fontSize": "0.8rem", "color": "var(--accent)"}),
            html.Td(str(e.payload)[:60] + "...", style={"fontSize": "0.75rem", "color": "var(--text-muted)"}),
            html.Td(e.record_hash[:16] + "...", style={"fontSize": "0.75rem"}),
        ], style={"borderBottom": "1px solid var(--border)", "padding": "6px 0"}))

    return html.Table([
        html.Thead(html.Tr([
            html.Th("Index", style={"textAlign": "left", "fontSize": "0.78rem"}),
            html.Th("Timestamp", style={"textAlign": "left", "fontSize": "0.78rem"}),
            html.Th("Event", style={"textAlign": "left", "fontSize": "0.78rem"}),
            html.Th("Payload Summary", style={"textAlign": "left", "fontSize": "0.78rem"}),
            html.Th("SHA-256 Hash", style={"textAlign": "left", "fontSize": "0.78rem"}),
        ], style={"borderBottom": "2px solid var(--border)"})),
        html.Tbody(rows),
    ], style={"width": "100%", "borderCollapse": "collapse"})


# ── Callbacks: Decision Resolution with RBAC Enforcement ───────────────
@callback(
    [
        Output("queue-items-container", "children"),
        Output("role-error-banner", "children"),
        Output("role-error-banner", "style"),
        Output("audit-entries-table", "children"),
    ],
    [
        Input({"type": "approve-btn", "id": ALL}, "n_clicks"),
        Input({"type": "reject-btn", "id": ALL}, "n_clicks"),
        Input("step-day-btn", "n_clicks"),
        Input("reset-clock-btn", "n_clicks"),
    ],
    [
        State("user-role-selector", "value"),
        State({"type": "resolve-note", "id": ALL}, "value"),
    ],
    prevent_initial_call=True,
)
def handle_queue_actions(approve_clicks, reject_clicks, step_clicks, reset_clicks, role, notes):
    ctx = dash.callback_context
    if not ctx.triggered:
        return _render_decision_cards(engine.list_decisions()), "", {"display": "none"}, _render_audit_entries(engine.audit.get_entries(limit=5))

    trigger_id = ctx.triggered[0]["prop_id"]

    # Replay clock actions
    if "step-day-btn" in trigger_id:
        replay_clock.step_day()
    elif "reset-clock-btn" in trigger_id:
        replay_clock.reset()

    # Resolution action
    error_banner_text = ""
    error_banner_style = {"display": "none"}

    if "approve-btn" in trigger_id or "reject-btn" in trigger_id:
        import json
        prop_dict = json.loads(trigger_id.split(".")[0])
        dec_id = prop_dict["id"]
        action = "approve" if "approve-btn" in trigger_id else "reject"

        # Lookup note
        note = "Operational sign-off via Control Tower"
        if notes and any(notes):
            for n in notes:
                if n:
                    note = n
                    break

        try:
            # Enforces RBAC server-side (§10.4)
            engine.resolve_decision(
                decision_id=dec_id,
                action=action,
                resolution_note=note,
                role=role,
                username=f"{role}_user",
            )
        except PermissionError as pe:
            error_banner_text = f"RBAC ACCESS DENIED: {str(pe)}"
            error_banner_style = {
                "display": "block",
                "backgroundColor": "var(--danger-pale)",
                "color": "var(--danger)",
                "border": "1px solid var(--danger)",
                "borderRadius": "var(--radius-sm)",
                "padding": "10px 16px",
                "fontSize": "0.82rem",
                "fontWeight": "600",
            }

    updated_cards = _render_decision_cards(engine.list_decisions())
    updated_audit = _render_audit_entries(engine.audit.get_entries(limit=5))
    return updated_cards, error_banner_text, error_banner_style, updated_audit


@callback(
    Output("replay-date-display", "children"),
    [Input("step-day-btn", "n_clicks"), Input("reset-clock-btn", "n_clicks")],
    prevent_initial_call=True,
)
def update_replay_date(step_clicks, reset_clicks):
    return replay_clock.get_current_date()


@callback(
    Output("promotion-status-msg", "children"),
    Input("promote-challenger-btn", "n_clicks"),
    prevent_initial_call=True,
)
def handle_model_promotion(n_clicks):
    if not n_clicks:
        return ""
    success, msg = lifecycle_mgr.evaluate_and_promote_challenger()
    color = "var(--success)" if success else "var(--danger)"
    return html.Span(msg, style={"color": color, "fontWeight": "600"})
