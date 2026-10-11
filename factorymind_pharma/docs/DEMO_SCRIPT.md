# FactoryMind AI (Pharma) — Demo Script

> Phase 6 handover artifact · §13.2
> **Scripted walkthrough: ~6 minutes**
> Audience: stakeholder / evaluator with Dashboard 11 open and the API running.

---

## Pre-flight (before starting the demo)

```bash
docker compose up -d
# Wait ~30 s for healthy status
curl http://localhost:8000/healthz
```
Open browser → `http://localhost:8501` → Dashboard 1 (Executive Overview).

---

## Scene 1 — Context: What is FactoryMind? (45 s)

> "FactoryMind AI is a five-layer decision-support platform for pharmaceutical manufacturing. It ingests 15 operational data sources, runs a stacked forecast ensemble, a digital twin for what-if simulation, policy gates, and a conversational AI copilot. Everything feeds a control tower that queues decisions for human review."

Point to Dashboard 1 (Executive Overview). Highlight:
- **Total Revenue** and **Net Profit**
- **Loss-Making Share: ~48%** — this is real in the data; we never hide it (C6)
- **QC Pass Rate ~81%**

---

## Scene 2 — Replay a Day (60 s)

Navigate to **Dashboard 11 (Control Tower)**. Point to the **Replay Clock** panel.

> "The dataset ends 2022-12-31. We simulate real time by advancing day-by-day over the held-out window."

Click **Step Day** once.

> "Day advanced. The system evaluated today's forecasts against actuals, updated rolling WAPE and bias, and checked for drift."

Point to **Model Health** sparklines:
> "Champion WAPE is 20.2% at product-day grain. The challenger sits at 17.8% — meeting our ≥2% improvement gate. Drift hasn't triggered yet."

---

## Scene 3 — QC Anomaly → BLOCK Decision (75 s)

Navigate to **Dashboard 7 (Supplier & QC)**.

> "Our Isolation Forest model flagged batch B100212. Let's ask the copilot why."

Open the **Copilot panel** (right side). Type:
```
Why did batch B100212 fail QC?
```

> "The copilot called `explain_batch_quality`. It returns fields outside the empirical P1–P99 range of PASS batches — note the disclaimer: these are *derived* reference ranges, not regulatory spec limits."

Show the provenance strip under the answer.

Back on Dashboard 11, point to the **Approval Queue**:
> "That batch generated a BLOCK-tier decision because `release_status = FAIL`. It cannot be auto-approved — only a QA Reviewer can resolve it."

Attempt to resolve as `planner` (show the 403 error in the queue UI or API):
> "Planners are blocked by RBAC — the decision is gated to the qa_reviewer role."

---

## Scene 4 — Scenario Run + Mitigation (75 s)

Navigate to **Dashboard 10 (Scenario Lab)**.

> "Suppose supplier S003 is 5 days late — what's the financial impact?"

In the scenario builder, add:
- Shock type: **SupplierDelay** · Supplier: S003 · Extra days: 5 · Start day: 1

Click **Run Scenario**.

> "The digital twin runs 500 stochastic paths over 30 days using common random numbers — baseline and shocked share the same seeds so differences are paired, not noise."

Once results appear, point to:
- **Fill-rate fan chart**: P5/P50/P95 — shocked vs. baseline
- **Profit delta**: expected loss due to the delay
- **Ranked mitigations**: expedite raw material saves ~X units at cost Y

> "The twin priced every outcome using the same financial model that generated Dashboard 6. All cost parameters tagged `assumed` are listed in the Assumptions drawer."

---

## Scene 5 — Copilot Question on Financials (45 s)

Still in the Copilot panel. Type:
```
What are the financial cost lines for the Cardiovascular category?
```

> "Copilot calls `get_financials`, retrieves the P&L breakdown, and narrates only the numbers in the tool result. The grounding checker post-processes every response — any number not traceable to a tool result causes the answer to be regenerated or refused."

Point to the **provenance strip**: "Based on: fact_financial | As-of: 2022-12-31"

---

## Scene 6 — Audit Chain Verification (30 s)

Navigate to **Dashboard 11**, **Audit Chain** card.

> "Every decision creation, resolution, model promotion, and rule change is written to a SHA-256 hash-chained log. Let's verify."

Click **Verify Chain**:
> "Green badge — chain intact from genesis to present. If any record were edited, the broken index would be reported here. This is an engineering integrity control; we do not claim it meets any regulatory standard."

---

## Scene 7 — Wrap-Up (15 s)

> "That's the full loop: sensing → forecast → twin simulation → policy gate → human decision → audit. The copilot answers any operational question grounded in the same data. 11 dashboards, 5 services, one token file — no hardcoded colours, no ungrounded numbers."

---

## Quick Reference — Key Numbers to Know

| Metric | Value | Source |
|---|---|---|
| Loss-making share | ~48.1% | C6 / `fact_financial` |
| Champion WAPE | 20.2% (product-day) | `models/registry.json` |
| Challenger WAPE | 17.8% | `models/registry.json` |
| QC FAIL batches | n = 175 (0.5%) | C8 |
| Cold-chain excursion rate | ~4.0% | `13_logistics` empirical |
| Twin paths default | 500 | `configs/twin.yaml` |
| Scenario runtime budget | ≤ 20 s | §4.9 |
