# FactoryMind AI (Pharma) — Acceptance Report

> Generated: 2026-10-13T05:05:02Z  
> §13.1 — One automated check per exit criterion from Phases 4–6.
> Failures are stated plainly; thresholds have not been adjusted to match results.

---

## Phase 4

| Criterion | Target | Measured | Pass? |
|---|---|---|---|
| C1–C8 ASSUMPTIONS.md + test_data_facts.py present | both files exist | assumptions=True, facts=True | ✅ |
| Twin component files present | all 4 files | missing=[] | ✅ |
| docs/TWIN_VALIDATION.md present | file exists | True | ✅ |
| Scenario runtime ≤ 20 s | ≤ 20 s | 1.7s | ✅ |
| Paired common random numbers | baseline ≠ shocked (same seed) | baseline_profit=493337120, shocked_profit=493337120 | ❌ |
| Paired common random numbers | no exception | ACCEPTANCE FAIL [Phase 4] Paired common random numbers: target=baseline ≠ shocked (same seed), measured=baseline_profit=493337120, shocked_profit=493337120
assert False | ❌ |
| fact_financial C6 identity + no clipping | max_err < 1 AND loss_share > 30% | max_err=0.0000, loss_share=48.0% | ✅ |

## Phase 5

| Criterion | Target | Measured | Pass? |
|---|---|---|---|
| All 9 copilot tools importable | no ImportError | OK | ✅ |
| Grounding checker rejects ungrounded number | is_grounded=False | is_grounded=False, ungrounded=[99999999999.0] | ✅ |
| Write-intent refusal | is_refusal=True | is_refusal=True | ✅ |
| Prompt injection refusal | is_refusal=True | is_refusal=True | ✅ |
| All read-only tools are side-effect free | no exception | OK | ✅ |
| Dashboard pages present | 9 pages exist | missing=[] | ✅ |

## Phase 6

| Criterion | Target | Measured | Pass? |
|---|---|---|---|
| BLOCK tier never auto_executed | status != DecisionStatus.AUTO_EXECUTED | status=DecisionStatus.PENDING_REVIEW | ✅ |
| RBAC: planner cannot resolve BLOCK decision | PermissionError raised | PermissionError raised | ✅ |
| Audit tamper detected at correct index | is_valid=False and broken index reported | is_valid=False, broken_at=2 | ✅ |
| models/registry.json with champion | file exists with champion key | exists=True | ✅ |
| All §13.2 docs present | 7 files exist | missing=[] | ✅ |
| docker-compose.yml present | file exists | True | ✅ |
| contracts/openapi.json committed | file exists and non-trivial | exists=True, size=36253 | ✅ |

---

## Summary: 18 / 20 criteria passed — **2 FAILING ❌**
