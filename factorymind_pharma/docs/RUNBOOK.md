# FactoryMind AI (Pharma) — Runbook

> Phase 6 handover artifact · §13.2

---

## 1. Start / Stop

### Full stack (Docker Compose)
```bash
# Start all services + dashboard
docker compose up -d

# Check health
curl http://localhost:8000/healthz
curl http://localhost:8501/  # Dash dashboard

# Stop
docker compose down

# Stop and wipe volumes (decisions, audit log)
docker compose down -v
```

### Local development (no Docker)
```bash
cd factorymind_pharma

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI service (port 8000)
uvicorn factorymind_pharma.services.tower_svc.api:app --host 0.0.0.0 --port 8000 --reload

# Start the Dash dashboard (port 8501) — separate terminal
python -m factorymind_pharma.dashboard.app
```

### Environment variables required
```bash
GROQ_API_KEY=<your-groq-api-key>       # Copilot LLM (optional: falls back to deterministic)
DATA_DIR=/path/to/data                  # Directory containing the 15 CSV source files
MODEL_DIR=/path/to/factorymind_pharma/models
```

---

## 2. Replay Clock

The dataset ends 2022-12-31. The replay clock simulates "real time" by advancing day-by-day over the held-out period (final 10%).

```bash
# Check current replay position
curl http://localhost:8000/v1/replay/status

# Advance one day (runs feedback, drift check)
curl -X POST http://localhost:8000/v1/replay/step

# Dashboard 11 provides play/pause/step controls in the UI
```

The clock resets to the start of the held-out window on service restart (in-memory state). To persist, write `replay_clock.current_date` to the SQLite DB before shutdown (future enhancement).

---

## 3. Retrain

Retraining is triggered automatically when drift is detected. To trigger manually:

```bash
# 1. Check current drift metrics
curl http://localhost:8000/v1/replay/status | python -m json.tool

# 2. Retrain involves re-running run_phase2.py with a new training window:
cd factorymind_pharma
python run_phase2.py --retrain --window-end 2022-11-15

# 3. The retrained model is saved as a challenger in models/
# It is NOT the champion until promoted.
```

---

## 4. Champion/Challenger Promotion

```bash
# Check current registry
curl http://localhost:8000/v1/models/registry | python -m json.tool

# Attempt to promote challenger (runs WAPE gate + bias gate automatically)
# Promotion requires: challenger WAPE improvement >= 2% AND bias <= threshold
curl -X POST "http://localhost:8000/v1/models/promote?challenger_id=ensemble-challenger-v2"

# Response: {"success": true, "message": "..."} or HTTP 400 with reason
```

**Promotion is logged to the audit chain automatically.** Champion is never overwritten in place — both files remain in `models/`.

---

## 5. Rollback

```bash
# Identify previous champion from audit log
curl http://localhost:8000/v1/audit/entries?limit=20 | python -m json.tool

# Manually update registry.json to re-promote previous champion:
# Edit models/registry.json: set previous champion status back to "champion"
# and current champion to "retired"
# Then restart the service — it reads registry.json on startup.
```

---

## 6. Audit Chain Verification

```bash
# Verify full chain integrity (recomputes all SHA-256 hashes)
curl http://localhost:8000/v1/audit/verify | python -m json.tool

# Expected healthy response:
# {"is_valid": true, "first_broken_index": null, "total_entries": N, "chain_head_hash": "..."}

# If broken:
# {"is_valid": false, "first_broken_index": 42, ...}
# → Entry 42 was tampered with. Restore from backup or investigate.

# View recent audit entries
curl "http://localhost:8000/v1/audit/entries?limit=50" | python -m json.tool
```

---

## 7. Rotate Secrets

```bash
# The only external secret is GROQ_API_KEY.
# To rotate:
# 1. Generate a new key at https://console.groq.com/
# 2. Update the environment variable (or .env file if used)
# 3. Restart the service — no code changes needed

# If using Docker Compose:
docker compose down
export GROQ_API_KEY=<new-key>
docker compose up -d
```

> Never commit API keys to source control. The codebase reads them from environment variables only.

---

## 8. Run Tests

```bash
cd factorymind_pharma

# Full test suite
pytest tests/ -v

# Data-facts tests only (C1–C8 assertions)
pytest tests/test_data_facts.py -v

# Copilot eval (deterministic, no API key required)
pytest tests/test_copilot_eval.py -v

# Phase 6 orchestration (RBAC, tamper detection, rule tiers)
pytest tests/test_phase6_orchestration.py -v

# Acceptance suite
pytest tests/acceptance/ -v --tb=short

# CI equivalent (lint + tests + metric gate)
flake8 factorymind_pharma/ --max-line-length=120
pytest tests/ --tb=short -q
```

---

## 9. Decision Queue Operations

```bash
# List all pending decisions
curl "http://localhost:8000/v1/decisions?status=pending_review" | python -m json.tool

# Resolve a decision (as qa_reviewer)
curl -X POST "http://localhost:8000/v1/decisions/{decision_id}/resolve" \
  -H "Content-Type: application/json" \
  -d '{"action": "approved", "resolution_note": "Reviewed and approved", "role": "qa_reviewer", "username": "analyst1"}'

# Note: BLOCK-tier decisions require qa_reviewer role. Planners receive HTTP 403.
```

---

## 10. Monitoring

- All API logs go to stdout in structured JSON with correlation IDs.
- Dashboard 11 shows: approval queue, live decision feed, model health sparklines, audit-chain status badge, replay clock controls.
- `/healthz` — liveness probe (returns 200 if service is running)
- `/readyz` — readiness probe (returns 200 if data and audit chain are connected)
