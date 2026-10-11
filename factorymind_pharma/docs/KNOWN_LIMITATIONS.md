# FactoryMind AI (Pharma) — Known Limitations

> Phase 6 handover artifact · §13.2
> **This system is built on synthetic data for demonstration purposes. It is not suitable for real pharmaceutical manufacturing decisions, clinical use, or regulatory submission.**

---

## L1: No Observed Stockout Events (C2)

`05_inventory.stockout_flag` is 0 on all 35,000 rows. Historical stockout analysis is impossible on the raw data. All stockout-risk metrics displayed on dashboards (labelled "derived") are computed as:

```
stockout_risk_event = closing_stock <= 0 OR closing_stock < safety_stock
```

Simulated stockout figures (Phase 3 re-measure, Dashboard 4) are produced by the digital twin — they are **simulated, not observed**. This is stated explicitly wherever these figures appear.

**Implication:** Phase 3 exit criteria #1–#2 (stockout reduction, fill rate) cannot be validated against historical truth. They are validated distributionally via the twin (§4.8).

---

## L2: Inventory Ledger Does Not Close (C3)

The inventory identity `opening + received + production_received − units_sold = closing_stock` holds on 0% of rows. Rows appear to be independently sampled rather than a true time-series ledger. The digital twin therefore **cannot** be validated by ledger replay; it is validated distributionally (§4.7). The twin cannot guarantee its stock levels are historically accurate — only that marginal distributions of demand, yield, and downtime are consistent with the source data.

---

## L3: Non-Informative Production Topology (C4)

Every product appears on all 10 production lines and every machine appears under all 10 lines in the raw data. No true machine→line or product→line mapping can be inferred. The twin uses a **configuration assumption** (`configs/twin.yaml`): lines L01–L10, machines M001–M040 assigned round-robin (4 per line). This is documented as `source: assumed`. Simulation results depend on this assumption and would change with a real topology.

---

## L4: Negative Profit is Real (C6)

48.1% of product-day rows in `15_financials` have negative profit. This is the actual data; it is not an error and is not clipped or corrected. The P&L identity `profit = revenue − (production_cost + raw_material_cost + labor_cost + energy_cost + storage_cost + logistics_cost)` holds on 100% of rows. The system presents the loss-making share prominently (Dashboard 6, KPI). Any financial projections derived from this data inherit this high variance and should be treated as directional, not precise.

---

## L5: Demand Forecast R² at Row Grain (C7)

`02_sales_demand` covers only ≈ 48% of possible product-days. Product-mean-only R² ≈ 0.29; lag-1/7 correlation ≈ 0.33. The ensemble achieves R² 0.65 at row (product-day) grain. The Phase 2 R² ≥ 0.80 bar is met at **product-week** grain. Any R² figure shown in dashboards or reports states its grain explicitly. Forecasts for low-volume or infrequent SKUs are less reliable.

---

## L6: FAIL Batch Label is Rare (C8)

FAIL batches represent 0.5% of QC records (n = 175). The anomaly detector is trained and evaluated on `RETEST ∪ FAIL` as "non-pass." FAIL recall is reported separately with its support (n = 175) stated. The model is not a reliable FAIL-specific classifier due to extreme class imbalance. It should be used as a screening tool, not as a release decision.

---

## L7: Audit Log is Not Regulatory Compliance

The SHA-256 hash-chained audit log provides engineering integrity traceability. It is **not** described as or claimed to meet any regulatory standard (e.g., 21 CFR Part 11, EU Annex 11, or similar). No validation documentation, user access controls, or audit trail requirements from those frameworks have been implemented.

---

## L8: Copilot is Decision Support Only

The AI Copilot does not have access to batch release controls, regulatory systems, or any write operations. Its answers are grounded only in tool results from this session's data snapshot. It is not a substitute for qualified pharmaceutical expertise, QA review, or regulatory judgment. All copilot answers carry the statement: "decision support, not a release decision."

---

## L9: Modelled Cost Estimates are Not Actuals

Cost of Poor Quality, Downtime Cost, and Supplier Risk Cost (Dashboard 6) are **modelled estimates**, not observed accounting figures. They depend on assumed parameters (holding-cost proxy, line throughput, batch value proxy) documented in `configs/twin.yaml` and in the dashboard tooltip. Small changes in assumptions produce materially different numbers.

---

## L10: Synthetic Data — Not for Real Pharma Use

All 15 source CSV files are synthetically generated for demonstration. Product names, batch IDs, supplier names, machine IDs, and all numeric values are fictitious. **This system must not be used for any real pharmaceutical manufacturing, clinical, or regulatory purpose.** Results would be meaningless and potentially misleading when applied to actual operations without retraining on real data and appropriate validation.
