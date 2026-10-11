# ASSUMPTIONS.md — FactoryMind Pharma Phase 0–1

Recorded per spec §3 and any additional decisions made during build.

## §3 Pre-Build Clarification Resolutions

### 3.1 Timezone/Date Handling
All dates loaded as UTC-naive (`pd.to_datetime(...)`). No TZ-aware conversions introduced anywhere in the pipeline. Confirmed: all 15 CSVs use plain date strings (no timezone offset).

### 3.2 Missing Values
- `ordered_quantity` and `closing_stock` nulls: forward-filled per `product_id` within the chronological sort order.
- A boolean `<col>_imputed` flag column is added for each imputed column (not silent overwrite).
- After forward-fill, any remaining NaN at the very start of a product's history (nothing to fill from) is set to 0 for numeric columns; this affects only the first few rows per product.

### 3.3 Warehouse Granularity
`05_inventory.csv` has `warehouse_id`; `02_sales_demand.csv` does not.
- **Master feature table** aggregates inventory across warehouses to the `(date, product_id)` grain using `sum`/`max`/`mean` (no warehouse join forced onto sales).
- **Operational dashboard** reads raw inventory directly to display per-warehouse rows.
- Sales remain at region grain throughout.

### 3.4 Currency/Unit
All monetary fields treated as a single generic unit. Displayed with `¤` prefix — no USD/INR symbol assumed.

## Additional Decisions

### Data Join
Left join anchor = `sales_demand` (densest table with ~35K rows). All other tables are left-joined onto it so the output row count equals the sales_demand row count exactly.

### Feature Columns
As specified in §4 Step 1.2:
- `day_of_week`, `month`, `lag_7`, `lag_28`, `rolling_mean_7`, `rolling_mean_28` of `units_sold`
- From demand_factors: `market_demand_index`, `disease_season_index`, `promotion_flag`, `competitor_price`

### Categorical Encoding
`region`, `channel`, `category`, `dosage_form`, `criticality` label-encoded at train time.

### LightGBM Early Stopping
50-round patience on val set to prevent overfitting. Verbosity set to -1 (silent).

### Stub Files (07–15)
All 9 Phase 2+ files have schema validated (columns confirmed present) but no features derived. Schema surprises found: none — all schemas match expectations. See CHANGELOG for details.


## Phase 2–3 Clarifications

### Warehouse Hierarchy (§4.1)
- Checked 05_inventory.csv for hierarchy fields (type/tier/level columns).
- Hierarchy found: NO — optimizing per-warehouse independently (noted simplification as per spec)

### Compliance Field Values — CORRECTED by C1 (Phase 4-6 spec §1)
- audit_status actual values (verified against 14_compliance.csv): Compliant (80.3%), Minor Deviation (16.9%), Major Deviation (2.9%). There is NO 'Passed' value.
- The Phase 3 gate rule `audit_status != 'Passed'` would have blocked every batch (stale assumption). 
- RESOLUTION: Replaced by the tiered gate in spec §10.2 (BLOCK=Major Deviation, REVIEW=Minor Deviation, AUTO-ELIGIBLE=Compliant). Applied in services/policy_svc/inventory_optimizer.py and services/tower_svc/ (Phase 6).
- deviation_count ranges 0-7 and is often >= 1 on Compliant rows (not a reliable sole gate indicator).

### Lead Time (§4.1)
- No explicit lead_time column found in inventory or orders data.
- Defaulted to 7 days for all SKUs (noted simplification).
- This can be updated once lead_time data is available.

### QC Features (§3.1) — CORRECTED by C8 (Phase 4-6 spec §1)
- QC numeric features used: assay_pct, dissolution_pct, impurity_level (available in 08_quality_control.csv).
- release_status actual values: PASS (81.44%), RETEST (18.06%), FAIL (0.50%, n=175).
- FAIL is too rare for reliable supervised classifier recall measurement.
- RESOLUTION: Use RETEST union FAIL as the 'non-pass' evaluation label. Report FAIL recall separately with its tiny support (n=175) stated explicitly.

### Maintenance Target (§3.2)
- maintenance_flag derived from maintenance event records within 7-day window if not directly available.
- Time-based split: first 80% dates -> train; last 20% -> test.


## Phase 4-6 Carry-Forward Corrections (spec §1, verified against raw CSVs)

### C1 — audit_status values
- Verified: 14_compliance.audit_status is Compliant / Minor Deviation / Major Deviation (no 'Passed').
- Measured: Compliant 80.3%, Minor Deviation 16.9%, Major Deviation 2.9%.
- deviation_count range: 0–7; ~63% of Compliant rows have deviation_count >= 1.
- Resolution applied: tiered gate (spec §10.2): BLOCK=Major Deviation, REVIEW=Minor Deviation OR deviation_count>=2, AUTO-ELIGIBLE=Compliant AND deviation_count<2.

### C2 — stockout_flag all zeros
- Verified: 05_inventory.stockout_flag = 0 on all 35,000 rows.
- Historical stockout count KPI = 0; ledger-replay backtest is impossible.
- Resolution applied: derived metric stockout_risk_event = closing_stock <= 0 OR closing_stock < safety_stock. Reported on dashboards as "derived". Phase 3 exit criteria re-measured via twin (§4.8).

### C3 — ledger identity failure
- Verified: opening + received + production_received - units_sold = closing_stock holds on 0.0% of rows. Rows are independently sampled.
- Resolution applied: twin validated distributionally (spec §4.7) — P10-P90 coverage and yield/downtime proximity — not by ledger replay.

### C4 — non-informative topology
- Verified: every product appears on all 10 production lines; every machine appears under all 10 lines. No true machine->line or product->line mapping exists.
- Resolution applied: topology defined by config assumption in configs/twin.yaml (lines L01–L10, 4 machines per line round-robin, all products on all lines). Tag: source=assumed.

### C5 — duplicate keys
- Verified: 15_financials has 7,204 duplicate (date, product_id) rows; 05_inventory has 794 duplicate (date, product_id, warehouse_id) rows.
- Resolution applied: financials aggregated by sum on (date, product_id) before any join. Inventory flow columns summed, closing_stock takes last value, was_duplicated flag added.

### C6 — profit identity and negative profit
- Verified: profit = revenue - sum(production_cost, raw_material_cost, labor_cost, energy_cost, storage_cost, logistics_cost) holds on 100% of rows.
- 48.1% of rows have negative profit. production_cost is NOT the sum of the other cost columns.
- Resolution applied: identity used as-is; negatives never clipped. Dashboard 6 surfaces loss-making share (approx 48%) as a top-level KPI.

### C7 — sales coverage and R2 grain
- Verified: 02_sales_demand has 35,000 rows, 27,908 unique (date, product_id) pairs vs. approx 73,000 possible product-days (approx 48% coverage).
- Product-mean-only R2 approx 0.29; lag-1/lag-7 correlation approx 0.33.
- Resolution applied: evaluation_report reports R2 at row, product-day, and product-week grain. Phase 2 R2 >= 0.80 gate grain confirmed with panel; grain stated explicitly in every R2 figure.

### C8 — QC FAIL is rare
- Verified: PASS 81.44% (n=28,503), RETEST 18.06% (n=6,322), FAIL 0.50% (n=175).
- Resolution applied: RETEST union FAIL used as non-pass evaluation label. FAIL recall reported separately with support n=175 stated. No supervised FAIL-only classifier built.

