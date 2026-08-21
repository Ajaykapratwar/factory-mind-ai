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
