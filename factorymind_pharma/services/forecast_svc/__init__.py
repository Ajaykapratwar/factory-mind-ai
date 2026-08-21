"""
forecast_svc — Baseline Forecast Service (Phase 1, Step 1.2)
Two models only (per spec §4):
  1. Naive seasonal  — last-year-same-day-of-week per product_id.
  2. LightGBM regression — on master_feature_table.

No ensembling, no quantile forecasting, no LSTM.
"""

from .trainer import run_forecast_pipeline

__all__ = ["run_forecast_pipeline"]
