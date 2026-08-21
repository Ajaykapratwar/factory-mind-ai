"""
sensing_svc — L1 Sensing: Ingestion, Validation & Normalization Service
Phase 0–1 scope: loads all 15 source CSVs, validates schemas/FK integrity,
and builds the master feature table.
"""

from .loader import load_all_files, validate_foreign_keys
from .feature_builder import build_master_feature_table

__all__ = [
    "load_all_files",
    "validate_foreign_keys",
    "build_master_feature_table",
]
