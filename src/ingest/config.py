"""Configuration for the Phase 2 ETL pipeline.

Edit values here or override from the CLI (`python src/ingest/etl.py --help`).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

YEARS = ["2021", "2022", "2023", "2024", "2025"]

SESSIONS: list[str] | None = None

WRITE_BRONZE = True

COMPRESSION = "zstd"

N_WORKERS = min(os.cpu_count() or 4, 8)

BRONZE_DIR = ROOT / "data" / "bronze" / "telemetry"
SILVER_DIR = ROOT / "data" / "silver"

NON_GP = {".git", ".github", ".devcontainer", "cache", "cache_joblib",
          "cache_preseason", "fastf1_cache", "__pycache__"}

SESSION_LEVEL_FILES = {"weather.json", "corners.json", "rcm.json",
                       "session_laptimes.json", "drivers.json"}
