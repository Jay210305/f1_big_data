"""DuckDB read layer for the Bronze/Silver lakehouse.

Returns a DuckDB connection with views registered over the Hive-partitioned
Parquet files so downstream phases (features, models, eval) can query with SQL.
"""
from __future__ import annotations

import duckdb
from pathlib import Path

from . import config as C


def _glob(base: Path) -> str:
    return str(base).replace("\\", "/") + "/**/*.parquet"


def open_lakehouse(base: Path | None = None) -> duckdb.DuckDBPyConnection:
    base = base or C.ROOT / "data"
    con = duckdb.connect()
    queries = {
        "telemetry": C.BRONZE_DIR,
        "stint_features": C.SILVER_DIR / "stint_features",
        "weather": C.SILVER_DIR / "race_events" / "weather",
        "corners": C.SILVER_DIR / "race_events" / "corners",
        "rcm": C.SILVER_DIR / "race_events" / "rcm",
    }
    for name, path in queries.items():
        g = _glob(path)
        con.execute(
            f"CREATE OR REPLACE VIEW {name} AS "
            f"SELECT * FROM read_parquet('{g}', hive_partitioning=1)"
        )
    return con
