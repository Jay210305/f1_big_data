"""Phase 2 verification: row counts, on-disk footprint, and a DuckDB read test.

Run after the ETL completes:
    python src/ingest/verify_lakehouse.py
"""
from __future__ import annotations

from pathlib import Path

from . import config as C
from .duckdb_layer import open_lakehouse


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*.parquet"))


def _fmt(n: int) -> str:
    return f"{n / (1024 ** 3):.2f} GB" if n else "0 GB"


def main() -> None:
    con = open_lakehouse()
    print("=" * 70)
    print("Phase 2 — Lakehouse verification")
    print("=" * 70)

    for view in ["telemetry", "stint_features", "weather", "corners", "rcm"]:
        try:
            n = con.execute(f"SELECT count(*) FROM {view}").fetchone()[0]
            print(f"  {view:18s} rows = {n:,}")
        except Exception as exc:
            print(f"  {view:18s} ERROR: {exc}")

    print("\nTelemetry rows per season:")
    try:
        for season, n in con.execute(
                "SELECT season, count(*) FROM telemetry "
                "GROUP BY season ORDER BY season").fetchall():
            print(f"  {season}: {n:,}")
    except Exception as exc:
        print(f"  ERROR: {exc}")

    print("\nStint features per season:")
    try:
        for season, n in con.execute(
                "SELECT season, count(*) FROM stint_features "
                "GROUP BY season ORDER BY season").fetchall():
            print(f"  {season}: {n:,}")
    except Exception as exc:
        print(f"  ERROR: {exc}")

    print("\nOn-disk footprint (Parquet only):")
    for label, path in [
        ("bronze/telemetry", C.BRONZE_DIR),
        ("silver/stint_features", C.SILVER_DIR / "stint_features"),
        ("silver/race_events", C.SILVER_DIR / "race_events"),
    ]:
        sz = _dir_size(path)
        print(f"  {label:24s} {_fmt(sz)}")

    total = _dir_size(C.ROOT / "data")
    print(f"  {'TOTAL data/':24s} {_fmt(total)}")

    print("\nSample query (top speeds, silver):")
    try:
        df = con.execute(
            "SELECT season, gp, driver, lap, compound, stint, life, "
            "laptime, speed_max, dirty_air_frac "
            "FROM stint_features ORDER BY speed_max DESC LIMIT 5"
        ).fetchdf()
        print(df.to_string(index=False))
    except Exception as exc:
        print(f"  ERROR: {exc}")

    print("\n" + "=" * 70)
    print("If row counts look right and the sample query returned rows -> GO.")
    print("=" * 70)


if __name__ == "__main__":
    main()
