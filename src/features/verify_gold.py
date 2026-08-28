"""Phase 3 verification: inspect the Gold feature store.

    python -m src.features.verify_gold
"""
from __future__ import annotations

import duckdb

from src.ingest.config import ROOT

GOLD = ROOT / "data" / "gold"


def _glob(name: str) -> str:
    return str(GOLD / f"{name}.parquet").replace("\\", "/")


def main() -> None:
    con = duckdb.connect()
    print("=" * 78)
    print("Phase 3 — Gold feature store verification")
    print("=" * 78)

    for name in ["telemetry_stats", "lap_features", "driver_style",
                 "stint_degradation", "environmental"]:
        path = GOLD / f"{name}.parquet"
        if not path.exists():
            print(f"  {name:22s} MISSING ({path})")
            continue
        n = con.execute(f"SELECT count(*) FROM read_parquet('{_glob(name)}')").fetchone()[0]
        print(f"  {name:22s} rows = {n:,}")

    print("\nLap features per season:")
    for season, n in con.execute(f"""
        SELECT season, count(*) FROM read_parquet('{_glob("lap_features")}')
        GROUP BY season ORDER BY season
    """).fetchall():
        print(f"  {season}: {n:,}")

    print("\nCompound distribution (lap_features):")
    for comp, n in con.execute(f"""
        SELECT compound, count(*) FROM read_parquet('{_glob("lap_features")}')
        GROUP BY compound ORDER BY count(*) DESC
    """).fetchall():
        print(f"  {comp}: {n:,}")

    print("\nNull check (key columns, lap_features):")
    for col, n in con.execute(f"""
        SELECT 'laptime', count(*) FROM read_parquet('{_glob("lap_features")}') WHERE laptime IS NULL
        UNION ALL
        SELECT 'compound', count(*) FROM read_parquet('{_glob("lap_features")}') WHERE compound IS NULL
        UNION ALL
        SELECT 'speed_mean', count(*) FROM read_parquet('{_glob("lap_features")}') WHERE speed_mean IS NULL
        UNION ALL
        SELECT 'track_temp_mean', count(*) FROM read_parquet('{_glob("lap_features")}') WHERE track_temp_mean IS NULL
    """).fetchall():
        print(f"  null {col}: {n:,}")

    print("\nSample: degradation curve for one stint (2025 Bahrain Race VER stint 1):")
    try:
        df = con.execute(f"""
            SELECT lap, life, laptime, laptime_delta_stint_start, cliff_flag,
                   track_temp_mean, dirty_air_frac
            FROM read_parquet('{_glob("lap_features")}')
            WHERE season='2025' AND gp='Bahrain Grand Prix' AND session='Race'
              AND driver='VER' AND stint=1
            ORDER BY lap LIMIT 15
        """).fetchdf()
        print(df.to_string(index=False))
    except Exception as exc:
        print(f"  ERROR: {exc}")

    print("\nSample: driver style (top 5 by lat_g_max_mean):")
    try:
        df = con.execute(f"""
            SELECT season, driver, lat_g_max_mean, brake_event_count_mean,
                   throttle_modulations_mean, dirty_air_frac_mean
            FROM read_parquet('{_glob("driver_style")}')
            ORDER BY lat_g_max_mean DESC LIMIT 5
        """).fetchdf()
        print(df.to_string(index=False))
    except Exception as exc:
        print(f"  ERROR: {exc}")

    print("\n" + "=" * 78)
    print("If row counts look right and the samples returned rows -> GO to Phase 4.")
    print("=" * 78)


if __name__ == "__main__":
    main()
