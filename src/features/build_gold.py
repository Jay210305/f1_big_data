"""Phase 3 — Build the Gold Feature Store.

Produces three deterministic Gold tables under data/gold/:

  1. lap_features.parquet      — one row per lap (master modeling table)
  2. driver_style.parquet      — one row per (season, driver) style profile
  3. stint_degradation.parquet — one row per stint degradation curve

Heavy per-lap telemetry stats are computed from the 325M-row Bronze via DuckDB
(out-of-core, per season to bound RAM). Degradation/environmental/style features
are computed with Polars over the small silver/lap table.

CLI:
    python -m src.features.build_gold                     # all years
    python -m src.features.build_gold --years 2025        # one season (test)
    python -m src.features.build_gold --skip-telemetry    # reuse existing telemetry_stats
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import duckdb
import polars as pl

from src.ingest.config import ROOT, SILVER_DIR, BRONZE_DIR

GOLD = ROOT / "data" / "gold"
GOLD.mkdir(parents=True, exist_ok=True)
YEARS = ["2021", "2022", "2023", "2024", "2025"]


# ---------------------------------------------------------------------------
# 1. Rich per-lap telemetry stats from Bronze (DuckDB, per season)
# ---------------------------------------------------------------------------
TELE_STATS_SQL = """
WITH t AS (
  SELECT season, gp, session, driver, lap, time, speed, throttle, brake,
         drs, gear, rpm, acc_x, acc_y, acc_z, distance, "DistanceToDriverAhead",
    CASE WHEN throttle > 0 AND coalesce(lag(throttle) OVER w, 0) <= 0
         THEN 1 ELSE 0 END AS thr_on,
    CASE WHEN brake > 0 AND coalesce(lag(brake) OVER w, 0) <= 0
         THEN 1 ELSE 0 END AS brake_on
  FROM read_parquet('{glob}', hive_partitioning=1)
  WINDOW w AS (PARTITION BY season, gp, session, driver, lap ORDER BY time)
)
SELECT
  season, gp, session, driver, lap,
  count(*)                                AS n_samples,
  avg(speed)                              AS speed_mean,
  max(speed)                              AS speed_max,
  min(speed)                              AS speed_min,
  stddev(speed)                           AS speed_std,
  avg(throttle)                           AS throttle_mean,
  stddev(throttle)                        AS throttle_std,
  avg(CASE WHEN throttle > 0 THEN 1.0 ELSE 0 END) AS throttle_active_frac,
  sum(thr_on)                             AS throttle_modulations,
  avg(CASE WHEN brake > 0 THEN 1.0 ELSE 0 END)    AS brake_frac,
  sum(brake_on)                           AS brake_event_count,
  avg(CASE WHEN brake > 0 THEN brake ELSE NULL END) AS brake_intensity,
  avg(CASE WHEN drs > 0 THEN 1.0 ELSE 0 END)      AS drs_frac,
  avg(gear)                               AS gear_mean,
  stddev(gear)                            AS gear_std,
  avg(rpm)                                AS rpm_mean,
  max(rpm)                                AS rpm_max,
  avg(abs(acc_x))                         AS lat_g_abs_mean,
  stddev(acc_x)                           AS lat_g_std,
  max(abs(acc_x))                         AS lat_g_max,
  avg(abs(acc_y))                         AS long_g_abs_mean,
  stddev(acc_y)                           AS long_g_std,
  max(abs(acc_y))                         AS long_g_max,
  avg(abs(acc_z))                         AS vert_g_abs_mean,
  avg(CASE WHEN "DistanceToDriverAhead" < 10 THEN 1.0 ELSE 0 END) AS dirty_air_frac,
  avg("DistanceToDriverAhead")            AS dirty_air_mean_dist,
  percentile_cont(0.1) WITHIN GROUP (ORDER BY "DistanceToDriverAhead") AS dirty_air_p10_dist,
  max(distance)                           AS lap_distance,
  avg(abs(acc_x)) * max(distance)         AS lat_load_integral,
  avg(abs(acc_y)) * max(distance)         AS long_load_integral
FROM t
GROUP BY season, gp, session, driver, lap
"""


def compute_telemetry_stats(years: list[str]) -> pl.DataFrame:
    print("[1/5] Computing rich per-lap telemetry stats from Bronze (DuckDB)...")
    frames = []
    con = duckdb.connect()
    for y in years:
        t0 = time.time()
        glob = str(BRONZE_DIR / f"season={y}" / "**" / "*.parquet").replace("\\", "/")
        df = con.execute(TELE_STATS_SQL.format(glob=glob)).fetchdf()
        frames.append(pl.from_pandas(df))
        print(f"   {y}: {len(df):,} laps  ({time.time()-t0:.0f}s)")
    stats = pl.concat(frames, how="diagonal_relaxed")
    out = GOLD / "telemetry_stats.parquet"
    stats.write_parquet(out, compression="zstd")
    print(f"   -> {out}  ({stats.shape[0]:,} x {stats.shape[1]})")
    return stats


# ---------------------------------------------------------------------------
# 2. Environmental features from weather (session-level)
# ---------------------------------------------------------------------------
def compute_environmental(years: list[str]) -> pl.DataFrame:
    print("[2/5] Computing environmental features from weather...")
    glob = str(SILVER_DIR / "race_events" / "weather" / "**" / "*.parquet").replace("\\", "/")
    yrs = ",".join(f"'{y}'" for y in years)
    con = duckdb.connect()
    df = con.execute(f"""
        SELECT season, gp, session,
               avg(wAT) AS ambient_temp_mean,
               max(wAT)  AS ambient_temp_max,
               avg(wTT) AS track_temp_mean,
               max(wTT)  AS track_temp_max,
               stddev(wTT) AS track_temp_std,
               max(CASE WHEN wR THEN 1 ELSE 0 END) AS rain_flag,
               avg(wWS) AS wind_speed_mean,
               avg(wWD) AS wind_dir_mean,
               count(*) AS weather_samples
        FROM read_parquet('{glob}', hive_partitioning=1)
        WHERE season IN ({yrs})
        GROUP BY season, gp, session
    """).fetchdf()
    env = pl.from_pandas(df)
    # track temperature slope: (last - first) over the session timeline
    env = env.with_columns([
        (pl.col("track_temp_max") - pl.col("track_temp_mean"))
            .alias("track_temp_range"),
    ])
    out = GOLD / "environmental.parquet"
    env.write_parquet(out, compression="zstd")
    print(f"   -> {out}  ({env.shape[0]:,} sessions)")
    return env


# ---------------------------------------------------------------------------
# 3. Degradation kinetics (Polars window within stint — no future leakage)
# ---------------------------------------------------------------------------
def add_degradation(lf: pl.LazyFrame) -> pl.LazyFrame:
    lf = lf.sort(["season", "gp", "session", "driver", "stint", "lap"])
    stint_keys = ["season", "gp", "session", "driver", "stint"]
    lf = lf.with_columns(
        pl.col("laptime").first().over(stint_keys).alias("stint_first_laptime"))
    lf = lf.with_columns([
        (pl.col("laptime") - pl.col("stint_first_laptime"))
            .alias("laptime_delta_stint_start"),
        (pl.col("laptime")
            - pl.col("laptime").shift(1).over(stint_keys))
            .alias("laptime_delta_prev"),
        (pl.col("s2") - pl.col("s1")).alias("sector_decay_21"),
        (pl.col("s3") - pl.col("s2")).alias("sector_decay_32"),
    ])
    lf = lf.with_columns(
        (pl.col("laptime_delta_stint_start") > 2.0).alias("cliff_flag"))
    return lf


# ---------------------------------------------------------------------------
# 4. Master lap_features table
# ---------------------------------------------------------------------------
def build_lap_features(stats: pl.DataFrame, env: pl.DataFrame,
                       years: list[str]) -> pl.DataFrame:
    print("[3/5] Building master lap_features table...")
    silver_glob = str(SILVER_DIR / "stint_features" / "**" / "*.parquet").replace("\\", "/")
    yrs = ",".join(f"'{y}'" for y in years)
    con = duckdb.connect()
    silver = pl.from_pandas(con.execute(f"""
        SELECT season, gp, session, driver, lap, stint, life, compound,
               laptime, s1, s2, s3
        FROM read_parquet('{silver_glob}', hive_partitioning=1)
        WHERE season IN ({yrs})
    """).fetchdf())

    # drop the lighter silver aggregates we are replacing with rich stats
    lf = silver.lazy()
    lf = add_degradation(lf)
    lap = lf.collect()

    keys = ["season", "gp", "session", "driver", "lap"]
    lap = lap.join(stats, on=keys, how="left")
    lap = lap.join(env, on=["season", "gp", "session"], how="left")

    out = GOLD / "lap_features.parquet"
    lap.write_parquet(out, compression="zstd")
    print(f"   -> {out}  ({lap.shape[0]:,} x {lap.shape[1]})")
    return lap


# ---------------------------------------------------------------------------
# 5. Driver style profile (per season-driver)
# ---------------------------------------------------------------------------
def build_driver_style(lap: pl.DataFrame) -> pl.DataFrame:
    print("[4/5] Building driver style profile...")
    style_cols = [
        "throttle_mean", "throttle_std", "throttle_modulations", "throttle_active_frac",
        "brake_frac", "brake_event_count", "brake_intensity",
        "speed_mean", "speed_std", "gear_mean", "gear_std", "rpm_mean",
        "lat_g_abs_mean", "lat_g_std", "lat_g_max",
        "long_g_abs_mean", "long_g_std", "long_g_max",
        "vert_g_abs_mean", "lat_load_integral", "long_load_integral",
        "dirty_air_frac",
    ]
    avail = [c for c in style_cols if c in lap.columns]
    g = lap.group_by(["season", "driver"]).agg([
        pl.col(c).mean().alias(c + "_mean") for c in avail
    ] + [
        pl.col(c).std().alias(c + "_std") for c in avail
    ])
    out = GOLD / "driver_style.parquet"
    g.write_parquet(out, compression="zstd")
    print(f"   -> {out}  ({g.shape[0]:,} driver-seasons)")
    return g


# ---------------------------------------------------------------------------
# 6. Stint degradation curves (per stint)
# ---------------------------------------------------------------------------
def build_stint_degradation(lap: pl.DataFrame) -> pl.DataFrame:
    print("[5/5] Building stint degradation curves...")
    stint_keys = ["season", "gp", "session", "driver", "stint"]
    g = lap.group_by(stint_keys).agg([
        pl.col("compound").first().alias("compound"),
        pl.len().alias("stint_length"),
        pl.col("life").max().alias("max_life"),
        pl.col("laptime").first().alias("pace_start"),
        pl.col("laptime").last().alias("pace_end"),
        pl.col("laptime_delta_stint_start").last().alias("pace_decay_end"),
        pl.col("cliff_flag").any().alias("had_cliff"),
        pl.col("cliff_flag").cum_sum().last().alias("cliff_laps"),  # noqa
        pl.col("laptime_delta_prev").mean().alias("mean_lap_delta"),
        pl.col("track_temp_mean").mean().alias("stint_track_temp"),
        pl.col("rain_flag").max().alias("stint_rain"),
        pl.col("lat_load_integral").mean().alias("stint_lat_load"),
        pl.col("long_load_integral").mean().alias("stint_long_load"),
    ])
    # degradation slope: (pace_end - pace_start) / (stint_length-1)
    g = g.with_columns(
        ((pl.col("pace_end") - pl.col("pace_start"))
         / (pl.col("stint_length") - 1)).alias("degradation_slope"))
    out = GOLD / "stint_degradation.parquet"
    g.write_parquet(out, compression="zstd")
    print(f"   -> {out}  ({g.shape[0]:,} stints)")
    return g


# ---------------------------------------------------------------------------
# 7. Feature dictionary manifest (machine-readable)
# ---------------------------------------------------------------------------
def write_manifest(lap, style, stint):
    def cols(df):
        return {c: str(df.schema[c]) for c in df.columns}
    manifest = {
        "tables": {
            "lap_features": {"rows": lap.shape[0], "columns": cols(lap)},
            "driver_style": {"rows": style.shape[0], "columns": cols(style)},
            "stint_degradation": {"rows": stint.shape[0], "columns": cols(stint)},
        },
        "leakage_notes": {
            "stint_length": "DO NOT use as RUL feature (it equals the label).",
            "max_life": "DO NOT use as RUL feature.",
            "pace_end / pace_decay_end / had_cliff": "stint-outcome fields; "
            "use for analysis/optimizer only, not per-lap RUL prediction.",
            "life": "tyre age so far = OK (known at prediction time).",
            "laptime_delta_*": "computed from current+past laps only = OK.",
        },
    }
    out = GOLD / "feature_dictionary.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"   manifest -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3: build Gold feature store")
    ap.add_argument("--years", nargs="*", default=YEARS)
    ap.add_argument("--skip-telemetry", action="store_true",
                    help="reuse existing data/gold/telemetry_stats.parquet")
    args = ap.parse_args()

    t0 = time.time()
    if args.skip_telemetry and (GOLD / "telemetry_stats.parquet").exists():
        print("[1/5] Reusing existing telemetry_stats.parquet")
        stats = pl.read_parquet(GOLD / "telemetry_stats.parquet")
    else:
        stats = compute_telemetry_stats(args.years)

    env = compute_environmental(args.years)
    lap = build_lap_features(stats, env, args.years)
    style = build_driver_style(lap)
    stint = build_stint_degradation(lap)
    write_manifest(lap, style, stint)

    print("-" * 70)
    print(f"Phase 3 COMPLETE in {time.time()-t0:.0f}s")
    print(f"  lap_features        : {lap.shape[0]:,} laps x {lap.shape[1]} cols")
    print(f"  driver_style        : {style.shape[0]:,} driver-seasons")
    print(f"  stint_degradation   : {stint.shape[0]:,} stints")
    print("  Run: python -m src.features.verify_gold")


if __name__ == "__main__":
    main()
