"""Phase 4 data prep v2 — refactored per the leakage/correctness spec.

Key changes vs v1:
  1. STINT FILTERING  — drops SC/VSC-truncated stints, DNF-ish stints, and
     unrepresentative short dry stints (strategy noise removal).
  2. FUEL CORRECTION  — fuel_corr = laptime + alpha*lap (Race/Sprint only)
     isolates tyre degradation from fuel-burn pace gain.
  3. ROLLING FEATURES — causal windows w in {3,5,8}: mean/std/min-max-spread,
     5-lap slope, EMA(alpha in {0.2,0.5}), sector deltas (S3 vs S1, per-sector
     degradation vs stint start).
  4. PERFORMANCE-BASED LABELS — T_end = first lap where rolling-3 fuel-corrected
     pace exceeds the stint's best pace + 1.5s (physical cliff, not strategy).
     target_rul = min(T_end - life, RUL_MAX[compound])  (piecewise clipping).
     cliff_within_3 = onset within next 3 laps (0 <= T_end - life <= 3).
  5. STRICT FEATURE EXCLUSION — ALL cliff-derived features removed from X
     (cliff_flag, pace_dev_* , stint_laptime_median/p25, stint_best_pace, ...).
     Only pure telemetry/pace-trend/environmental features remain.

Temporal split: train 2021-2023 / val 2024 / test 2025.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from src.ingest.config import ROOT, BRONZE_DIR, SILVER_DIR

GOLD = ROOT / "data" / "gold"
YEARS = ["2021", "2022", "2023", "2024", "2025"]
SPLIT = {"train": ["2021", "2022", "2023"], "val": ["2024"], "test": ["2025"]}
STINT_KEYS = ["season", "gp", "session", "driver", "stint"]
CATEGORICAL = ["compound", "session_type", "gp", "driver"]

FUEL_ALPHA = 0.03          # s per lap of fuel burned (Race/Sprint)
CLIFF_THR = 1.5            # s above stint best (fuel-corrected) pace
RUL_MAX = {"SOFT": 15, "MEDIUM": 22, "HARD": 30, "INTERMEDIATE": 20, "WET": 20}
DEFAULT_RUL_MAX = 20
MIN_DRY_STINT_LAPS = 5     # drop shorter dry stints
MIN_VALID_FRAC = 0.7       # drop stints with >30% null laptimes

IDENTIFIERS = {"season", "session", "lap"}
LABELS = {"rul", "rul_physical", "target_rul", "cliff_within_3", "t_end_life"}
# EVERY feature derived from the cliff labelling heuristic (spec Task 2.1)
CLIFF_DERIVED = {
    "cliff_flag", "cliff_sustained", "cliff_within_3",
    "pace_dev_median", "pace_dev_p25", "pace_above_median_count",
    "stint_laptime_median", "stint_laptime_p25", "stint_best_pace",
    "rul", "rul_physical", "target_rul", "t_end_life",
}

# Features pruned to eliminate artificial structural multicollinearity,
# formula redundancies, spurious proxies, and circular errors.
PRUNED_CORRELATIONS = {
    # 1, 4 & 5: Baseline levels and structural identities (t = t0 + delta)
    "stint_first_laptime", "laptime", "laptime_delta_stint_start",
    "pace_best_so_far", "pace_above_best",
    # 2: Sample count (telemetry duration / acquisition proxy, not wear physics)
    "n_samples", "weather_samples",
    # 3: Circular angle (replaced by wind_dir_sin and wind_dir_cos)
    "wind_dir_mean",
    # 6: Formula differences & raw sector levels
    "s1", "s2", "s3", "sector_decay_21", "sector_decay_32", "s3_s1_gap",
    # 7 & 8: Collinear rolling pace windows
    "pace_roll_mean_3", "pace_roll_std_3", "pace_roll_mean_8", "pace_roll_std_8",
    "pace_ema_02", "pace_ema_05", "pace_above_best_roll3", "pace_above_best_roll8",
    # 9: Collinear temperatures (range = max - mean; max has r > 0.97 with mean)
    "track_temp_max", "ambient_temp_max", "track_temp_range",
    # 10: Collinear load products (lat_g * lap_distance)
    "lat_load_integral", "long_load_integral",
    # 11: Lap distance (circuit identifier leakage into trees)
    "lap_distance",
    # Zero-variance constant
    "brake_intensity",
}

EXCLUDED = IDENTIFIERS | LABELS | CLIFF_DERIVED | PRUNED_CORRELATIONS


def _session_type(name: str) -> str:
    n = (name or "").lower()
    if n.startswith("race"):
        return "Race"
    if n.startswith("sprint"):
        return "Sprint"
    if n.startswith("qualifying") or "shootout" in n:
        return "Qualifying"
    if n.startswith("practice"):
        return "Practice"
    return "Testing"


def load_lap_features() -> pl.DataFrame:
    return pl.read_parquet(GOLD / "lap_features.parquet")


# ---------------------------------------------------------------------------
# SC/VSC lap detection from race control messages (spec Task 1.1)
# ---------------------------------------------------------------------------
def load_sc_laps() -> pl.DataFrame:
    """Return (season, gp, session, lap) rows for SC/VSC/red-flag-affected laps.

    Captures:
      - Category signatures: 'SafetyCar', 'VirtualSafetyCar', 'VSC'
      - Verified deployment, ending, and pit-procedure message patterns for SC & VSC
        (guarding against steward infringement/penalty notices)
      - Full period expansion (generate_series from start_lap to end_lap) for all
        active SC/VSC intervals across races
      - Flag signatures: DOUBLE YELLOW, RED
    """
    glob = str(SILVER_DIR / "race_events" / "rcm" / "**" / "*.parquet").replace("\\", "/")
    con = duckdb.connect()
    df = con.execute(f"""
        WITH raw_events AS (
            SELECT DISTINCT season, gp, session, msg, lap,
                   row_number() OVER (PARTITION BY season, gp, session ORDER BY lap, time) as rn
            FROM read_parquet('{glob}', hive_partitioning=1)
            WHERE (
                cat IN ('SafetyCar', 'VirtualSafetyCar', 'VSC')
                OR (
                    upper(msg) LIKE '%SAFETY CAR%'
                    AND upper(msg) NOT LIKE '%INFRINGEMENT%'
                    AND upper(msg) NOT LIKE '%PENALTY%'
                    AND upper(msg) NOT LIKE '%UNDER INVESTIGATION%'
                    AND upper(msg) NOT LIKE '%NOTED%'
                    AND upper(msg) NOT LIKE '%REVIEWED%'
                    AND upper(msg) NOT LIKE '%LIGHTS ON%'
                )
                OR (
                    (upper(msg) LIKE '%VIRTUAL SAFETY CAR%' OR upper(msg) LIKE '%VSC%')
                    AND upper(msg) NOT LIKE '%INFRINGEMENT%'
                    AND upper(msg) NOT LIKE '%PENALTY%'
                    AND upper(msg) NOT LIKE '%UNDER INVESTIGATION%'
                    AND upper(msg) NOT LIKE '%NOTED%'
                    AND upper(msg) NOT LIKE '%REVIEWED%'
                    AND upper(msg) NOT LIKE '%TEST%'
                )
            )
            AND lap IS NOT NULL
        ),
        deploy_events AS (
            SELECT season, gp, session, lap as start_lap, rn,
                   CASE WHEN upper(msg) LIKE '%VIRTUAL%' THEN 'VSC' ELSE 'SC' END as sc_type
            FROM raw_events
            WHERE upper(msg) LIKE '%DEPLOYED%'
        ),
        end_events AS (
            SELECT season, gp, session, lap as end_lap, rn,
                   CASE WHEN upper(msg) LIKE '%VIRTUAL%' THEN 'VSC' ELSE 'SC' END as sc_type
            FROM raw_events
            WHERE upper(msg) LIKE '%ENDING%'
               OR upper(msg) LIKE '%IN THIS LAP%'
               OR upper(msg) LIKE '%WILL ENTER PITS%'
        ),
        matched_periods AS (
            SELECT d.season, d.gp, d.session, d.start_lap,
                   coalesce(
                       (SELECT min(e.end_lap) FROM end_events e
                        WHERE e.season = d.season AND e.gp = d.gp AND e.session = d.session
                          AND e.end_lap >= d.start_lap AND e.rn > d.rn AND e.sc_type = d.sc_type),
                       d.start_lap
                   ) as end_lap
            FROM deploy_events d
        ),
        expanded_laps AS (
            SELECT season, gp, session, unnest(generate_series(start_lap, end_lap)) as lap
            FROM matched_periods
        ),
        discrete_laps AS (
            SELECT DISTINCT season, gp, session, lap
            FROM read_parquet('{glob}', hive_partitioning=1)
            WHERE (
                flag IN ('DOUBLE YELLOW', 'RED')
                OR (cat IN ('SafetyCar', 'VirtualSafetyCar', 'VSC') AND lap IS NOT NULL)
                OR (
                    upper(msg) LIKE '%SAFETY CAR%'
                    AND upper(msg) NOT LIKE '%INFRINGEMENT%'
                    AND upper(msg) NOT LIKE '%PENALTY%'
                    AND upper(msg) NOT LIKE '%UNDER INVESTIGATION%'
                    AND upper(msg) NOT LIKE '%NOTED%'
                    AND upper(msg) NOT LIKE '%REVIEWED%'
                    AND upper(msg) NOT LIKE '%LIGHTS ON%'
                    AND lap IS NOT NULL
                )
            )
        )
        SELECT DISTINCT season, gp, session, lap FROM expanded_laps
        UNION
        SELECT DISTINCT season, gp, session, lap FROM discrete_laps
    """).fetchdf()
    return pl.from_pandas(df).with_columns(pl.col("season").cast(pl.Utf8))


# ---------------------------------------------------------------------------
# Lateral tyre-energy integral  sum(a_lat^2 * v)  (spec Task 2, cached)
# ---------------------------------------------------------------------------
def load_lat_energy() -> pl.DataFrame:
    """avg(acc_x^2 * speed) per lap from Bronze. Cached to gold/lat_energy.parquet."""
    cache = GOLD / "lat_energy.parquet"
    if cache.exists():
        return pl.read_parquet(cache).with_columns(pl.col("season").cast(pl.Utf8))
    frames = []
    con = duckdb.connect()
    for y in YEARS:
        glob = str(BRONZE_DIR / f"season={y}" / "**" / "*.parquet").replace("\\", "/")
        df = con.execute(f"""
            SELECT season, gp, session, driver, lap,
                   avg(acc_x * acc_x * speed) AS lat_energy_integral
            FROM read_parquet('{glob}', hive_partitioning=1)
            GROUP BY season, gp, session, driver, lap
        """).fetchdf()
        frames.append(pl.from_pandas(df))
        print(f"   lat_energy {y}: {len(df):,} laps")
    out = pl.concat(frames, how="diagonal_relaxed")
    out = out.with_columns(pl.col("season").cast(pl.Utf8))
    out.write_parquet(cache, compression="zstd")
    return out


# ---------------------------------------------------------------------------
# Fuel correction (spec Task 2)
# ---------------------------------------------------------------------------
def add_fuel_corr(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("session_type").is_in(["Race", "Sprint"]))
        .then(pl.col("laptime") + FUEL_ALPHA * pl.col("lap"))
        .otherwise(pl.col("laptime"))
        .alias("fuel_corr"))


# ---------------------------------------------------------------------------
# Causal rolling features per stint (spec Task 4)
# ---------------------------------------------------------------------------
def _compute_rolling_features_raw(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(STINT_KEYS + ["life"])
    fc = pl.col("fuel_corr")

    df = df.with_columns(
        pl.int_range(pl.len()).over(STINT_KEYS).alias("_stint_pos"))

    exprs = []
    # 5-lap window for mean and std
    exprs.append(fc.rolling_mean(window_size=5, min_samples=1)
                 .over(STINT_KEYS).alias("pace_roll_mean_5"))
    exprs.append(fc.rolling_std(window_size=5, min_samples=2)
                 .over(STINT_KEYS).fill_null(0.0)
                 .alias("pace_roll_std_5"))
    # robust median (out-lap insensitive) — also used for the T_end label rule
    exprs.append(fc.rolling_median(window_size=3, min_samples=1)
                 .over(STINT_KEYS).alias("pace_roll_med_3"))
    # consistency index: min-max spread over last 5 valid laps
    exprs.append(
        (fc.rolling_max(window_size=5, min_samples=1).over(STINT_KEYS)
         - fc.rolling_min(window_size=5, min_samples=1).over(STINT_KEYS))
        .alias("pace_roll_spread_5"))
    # degradation slope over last 5 laps (simple linear approx)
    exprs.append(
        ((fc - fc.shift(4).over(STINT_KEYS)) / 4.0).alias("pace_slope_5"))

    # Out-lap-aware fuel-corrected baseline (first flying lap with pos >= 1; pos 0 gets 0.0)
    fc_base = pl.coalesce([
        pl.when(pl.col("_stint_pos") >= 1).then(fc).drop_nulls().first().over(STINT_KEYS),
        fc.first().over(STINT_KEYS)
    ])
    exprs.append(
        pl.when(pl.col("_stint_pos") == 0)
        .then(0.0)
        .otherwise(fc - fc_base)
        .alias("fuel_corr_delta_start")
    )

    # Causal running-best pace and the degradation gap to it:
    exprs.append(fc.cum_min().over(STINT_KEYS).alias("pace_best_so_far"))
    df = df.with_columns(exprs)
    df = df.with_columns(
        (pl.col("fuel_corr") - pl.col("pace_best_so_far"))
        .alias("pace_above_best"))
    df = df.with_columns(
        pl.col("pace_above_best").rolling_mean(window_size=5, min_samples=1)
        .over(STINT_KEYS).alias("pace_above_best_roll5"))
    df = df.with_columns(
        ((pl.col("pace_above_best_roll5")
          - pl.col("pace_above_best_roll5").shift(4).over(STINT_KEYS)) / 4.0)
        .alias("pace_above_best_slope5"))

    # Sector deltas: per-sector degradation from first flying lap (out-lap aware)
    exprs2 = []
    for s in ("s1", "s2", "s3"):
        s_base = pl.coalesce([
            pl.when(pl.col("_stint_pos") >= 1).then(pl.col(s)).drop_nulls().first().over(STINT_KEYS),
            pl.col(s).first().over(STINT_KEYS)
        ])
        exprs2.append(
            pl.when(pl.col("_stint_pos") == 0)
            .then(0.0)
            .otherwise(pl.col(s) - s_base)
            .alias(f"{s}_delta_start")
        )
    df = df.with_columns(exprs2)
    # rolling S3 degradation (traction-limited sector wears tyres fastest)
    df = df.with_columns(
        pl.col("s3_delta_start").rolling_mean(window_size=5, min_samples=1)
        .over(STINT_KEYS).alias("s3_delta_roll_5"))
    return df.drop(["_stint_pos"])


def compute_rolling_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(STINT_KEYS + ["life"])
    # Option B (Task 8.1.2): If is_sc_lap is present and carries SC laps, compute
    # rolling pace features over green-flag racing laps only, then re-join and
    # forward-fill across SC laps. This prevents 20-40s SC delta laptimes from
    # corrupting rolling pace windows while preserving sequence continuity for tyre life.
    has_sc = "is_sc_lap" in df.columns and (df["is_sc_lap"] == 1).any()
    if has_sc:
        orig_cols = set(df.columns)
        green = df.filter(pl.col("is_sc_lap") == 0)
        green = _compute_rolling_features_raw(green)
        roll_cols = [c for c in green.columns if c not in orig_cols]
        df = df.join(green.select(STINT_KEYS + ["life"] + roll_cols),
                     on=STINT_KEYS + ["life"], how="left")
        # Forward-fill (and backward-fill if stint starts under SC) per stint
        for c in roll_cols:
            df = df.with_columns(
                pl.col(c).forward_fill().backward_fill().over(STINT_KEYS))
        return df
    return _compute_rolling_features_raw(df)


# ---------------------------------------------------------------------------
# Stint filtering (spec Task 1)
# ---------------------------------------------------------------------------
def filter_stints(df: pl.DataFrame, sc_laps: pl.DataFrame) -> pl.DataFrame:
    n0 = df.height
    df = df.with_columns(
        (pl.col("laptime").is_not_null().sum().over(STINT_KEYS))
        .alias("_valid_laps"))
    df = df.with_columns(
        (pl.col("_valid_laps") / pl.len().over(STINT_KEYS)).alias("_valid_frac"))

    # SC/VSC-affected lap flag (join on season/gp/session/lap)
    sc = sc_laps.filter(pl.col("lap").is_not_null()).with_columns(
        pl.lit(1).cast(pl.Int32).alias("is_sc_lap"))
    df = df.join(sc, on=["season", "gp", "session", "lap"], how="left")
    df = df.with_columns(pl.col("is_sc_lap").fill_null(0))

    # stint ends under SC/VSC: any of last 2 laps (by life) affected
    df = df.with_columns(
        ((pl.col("life") >= pl.col("life").max().over(STINT_KEYS) - 1)
         & (pl.col("is_sc_lap") == 1)).any().over(STINT_KEYS).alias("_sc_end"))

    rain = (pl.col("rain_flag").first().over(STINT_KEYS).fill_null(0) > 0)
    keep = (
        ((pl.col("_valid_laps") >= MIN_DRY_STINT_LAPS) | rain)
        & (pl.col("_valid_frac") > MIN_VALID_FRAC)
        & ~pl.col("_sc_end")
        & (pl.col("_valid_laps") >= 3)
    ).first().over(STINT_KEYS)
    df = df.filter(keep)
    # keep only laps with a valid laptime (labels need fuel_corr)
    df = df.filter(pl.col("laptime").is_not_null())
    # Option B (Task 8.1.2): Retain is_sc_lap as an explicit feature in X
    df = df.drop(["_valid_laps", "_valid_frac", "_sc_end"])
    n1 = df.height
    nst0, nst1 = n0, n1
    print(f"   stint filtering: {n0:,} -> {n1:,} laps "
          f"({100*(1-n1/max(n0,1)):.1f}% removed)")
    return df


# ---------------------------------------------------------------------------
# Performance-based labels (spec Task 3)
# ---------------------------------------------------------------------------
def compute_labels(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(STINT_KEYS + ["life"])
    has_sc = "is_sc_lap" in df.columns
    # stint best fuel-corrected pace (LABEL-side only — uses future, allowed)
    # Option B: derive stint best pace strictly from non-SC racing laps
    if has_sc:
        df = df.with_columns(
            pl.when(pl.col("is_sc_lap") == 0)
            .then(pl.col("fuel_corr"))
            .otherwise(None)
            .min()
            .over(STINT_KEYS)
            .alias("stint_best_pace"))
        df = df.with_columns(
            pl.col("stint_best_pace").fill_null(
                pl.col("fuel_corr").min().over(STINT_KEYS)))
    else:
        df = df.with_columns(
            pl.col("fuel_corr").min().over(STINT_KEYS).alias("stint_best_pace"))

    # position within stint (0-based, by life order) — out-lap/standing-start = 0
    df = df.with_columns(
        pl.int_range(pl.len()).over(STINT_KEYS).alias("_pos"))

    # T_end: first lap (from stint position 2 = 3rd lap onward, so the out-lap
    # cannot trigger it) where the out-lap-robust rolling median of fuel-
    # corrected pace exceeds the stint best pace + CLIFF_THR, SUSTAINED for
    # 3 consecutive laps (immune to single traffic laps / in-laps).
    # Option B mask: an SC lap cannot trigger _above or contribute to a false cliff!
    if has_sc:
        above_expr = (
            (pl.col("pace_roll_med_3") > pl.col("stint_best_pace") + CLIFF_THR)
            & (pl.col("_pos") >= 2)
            & (pl.col("is_sc_lap") == 0)
        )
    else:
        above_expr = (
            (pl.col("pace_roll_med_3") > pl.col("stint_best_pace") + CLIFF_THR)
            & (pl.col("_pos") >= 2)
        )
    df = df.with_columns(above_expr.alias("_above"))

    df = df.with_columns(
        (pl.col("_above")
         & pl.col("_above").shift(-1).over(STINT_KEYS).fill_null(False)
         & pl.col("_above").shift(-2).over(STINT_KEYS).fill_null(False))
        .alias("_cross"))
    df = df.with_columns(
        pl.when(pl.col("_cross")).then(pl.col("life")).alias("_cross_life"))
    df = df.with_columns(
        pl.col("_cross_life").min().over(STINT_KEYS).alias("t_end_life"))
    max_life = pl.col("life").max().over(STINT_KEYS)

    # rul_physical: laps until physical cliff; censored stints -> stint end + 1
    df = df.with_columns(
        pl.when(pl.col("t_end_life").is_not_null())
        .then(pl.col("t_end_life") - pl.col("life"))
        .otherwise(max_life - pl.col("life") + 1)
        .clip(lower_bound=0)
        .alias("rul_physical"))

    # piecewise clipping per compound (spec 3.2)
    cap = (pl.when(pl.col("compound") == "SOFT").then(pl.lit(RUL_MAX["SOFT"]))
           .when(pl.col("compound") == "MEDIUM").then(pl.lit(RUL_MAX["MEDIUM"]))
           .when(pl.col("compound") == "HARD").then(pl.lit(RUL_MAX["HARD"]))
           .when(pl.col("compound") == "INTERMEDIATE").then(pl.lit(RUL_MAX["INTERMEDIATE"]))
           .when(pl.col("compound") == "WET").then(pl.lit(RUL_MAX["WET"]))
           .otherwise(pl.lit(DEFAULT_RUL_MAX)))
    df = df.with_columns(
        pl.min_horizontal(pl.col("rul_physical"), cap).alias("target_rul"))

    # cliff_within_3: tyre is in the cliff OR enters it within the next 3 laps
    # (t_end - life <= 3, no lower bound — post-onset laps are positive because
    # the tyre stays degraded; temporal hysteresis is signal, not leakage,
    # since NO cliff-derived features are in X)
    df = df.with_columns(
        (pl.col("t_end_life").is_not_null()
         & (pl.col("t_end_life") - pl.col("life") <= 3))
        .alias("cliff_within_3"))
    df = df.drop(["_cross", "_cross_life", "_pos", "_above"])
    return df


# ---------------------------------------------------------------------------
# Feature selection / encoding / splits
# ---------------------------------------------------------------------------
def feature_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDED]


def make_tabular(df: pl.DataFrame):
    feats = feature_cols(df)
    codes: dict[str, dict] = {}
    for c in CATEGORICAL:
        df = df.with_columns(pl.col(c).fill_null("UNKNOWN").alias(c))
        mapping = {v: i for i, v in enumerate(sorted(df[c].unique().to_list()))}
        codes[c] = mapping
        df = df.with_columns(pl.col(c).replace(mapping).cast(pl.Int32).alias(c))
    out = {}
    for split, years in SPLIT.items():
        sub = df.filter(pl.col("season").is_in(years)).sort(STINT_KEYS + ["life"])
        X = sub.select(feats)
        out[split] = (X, sub["target_rul"].to_numpy().astype(np.float32),
                      sub["cliff_within_3"].to_numpy().astype(np.float32))
    return out, feats, codes


def make_sequences(df: pl.DataFrame, feats: list[str], codes: dict):
    for c in CATEGORICAL:
        df = df.with_columns(pl.col(c).fill_null("UNKNOWN").alias(c))
        df = df.with_columns(
            pl.col(c).replace(codes[c]).cast(pl.Int32).alias(c))
    num_feats = [f for f in feats if f not in CATEGORICAL]
    train_df = df.filter(pl.col("season").is_in(SPLIT["train"]))
    means = {c: float(train_df[c].mean() or 0.0) for c in num_feats}
    stds = {c: float(train_df[c].std() or 1.0) or 1.0 for c in num_feats}

    def encode(sub: pl.DataFrame):
        sub = sub.sort(STINT_KEYS + ["life"])
        seqs, ruls, cliffs, sids = [], [], [], []
        for key, grp in sub.group_by(STINT_KEYS, maintain_order=True):
            grp = grp.sort("life")
            mat = np.zeros((grp.height, len(feats)), dtype=np.float32)
            for j, f in enumerate(feats):
                if f in CATEGORICAL:
                    mat[:, j] = grp[f].to_numpy()
                else:
                    col = grp[f].to_numpy().astype(np.float32)
                    col = np.where(np.isfinite(col), col, 0.0)
                    mat[:, j] = (col - means[f]) / stds[f]
            seqs.append(mat)
            ruls.append(grp["target_rul"].to_numpy().astype(np.float32))
            cliffs.append(grp["cliff_within_3"].to_numpy().astype(np.float32))
            sids.append(key)
        return {"X": seqs, "y_rul": ruls, "y_cliff": cliffs, "sids": sids}

    return {sp: encode(df.filter(pl.col("season").is_in(yrs)))
            for sp, yrs in SPLIT.items()}, feats, (means, stds)


def prepare():
    df = load_lap_features()
    df = df.with_columns(pl.col("season").cast(pl.Utf8))
    df = df.with_columns(
        pl.col("session").map_elements(_session_type, return_dtype=pl.Utf8)
        .alias("session_type"))
    # restrict to the race-strategy domain: practice/qualifying stint semantics
    # (short experimental runs, quali-sim baselines) are strategy noise (spec Task 1)
    df = df.filter(pl.col("session_type").is_in(["Race", "Sprint"]))
    if "wind_dir_sin" not in df.columns and "wind_dir_mean" in df.columns:
        rad = pl.col("wind_dir_mean") * (np.pi / 180.0)
        df = df.with_columns([
            rad.sin().alias("wind_dir_sin"),
            rad.cos().alias("wind_dir_cos"),
        ])
    print(f"[prep] Race/Sprint laps: {df.height:,}")
    print("[prep] loading SC/VSC laps from rcm...")
    sc_laps = load_sc_laps()
    print(f"   sc laps: {sc_laps.height:,} session-laps affected")
    print("[prep] loading lateral tyre-energy integral (cached Bronze scan)...")
    lat = load_lat_energy()
    df = df.join(lat, on=["season", "gp", "session", "driver", "lap"],
                 how="left")
    print("[prep] stint filtering (SC/VSC ends, DNF-ish, short dry stints)...")
    df = filter_stints(df, sc_laps)
    print("[prep] fuel correction + rolling features...")
    df = add_fuel_corr(df)
    df = compute_rolling_features(df)
    print("[prep] performance-based labels (T_end, piecewise RUL, cliff)...")
    df = compute_labels(df)
    tab, feats, codes = make_tabular(df)
    seq, feats, norm = make_sequences(df, feats, codes)
    return tab, seq, feats, codes, norm


if __name__ == "__main__":
    tab, seq, feats, codes, norm = prepare()
    for sp in ["train", "val", "test"]:
        X, yr, yc = tab[sp]
        print(f"{sp:5s} tabular: X={X.shape} target_rul mean={yr.mean():.2f} "
              f"cliff+={yc.mean():.3f}")
        sq = seq[sp]
        print(f"      seqs: {len(sq['X'])} stints")
    print(f"\nfeatures ({len(feats)}): {feats}")
    print(f"compound codes: {codes}")
    ex = sorted(CLIFF_DERIVED & set(feats))
    print(f"leak check (cliff-derived in X): {ex if ex else 'NONE - OK'}")

    # label sanity diagnostics
    df = load_lap_features()
    df = df.with_columns(pl.col("season").cast(pl.Utf8))
    df = df.with_columns(
        pl.col("session").map_elements(_session_type, return_dtype=pl.Utf8)
        .alias("session_type"))
    df = df.filter(pl.col("session_type").is_in(["Race", "Sprint"]))
    df = df.join(load_lat_energy(), on=["season", "gp", "session", "driver", "lap"],
                 how="left")
    df = filter_stints(df, load_sc_laps())
    df = add_fuel_corr(df)
    df = compute_rolling_features(df)
    df = compute_labels(df)
    n_stints = df.select(pl.struct(STINT_KEYS)).n_unique()
    n_cliff = df.filter(pl.col("t_end_life").is_not_null()).select(
        pl.struct(STINT_KEYS)).n_unique()
    print(f"\nlabel diagnostics: {n_stints:,} stints, "
          f"{n_cliff:,} with physical T_end ({100*n_cliff/max(n_stints,1):.0f}%)")
    tr = df["target_rul"]
    print(f"target_rul: mean={tr.mean():.2f} p25={tr.quantile(0.25):.1f} "
          f"p50={tr.quantile(0.5):.1f} p75={tr.quantile(0.75):.1f} max={tr.max():.0f}")
    print(f"cliff_within_3 rate: {df['cliff_within_3'].mean():.3f}")
