"""Phase 5 — Driving style data prep (transition-spec compliant).

Action item 1 (strict in/out lap exclusion):
  - drop laps with null laptime
  - drop out-laps (first lap of a stint = pit exit / standing start)
  - drop in-laps (last lap of a stint when the driver pits again)
  - keep hot laps only (laptime <= 107% of the driver's session best)
  - drop SC/VSC laps (rcm-derived lap set)

Action item 2 (track evolution proxy):
  - cum_session_laps   = valid laps by ALL drivers in the session up to this lap
  - weekend_prior_laps = weighted cumulative laps from earlier weekend sessions
  - track_evolution_index = session_weight * cum_session_laps + weekend_prior

Action item 4 (Option B): per-lap micro-sector vectors (10 bins x 4 channels)
joined from data/gold/micro_sectors.parquet.

AE input = micro-sector vector + track_evolution_index + track_temp_mean +
compound one-hot + session_type one-hot. Standardized with TRAIN-season stats.
"""
from __future__ import annotations

import polars as pl

from src.features.micro_sectors import build_micro_sectors
from src.models.rul_data import load_lap_features, load_sc_laps, _session_type

STINT_KEYS = ["season", "gp", "session", "driver", "stint"]
SPLIT_YEARS = {"train": ["2021", "2022", "2023", "2024"], "test": ["2025"]}
HOT_LAP_PCT = 1.07

SESS_WEIGHT = {
    "Practice 1": 0.6, "Practice 2": 0.8, "Practice 3": 1.0,
    "Sprint Qualifying": 1.1, "Sprint Shootout": 1.1,
    "Qualifying": 1.2, "Sprint": 1.3, "Race": 1.5,
}
SESS_ORDER = {
    "Practice 1": 1, "Practice 2": 2, "Practice 3": 3,
    "Sprint Qualifying": 4, "Sprint Shootout": 4,
    "Qualifying": 5, "Sprint": 6, "Race": 7,
}
GAMMA_WASH = 0.3      # wet session washes away 70% of accumulated rubber
WET_EFFICIENCY = 0.5  # wet tyres lay only 50% of usable rubber
INSTANT_WET = 0.6     # rain during the current lap: grip x0.6
WET_COMPOUNDS = ["INTERMEDIATE", "WET"]


def _flag_pit_laps(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(STINT_KEYS + ["life"])
    df = df.with_columns(
        pl.int_range(pl.len()).over(STINT_KEYS).alias("_pos"))
    df = df.with_columns(
        pl.col("stint").max().over(["season", "gp", "session", "driver"])
        .alias("_max_stint"))
    df = df.with_columns([
        (pl.col("_pos") == 0).alias("is_pit_out_lap"),
        ((pl.col("_pos") == pl.len().over(STINT_KEYS) - 1)
         & (pl.col("stint") < pl.col("_max_stint"))).alias("is_pit_in_lap"),
    ])
    return df.drop(["_pos", "_max_stint"])


def _track_evolution(df_all: pl.DataFrame) -> pl.DataFrame:
    """TEI with rain wash-out (spec update).

    Weekend fold (chronological, per spec):
        R_prior(S_k) = [R_prior(S_{k-1}) + w_{k-1} V_{k-1} (0.5)^W_{k-1}]
                       * (gamma_wash)^W_k
    Lap level inside session k at lap t:
        TEI(t) = [R_prior(S_k) + w_k * cum_session_laps(t)] * (0.6)^r_t
    where r_t = 1 if any driver is on INTERMEDIATE/WET at lap number t
    (per-lap rain proxy). Wet session flag W_s = wet-compound lap fraction
    > 0.02 OR any rain recorded in the session (rain_flag).
    """
    valid = df_all.filter(pl.col("laptime").is_not_null())

    # lap-number level: cumulative laps + any-wet-tyres indicator (r_t)
    per_lapnum = (valid
                  .group_by(["season", "gp", "session", "lap"])
                  .agg(pl.len().alias("n_laps"),
                       pl.col("compound").is_in(WET_COMPOUNDS).any()
                       .alias("is_wet_lap"))
                  .sort(["season", "gp", "session", "lap"]))
    per_lapnum = per_lapnum.with_columns(
        pl.col("n_laps").cum_sum().over(["season", "gp", "session"])
        .alias("cum_session_laps"))

    # session-level summaries: total laps, wet flag, weight, weekend order
    sess = (valid
            .group_by(["season", "gp", "session"])
            .agg(pl.len().alias("V"),
                 (pl.col("compound").is_in(WET_COMPOUNDS).mean() > 0.02)
                 .alias("wet_by_compound"),
                 (pl.col("rain_flag").max() > 0).alias("wet_by_rain")))
    sess = sess.with_columns(
        (pl.col("wet_by_compound") | pl.col("wet_by_rain"))
        .alias("is_wet_session"))
    sess = sess.with_columns(
        pl.col("session").map_elements(
            lambda s: SESS_WEIGHT.get(s, 1.0), return_dtype=pl.Float64)
        .alias("sess_w"),
        pl.col("session").map_elements(
            lambda s: SESS_ORDER.get(s, 8), return_dtype=pl.Int32)
        .alias("sess_ord"))
    sess = sess.sort(["season", "gp", "sess_ord"])

    # chronological fold over weekend sessions (spec Step 2)
    r_prior_map = {}
    state = {}
    for row in sess.to_dicts():
        wk = (row["season"], row["gp"])
        if wk not in state:
            state[wk] = 0.0
        if row["is_wet_session"]:
            state[wk] *= GAMMA_WASH
        r_prior_map[(row["season"], row["gp"], row["session"])] = state[wk]
        eff = WET_EFFICIENCY if row["is_wet_session"] else 1.0
        state[wk] += row["sess_w"] * row["V"] * eff

    prior = pl.DataFrame({
        "season": [k[0] for k in r_prior_map],
        "gp": [k[1] for k in r_prior_map],
        "session": [k[2] for k in r_prior_map],
        "r_prior_rubber": [v for v in r_prior_map.values()],
    })
    tei = (per_lapnum
           .join(prior, on=["season", "gp", "session"], how="left")
           .join(sess.select(["season", "gp", "session", "sess_w"]),
                 on=["season", "gp", "session"], how="left"))
    tei = tei.with_columns(
        (pl.col("r_prior_rubber") + pl.col("sess_w")
         * pl.col("cum_session_laps")).alias("_tei_base"))
    tei = tei.with_columns(
        pl.when(pl.col("is_wet_lap"))
        .then(pl.col("_tei_base") * INSTANT_WET)
        .otherwise(pl.col("_tei_base"))
        .alias("track_evolution_index"))
    return tei.select(["season", "gp", "session", "lap",
                       "cum_session_laps", "r_prior_rubber",
                       "is_wet_lap", "track_evolution_index"])


def prepare_style(verbose: bool = True):
    """Returns (splits, feature_names, meta) for the style Autoencoder."""
    if verbose:
        print("[style] loading lap features + micro sectors...")
    df = load_lap_features()
    df = df.with_columns(pl.col("season").cast(pl.Utf8))
    df = df.with_columns(
        pl.col("session").map_elements(_session_type, return_dtype=pl.Utf8)
        .alias("session_type"))
    ms = build_micro_sectors().with_columns(pl.col("season").cast(pl.Utf8))
    ms_cols = [c for c in ms.columns
               if c.startswith("ms") or "_" in c and c.split("_")[0].isdigit()]
    ms_cols = [c for c in ms.columns if c not in
               ["season", "gp", "session", "driver", "lap"]]
    df = df.join(ms, on=["season", "gp", "session", "driver", "lap"],
                 how="inner")

    n0 = df.height
    df = _flag_pit_laps(df)
    best = (df.filter(pl.col("laptime").is_not_null())
            .group_by(["season", "gp", "session", "driver"])
            .agg(pl.col("laptime").min().alias("_best")))
    df = df.join(best, on=["season", "gp", "session", "driver"], how="left")
    df = df.with_columns(
        (pl.col("laptime") <= HOT_LAP_PCT * pl.col("_best"))
        .alias("_is_hot"))
    sc = load_sc_laps().with_columns(pl.lit(True).alias("_is_sc"))
    df = df.join(sc, on=["season", "gp", "session", "lap"], how="left")

    keep = (
        pl.col("laptime").is_not_null()
        & ~pl.col("is_pit_out_lap")
        & ~pl.col("is_pit_in_lap")
        & pl.col("_is_hot")
        & pl.col("_is_sc").fill_null(False).__invert__()
    )
    stats = {
        "start": n0,
        "null_laptime": df.filter(pl.col("laptime").is_null()).height,
        "pit_out": df.filter(pl.col("is_pit_out_lap")).height,
        "pit_in": df.filter(pl.col("is_pit_in_lap")).height,
        "not_hot": df.filter(~pl.col("_is_hot")
                             & pl.col("laptime").is_not_null()).height,
        "sc_laps": df.filter(pl.col("_is_sc").fill_null(False)).height,
    }
    df = df.filter(keep)
    stats["kept"] = df.height
    if verbose:
        print(f"[style] lap filtering: {stats}")
        print(f"       {n0:,} -> {df.height:,} laps "
              f"({100*(1-df.height/max(n0,1)):.1f}% removed)")

    if verbose:
        print("[style] track evolution index...")
    tei = _track_evolution(load_lap_features().with_columns(
        pl.col("season").cast(pl.Utf8)))
    df = df.join(tei, on=["season", "gp", "session", "lap"], how="left")

    for c in ["compound", "session_type"]:
        df = df.with_columns(pl.col(c).fill_null("UNKNOWN").alias(c))
    comp_levels = sorted(df["compound"].unique().to_list())
    st_levels = sorted(df["session_type"].unique().to_list())
    for lv in comp_levels:
        df = df.with_columns(
            (pl.col("compound") == lv).cast(pl.Float32)
            .alias(f"comp_{lv}"))
    for lv in st_levels:
        df = df.with_columns(
            (pl.col("session_type") == lv).cast(pl.Float32)
            .alias(f"st_{lv}"))

    feats = (ms_cols
             + ["track_evolution_index", "track_temp_mean"]
             + [f"comp_{c}" for c in comp_levels]
             + [f"st_{s}" for s in st_levels])

    tr = df.filter(pl.col("season").is_in(SPLIT_YEARS["train"]))
    means = {c: float(tr[c].mean() or 0.0) for c in feats}
    stds = {c: float(tr[c].std() or 1.0) or 1.0 for c in feats}

    def mat(sub: pl.DataFrame):
        import numpy as np
        X = np.zeros((sub.height, len(feats)), dtype=np.float32)
        for j, f in enumerate(feats):
            col = sub[f].to_numpy().astype(np.float32)
            col = np.where(np.isfinite(col), col, 0.0)
            X[:, j] = (col - means[f]) / stds[f]
        return X

    splits = {}
    for sp, yrs in SPLIT_YEARS.items():
        sub = df.filter(pl.col("season").is_in(yrs))
        splits[sp] = {
            "X": mat(sub),
            "keys": sub.select(["season", "gp", "session", "driver", "lap",
                                "stint", "life", "compound"]),
        }
    meta = {"features": feats, "means": means, "stds": stds,
            "compounds": comp_levels, "session_types": st_levels,
            "filter_stats": stats}
    return splits, meta, df


if __name__ == "__main__":
    splits, meta, df = prepare_style()
    for sp, d in splits.items():
        print(f"{sp}: X={d['X'].shape}")
    print(f"features ({len(meta['features'])}): {meta['features'][:12]} ...")
    print(f"filter stats: {meta['filter_stats']}")
    print(f"driver-seasons kept: "
          f"{df.group_by(['season', 'driver']).agg(pl.len()).height}")
