"""Phase 6 — Fit the tyre degradation model for the strategy optimizer.

From the Phase 4 labeled frame (Race/Sprint, filtered stints, fuel-corrected
pace) + Phase 5 style clusters, fit:
  - beta[compound]        : median fuel-corrected degradation slope (s/lap)
  - cluster_offset[cl]    : style-cluster adjustment to beta
  - temp_coef             : s/lap per degree C of track temperature
  - sigma[compound]       : slope dispersion (Monte Carlo noise)
  - cliff_dist[compound]  : empirical physical-cliff lap distribution
                            (quantiles + censoring probability)

Saved to data/models/degradation_model.json.

    python -m src.models.fit_degradation
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl

from src.ingest.config import ROOT
from src.models.eval_rul import build_labeled_frame
from src.models.rul_data import STINT_KEYS

GOLD = ROOT / "data" / "gold"
MODELS_DIR = ROOT / "data" / "models"
DRY = ["SOFT", "MEDIUM", "HARD"]
N_QUANT = 25


def fit_degradation(df=None, save=True):
    print("[deg] building labeled frame (Race/Sprint, filtered stints)...")
    if df is None:
        df = build_labeled_frame()
    emb = pl.read_parquet(GOLD / "driver_embeddings.parquet")
    emb = emb.with_columns(pl.col("season").cast(pl.Utf8))
    df = df.join(emb.select(["season", "driver", "cluster"]),
                 on=["season", "driver"], how="left")
    df = df.with_columns(pl.col("cluster").fill_null(-1).cast(pl.Int32))

    dry = df.filter(pl.col("compound").is_in(DRY))
    dry = dry.sort(STINT_KEYS + ["life"])
    # stint position: skip out-lap + first 2 flying laps (tyre warm-up and
    # race-start traffic bias the slope negative / trigger false cliffs)
    dry = dry.with_columns(
        pl.int_range(pl.len()).over(STINT_KEYS).alias("_pos"))
    dry = dry.filter(pl.col("_pos") >= 3)
    # per-stint slope via cov/var (equivalent to OLS slope on life)
    dry = dry.with_columns(
        ((pl.cov("life", "fuel_corr") / pl.var("life")).over(STINT_KEYS))
        .alias("beta"),
        pl.len().over(STINT_KEYS).alias("n_laps"),
        pl.col("t_end_life").first().over(STINT_KEYS).alias("t_end"),
        pl.col("life").min().over(STINT_KEYS).alias("life0"),
        pl.col("track_temp_mean").mean().over(STINT_KEYS).alias("temp"),
        pl.col("compound").first().over(STINT_KEYS).alias("comp"),
        pl.col("cluster").first().over(STINT_KEYS).alias("clu"),
    )
    st = (dry.filter((pl.col("n_laps") >= 4) & pl.col("beta").is_not_null())
          .group_by(STINT_KEYS)
          .agg(pl.col("beta").first(), pl.col("n_laps").first(),
               pl.col("t_end").first(), pl.col("life0").first(),
               pl.col("temp").first(), pl.col("comp").first(),
               pl.col("clu").first()))
    print(f"[deg] {st.shape[0]:,} dry stints (pos>=3, warm-up excluded)")

    beta_g = float(st["beta"].median())
    params = {
        "beta_global": beta_g,
        "beta": {}, "sigma": {}, "cluster_offset": {},
        "temp_coef": 0.0, "temp_mean": float(st["temp"].mean() or 0.0),
        "cliff": {}, "n_stints": st.shape[0],
        "cliff_jump": 1.5, "cliff_slope": 0.15,
        "notes": ("slopes fitted on stint positions >=3 (warm-up/traffic "
                  "excluded); cliff distribution excludes cliff_at < 3 "
                  "(race-start traffic artifacts, counted as censored); "
                  "cluster offsets only for clusters with >=30 race stints; "
                  "sigma is IQR-robust (raw std destroyed by traffic outliers); "
                  "cliff cost = one-time jump + 0.15 s/lap beyond the cliff"),
    }
    for c in DRY:
        sub = st.filter(pl.col("comp") == c)
        if sub.height < 5:
            params["beta"][c] = beta_g
            params["sigma"][c] = float(st["beta"].std() or 0.05)
            continue
        params["beta"][c] = float(sub["beta"].median())
        # robust sigma (IQR-based): raw std is destroyed by traffic-locked
        # stint outliers (e.g. sigma_MEDIUM ~ 8 s/lap raw vs ~0.1 robust)
        q25, q75 = sub["beta"].quantile(0.25), sub["beta"].quantile(0.75)
        sig = float((q75 - q25) / 1.349)
        params["sigma"][c] = max(min(sig, 0.5), 0.02)
        te = sub.filter(pl.col("t_end").is_not_null())
        # cliff_at relative to stint start; drop early artifacts (<3) as censored
        rel = (te.with_columns(
            (pl.col("t_end") - pl.col("life0")).alias("cliff_at"))
            .filter(pl.col("cliff_at") >= 3))["cliff_at"].to_numpy()
        cen = 1.0 - len(rel) / max(sub.height, 1)
        qs = np.quantile(rel, np.linspace(0.05, 0.95, N_QUANT)) if len(rel) \
            else np.array([20.0])
        params["cliff"][c] = {
            "censor_prob": round(cen, 3),
            "quantiles": [round(float(q), 1) for q in qs],
        }
    for cl in sorted(st["clu"].unique().to_list()):
        sub = st.filter(pl.col("clu") == cl)
        if sub.height >= 30:
            params["cluster_offset"][str(cl)] = \
                round(float(sub["beta"].median() - beta_g), 5)
    # temperature coefficient: cov(beta, temp)/var(temp)
    b = st["beta"].to_numpy()
    t = st["temp"].to_numpy()
    mask = np.isfinite(b) & np.isfinite(t)
    if mask.sum() > 50:
        tv = t[mask].astype(float)
        bv = b[mask].astype(float)
        var = float(np.var(tv))
        if var > 1e-6:
            params["temp_coef"] = float(np.cov(tv, bv)[0, 1] / var)

    # compound pace offsets: pairwise within-driver stint-best differences
    # (same driver, same session, both compounds present).
    # SOFT-MEDIUM is physical (measured). HARD-MEDIUM is contaminated by track
    # evolution (M->H one-stoppers run the HARD stint late on a rubbered track:
    # measured -0.38 with an inconsistent triangle) -> symmetric prior used.
    laps = df.filter(pl.col("compound").is_in(DRY))
    best = (laps.group_by(["season", "gp", "session", "driver", "compound"])
            .agg(pl.col("fuel_corr").min().alias("best")))
    piv = best.pivot(on="compound", index=["season", "gp", "session", "driver"],
                     values="best", aggregate_function="first")
    off_s, off_h = -0.35, 0.35
    if "SOFT" in piv.columns and "MEDIUM" in piv.columns:
        d = (piv.drop_nulls(["SOFT", "MEDIUM"])
             .with_columns((pl.col("SOFT") - pl.col("MEDIUM")).alias("d"))["d"])
        if d.len() >= 10:
            off_s = float(d.median())
            off_h = abs(off_s)
    params["pace_offset"] = {"SOFT": round(off_s, 3), "MEDIUM": 0.0,
                             "HARD": round(off_h, 3)}
    params["pace_offset_notes"] = (
        "SOFT-MEDIUM measured pairwise within-driver; HARD set to the "
        "symmetric prior (+|SOFT-MEDIUM|) because the measured M->H gap "
        "(-0.38) is track-evolution contaminated (late-race rubbering); "
        "override via config if desired.")

    if not save:
        return params
    out = MODELS_DIR / "degradation_model.json"
    with open(out, "w") as fh:
        json.dump(params, fh, indent=2)
    print(f"[deg] saved -> {out}")
    print(f"     beta: {params['beta']}")
    print(f"     pace offsets (vs session median): {params['pace_offset']}")
    print(f"     cluster offsets: {params['cluster_offset']}")
    print(f"     temp coef: {params['temp_coef']:.5f} s/lap/degC "
          f"(mean {params['temp_mean']:.1f} degC)")
    for c in DRY:
        print(f"     cliff {c}: censor_prob="
              f"{params['cliff'][c]['censor_prob']} "
              f"q50={params['cliff'][c]['quantiles'][12]}")
    return params


if __name__ == "__main__":
    fit_degradation()
