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
from src.models.rul_data import STINT_KEYS, load_lap_features
from src.models.style_data import _track_evolution

GOLD = ROOT / "data" / "gold"
MODELS_DIR = ROOT / "data" / "models"
DRY = ["SOFT", "MEDIUM", "HARD"]
N_QUANT = 25

# scenario dirty-air anchors (Task 8.0.4 Option A). Because dirty_air_frac is
# sparse (90th percentile ~0.05, max ~0.26 across dry stints), the DR/backmarker
# anchors are taken from the OBSERVED distribution (percentiles), not arbitrary
# hand-set values, and the multiplier is capped so the spline cannot explode
# when extrapolating to traffic extremes not present in the telemetry.
MULT_CEILING = 2.5   # physical upper bound on the dirty-air degradation penalty
N_TEI_BINS = 8


def _theil_sen(x, y):
    """Robust slope/intercept via pairwise median slopes (resists extreme bins)."""
    n = len(x)
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[j] - x[i]
            if abs(dx) > 1e-9:
                slopes.append((y[j] - y[i]) / dx)
    if not slopes:
        return 0.0, float(np.median(y))
    m = float(np.median(slopes))
    b = float(np.median(y - m * x))
    return m, b


def _fit_dirty_air(stinfo: pl.DataFrame) -> dict:
    """Fit beta(dirty_air_frac) via binned medians + robust linear regression.

    Returns {"fit": {...}, "scenario_beta_mult": {...}, "notes": str}. The
    fitted curve is evaluated at empirical dirty_air_frac percentile anchors
    (clean=0, drs_train=P90, backmarker=P99) and normalized by the clean-air
    slope so clean_air == 1.0 exactly, then clipped to [1.0, MULT_CEILING].
    """
    st = stinfo.filter(pl.col("beta").is_not_null()
                       & pl.col("daf").is_not_null())
    daf = st["daf"].to_numpy().astype(float)
    beta = st["beta"].to_numpy().astype(float)
    mask = np.isfinite(daf) & np.isfinite(beta) & (beta > 0) & (beta < 1.0)
    daf, beta = daf[mask], beta[mask]
    n = int(mask.sum())

    def _fallback():
        return {
            "fit": {"method": "fallback", "n": n,
                    "beta_baseline": float(np.median(beta)) if n else None,
                    "reason": "insufficient dirty-air samples; static priors"},
            "scenario_beta_mult": {"clean_air": 1.0, "drs_train": 1.15,
                                   "backmarker": 1.25},
            "notes": ("dirty_air_frac sample too small to fit; retained "
                      "hand-set multipliers 1.00/1.15/1.25"),
        }

    if n < 30:
        return _fallback()

    # binned medians (quantile bins so sparse traffic extremes do not dominate)
    n_bins = min(N_TEI_BINS, max(2, n // 100))
    q_edges = np.quantile(daf, np.linspace(0, 1, n_bins + 1))
    q_edges[-1] += 1e-6
    idx = np.digitize(daf, q_edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    centers, medians = [], []
    for b in range(n_bins):
        sel = beta[idx == b]
        if sel.size >= 3:
            centers.append(float(np.mean(daf[idx == b])))
            medians.append(float(np.median(sel)))
    if len(centers) < 3:
        return _fallback()
    centers = np.asarray(centers)
    medians = np.asarray(medians)

    # robust linear fit beta(daf) = m*daf + c  (c = clean-air degradation slope)
    m, c = _theil_sen(centers, medians)

    def curve(d):
        return m * d + c

    ref = max(curve(0.0), 1e-6)

    # empirical percentile anchors from observed dirty-air distribution
    anchors = {
        "clean": 0.0,
        "drs_train": float(np.quantile(daf, 0.90)),
        "backmarker": float(np.quantile(daf, 0.99)),
    }

    def _mult(anchor):
        return float(min(max(curve(anchor) / ref, 1.0), MULT_CEILING))

    resid = medians - curve(centers)
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((medians - medians.mean()) ** 2).sum())
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-9 else 0.0

    scenario_beta_mult = {
        "clean_air": 1.0,
        "drs_train": round(_mult(anchors["drs_train"]), 4),
        "backmarker": round(_mult(anchors["backmarker"]), 4),
    }
    fit = {
        "method": "binned_median_theil_sen",
        "n_stints": n, "n_bins": n_bins,
        "coefs": {"slope": round(float(m), 6),
                  "intercept": round(float(c), 6)},
        "base_slope": round(float(c), 6),
        "anchors": {k: round(v, 4) for k, v in anchors.items()},
        "mult_ceiling": MULT_CEILING,
        "r2": round(r2, 4),
        "bin_centers": [round(float(x), 4) for x in centers],
        "bin_medians": [round(float(x), 6) for x in medians],
    }
    notes = (f"beta(dirty_air_frac) fitted on {n} dry stints "
             f"({n_bins} quantile bins, robust Theil-Sen fit, R^2={r2:.3f}); "
             "multipliers normalized to clean air and evaluated at empirical "
             f"P90/P99 dirty-air anchors "
             f"({anchors['drs_train']:.3f}/{anchors['backmarker']:.3f}); "
             f"capped at {MULT_CEILING:.1f}x so sparse traffic extremes "
             "cannot blow up the spline.")
    return {"fit": fit, "scenario_beta_mult": scenario_beta_mult, "notes": notes}


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
        pl.col("dirty_air_frac").mean().over(STINT_KEYS).alias("daf"),
    )
    st = (dry.filter((pl.col("n_laps") >= 4) & pl.col("beta").is_not_null())
          .group_by(STINT_KEYS)
          .agg(pl.col("beta").first(), pl.col("n_laps").first(),
               pl.col("t_end").first(), pl.col("life0").first(),
               pl.col("temp").first(), pl.col("comp").first(),
               pl.col("clu").first(), pl.col("daf").first()))
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

    # compound pace offsets (Task 8.0.3 Option A): pairwise within-driver
    # stint-best differences de-contaminated by regressing the measured pace
    # gap against stint TEI and extrapolating to TEI ~ 0.
    #   DeltaPace = beta0 + beta1 * TEI + eps  ->  beta0 = intrinsic grip delta
    # Track evolution is an environmental variable that speeds the whole
    # circuit; separating it from the tyre's mechanical offset prevents
    # double-counting late-race rubbering (which otherwise corrupts the
    # HARD-MEDIUM gap, where M->H one-stoppers run HARD late on a rubbered track).
    tei = _track_evolution(load_lap_features().with_columns(
        pl.col("season").cast(pl.Utf8)))
    laps = df.filter(pl.col("compound").is_in(DRY)).join(
        tei, on=["season", "gp", "session", "lap"], how="left")
    stint_agg = (laps.group_by(["season", "gp", "session", "driver", "compound"])
                 .agg(pl.col("fuel_corr").min().alias("best"),
                      pl.col("track_evolution_index").mean().alias("tei")))
    piv = stint_agg.pivot(on="compound",
                          index=["season", "gp", "session", "driver"],
                          values=["best", "tei"],
                          aggregate_function="first")

    def _fit_offset(piv: pl.DataFrame, comp: str) -> dict:
        """Regress (comp_best - MEDIUM_best) against the paired TEI difference.

        gap = beta0 + beta1 * (TEI_comp - TEI_MEDIUM) + eps
        beta0 (intercept at equal track rubbering) is the intrinsic compound
        grip deficit — the chemical offset with track evolution removed. The
        within-session Delta-TEI is bounded (unlike absolute TEI, which spans
        ~64..3584 and makes a global extrapolation to TEI=0 numerically
        unstable), so this is the well-posed continuous-regression form.

        Returns {"offset": beta0, "slope": beta1, "r2": .., "n": ..} or a
        median fallback {"offset": median, ...} when too few pairs.
        """
        best_col = f"best_{comp}"
        tei_col = f"tei_{comp}"
        if best_col not in piv.columns:
            return {"offset": None, "n": 0}
        d = piv.drop_nulls([best_col, "best_MEDIUM", tei_col, "tei_MEDIUM"])
        gap = (d[best_col] - d["best_MEDIUM"]).to_numpy().astype(float)
        dtei = (d[tei_col] - d["tei_MEDIUM"]).to_numpy().astype(float)
        mask = np.isfinite(gap) & np.isfinite(dtei)
        gap, dtei = gap[mask], dtei[mask]
        n = int(mask.sum())
        if n >= 20:
            b1, b0 = np.polyfit(dtei, gap, 1)
            resid = gap - (b0 + b1 * dtei)
            ss_res = float((resid ** 2).sum())
            ss_tot = float(((gap - gap.mean()) ** 2).sum())
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-9 else 0.0
            return {"offset": float(b0), "slope": float(b1),
                    "r2": r2, "n": n, "median_gap": float(np.median(gap)),
                    "median_dtei": float(np.median(dtei))}
        off = float(np.median(gap)) if n else None
        return {"offset": off, "n": n}

    fit_s = _fit_offset(piv, "SOFT")
    fit_h = _fit_offset(piv, "HARD")

    # intrinsic (TEI-decontaminated) offsets; deterministic priors as fallback
    off_s = fit_s["offset"] if fit_s["offset"] is not None else -0.35
    off_h = fit_h["offset"] if fit_h["offset"] is not None else abs(off_s)
    # physical sanity: SOFT is faster than MEDIUM (negative), HARD is slower
    # than MEDIUM (positive). Clip de-contaminated values to a sane window to
    # guard against sparse-pair regression drift.
    off_s = float(np.clip(off_s, -2.0, -0.01))
    off_h = float(np.clip(off_h, 0.01, 2.0))
    params["pace_offset"] = {"SOFT": round(off_s, 4), "MEDIUM": 0.0,
                             "HARD": round(off_h, 4)}
    params["pace_offset_tei_fit"] = {
        "SOFT": {k: (round(v, 4) if isinstance(v, float) else v)
                 for k, v in fit_s.items()},
        "HARD": {k: (round(v, 4) if isinstance(v, float) else v)
                 for k, v in fit_h.items()},
    }
    params["pace_offset_notes"] = (
        "SOFT-MEDIUM and HARD-MEDIUM offsets obtained by regressing the "
        "within-session pace gap against Delta-TEI and taking the intercept "
        "at equal rubbering (Task 8.0.3 Option A: continuous regression "
        "against track_evolution_index); MEDIUM anchored at 0.")

    # Task 8.0.4 Option A: replace hand-set scenario multipliers (1.00/1.15/1.25)
    # with an empirically fitted beta(dirty_air_frac) relationship. Running in
    # dirty air costs front-axle downforce -> micro-sliding + thermal surface
    # degradation, a non-linear penalty captured by a binned/spline fit of the
    # per-stint degradation slope against dirty-air fraction.
    daf_fit = _fit_dirty_air(stinfo=st)
    params["dirty_air_fit"] = daf_fit["fit"]
    params["scenario_beta_mult"] = daf_fit["scenario_beta_mult"]
    params["scenario_notes"] = daf_fit["notes"]

    if not save:
        return params
    out = MODELS_DIR / "degradation_model.json"
    with open(out, "w") as fh:
        json.dump(params, fh, indent=2)
    print(f"[deg] saved -> {out}")
    print(f"     beta: {params['beta']}")
    print(f"     pace offsets (TEI-decontaminated): {params['pace_offset']}")
    print(f"       TEI fit: {params['pace_offset_tei_fit']}")
    print(f"     scenario beta_mult (dirty-air fitted): "
          f"{params['scenario_beta_mult']}")
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
