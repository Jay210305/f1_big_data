"""Phase 6 — Prescriptive Dynamic Pit-Window & Stint Optimizer.

Components:
  1. Degradation model (data/models/degradation_model.json, fitted by
     fit_degradation.py): beta per compound (+style/temp), sigma, empirical
     cliff-lap distributions, compound pace offsets.
  2. Monte Carlo strategy evaluator: analytic stint times (no lap loops),
     N simulations per candidate, pit loss + warm-up per stop.
  3. Candidate search: 0/1/2-stop grids over pit laps x compounds, with the
     F1 two-compound rule enforced.
  4. Scenario context: clean_air / drs_train / backmarker multipliers on
     degradation rate and pace.
  5. In-race Pit Monitor (backtest on real stints): Phase 4 RUL (XGBoost) +
     cliff probability (XGB+LGB ensemble) per lap, with the transition-spec
     rules: debouncing (>=2 consecutive laps above threshold) and the joint
     cliff/RUL anomaly rule (cliff alert + RUL > 15 => thermal anomaly, no pit).

CLI:
    python -m src.models.optimizer                       # demo: 2025 Bahrain VER
    python -m src.models.optimizer --gp "Spanish Grand Prix" --driver HAM --lap 20
    python -m src.models.optimizer --n-sims 1000 --pit-loss 21
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import polars as pl

from src.ingest.config import ROOT
from src.models.eval_rul import build_labeled_frame
from src.models.rul_data import STINT_KEYS

MODELS_DIR = ROOT / "data" / "models"
DRY = ["SOFT", "MEDIUM", "HARD"]

SCENARIOS = {
    "clean_air": {"beta_mult": 1.00, "pace_pen": 0.00},
    "drs_train": {"beta_mult": 1.15, "pace_pen": 0.25},
    "backmarker": {"beta_mult": 1.25, "pace_pen": 0.50},
}
PIT_LOSS = 21.0          # s lost per stop (circuit-dependent; CLI flag)
NEW_STINT_WARMUP = 2.0   # s lump per stop (pit exit + tyre warm-up)
CLIFF_BETA_MULT = 3.0    # degradation accelerates 3x after the cliff
JOINT_RUL_LIMIT = 15     # cliff alert + RUL > this => anomaly, no pit
RUL_IMMINENT = 2         # RUL <= this => pit now regardless


# ---------------------------------------------------------------------------
# Phase 4 model loading + per-lap inference
# ---------------------------------------------------------------------------
def load_phase4():
    import xgboost as xgb
    import lightgbm as lgb
    with open(MODELS_DIR / "feature_meta.json") as fh:
        meta = json.load(fh)
    with open(MODELS_DIR / "phase4_metrics.json") as fh:
        m4 = json.load(fh)
    rul_m = xgb.Booster()
    rul_m.load_model(str(MODELS_DIR / "xgb_rul.json"))
    cliff_x = xgb.Booster()
    cliff_x.load_model(str(MODELS_DIR / "xgb_cliff.json"))
    cliff_l = lgb.Booster(model_file=str(MODELS_DIR / "lgb_cliff.txt"))
    thr = float(m4["xgb_cliff"]["threshold"])
    return meta, rul_m, cliff_x, cliff_l, thr


def encode_frame(df: pl.DataFrame, meta) -> np.ndarray:
    X = df.select(meta["features"]).clone()
    for c, codes in meta["codes"].items():
        X = X.with_columns(pl.col(c).replace(codes).cast(pl.Int32).alias(c))
    return X.to_pandas().to_numpy(dtype=np.float32, na_value=np.nan)


def predict_laps(df: pl.DataFrame, meta, rul_m, cliff_x, cliff_l):
    import xgboost as xgb
    Xn = encode_frame(df, meta)
    rul = np.clip(rul_m.predict(xgb.DMatrix(Xn)), 0, None)
    prob = 0.5 * (cliff_x.predict(xgb.DMatrix(Xn)) + cliff_l.predict(Xn))
    return rul, prob


# ---------------------------------------------------------------------------
# Monte Carlo strategy simulation (analytic stint times)
# ---------------------------------------------------------------------------
def _sample_cliff(rng, cliff_cfg, n):
    """cliff_rem: laps until cliff (relative to stint/now start); inf if none."""
    out = np.full(n, np.inf)
    cen = cliff_cfg["censor_prob"]
    qs = np.array(cliff_cfg["quantiles"])
    wet = rng.random(n) < cen
    out[~wet] = qs[rng.integers(0, len(qs), size=int((~wet).sum()))]
    return out


def _sample_beta(rng, mu, sigma, n):
    """Beta sampler with a physical cap: the raw slope distribution has
    traffic-locked outliers, so uncapped samples + the lower clip would
    explode E[beta] far above the fitted median rate."""
    lo, hi = 0.005, min(mu * 4.0 + 0.15, 1.0)
    return np.clip(rng.normal(mu, sigma, n), lo, hi)


def stint_time(base, pen, beta, L, cliff_rem, jump, slope):
    """Vectorized total seconds for one stint of L laps.

    lap k (k=1..L from stint/now start): base + pen + beta*k
    cliff at k > cliff_rem: one-time +jump at crossing, then +slope per lap.
    deg term = beta * L(L+1)/2 ; cliff = jump*m + slope*m(m+1)/2 with
    m = laps in cliff = max(0, L - max(cliff_rem, 0)).
    """
    deg = beta * L * (L + 1) / 2.0
    m = np.maximum(L - np.maximum(cliff_rem, 0.0), 0.0)
    cliff_cost = jump * m + slope * m * (m + 1) / 2.0
    return L * (base + pen) + deg + cliff_cost


def simulate(strategy, state, params, scen, n_sims, rng):
    """Returns (n_sims,) total seconds for the remaining race."""
    totals = np.zeros(n_sims)
    jump = params.get("cliff_jump", 1.5)
    slope = params.get("cliff_slope", 0.15)
    scen_pen = scen["pace_pen"]
    for i, (comp, end_lap, is_current) in enumerate(strategy):
        L = end_lap - (state["cur_lap"] if is_current else
                       strategy[i - 1][1])
        L = max(int(L), 0)
        if L == 0:
            continue
        if is_current:
            base = state["base_pace"]
            beta_mu = (params["beta"][comp]
                       + params["cluster_offset"].get(str(state["cluster"]), 0.0)
                       + params["temp_coef"] * (state["temp"] - params["temp_mean"]))
            beta = _sample_beta(rng, beta_mu, params["sigma"][comp], n_sims)
            cliff_rem = _sample_cliff(rng, params["cliff"][comp], n_sims) \
                - state["laps_into_stint"]
            if state["in_cliff_now"]:
                cliff_rem = np.minimum(cliff_rem, -1.0)
        else:
            base = (state["base_pace"]
                    + params["pace_offset"][comp]
                    - params["pace_offset"][state["compound"]])
            beta_mu = (params["beta"][comp]
                       + params["cluster_offset"].get(str(state["cluster"]), 0.0)
                       + params["temp_coef"] * (state["temp"] - params["temp_mean"]))
            beta = _sample_beta(rng, beta_mu, params["sigma"][comp], n_sims)
            cliff_rem = _sample_cliff(rng, params["cliff"][comp], n_sims)
        beta = beta * scen["beta_mult"]
        totals += stint_time(base, scen_pen, beta, L, cliff_rem, jump, slope)
    n_stops = sum(1 for c, e, cur in strategy if not cur and e > state["cur_lap"])
    totals += n_stops * (PIT_LOSS + NEW_STINT_WARMUP)
    return totals


def enumerate_strategies(state):
    """Strategies as [(compound, end_lap, is_current), ...]."""
    cur, n = state["cur_lap"], state["n_laps"]
    used = set(state["compounds_used"])
    out = [[(state["compound"], n, True)]]
    for p1 in range(cur + 2, n - 2, 2):
        for c1 in DRY:
            if len(used | {c1}) < 2 and n - p1 > 3:
                continue
            out.append([(state["compound"], p1, True), (c1, n, False)])
    for p1 in range(cur + 3, n - 10, 4):
        for p2 in range(p1 + 6, n - 2, 4):
            for c1 in DRY:
                for c2 in DRY:
                    if len(used | {c1, c2}) < 2:
                        continue
                    out.append([(state["compound"], p1, True),
                                (c1, p2, False), (c2, n, False)])
    if len(out) > 400:
        idx = np.linspace(0, len(out) - 1, 400).astype(int)
        out = [out[i] for i in idx]
    return out


def optimize(state, params, n_sims, scenarios):
    rng = np.random.default_rng(42)
    results = {}
    cands = enumerate_strategies(state)
    for sname, scen in scenarios.items():
        rows = []
        for strat in cands:
            tot = simulate(strat, state, params, scen, n_sims, rng)
            desc = " | ".join(
                f"{c}->{e}" for c, e, cur in strat)
            rows.append({
                "strategy": desc,
                "stops": sum(1 for c, e, cur in strat if not cur),
                "mean": float(tot.mean()), "std": float(tot.std()),
                "p5": float(np.quantile(tot, 0.05)),
                "p95": float(np.quantile(tot, 0.95)),
                "_strat": strat,
            })
        rows.sort(key=lambda r: r["mean"])
        best = rows[0]["mean"]
        for r in rows:
            r["delta_vs_best"] = r["mean"] - best
        results[sname] = rows
    return results


# ---------------------------------------------------------------------------
# In-race Pit Monitor (transition-spec rules) — backtest on real stints
# ---------------------------------------------------------------------------
def run_monitor(df_driver, meta, rul_m, cliff_x, cliff_l, thr, label=""):
    """Applies debounced cliff + joint RUL rules lap by lap per stint."""
    print(f"\n[pit-monitor] {label}")
    df_driver = df_driver.sort(STINT_KEYS + ["life"])
    for (key, grp) in df_driver.group_by(STINT_KEYS, maintain_order=True):
        season, gp, session, driver, stint = key
        grp = grp.sort("life")
        rul, prob = predict_laps(grp, meta, rul_m, cliff_x, cliff_l)
        laps = grp["lap"].to_numpy()
        life = grp["life"].to_numpy()
        n = len(laps)
        above = prob > thr
        trigger = None
        anomaly = None
        imminent = None
        deb = 0
        for i in range(n):
            if rul[i] <= RUL_IMMINENT:
                imminent = int(laps[i])
                break
            if above[i]:
                deb += 1
                if deb >= 2:
                    if rul[i] > JOINT_RUL_LIMIT:
                        anomaly = int(laps[i])
                        deb = 0
                    else:
                        trigger = int(laps[i])
                        break
            else:
                deb = 0
        actual_end = int(laps[-1])
        comp = grp["compound"][0]
        print(f"  stint {stint} ({comp}, {n} laps, laps {laps[0]}-{actual_end}): "
              f"pit_trigger={trigger if trigger else '-'} "
              f"anomaly_flag={anomaly if anomaly else '-'} "
              f"rul_imminent={imminent if imminent else '-'} "
              f"| actual stint end lap {actual_end}")


# ---------------------------------------------------------------------------
# Demo: real race state -> optimization + monitor backtest
# ---------------------------------------------------------------------------
def get_state(df, season, gp, session, driver, lap, lapf=None):
    sub = df.filter((pl.col("season") == season) & (pl.col("gp") == gp)
                    & (pl.col("session") == session)
                    & (pl.col("driver") == driver))
    if sub.height == 0:
        raise SystemExit(f"no data for {season} {gp} {session} {driver}")
    # race length from the RAW lap table (the filtered labeled frame can drop
    # short final stints and undercount the race distance)
    if lapf is not None:
        n_laps = int(lapf.filter((pl.col("season") == season)
                                 & (pl.col("gp") == gp)
                                 & (pl.col("session") == session))["lap"].max())
    else:
        n_laps = int(df.filter((pl.col("season") == season)
                               & (pl.col("gp") == gp)
                               & (pl.col("session") == session))["lap"].max())
    used = sorted(sub.filter(pl.col("lap") <= lap)["compound"]
                  .unique().to_list())
    rows = (sub.filter(pl.col("lap") == lap).sort("life").to_dicts())
    if not rows:
        raise SystemExit(f"lap {lap} not found for {driver} in {gp} {session}")
    row = rows[-1]
    stint = row["stint"]
    sst = sub.filter(pl.col("stint") == stint).sort("life")
    pos = int((sst["lap"] <= lap).sum()) - 1
    return {
        "season": season, "gp": gp, "session": session, "driver": driver,
        "cur_lap": int(lap), "n_laps": n_laps, "compound": row["compound"],
        "stint": stint, "laps_into_stint": max(pos, 0),
        "base_pace": float(row["fuel_corr"]),
        "in_cliff_now": bool((row["pace_above_best"] or 0) > 1.5),
        "temp": float(row["track_temp_mean"] or 36.0),
        "cluster": int(row["cluster"]),
        "compounds_used": used,
    }


def main():
    global PIT_LOSS
    ap = argparse.ArgumentParser()
    ap.add_argument("--gp", default="Bahrain Grand Prix")
    ap.add_argument("--driver", default="VER")
    ap.add_argument("--season", default="2025")
    ap.add_argument("--lap", type=int, default=12)
    ap.add_argument("--n-sims", type=int, default=400)
    ap.add_argument("--pit-loss", type=float, default=PIT_LOSS)
    args = ap.parse_args()
    PIT_LOSS = args.pit_loss
    t0 = time.time()

    with open(MODELS_DIR / "degradation_model.json") as fh:
        params = json.load(fh)
    meta, rul_m, cliff_x, cliff_l, thr = load_phase4()

    print("[opt] building labeled frame...")
    df = build_labeled_frame()
    emb = pl.read_parquet(ROOT / "data" / "gold" / "driver_embeddings.parquet")
    emb = emb.with_columns(pl.col("season").cast(pl.Utf8))
    df = df.join(emb.select(["season", "driver", "cluster"]),
                 on=["season", "driver"], how="left")
    df = df.with_columns(pl.col("cluster").fill_null(-1).cast(pl.Int32))

    state = get_state(df, args.season, args.gp, "Race", args.driver, args.lap,
                      lapf=pl.read_parquet(ROOT / "data" / "gold"
                                           / "lap_features.parquet"))
    print(f"\n[state] {args.season} {args.gp} Race | {args.driver} @ lap "
          f"{state['cur_lap']}/{state['n_laps']}")
    print(f"        stint {state['stint']} on {state['compound']} "
          f"({state['laps_into_stint']} laps in, in_cliff={state['in_cliff_now']})")
    print(f"        track {state['temp']:.1f}C, style cluster {state['cluster']}, "
          f"compounds used so far: {state['compounds_used']}")

    results = optimize(state, params, args.n_sims, SCENARIOS)
    for sname, rows in results.items():
        print(f"\n[strategy] scenario: {sname} "
              f"(beta x{SCENARIOS[sname]['beta_mult']}, "
              f"pace +{SCENARIOS[sname]['pace_pen']}s/lap)")
        print(f"  {'strategy':32s} {'stops':>5s} {'mean':>8s} {'std':>6s} "
              f"{'p5':>8s} {'p95':>8s} {'dvs best':>9s}")
        for r in rows[:6]:
            print(f"  {r['strategy']:32s} {r['stops']:5d} {r['mean']:8.1f} "
                  f"{r['std']:6.1f} {r['p5']:8.1f} {r['p95']:8.1f} "
                  f"{r['delta_vs_best']:+9.1f}")

    # monitor backtest on the same driver's actual race stints
    drv_df = df.filter((pl.col("season") == args.season)
                       & (pl.col("gp") == args.gp)
                       & (pl.col("session") == "Race")
                       & (pl.col("driver") == args.driver))
    run_monitor(drv_df, meta, rul_m, cliff_x, cliff_l, thr,
                f"{args.season} {args.gp} Race {args.driver} "
                f"(threshold {thr:.3f})")

    out = {
        "state": {k: v for k, v in state.items()},
        "n_sims": args.n_sims, "pit_loss": args.pit_loss,
        "scenarios": {s: [{k: v for k, v in r.items() if k != "_strat"}
                          for r in rows[:10]] for s, rows in results.items()},
    }
    with open(MODELS_DIR / "phase6_demo.json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\n[opt] demo saved -> {MODELS_DIR / 'phase6_demo.json'} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
