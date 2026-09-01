"""Phase 7 â€” Backtest: did the optimizer's strategy beat the actual one?

Design (honest, out-of-sample):
  - Decision point per (2025 race, driver): the lap where stint 1 actually
    ended (the first real pit window). Drivers with <4-lap stint 1 or <8
    remaining laps are skipped.
  - Degradation parameters are REFIT on 2021-2024 only (2025 excluded) so the
    backtest is out-of-sample for the strategy model too.
  - The ACTUAL remaining strategy (compounds + stint end laps from raw
    lap_features) and the OPTIMIZER's best candidate are evaluated through the
    SAME Monte Carlo simulator with common random numbers -> paired delta.
  - delta = mean(actual) - mean(recommended); delta > 0 => optimizer wins.

Caveat (documented): the recommended strategy is chosen BY the simulator, so
the comparison carries selection bias in the simulator's favor; it measures
consistency with the fitted degradation/cliff physics, not counterfactual
race outcomes (impossible without a re-simulation of reality).

Ablations: (a) no cliff risk (cliff_penalty=0), (b) no compound pace offsets,
(c) no style cluster offsets â€” how often the recommendation changes.

    python -m src.models.backtest [--n-sims 250] [--races 8]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import polars as pl

from src.ingest.config import ROOT
from src.models import optimizer as OPT
from src.models.eval_rul import build_labeled_frame
from src.models.fit_degradation import fit_degradation
from src.models.rul_data import STINT_KEYS

MODELS_DIR = ROOT / "data" / "models"
GOLD = ROOT / "data" / "gold"


def actual_strategies(lapf: pl.DataFrame):
    """Per (season,gp,session,driver): list of (compound, start_lap, end_lap)."""
    out = {}
    sub = lapf.filter(
        (pl.col("season") == "2025") & (pl.col("session") == "Race")
        & pl.col("stint").is_not_null() & pl.col("laptime").is_not_null())
    for key, grp in sub.group_by(["season", "gp", "session", "driver"]):
        stints = []
        for stint, sg in grp.group_by("stint", maintain_order=True):
            sg = sg.sort("lap")
            stints.append((sg["compound"][0], int(sg["lap"].min()),
                           int(sg["lap"].max())))
        stints.sort(key=lambda s: s[1])
        out[key] = stints
    return out


def make_state(frame, race, driver, e0, lapf):
    """State at the decision lap, referencing the last FLYING lap (e0 is the
    in-lap: slow pace + always-high pace_above_best)."""
    state = OPT.get_state(frame, "2025", race, "Race", driver, e0, lapf=lapf)
    prev = frame.filter(
        (pl.col("season") == "2025") & (pl.col("gp") == race)
        & (pl.col("session") == "Race") & (pl.col("driver") == driver)
        & (pl.col("lap") == e0 - 1))
    if prev.height:
        state["base_pace"] = float(prev["fuel_corr"][0])
        state["in_cliff_now"] = bool((prev["pace_above_best"][0] or 0) > 1.5)
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sims", type=int, default=250)
    ap.add_argument("--races", type=int, default=0, help="cap # races (0=all)")
    args = ap.parse_args()
    t0 = time.time()
    ns = args.n_sims

    print("[bt] building labeled frame...")
    frame = build_labeled_frame()
    emb = pl.read_parquet(GOLD / "driver_embeddings.parquet")
    emb = emb.with_columns(pl.col("season").cast(pl.Utf8))
    frame = frame.join(emb.select(["season", "driver", "cluster"]),
                       on=["season", "driver"], how="left")
    frame = frame.with_columns(pl.col("cluster").fill_null(-1).cast(pl.Int32))

    print("[bt] refitting degradation params on 2021-2024 (2025 held out)...")
    params = fit_degradation(df=frame.filter(pl.col("season") != "2025"),
                             save=False)

    lapf = pl.read_parquet(GOLD / "lap_features.parquet")
    lapf = lapf.with_columns(pl.col("season").cast(pl.Utf8))
    actual = actual_strategies(lapf)
    print(f"[bt] {len(actual)} driver-races in 2025")

    # exclude rain-affected races: wet-compound strategies cannot be scored
    # by the dry degradation model
    wet = (lapf.filter((pl.col("season") == "2025") & (pl.col("session") == "Race"))
           .group_by("gp").agg(
               (pl.col("compound").is_in(["INTERMEDIATE", "WET"]).mean())
               .alias("wet_frac")))
    wet_races = {r["gp"] for r in wet.to_dicts() if r["wet_frac"] > 0.02}
    print(f"[bt] excluding {len(wet_races)} rain-affected races: "
          f"{sorted(wet_races)}")

    races = sorted({k[1] for k in actual} - wet_races)
    if args.races:
        races = races[:args.races]

    rows = []
    skipped = 0
    for race in races:
        for key, stints in actual.items():
            if key[1] != race or len(stints) < 2:
                continue
            comp0, s0, e0 = stints[0]
            if e0 - s0 + 1 < 4:
                skipped += 1
                continue
            n_laps = int(lapf.filter((pl.col("gp") == race)
                                     & (pl.col("session") == "Race"))["lap"].max())
            remaining = n_laps - e0
            if remaining < 8:
                skipped += 1
                continue
            # DNF / truncated data: actual stints must reach the race end
            if stints[-1][2] < n_laps - 3:
                skipped += 1
                continue
            try:
                state = make_state(frame, race, key[3], e0, lapf)
            except SystemExit:
                skipped += 1
                continue
            if state["compound"] not in ("SOFT", "MEDIUM", "HARD"):
                skipped += 1
                continue
            # actual remaining strategy: pit now, then the real stints
            actual_strat = [(state["compound"], e0, True)] + [
                (c, e, False) for c, s, e in stints[1:] if c in
                ("SOFT", "MEDIUM", "HARD")]
            if not any(not cur for _, _, cur in actual_strat):
                skipped += 1
                continue
            # paired MC evaluation (common random numbers)
            rng_a = np.random.default_rng(7)
            act = OPT.simulate(actual_strat, state, params,
                               OPT.SCENARIOS["clean_air"], ns, rng_a)
            rec = OPT.optimize(state, params, ns,
                               {"clean_air": OPT.SCENARIOS["clean_air"]})
            best = rec["clean_air"][0]
            rows.append({
                "gp": race, "driver": key[3],
                "actual_stops": sum(1 for c, e, cur in actual_strat if not cur),
                "rec_stops": best["stops"],
                "actual_strategy": " | ".join(
                    f"{c}->{e}" for c, e, cur in actual_strat),
                "rec_strategy": best["strategy"],
                "actual_mean": round(float(act.mean()), 1),
                "rec_mean": round(best["mean"], 1),
                "delta": round(float(act.mean() - best["mean"]), 1),
            })

    print(f"[bt] {len(rows)} decisions evaluated ({skipped} skipped), "
          f"{time.time()-t0:.0f}s")

    deltas = np.array([r["delta"] for r in rows])
    wins = int((deltas > 0).sum())
    summary = {
        "n_decisions": len(rows), "skipped": skipped,
        "n_races": len(races), "n_sims": ns,
        "win_rate": round(wins / max(len(rows), 1), 3),
        "delta_mean": round(float(deltas.mean()), 1) if len(rows) else None,
        "delta_median": round(float(np.median(deltas)), 1) if len(rows) else None,
        "delta_p10": round(float(np.quantile(deltas, 0.10)), 1) if len(rows) else None,
        "delta_p90": round(float(np.quantile(deltas, 0.90)), 1) if len(rows) else None,
        "mean_actual_stops": round(float(np.mean([r["actual_stops"] for r in rows])), 2) if rows else None,
        "mean_rec_stops": round(float(np.mean([r["rec_stops"] for r in rows])), 2) if rows else None,
    }
    print("\n" + "=" * 78)
    print("PHASE 7 â€” BACKTEST (2025 races, decision at stint-1 end, clean air)")
    print("=" * 78)
    print(f"  decisions      : {summary['n_decisions']}  "
          f"({summary['skipped']} skipped)")
    print(f"  optimizer wins : {wins} ({summary['win_rate']:.1%})")
    print(f"  delta (actual - recommended, seconds):")
    print(f"    mean={summary['delta_mean']}  median={summary['delta_median']}  "
          f"p10={summary['delta_p10']}  p90={summary['delta_p90']}")
    print(f"  avg stops: actual={summary['mean_actual_stops']}  "
          f"recommended={summary['mean_rec_stops']}")

    # --- ablations on a subsample (how often the recommendation changes) ---
    print("\n[bt] ablations (recommendation sensitivity)...")
    abl = {}
    sub_rows = rows[:120] if len(rows) > 120 else rows
    variants = {
        "no_cliff": {**params, "cliff_jump": 0.0, "cliff_slope": 0.0},
        "no_pace_offsets": {**params, "pace_offset": {
            "SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}},
        "no_style_cluster": {**params, "cluster_offset": {}},
    }
    for vname, vp in variants.items():
        changed = 0
        d_ws = []
        for r in sub_rows:
            e0 = int(r["actual_strategy"].split("->")[1].split(" ")[0])
            state = make_state(frame, r["gp"], r["driver"], e0, lapf)
            rec = OPT.optimize(state, vp, ns,
                               {"clean_air": OPT.SCENARIOS["clean_air"]})
            b = rec["clean_air"][0]
            if b["strategy"] != r["rec_strategy"]:
                changed += 1
            d_ws.append(r["actual_mean"] - b["mean"])
        abl[vname] = {"changed_recommendation": changed,
                      "pct": round(changed / max(len(sub_rows), 1), 3),
                      "delta_mean_vs_actual": round(float(np.mean(d_ws)), 1)}
        print(f"  {vname:18s} recommendation changed in {changed}/"
              f"{len(sub_rows)} decisions; delta={abl[vname]['delta_mean_vs_actual']}s")

    out = {"summary": summary, "ablations": abl,
           "decisions": rows[:200],
           "notes": ("Degradation params refit on 2021-2024 only; paired MC "
                     "with common random numbers; recommended strategy chosen "
                     "by the same simulator (selection bias documented).")}
    with open(MODELS_DIR / "phase7_backtest.json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\n[bt] saved -> {MODELS_DIR / 'phase7_backtest.json'} "
          f"({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()

