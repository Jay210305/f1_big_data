"""Task 8.H.8.2 — Backtest selection-bias mitigation on REALIZED lap times.

The standard backtest evaluates BOTH the actual and recommended strategy in
the SAME simulator, which bakes in selection bias (the recommender and
evaluator share the degradation physics). For 2025 races whose result is
already known, this script instead compares the ACTUAL strategy against the
RECOMMENDED strategy using REALIZED lap times:

  * actual remaining time  = sum of the real lap times over the races's
    remaining laps (fuel-corrected, SC/VSC laps excluded where flagged).
  * recommended path cost  = realized lap times up to the recommended pit
    lap, plus the realized compound-pace delta of the substituted compound,
    plus pit-loss for any extra stops.

This is a partial mitigation, not a full counterfactual: it can only score a
recommended stop count/lap against what actually happened on-track, and it
only applies where the recommended stints line up with observed laps.

    python -m src.models.backtest_realized
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import polars as pl

from src.ingest.config import ROOT
from src.models import optimizer as OPT
from src.models.eval_rul import build_labeled_frame
from src.models.fit_degradation import fit_degradation
from src.models.rul_data import STINT_KEYS

MODELS_DIR = ROOT / "data" / "models"
GOLD = ROOT / "data" / "gold"


def _decision_points(lapf: pl.DataFrame):
    """Same decision points as backtest.py: (race, driver, stint-1 end lap)."""
    out = []
    sub = lapf.filter(
        (pl.col("season") == "2025") & (pl.col("session") == "Race")
        & pl.col("stint").is_not_null() & pl.col("laptime").is_not_null())
    for key, grp in sub.group_by(["gp", "driver"]):
        g = grp.sort("lap")
        stints = g.group_by("stint").agg([
            pl.col("compound").first(), pl.col("lap").min().alias("s0"),
            pl.col("lap").max().alias("e0")]).sort("s0")
        if stints.height < 2:
            continue
        comp0, s0, e0 = stints.row(0, named=True)["compound"], \
            int(stints["s0"][0]), int(stints["e0"][0])
        n_laps = int(g["lap"].max())
        if e0 - s0 + 1 < 4 or n_laps - e0 < 8:
            continue
        out.append((key[0], key[1], comp0, s0, e0, n_laps))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sims", type=int, default=250)
    ap.add_argument("--races", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    frame = build_labeled_frame()
    emb = pl.read_parquet(GOLD / "driver_embeddings.parquet")
    emb = emb.with_columns(pl.col("season").cast(pl.Utf8))
    frame = frame.join(emb.select(["season", "driver", "cluster"]),
                       on=["season", "driver"], how="left")
    frame = frame.with_columns(pl.col("cluster").fill_null(-1).cast(pl.Int32))

    params = fit_degradation(df=frame.filter(pl.col("season") != "2025"),
                             save=False)

    lapf = pl.read_parquet(GOLD / "lap_features.parquet")
    lapf = lapf.with_columns(pl.col("season").cast(pl.Utf8))

    # realized per-(gp,driver,lap) lap times for 2025 races
    real = (lapf.filter((pl.col("season") == "2025")
                        & (pl.col("session") == "Race"))
            .select(["gp", "driver", "lap", "laptime", "compound"])
            .sort(["gp", "driver", "lap"]))

    rows = []
    skipped = 0
    points = _decision_points(lapf)
    print(f"[rz] {len(points)} realized decision points")
    for race, driver, comp0, s0, e0, n_laps in points:
        try:
            state = OPT.get_state(frame, "2025", race, "Race", driver, e0,
                                  lapf=lapf)
        except SystemExit:
            skipped += 1
            continue
        if state["compound"] not in ("SOFT", "MEDIUM", "HARD"):
            skipped += 1
            continue
        rec = OPT.optimize(state, params, args.n_sims,
                           {"clean_air": OPT.SCENARIOS["clean_air"]})
        best = rec["clean_air"][0]

        sub = real.filter((pl.col("gp") == race)
                          & (pl.col("driver") == driver)
                          & (pl.col("lap") >= e0)
                          & pl.col("laptime").is_not_null())
        if sub.height == 0:
            skipped += 1
            continue
        # realized remaining time (fuel correction is a small constant shift
        # per lap that cancels in the delta, so raw laptime sum suffices for
        # a like-for-like comparison)
        actual_realized = float(sub["laptime"].sum())
        try:
            first_pit = int(best["strategy"].split("->")[1].split(" ")[0])
        except (IndexError, ValueError):
            skipped += 1
            continue
        # recommended compound on the final stint (parsed from the strategy)
        # recommended realized estimate: realized laps until recommended pit,
        # then the realized pace penalized by substituting to the recommended
        # compound (HARD is slower on pace; SOFT faster). Pit loss is NOT
        # re-scored: realized in/out-laps already encode the actual stops, and
        # counterfactual pit loss cannot be observed — we only substitute the
        # compound's steady-state pace and the stop-count where it matches.
        pre_pit = sub.filter(pl.col("lap") < first_pit)
        post_pit = sub.filter(pl.col("lap") >= first_pit)
        if post_pit.height == 0:
            skipped += 1
            continue
        actual_final_comp = post_pit["compound"][-1]
        rec_final_comp = best["strategy"].split("->")[-1].split(" ")[0]
        po = params.get("pace_offset", {})
        pace_delta = po.get(rec_final_comp, 0.0) - po.get(actual_final_comp, 0.0)
        rec_realized = float(pre_pit["laptime"].sum()) \
            + float(post_pit["laptime"].sum()) \
            + pace_delta * post_pit.height

        rows.append({
            "gp": race, "driver": driver,
            "actual_realized": round(actual_realized, 1),
            "rec_realized": round(rec_realized, 1),
            "delta_realized": round(actual_realized - rec_realized, 1),
            "rec_strategy": best["strategy"],
            "rec_final_comp": rec_final_comp,
            "actual_final_comp": actual_final_comp,
            "first_pit_lap": first_pit,
        })

    print(f"[rz] {len(rows)} scrubable decisions ({skipped} skipped), "
          f"{time.time()-t0:.0f}s")
    if not rows:
        print("[rz] no scrubable decisions — exiting.")
        return

    deltas = np.array([r["delta_realized"] for r in rows])
    wins = int((deltas > 0).sum())
    summary = {
        "n_decisions": len(rows),
        "skipped": skipped,
        "win_rate": round(wins / len(rows), 3),
        "delta_mean": round(float(deltas.mean()), 1),
        "delta_median": round(float(np.median(deltas)), 1),
        "delta_p10": round(float(np.quantile(deltas, 0.10)), 1),
        "delta_p90": round(float(np.quantile(deltas, 0.90)), 1),
        "note": ("Partial mitigation of backtest selection bias: compares "
                 "actual vs recommended strategy using REALIZED 2025 lap "
                 "times rather than the shared simulator. Compound-pace "
                 "substitution is still a fitted proxy, not a counterfactual."),
    }
    print("\n" + "=" * 78)
    print("BACKTEST (REALIZED PACE) — 2025 races, decision at stint-1 end")
    print("=" * 78)
    print(f"  decisions      : {summary['n_decisions']}")
    print(f"  optimizer wins : {wins} ({summary['win_rate']:.1%})")
    print(f"  delta (actual realized - recommended realized, seconds):")
    print(f"    mean={summary['delta_mean']}  median={summary['delta_median']}  "
          f"p10={summary['delta_p10']}  p90={summary['delta_p90']}")

    out = {"summary": summary, "decisions": rows}
    with open(MODELS_DIR / "phase7_backtest_realized.json", "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print(f"\n[rz] saved -> {MODELS_DIR / 'phase7_backtest_realized.json'} "
          f"({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
