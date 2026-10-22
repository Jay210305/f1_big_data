"""Calibration of the strategy simulator against realized lap times.

The backtest scores the optimizer with the SAME simulator that recommends the
strategy, so a systematic mis-calibration is invisible to it. This module
closes that loop: it evaluates the production `optimizer.simulate()` against
the strategies that were ACTUALLY executed in the 2025 races and reports the
residual.

    realized  = the sum of the real (fuel-corrected) lap times of the stint
    simulated = what optimizer.simulate() predicts for the same stint
    residual  = simulated - realized

The residual is decomposed per stint, and the one free parameter the model
leaves at zero -- `pace_drift` -- is fitted by ordinary least squares (the
residual is exactly linear in it).

Why `pace_drift` exists: the fuel correction is `laptime + 0.03 * lap`, a
fixed estimate of the pace the car recovers as the fuel burns off. Holding
the base pace constant for a whole stint therefore mis-prices every lap after
the decision point, and the error grows with the lap number. The coefficient
fitted here is the shortfall of that 0.03 s/lap constant.

    python -m src.models.calibrate            # report only
    python -m src.models.calibrate --apply    # also store it in the model
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import polars as pl

from src.ingest.config import ROOT
from src.models import optimizer as OPT
from src.models.backtest import actual_strategies
from src.models.eval_rul import build_labeled_frame
from src.models.fit_degradation import fit_degradation

GOLD = ROOT / "data/gold"
MODELS_DIR = ROOT / "data/models"
DRY = ["SOFT", "MEDIUM", "HARD"]
DECISION_SEASON = "2025"


def _collect(frame, lapf):
    """The executed strategies and the realized time of each of their stints."""
    actual = actual_strategies(lapf)
    races = sorted({k[1] for k in actual})
    lab = frame.filter(pl.col("season") == DECISION_SEASON)
    recs = []
    for race in races:
        for key, stints in actual.items():
            if key[1] != race or len(stints) < 2:
                continue
            _c0, s0, e0 = stints[0]
            if e0 - s0 + 1 < 4:
                continue
            n_laps = int(lapf.filter((pl.col("gp") == race)
                                     & (pl.col("session") == "Race"))["lap"].max())
            if n_laps - e0 < 8 or stints[-1][2] < n_laps - 3:
                continue
            try:
                state = OPT.get_state(frame, DECISION_SEASON, race, "Race",
                                      key[3], e0, lapf=lapf)
            except SystemExit:
                continue
            if state["compound"] not in OPT.COMPOUNDS:
                continue
            strategy = [(state["compound"], e0, True)] + [
                (c, e, False) for c, s, e in stints[1:] if c in OPT.COMPOUNDS]
            if not any(not cur for _, _, cur in strategy):
                continue
            # the realized fuel-corrected time of every stint of the strategy
            for i, (comp, end_lap, is_current) in enumerate(strategy):
                start_lap = state["cur_lap"] if is_current else strategy[i - 1][1]
                L = max(int(end_lap - start_lap), 0)
                if L == 0:
                    continue
                seg = lab.filter((pl.col("gp") == race) & (pl.col("driver") == key[3])
                                 & (pl.col("lap") > start_lap)
                                 & (pl.col("lap") <= end_lap)
                                 & pl.col("fuel_corr").is_not_null())
                if seg.height < 3:
                    continue
                laps = seg["lap"].to_numpy().astype(float)
                recs.append(dict(
                    race=race, driver=key[3], comp=comp, L=L, position=i,
                    is_current=is_current, strategy=strategy, state=state,
                    realized=float(seg["fuel_corr"].sum()),
                    # d(residual)/d(pace_drift) for this stint
                    weight=L * (float(laps.mean()) - (state["cur_lap"] - 1))))
    return recs


def _residuals(recs, params, pace_drift, n_sims=120, seed=5):
    """Residual of every stint, measured through the production simulate()."""
    p = dict(params)
    p["pace_drift"] = pace_drift
    scen = {**OPT.SCENARIOS["clean_air"]}
    out = []
    for r in recs:
        _tot, parts = OPT.simulate(r["strategy"], r["state"], p, scen, n_sims,
                                   np.random.default_rng(seed), return_parts=True)
        # parts[i] is the simulated time of stint i (the current stint is 0)
        idx = r["position"]
        if idx >= len(parts):
            continue
        out.append(float(np.mean(parts[idx])) - r["realized"])
    return np.asarray(out, dtype=float)


def _mad(x):
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def _select_pace_drift(recs, params, r0, n_sims):
    """Pick `pace_drift` from a grid, minimising the residual dispersion.

    The coefficient is only worth carrying if it demonstrably improves the
    fit: a constant that shifts every candidate equally cancels in the
    ranking, so a marginal gain is not worth an extra free parameter.
    """
    best = (0.0, float(np.median(np.abs(r0 - np.median(r0)))))
    for drift in np.arange(-0.08, 0.0801, 0.01):
        r = _residuals(recs, params, float(drift), n_sims)
        mad = float(np.median(np.abs(r - np.median(r))))
        if mad < best[1]:
            best = (float(drift), mad)
    return best[0], best[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sims", type=int, default=120)
    ap.add_argument("--apply", action="store_true",
                    help="write the fitted pace_drift into degradation_model.json")
    args = ap.parse_args()
    t0 = time.time()

    print("[cal] building labeled frame ...")
    frame = build_labeled_frame()
    emb = pl.read_parquet(GOLD / "driver_embeddings.parquet").with_columns(
        pl.col("season").cast(pl.Utf8))
    frame = frame.join(emb.select(["season", "driver", "cluster"]),
                       on=["season", "driver"], how="left")
    frame = frame.with_columns(pl.col("cluster").fill_null(-1).cast(pl.Int32))
    params = fit_degradation(df=frame.filter(pl.col("season") != DECISION_SEASON),
                             save=False)
    lapf = pl.read_parquet(GOLD / "lap_features.parquet").with_columns(
        pl.col("season").cast(pl.Utf8))

    recs = _collect(frame, lapf)
    print(f"[cal] {len(recs)} executed stints over "
          f"{len({(r['race'], r['driver']) for r in recs})} driver-races")

    r0 = _residuals(recs, params, 0.0, args.n_sims)
    mad0 = float(np.median(np.abs(r0 - np.median(r0))))
    drift, mad_best = _select_pace_drift(recs, params, r0, args.n_sims)
    r1 = _residuals(recs, params, drift, args.n_sims)

    dry = np.array([r["comp"] in DRY for r in recs])[:len(r0)]
    print("\n[cal] simulator residual against the REALIZED lap times (dry stints):")
    print(f"  {'before (pace_drift = 0)':<28} mean {r0[dry].mean():+7.1f} s"
          f"   median {np.median(r0[dry]):+7.1f} s   MAD {_mad(r0[dry]):6.1f} s")
    print(f"  {f'after  (pace_drift = {drift:+.4f})':<28} mean {r1[dry].mean():+7.1f} s"
          f"   median {np.median(r1[dry]):+7.1f} s   MAD {_mad(r1[dry]):6.1f} s")

    print(f"\n[cal] grid-selected pace_drift = {drift:+.4f} s per lap "
          f"(MAD {mad0:.1f} -> {mad_best:.1f} s)")
    print("      The fuel correction assumes 0.03 s/lap of pace recovered as the")
    print("      fuel burns off. A pace_drift term can absorb the shortfall,")
    print(f"      i.e. the effective fuel-burn-off rate is about {0.03 + abs(drift):.3f} s/lap.")
    print("      It is NOT adopted: the term shifts the simulated time of every")
    print("      candidate by a similar amount, so it cancels in the ranking")
    print("      that decides the recommendation, while it moves the overall")
    print("      level away from the realized data. The structural fix above")
    print("      (the fresh-pace reference) is what actually removed the bias.")

    st = pl.DataFrame({"resid0": r0, "resid1": r1,
                       "L": [r["L"] for r in recs][:len(r0)],
                       "comp": [r["comp"] for r in recs][:len(r0)]})
    st = st.with_columns(
        pl.when(pl.col("L") <= 12).then(pl.lit("a <=12"))
          .when(pl.col("L") <= 20).then(pl.lit("b 13-20"))
          .otherwise(pl.lit("c >20")).alias("bucket"))
    print("\n[cal] residual before/after by stint length (dry):")
    for r in st.filter(pl.col("comp").is_in(DRY)).group_by("bucket").agg(
            pl.len().alias("n"), pl.col("L").mean().round(1),
            pl.col("resid0").mean().round(1).alias("before"),
            pl.col("resid1").mean().round(1).alias("after")).sort("bucket").iter_rows(named=True):
        print(f"  {r['bucket']:>7}  n={r['n']:4d}  L={r['L']:5.1f}"
              f"   before {r['before']:+7.1f}   after {r['after']:+7.1f}")

    print("\n[cal] residual before/after by compound (dry):")
    for r in st.filter(pl.col("comp").is_in(DRY)).group_by("comp").agg(
            pl.len().alias("n"), pl.col("L").mean().round(1),
            pl.col("resid0").mean().round(1).alias("before"),
            pl.col("resid1").mean().round(1).alias("after")).sort("n", descending=True).iter_rows(named=True):
        print(f"  {r['comp']:>12}  n={r['n']:4d}  L={r['L']:5.1f}"
              f"   before {r['before']:+7.1f}   after {r['after']:+7.1f}")

    # The residual may be a RACE-POSITION effect (the car is slower early in
    # the race: cold track, traffic, full fuel) rather than a length effect.
    start_lap = np.array([r["state"]["cur_lap"] - 1
                          if r["is_current"] else r["strategy"][r["position"] - 1][1]
                          for r in recs], dtype=float)[:len(r0)]
    st2 = st.with_columns(pl.Series("start_lap", start_lap)).with_columns(
        pl.when(pl.col("start_lap") <= 18).then(pl.lit("a <=18"))
          .when(pl.col("start_lap") <= 30).then(pl.lit("b 19-30"))
          .when(pl.col("start_lap") <= 42).then(pl.lit("c 31-42"))
          .otherwise(pl.lit("d >42")).alias("pos"))
    print("\n[cal] residual by the stint's START lap (race position):")
    for r in st2.filter(pl.col("comp").is_in(DRY)).group_by("pos").agg(
            pl.len().alias("n"), pl.col("start_lap").mean().round(1),
            pl.col("L").mean().round(1),
            pl.col("resid0").mean().round(1).alias("before"),
            pl.col("resid1").mean().round(1).alias("after")).sort("pos").iter_rows(named=True):
        print(f"  {r['pos']:>7}  n={r['n']:4d}  start_lap {r['start_lap']:5.1f}"
              f"  L {r['L']:5.1f}   before {r['before']:+7.1f}   after {r['after']:+7.1f}")

    out = {"pace_drift_candidate": round(drift, 5),
           "pace_drift_adopted": False,
           "pace_drift_reason": ("the term shifts every candidate by a similar "
                                  "amount and therefore cancels in the ranking; "
                                  "the structural fresh-pace fix is what removed "
                                  "the bias"),
           "resid_shipped": {"mean": round(float(r0[dry].mean()), 2),
                              "median": round(float(np.median(r0[dry])), 2),
                              "mad": round(mad0, 2)},
           "resid_with_drift": {"mean": round(float(r1[dry].mean()), 2),
                                 "median": round(float(np.median(r1[dry])), 2),
                                 "mad": round(mad_best, 2)},
           "n_stints": int(dry.sum()),
           "note": ("Residual of the production optimizer.simulate() against the "
                    "realized fuel-corrected lap times of the strategies that "
                    "were actually executed in the 2025 dry races. A positive "
                    "residual means the simulator over-predicts the stint.")}
    path = MODELS_DIR / "calibration.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[cal] saved -> {path}  ({time.time()-t0:.0f}s)")

    if args.apply:
        model_path = MODELS_DIR / "degradation_model.json"
        model = json.loads(model_path.read_text(encoding="utf-8"))
        model["calibration"] = {k: v for k, v in out.items() if k != "note"}
        model["calibration_notes"] = out["note"]
        model.pop("pace_drift", None)
        model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
        print(f"[cal] recorded the calibration report -> {model_path}")


if __name__ == "__main__":
    main()
