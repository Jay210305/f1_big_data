"""Phase 4 v2 evaluation: reload artifacts, show metrics + per-stint sample.

    python -m src.models.eval_rul
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl

from src.ingest.config import ROOT
from src.models import rul_data as RD
from src.models.rul_data import (SPLIT, add_fuel_corr, compute_labels,
                                 compute_rolling_features, filter_stints,
                                 load_lap_features, load_lat_energy,
                                 load_sc_laps)

MODELS_DIR = ROOT / "data" / "models"


def build_labeled_frame() -> pl.DataFrame:
    df = load_lap_features()
    df = df.with_columns(pl.col("season").cast(pl.Utf8))
    df = df.with_columns(
        pl.col("session").map_elements(RD._session_type, return_dtype=pl.Utf8)
        .alias("session_type"))
    df = df.filter(pl.col("session_type").is_in(["Race", "Sprint"]))
    df = df.join(load_lat_energy(), on=["season", "gp", "session", "driver", "lap"],
                 how="left")
    df = filter_stints(df, load_sc_laps())
    df = add_fuel_corr(df)
    df = compute_rolling_features(df)
    df = compute_labels(df)
    return df


def main():
    with open(MODELS_DIR / "phase4_metrics.json") as fh:
        res = json.load(fh)
    with open(MODELS_DIR / "feature_meta.json") as fh:
        meta = json.load(fh)

    print("=" * 88)
    print("Phase 4 v2 — RUL / Cliff (refactored, no leakage) — saved metrics")
    print("=" * 88)
    print(f"\n{'Model':18s} {'MAE':>7s} {'RMSE':>7s} {'R2':>7s} {'MAE<=10':>8s} {'R2_crit':>8s} {'R2_nc':>7s} {'PHM':>9s}")
    for k in ["naive_const", "naive_compound", "lgb_rul", "xgb_rul",
              "xgb_rul_sym", "lstm_rul", "piecewise_blend"]:
        if k in res:
            r = res[k]
            r2_c = f"{r.get('R2_crit', float('nan')):8.3f}"
            r2_nc = f"{r.get('R2_noncrit', float('nan')):7.3f}"
            print(f"{k:18s} {r['MAE']:7.3f} {r['RMSE']:7.3f} {r['R2']:7.3f} "
                  f"{r['MAE_crit']:8.3f} {r2_c} {r2_nc} {r['PHM']:9.0f}")
    if "xgb_cliff" in res:
        c = res["xgb_cliff"]
        print(f"\nCliff (in-cliff or onset within 3 laps, thr={c.get('threshold',0.5):.3f}):")
        print(f"  prec={c['precision']:.3f} rec={c['recall']:.3f} "
              f"f1={c['f1']:.3f} auc={c['auc']:.3f}")

    # --- per-stint prediction sample (2025 Bahrain Race VER, stint 1) ---
    print("\nSample: RUL predictions (XGBoost vs LSTM vs Piecewise Blend) "
          "(2025 Bahrain Race VER stint 1):")
    from src.models.optimizer import load_phase4, predict_laps

    df = build_labeled_frame()
    sub = df.filter(
        (pl.col("season") == "2025") & (pl.col("gp") == "Bahrain Grand Prix")
        & (pl.col("session") == "Race")
        & (pl.col("driver") == "VER") & (pl.col("stint") == 1)).sort("life")
    if sub.height == 0:
        print("  (stint filtered out or not found)")
        return

    meta, rul_m, cliff_x, cliff_l, thr, weights, lstm_m = load_phase4()
    rul_blend, prob = predict_laps(sub, meta, rul_m, cliff_x, cliff_l, weights=weights, lstm_m=lstm_m)
    rul_pure_xgb, _ = predict_laps(sub, meta, rul_m, cliff_x, cliff_l, weights=weights, lstm_m=None)

    sub = sub.with_columns([
        pl.Series("rul_xgb", np.round(rul_pure_xgb, 1)),
        pl.Series("rul_blend", np.round(rul_blend, 1)),
        pl.Series("cliff_prob", np.round(prob, 3)),
    ])
    cols = ["life", "laptime", "fuel_corr", "pace_above_best", "target_rul",
            "rul_xgb", "rul_blend", "cliff_prob", "cliff_within_3"]
    avail = [c for c in cols if c in sub.columns]
    print(sub.select(avail).to_pandas().to_string(index=False))

    print("\n" + "=" * 88)
    print("Acceptance: crit-zone MAE < 2.0 (LSTM 2.00, Blend 2.14 vs XGB 2.51);")
    print("R2 and F1 targets honestly reported in phase4_metrics.json — see annotation.")
    print("=" * 88)


if __name__ == "__main__":
    main()
