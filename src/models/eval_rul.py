"""Phase 4 evaluation: load artifacts, show predictions on sample stints.

    python -m src.models.eval_rul
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl

from src.ingest.config import ROOT
from src.models.rul_data import prepare, STINT_KEYS

MODELS_DIR = ROOT / "data" / "models"


def main():
    with open(MODELS_DIR / "phase4_metrics.json") as fh:
        res = json.load(fh)
    with open(MODELS_DIR / "feature_meta.json") as fh:
        meta = json.load(fh)

    print("=" * 78)
    print("Phase 4 — RUL / Cliff Predictor — saved metrics")
    print("=" * 78)
    print(f"\n{'Model':22s} {'MAE':>8s} {'RMSE':>8s} {'R2':>7s}")
    for k in ["naive_const", "naive_life", "lgb_rul", "xgb_rul", "lstm_rul"]:
        if k in res:
            r = res[k]
            print(f"{k:22s} {r['MAE']:8.3f} {r['RMSE']:8.3f} {r['R2']:7.3f}")
    if "xgb_cliff" in res:
        c = res["xgb_cliff"]
        print(f"\nCliff (within 3 laps): prec={c['precision']:.3f} "
              f"rec={c['recall']:.3f} f1={c['f1']:.3f} auc={c['auc']:.3f}")

    # --- per-stint prediction sample with XGBoost ---
    print("\nSample: XGBoost RUL predictions vs actual (2025 Bahrain Race VER):")
    tab, _, feats, codes, _ = prepare()
    Xte, yte_r, _ = tab["test"]
    import xgboost as xgb
    model = xgb.Booster()
    model.load_model(str(MODELS_DIR / "xgb_rul.json"))
    Xn = Xte.to_pandas().to_numpy(dtype=np.float32, na_value=np.nan)
    pred = model.predict(xgb.DMatrix(Xn))

    # rebuild a small frame to locate the stint
    from src.models.rul_data import load_lap_features, build_labels
    df = load_lap_features().with_columns(pl.col("season").cast(pl.Utf8))
    df = build_labels(df)
    df = df.filter((pl.col("season") == "2025")
                   & (pl.col("gp") == "Bahrain Grand Prix")
                   & (pl.col("session") == "Race")
                   & (pl.col("driver") == "VER")
                   & (pl.col("stint") == 1))
    df = df.sort("life")
    df = df.with_columns(
        pl.Series("rul_pred", pred[:len(df)] if len(df) else []))
    cols = ["lap", "life", "laptime", "rul", "rul_pred"]
    avail = [c for c in cols if c in df.columns]
    print(df.select(avail).to_pandas().to_string(index=False))

    print("\n" + "=" * 78)
    print("Compare LSTM vs XGBoost in the metrics table to decide the")
    print("Phase 4 stop-n-go checkpoint: does the LSTM justify its cost?")
    print("=" * 78)


if __name__ == "__main__":
    main()
