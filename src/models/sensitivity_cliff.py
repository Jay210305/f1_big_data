"""Task 8.H.9 — Sensitivity analysis of the cliff threshold & RUL cap.

Sweeps CLIFF_THR (the seconds-above-best pace that marks cliff onset) and the
sustain-window (consecutive laps above threshold required to declare onset)
and reports how the generated labels and a downstream cliff classifier respond.

    python -m src.models.sensitivity_cliff
    python -m src.models.sensitivity_cliff --no-fit
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import polars as pl

from src.ingest.config import ROOT
from src.models import rul_data as RD

MODELS_DIR = ROOT / "data" / "models"

CLIFF_THR_GRID = [1.0, 1.5, 2.0]
SUSTAIN_GRID = [2, 3, 4]


def _compute_labels(df: pl.DataFrame, thr: float, sustain: int) -> pl.DataFrame:
    """Recompute performance-based labels for a given (CLIFF_THR, sustain)."""
    df = df.sort(RD.STINT_KEYS + ["life"])
    df = df.with_columns(
        pl.col("fuel_corr").min().over(RD.STINT_KEYS).alias("stint_best_pace"))
    df = df.with_columns(
        pl.int_range(pl.len()).over(RD.STINT_KEYS).alias("_pos"))
    df = df.with_columns(
        ((pl.col("pace_roll_med_3") > pl.col("stint_best_pace") + thr)
         & (pl.col("_pos") >= 2)).alias("_above"))
    # sustain window: `sustain` consecutive `_above` laps
    cross = pl.col("_above")
    for k in range(1, sustain):
        cross = cross & pl.col("_above").shift(-k).over(RD.STINT_KEYS) \
            .fill_null(False)
    df = df.with_columns(cross.alias("_cross"))
    df = df.with_columns(
        pl.when(pl.col("_cross")).then(pl.col("life")).alias("_cross_life"))
    df = df.with_columns(
        pl.col("_cross_life").min().over(RD.STINT_KEYS).alias("t_end_life"))
    max_life = pl.col("life").max().over(RD.STINT_KEYS)
    df = df.with_columns(
        pl.when(pl.col("t_end_life").is_not_null())
        .then(pl.col("t_end_life") - pl.col("life"))
        .otherwise(max_life - pl.col("life") + 1)
        .clip(lower_bound=0).alias("rul_physical"))
    df = df.with_columns(
        (pl.col("t_end_life").is_not_null()
         & (pl.col("t_end_life") - pl.col("life") <= 3))
        .alias("cliff_within_3"))
    df = df.drop(["_above", "_cross", "_cross_life", "_pos"])
    return df


def _label_stats(df: pl.DataFrame) -> dict:
    n_stints = df.select(pl.struct(RD.STINT_KEYS)).n_unique()
    n_cliff = df.filter(pl.col("t_end_life").is_not_null()) \
        .select(pl.struct(RD.STINT_KEYS)).n_unique()
    pos_rate = float(df["cliff_within_3"].mean())
    rul = df["target_rul"] if "target_rul" in df.columns else df["rul_physical"]
    return {
        "n_stints": n_stints,
        "n_cliff_stints": n_cliff,
        "cliff_stint_rate": round(n_cliff / max(n_stints, 1), 4),
        "cliff_positive_rate": round(pos_rate, 5),
        "rul_mean": round(float(rul.mean()), 2),
        "rul_p50": round(float(rul.quantile(0.5)), 2),
        "rul_p75": round(float(rul.quantile(0.75)), 2),
    }


def _fit_cliff_classifier(df: pl.DataFrame) -> dict | None:
    """Train XGBoost (CPU) cliff classifier on the current labels and return
    test F1/AUC. Falls back to None if data is degenerate."""
    import xgboost as xgb
    from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

    feats = RD.feature_cols(df)
    codes = {}
    for c in RD.CATEGORICAL:
        df = df.with_columns(pl.col(c).fill_null("UNKNOWN").alias(c))
        mapping = {v: i for i, v in enumerate(sorted(df[c].unique().to_list()))}
        codes[c] = mapping
        df = df.with_columns(pl.col(c).replace(mapping).cast(pl.Int32).alias(c))

    def to_split(years):
        s = df.filter(pl.col("season").is_in(years))
        X = s.select(feats).to_pandas().to_numpy(dtype=np.float32, na_value=np.nan)
        y = s["cliff_within_3"].to_numpy().astype(np.float32)
        return X, y

    Xtr, ytr = to_split(["2021", "2022", "2023"])
    Xva, yva = to_split(["2024"])
    Xte, yte = to_split(["2025"])
    if ytr.sum() < 10 or yte.sum() < 10:
        return None
    pos = int(ytr.sum())
    neg = int(len(ytr) - pos)
    sw = np.where(ytr > 0, float(np.sqrt(neg / max(pos, 1))), 1.0)
    dtr = xgb.DMatrix(Xtr, label=ytr, weight=sw.astype(np.float32))
    dte = xgb.DMatrix(Xte, label=yte)
    dva = xgb.DMatrix(Xva, label=yva)
    params = dict(objective="binary:logistic", tree_method="hist",
                  max_depth=7, eta=0.06, eval_metric="aucpr",
                  min_child_weight=3, max_delta_step=1)
    m = xgb.train(params, dtr, num_boost_round=300, evals=[(dva, "val")],
                  early_stopping_rounds=30, verbose_eval=False)
    prob = m.predict(dte)
    pred = (prob > 0.5).astype(int)
    return {
        "auc": round(float(roc_auc_score(yte, prob)), 3),
        "f1": round(float(f1_score(yte, pred)), 3),
        "precision": round(float(precision_score(yte, pred, zero_division=0)), 3),
        "recall": round(float(recall_score(yte, pred)), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fit", action="store_true",
                    help="skip classifier fitting (label stats only)")
    args = ap.parse_args()

    print("[sens] building labeled frame base (lap_features -> filtered)...")
    df = RD.load_lap_features()
    df = df.with_columns(pl.col("season").cast(pl.Utf8))
    df = df.with_columns(
        pl.col("session").map_elements(RD._session_type, return_dtype=pl.Utf8)
        .alias("session_type"))
    df = df.filter(pl.col("session_type").is_in(["Race", "Sprint"]))
    df = df.join(RD.load_lat_energy(),
                 on=["season", "gp", "session", "driver", "lap"], how="left")
    df = RD.filter_stints(df, RD.load_sc_laps())
    df = RD.add_fuel_corr(df)
    df = RD.compute_rolling_features(df)
    base = df

    print("=" * 78)
    print("Cliff threshold (CLIFF_THR) x sustain-window sensitivity")
    print("=" * 78)

    default = None
    results = {}
    for thr in CLIFF_THR_GRID:
        for sustain in SUSTAIN_GRID:
            key = f"thr{thr:.1f}_win{sustain}"
            labeled = _compute_labels(base.clone(), thr, sustain)
            label_stats = _label_stats(labeled)
            clf = None if args.no_fit else _fit_cliff_classifier(labeled)
            results[key] = {"CLIFF_THR": thr, "sustain_window": sustain,
                            "labels": label_stats, "classifier": clf}
            is_default = (thr == RD.CLIFF_THR and sustain == 3)
            tag = "  <-- DEFAULT" if is_default else ""
            clf_s = ""
            if clf:
                clf_s = (f"auc={clf['auc']} f1={clf['f1']} "
                         f"prec={clf['precision']} rec={clf['recall']}")
            print(f"\n[{key}]{tag}")
            print(f"  labels: cliff_stints={label_stats['cliff_stint_rate']:.3f} "
                  f"pos_rate={label_stats['cliff_positive_rate']:.4f} "
                  f"rul_p50={label_stats['rul_p50']} rul_p75={label_stats['rul_p75']}")
            if clf:
                print(f"  classifier (test 2025): {clf_s}")

    # ---- sensitivity table (markdown) ----
    lines = ["# Cliff threshold & sustain-window sensitivity",
             "",
             "| CLIFF_THR (s) | Window (laps) | cliff stint rate | "
             "positive rate | RUL p50 | RUL p75 | AUC | F1 |", 
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for thr in CLIFF_THR_GRID:
        for sustain in SUSTAIN_GRID:
            key = f"thr{thr:.1f}_win{sustain}"
            r = results[key]
            ls = r["labels"]
            c = r["classifier"] or {}
            lines.append(
                f"| {thr:.1f} | {sustain} | {ls['cliff_stint_rate']:.3f} | "
                f"{ls['cliff_positive_rate']:.4f} | {ls['rul_p50']} | "
                f"{ls['rul_p75']} | {c.get('auc', '—')} | {c.get('f1', '—')} |")
    md = "\n".join(lines) + "\n"
    out_md = MODELS_DIR / "cliff_sensitivity.md"
    out_md.write_text(md, encoding="utf-8")
    with open(MODELS_DIR / "cliff_sensitivity.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved -> {out_md}")
    print(f"Saved -> {MODELS_DIR / 'cliff_sensitivity.json'}")


if __name__ == "__main__":
    main()
