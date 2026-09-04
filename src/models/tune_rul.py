"""Task 8.H.7 — Hyperparameter search on the flagship RUL regressor.

Two stages:
  1. SANITY CHECK (default): fit a few small models and compare the residual
     standard error (RMSE) against a telemetry noise floor to decide whether a
     full Optuna study is justified. Passive — only reads telemetry stats.
  2. OPTUNA STUDY (--study N): if justified, run an N-trial Optuna study that
     optimizes VALIDATION RMSE of the XGBoost RUL regressor (same temporal
     split as train_rul.py) and save the best hyperparameters.

    python -m src.models.tune_rul                  # sanity check
    python -m src.models.tune_rul --study 30       # 30-trial Optuna search
    python -m src.models.tune_rul --study 30 --reps 3
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from src.ingest.config import ROOT

MODELS_DIR = ROOT / "data" / "models"

# noise floor estimate: telemetry sampling jitter (speed/time resolution) maps
# to a per-lap timing uncertainty of a couple tenths of a second; the RUL
# target is quantized in whole laps, so a fitted RMSE within a few laps of the
# theoretical floor leaves little room for hyperparameters to matter.
NOISE_FLOOR_RMSE = 1.0  # s (per-lap telemetry timing noise, order of magnitude)


def _report_noise_floor() -> float:
    """Estimate the telemetry noise ceiling from Bronze speed resolution."""
    import duckdb
    from src.ingest.config import BRONZE_DIR
    try:
        glob = str(BRONZE_DIR / "season=2025" / "**" / "*.parquet") \
            .replace("\\", "/")
        q = f"""SELECT stddev(speed) AS speed_std
                FROM read_parquet('{glob}', hive_partitioning=1)
                USING SAMPLE 0.001"""
        speed_std = duckdb.connect().execute(q).fetchone()[0]
        # fractional speed noise ~ speed_std / 350 km/h maps to laptime noise
        frac = (speed_std or 0.0) / 350.0
        return max(frac * 90.0, NOISE_FLOOR_RMSE)  # ~90s lap
    except Exception:
        return NOISE_FLOOR_RMSE


def sanity_check() -> None:
    from src.models.train_rul import reg_metrics, train_xgb_rul
    from src.models.rul_data import prepare
    floor = _report_noise_floor()
    print("=" * 78)
    print("Task 8.H.7.1 — residual-error sanity check vs telemetry noise")
    print("=" * 78)
    print(f"  telemetry noise floor (per-lap, est.): {floor:.3f} s")

    tab, _, _, _, _ = prepare()
    Xtr, ytr_r, _ = tab["train"]
    Xva, yva_r, _ = tab["val"]
    Xte, yte_r, _ = tab["test"]
    from src.models.train_rul import to_numpy

    # budget-friendly: train with fewer rounds for the sanity check
    import xgboost as xgb
    Xtr_n, Xva_n, Xte_n = to_numpy(Xtr), to_numpy(Xva), to_numpy(Xte)
    sample = min(len(Xtr_n), 20000)
    idx = np.random.default_rng(0).choice(len(Xtr_n), sample, replace=False)
    dtr = xgb.DMatrix(Xtr_n[idx], label=ytr_r[idx])
    dva = xgb.DMatrix(Xva_n, label=yva_r)
    dte = xgb.DMatrix(Xte_n, label=yte_r)
    params = dict(objective="reg:squarederror", tree_method="hist",
                  max_depth=8, eta=0.1, subsample=0.9, colsample_bytree=0.9,
                  min_child_weight=4)
    m = xgb.train(params, dtr, num_boost_round=150, evals=[(dva, "val")],
                  early_stopping_rounds=20, verbose_eval=False)
    pred = np.clip(m.predict(dte), 0, None)
    rmse = reg_metrics(yte_r, pred)["RMSE"]
    print(f"  budget sanity model RMSE (test 2025): {rmse:.3f} laps-equivalent")

    gap = rmse - floor
    justified = rmse > 2.0 * floor
    verdict = ("JUSTIFIED: residual RMSE is well above the noise floor — "
               "a hyperparameter search can still move the needle." if justified
               else "NOT clearly justified: residual RMSE is close to the "
                    "noise floor; diminishing returns expected.")
    print(f"  gap above floor: {gap:.3f} laps")
    print(f"  -> {verdict}")
    with open(MODELS_DIR / "tune_sanity_check.json", "w") as fh:
        json.dump({"noise_floor_rmse": floor, "budget_test_rmse": rmse,
                   "justified": justified}, fh, indent=2)
    print(f"  saved -> {MODELS_DIR / 'tune_sanity_check.json'}")


def study(n_trials: int, reps: int) -> None:
    import optuna
    import xgboost as xgb
    from src.models.train_rul import (reg_metrics, to_numpy, xgb_asym_obj)
    from src.models.rul_data import prepare

    tab, _, _, _, _ = prepare()
    Xtr_n = to_numpy(tab["train"][0])
    ytr = tab["train"][1]
    Xva_n = to_numpy(tab["val"][0])
    yva = tab["val"][1]
    Xte_n = to_numpy(tab["test"][0])
    yte = tab["test"][1]

    # subsample train for trial speed (full train is fine but slower)
    n = min(len(Xtr_n), 40000)
    idx = np.random.default_rng(1).choice(len(Xtr_n), n, replace=False)
    dtr = xgb.DMatrix(Xtr_n[idx], label=ytr[idx])
    dva = xgb.DMatrix(Xva_n, label=yva)
    dte = xgb.DMatrix(Xte_n, label=yte)

    def objective(trial):
        p = dict(
            objective="reg:squarederror", tree_method="hist",
            max_depth=trial.suggest_int("max_depth", 4, 12),
            eta=trial.suggest_float("eta", 0.02, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 16),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        )
        val_rmses = []
        for _ in range(reps):
            m = xgb.train(p, dtr, num_boost_round=600, evals=[(dva, "val")],
                          early_stopping_rounds=40, verbose_eval=False)
            val_rmses.append(float(m.eval(dva).split(":")[1].strip().rstrip("]")))
        return float(np.mean(val_rmses))

    study_obj = optuna.create_study(direction="minimize")
    study_obj.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = study_obj.best_params
    print("\n" + "=" * 78)
    print(f"Task 8.H.7.2 — Optuna study complete ({n_trials} trials)")
    print("=" * 78)
    print(f"  best params: {best}")
    print(f"  best val RMSE: {study_obj.best_value:.3f}")

    # refit on full train with best params, evaluate on test
    full_dtr = xgb.DMatrix(Xtr_n, label=ytr)
    p = dict(objective="reg:squarederror", tree_method="hist", **best)
    m = xgb.train(p, full_dtr, num_boost_round=600, evals=[(dva, "val")],
                  early_stopping_rounds=40, verbose_eval=False)
    pred = np.clip(m.predict(dte), 0, None)
    test_rmse = reg_metrics(yte, pred)["RMSE"]
    print(f"  test RMSE with best params: {test_rmse:.3f}")

    out = {"best_params": best, "best_val_rmse": float(study_obj.best_value),
           "test_rmse": test_rmse, "n_trials": n_trials, "reps": reps}
    with open(MODELS_DIR / "tune_rul_best_params.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"  saved -> {MODELS_DIR / 'tune_rul_best_params.json'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", type=int, default=0,
                    help="run an Optuna study with N trials")
    ap.add_argument("--reps", type=int, default=1)
    args = ap.parse_args()
    if args.study:
        study(args.study, args.reps)
    else:
        sanity_check()


if __name__ == "__main__":
    main()
