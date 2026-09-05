"""Plotting module — figures for the strategy-engine report & diagnostics.

Headless (Agg) rendering of diagnostic visuals derived from the fitted model
artifacts. Task 8.0.6.7 (conditional subtask) delivers:

  scenario_multiplier_regression.png  beta(dirty_air_frac) fitted curve
                                      overlaid with empirical stint-bin medians
                                      and the clean/drs/backmarker anchors ->
                                      validates the spline does not overfit
                                      sparse traffic extremes (Task 8.0.4).
  critical_zone_r2.png                critical-zone (RUL <= 10) vs non-critical
                                      R^2 per model, exposing which engine is
                                      trustworthy right at the cliff threshold
                                      (Tasks 8.0.2 / 8.0.6.7).

Output folder: <repo-root>/eval_plots/

Usage:
    python -m src.eval.plots
    python -m src.eval.plots --only scenario_multiplier_regression
    python -m src.eval.plots --only critical_zone_r2
"""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.ingest.config import ROOT  # noqa: E402

OUT_DIR = ROOT / "eval_plots"
MODELS_DIR = ROOT / "data" / "models"

plt.rcParams.update({"font.size": 10, "axes.grid": True})


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   -> {path}")


def _load_json(name: str) -> dict:
    with open(MODELS_DIR / name) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# 8.0.6.7a — scenario multiplier regression  beta(dirty_air_frac)
# ---------------------------------------------------------------------------
def scenario_multiplier_regression() -> None:
    params = _load_json("degradation_model.json")
    fit = params.get("dirty_air_fit", {})
    mult = params.get("scenario_beta_mult", {})

    if fit.get("method") == "fallback" or "coefs" not in fit:
        print("[plots] dirty_air_fit unavailable — skipping "
              "scenario_multiplier_regression")
        return

    slope = fit["coefs"]["slope"]
    intercept = fit["coefs"]["intercept"]
    centers = np.asarray(fit.get("bin_centers", []))
    medians = np.asarray(fit.get("bin_medians", []))
    anchors = fit.get("anchors", {})
    ceiling = fit.get("mult_ceiling", 2.5)

    fig, ax = plt.subplots(figsize=(9, 6))

    # fitted curve across the observed + marginal extrapolation domain
    dmax = max(centers.max() if centers.size else 0.1,
               max(anchors.values(), default=0.0)) * 1.1
    d = np.linspace(0.0, dmax, 200)
    ax.plot(d, intercept + slope * d, "-", color="#1f77b4", lw=2,
            label="fitted $\\beta(dirty\\_air\\_frac)$ (Theil–Sen)")
    ax.plot(d, intercept + slope * np.clip(d, 0, 1) * 0 + intercept,
            "--", color="gray", lw=1, alpha=0.6, label="clean-air $\\beta_0$")

    # empirical stint bins
    if centers.size:
        ax.scatter(centers, medians, s=70, color="#d62728", zorder=3,
                   label="empirical stint-bin medians", edgecolors="k", linewidths=0.5)

    # scenario anchors (clean / drs_train / backmarker)
    for key, col, lab in [("drs_train", "#ff7f0e", "drs_train (P90)"),
                          ("backmarker", "#9467bd", "backmarker (P99)")]:
        a = anchors.get(key)
        if a is None:
            continue
        y = intercept + slope * a
        ax.axvline(a, color=col, ls=":", lw=1.2, alpha=0.8)
        ax.plot(a, y, "o", color=col, ms=8, zorder=4)
        mlabel = f"{lab}  ×{mult.get(key):.2f}"
        ax.annotate(mlabel, (a, y), textcoords="offset points",
                    xytext=(6, 8), fontsize=9, color=col)

    ax.axvline(0.0, color="#2ca02c", ls=":", lw=1.2, alpha=0.8)
    ax.plot(0.0, intercept, "o", color="#2ca02c", ms=8, zorder=4)
    ax.annotate("clean_air  ×1.00", (0.0, intercept), textcoords="offset points",
                xytext=(6, -14), fontsize=9, color="#2ca02c")

    ax.set_xlabel("dirty_air_frac (share of lap within 10 m of car ahead)")
    ax.set_ylabel("degradation slope $\\beta$ (s/lap)")
    ax.set_title("Scenario degradation multipliers — fitted "
                 "$\\beta(dirty\\_air\\_frac)$ (Task 8.0.4 Option A)")
    ax.legend(loc="best", fontsize=8)
    r2 = fit.get("r2", 0.0)
    ax.text(0.99, 0.01, f"n={fit.get('n_stints')} stints, "
            f"{fit.get('n_bins')} bins, bin-median $R^2$={r2:.3f}, "
            f"ceiling={ceiling}x",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="gray")
    fig.tight_layout()
    _save(fig, "scenario_multiplier_regression.png")


# ---------------------------------------------------------------------------
# 8.0.6.7b — critical-zone R^2 bar chart
# ---------------------------------------------------------------------------
CRIT_DISPLAY = [
    ("naive_const", "Naive (const)"),
    ("naive_compound", "Naive (compound)"),
    ("lgb_rul", "LightGBM"),
    ("xgb_rul", "XGBoost (asym)"),
    ("xgb_rul_sym", "XGBoost (sym)"),
    ("lstm_rul", "LSTM"),
    ("piecewise_blend", "Blend (prod)"),
]


def critical_zone_r2() -> None:
    m4 = _load_json("phase4_metrics.json")

    labels, r2_crit, r2_noncrit = [], [], []
    for key, disp in CRIT_DISPLAY:
        r = m4.get(key)
        if not isinstance(r, dict) or "R2_crit" not in r:
            continue
        labels.append(disp)
        r2_crit.append(float(r["R2_crit"]))
        r2_noncrit.append(float(r["R2_noncrit"]))

    if not labels:
        print("[plots] no segmented R^2 in phase4_metrics.json — skipping "
              "critical_zone_r2")
        return

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.5))
    b1 = ax.bar(x - width / 2, r2_crit, width, color="#d62728",
                label="critical zone ($RUL\\leq 10$)")
    b2 = ax.bar(x + width / 2, r2_noncrit, width, color="#1f77b4",
                label="non-critical ($RUL>10$)")

    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("$R^2$")
    ax.set_title("Remaining-useful-life $R^2$ by degradation zone "
                 "(Task 8.0.6.7 diagnostic)")
    ax.legend(loc="lower left", fontsize=9)

    for rect in list(b1) + list(b2):
        h = rect.get_height()
        ax.annotate(f"{h:.2f}", (rect.get_x() + rect.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 2 if h >= 0 else -10),
                    ha="center", fontsize=7, rotation=90)

    ax.set_ylim(top=max(max(r2_crit, default=0), 0.5) + 0.2)
    fig.tight_layout()
    _save(fig, "critical_zone_r2.png")


# ---------------------------------------------------------------------------
FIGURES = {
    "scenario_multiplier_regression": scenario_multiplier_regression,
    "critical_zone_r2": critical_zone_r2,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate strategy-engine plots")
    ap.add_argument("--only", choices=list(FIGURES), default=None)
    args = ap.parse_args()

    print(f"Output folder: {OUT_DIR}")
    for name, fn in FIGURES.items():
        if args.only in (None, name):
            fn()
    print(f"Done. Figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
