"""Plotting module — figures for the strategy-engine report & diagnostics.

Headless (Agg) rendering of diagnostic visuals derived from the fitted model
artifacts.

Core figures (Task 8.0.6):
  rul_predicted_vs_actual.png  predicted vs actual remaining useful life on
                                the held-out 2025 split, per engine and
                                coloured by compound
  degradation_curves.png        measured per-compound degradation against
                                the fitted beta line
  cluster_embeddings.png       2-D projection of the driver style latent
                                space, coloured by HDBSCAN cluster
  backtest_delta_histogram.png  distribution of the paired backtest deltas

Conditional figures (Task 8.0.6.7):
  scenario_multiplier_regression.png  beta(dirty_air_frac) fitted curve
                                      overlaid with empirical stint-bin medians
                                      and the clean/drs/backmarker anchors ->
                                      validates the spline does not overfit
                                      sparse traffic extremes (Task 8.0.4).
  critical_zone_r2.png                critical-zone (RUL <= 10) vs non-critical
                                      R^2 per model, exposing which engine is
                                      trustworthy right at the cliff threshold
                                      (Tasks 8.0.2 / 8.0.6.7).

Phase 9 figures:
  mapreduce_shuffle.png              complete counters with/without combiner
  mongo_indexes.png                  controlled COLLSCAN/IXSCAN execution stats
  three_way_parity.png               independent per-stint MR/Polars/Mongo slopes

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
    fig.savefig(path, dpi=150, bbox_inches="tight", pil_kwargs={"optimize": True})
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


def mapreduce_shuffle():
    """Complete counters, rather than sampled traces, measure the combiner."""
    base = ROOT / "data/mapreduce/output"
    combined_path = base / "lap_aggregate/metrics.json"
    plain_path = base / "lap_aggregate_plain/metrics.json"
    if not combined_path.exists() or not plain_path.exists():
        print("[plots] run lap_aggregate with and without combiner first")
        return
    combined, plain = (json.loads(path.read_text()) for path in (combined_path, plain_path))
    partitions = list(range(max(combined["reduce_tasks"], plain["reduce_tasks"])))
    def values(metrics, field):
        table = {r["partition"]: r[field] for r in metrics["partitions"]}
        return [table.get(p, 0) for p in partitions]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(partitions))
    for ax, field, label in zip(axes, ("values", "keys"), ("Shuffled KV pairs", "Distinct keys")):
        ax.bar(x - .2, values(plain, field), .4, label="Without combiner")
        ax.bar(x + .2, values(combined, field), .4, label="With combiner")
        ax.set(xticks=x, xticklabels=partitions, xlabel="Reducer partition", ylabel=label)
        ax.legend(fontsize=8)
    saved = 100 * (1 - combined["bytes_after_combine"] / plain["bytes_after_combine"])
    fig.suptitle(f"2025 Bronze → Silver: {saved:.1f}% fewer serialized shuffle bytes")
    fig.tight_layout()
    _save(fig, "mapreduce_shuffle.png")


def mongo_indexes():
    path = ROOT / "data/mongo/explain.json"
    if not path.exists():
        print("[plots] run src.nosql.cli explain first")
        return
    rows = json.loads(path.read_text())
    benchmark_path = ROOT / "data/mongo/benchmark.json"
    benchmark = json.loads(benchmark_path.read_text()) if benchmark_path.exists() else None
    fig, axes = plt.subplots(1, 3 if benchmark else 2, figsize=(16 if benchmark else 12, 4))
    x = np.arange(len(rows))
    for ax, metric, ylabel in zip(axes, ("totalDocsExamined", "totalKeysExamined"),
                                  ("Documents examined", "Index keys examined")):
        ax.bar(x - .2, [r["before"][metric] for r in rows], .4, label="COLLSCAN ($natural)")
        ax.bar(x + .2, [r["after"][metric] for r in rows], .4, label="IXSCAN (index hint)")
        ax.set(xticks=x, xticklabels=[r["query"] for r in rows], ylabel=ylabel)
        ax.tick_params(axis="x", rotation=35)
        ax.legend(fontsize=8)
    if benchmark:
        storage = axes[2]
        categories = np.arange(2)
        for offset, name, label in ((-.2, "without_indexes", "_id index only"),
                                    (.2, "with_indexes", "With secondary indexes")):
            stats = benchmark[name]["stats"]
            storage.bar(categories + offset, [stats[k] / 1e6 for k in ("size", "totalIndexSize")],
                        .4, label=label)
        storage.set(xticks=categories, xticklabels=["Logical BSON", "Index storage"], ylabel="MB (100,000 synthetic docs)")
        storage.legend(fontsize=7)
    fig.suptitle("MongoDB: controlled index access paths on the same collection")
    fig.tight_layout()
    _save(fig, "mongo_indexes.png")


def three_way_parity():
    path = ROOT / "data/mapreduce/parity.json"
    if not path.exists():
        print("[plots] run src.mapreduce.evidence --mongo first")
        return
    evidence = json.loads(path.read_text())
    if "mongo" not in evidence:
        print("[plots] Mongo parity unavailable; no three-way figure generated")
        return
    rows = evidence["mongo"]["stints"]
    reference = np.array([r["polars"] for r in rows])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(reference, [r["mapreduce"] for r in rows], s=14, alpha=.5, label="MapReduce")
    ax.scatter(reference, [r["mongo"] for r in rows], s=10, alpha=.5, marker="x", label="Mongo $group")
    lo, hi = reference.min(), reference.max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set(xlabel="Polars β (s/lap)", ylabel="Independent engine β (s/lap)",
           title=f"{len(rows):,} stints; max Mongo error {evidence['mongo']['max_abs_diff']:.2e}")
    ax.legend()
    fig.tight_layout()
    _save(fig, "three_way_parity.png")


# ---------------------------------------------------------------------------
# 8.0.6 core figures — RUL quality, degradation physics, style archetypes,
# and the backtest delta distribution.
# ---------------------------------------------------------------------------
def _rul_test_predictions():
    """Per-lap RUL predictions on the 2025 test split.

    Returns a dict with the truth, the XGBoost prediction and the piecewise
    XGBoost/LSTM blend, computed stint by stint so the sequence model sees a
    whole stint at a time.
    """
    import torch
    import polars as pl

    from src.models.eval_rul import build_labeled_frame
    from src.models.optimizer import encode_sequence, load_phase4, predict_laps
    from src.models.rul_data import STINT_KEYS

    meta, rul_m, cliff_x, cliff_l, _thr, weights, lstm_m = load_phase4()
    if lstm_m is None:
        return None
    df = build_labeled_frame()
    test = df.filter(pl.col("season") == "2025").sort(STINT_KEYS + ["life"])
    if test.height == 0:
        return None

    xgb_pred, _ = predict_laps(test, meta, rul_m, cliff_x, cliff_l,
                               weights=weights, lstm_m=None)
    lstm_pred = np.zeros(test.height, dtype=float)
    offset = 0
    for _key, grp in test.group_by(STINT_KEYS, maintain_order=True):
        grp = grp.sort("life")
        mat = encode_sequence(grp, meta)
        with torch.no_grad():
            out = lstm_m(torch.from_numpy(mat)).squeeze(0).numpy()
        lstm_pred[offset:offset + grp.height] = np.clip(out, 0, None)
        offset += grp.height

    w = np.clip((12.0 - xgb_pred) / 6.0, 0.0, 1.0)
    blend = w * lstm_pred + (1.0 - w) * xgb_pred
    return {"truth": test["target_rul"].to_numpy().astype(float),
            "xgb": np.asarray(xgb_pred, dtype=float),
            "lstm": lstm_pred,
            "blend": blend,
            "compound": test["compound"].to_list()}


def rul_predicted_vs_actual() -> None:
    """Predicted-vs-actual remaining useful life on the held-out 2025 split."""
    payload = _rul_test_predictions()
    if payload is None:
        print("[plots] RUL models or labeled frame unavailable — skipping "
              "rul_predicted_vs_actual")
        return
    truth = payload["truth"]
    compounds = payload["compound"]
    palette = {"SOFT": "#d62728", "MEDIUM": "#ff7f0e", "HARD": "#2ca02c",
               "unknown": "#7f7f7f"}
    compounds = ["unknown" if c is None else c for c in compounds]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharex=True, sharey=True)
    for ax, key, title in ((axes[0], "xgb", "XGBoost (asymmetric)"),
                           (axes[1], "blend", "Production piecewise blend (XGB+LSTM)")):
        pred = payload[key]
        for compound in sorted(set(compounds)):
            mask = np.array([c == compound for c in compounds])
            ax.scatter(truth[mask], pred[mask], s=3, alpha=.25,
                       color=palette.get(compound, "#9467bd"),
                       linewidths=0,
                       label=f"{compound} (n={mask.sum():,})")
        hi = float(max(truth.max(), pred.max())) * 1.05
        ax.plot([0, hi], [0, hi], "k--", lw=1, label="y = x")
        ax.axvspan(0, 10, color="#d62728", alpha=.07)
        ax.set(xlabel="Actual remaining useful life (laps)",
               ylabel="Predicted RUL (laps)", title=title)
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        mae = float(np.abs(truth - pred).mean())
        ax.text(0.03, 0.95, f"MAE = {mae:.2f}", transform=ax.transAxes,
                va="top", fontsize=9)
    axes[0].legend(fontsize=8, markerscale=3, loc="lower right")
    axes[0].text(0.03, 0.88, "shaded: critical zone (RUL $\\leq$ 10)",
                 transform=axes[0].transAxes, fontsize=8, color="#d62728")
    fig.suptitle("Model 1 — remaining useful life, test split 2025 (out of sample)")
    fig.tight_layout()
    _save(fig, "rul_predicted_vs_actual.png")


def degradation_curves() -> None:
    """Observed per-compound degradation against the fitted beta line."""
    import polars as pl

    from src.models.eval_rul import build_labeled_frame
    from src.models.rul_data import STINT_KEYS

    params = _load_json("degradation_model.json")
    beta = params.get("beta", {})
    df = build_labeled_frame()
    frame = df.filter(pl.col("compound").is_in(["SOFT", "MEDIUM", "HARD"])
                      & pl.col("fuel_corr").is_not_null()
                      & pl.col("is_sc_lap").is_not_null()
                      & (pl.col("life") >= 1))
    if frame.height == 0:
        print("[plots] labeled frame unavailable — skipping degradation_curves")
        return
    frame = frame.with_columns(
        (pl.col("fuel_corr") - pl.col("fuel_corr").min().over(STINT_KEYS))
        .alias("pace_delta"))
    frame = frame.with_columns(
        pl.col("pace_delta").median().over([pl.col("compound"), "life"])
        .alias("pace_delta_med"))
    curve = (frame.group_by(["compound", "life"])
             .agg(pl.col("pace_delta_med").first())
             .sort(["compound", "life"]))

    fig, ax = plt.subplots(figsize=(9.5, 6))
    colors = {"SOFT": "#d62728", "MEDIUM": "#ff7f0e", "HARD": "#2ca02c"}
    for compound in ("SOFT", "MEDIUM", "HARD"):
        sub = curve.filter(pl.col("compound") == compound)
        if sub.height == 0:
            continue
        life = sub["life"].to_numpy().astype(float)
        observed = sub["pace_delta_med"].to_numpy().astype(float)
        ax.plot(life, observed, "-o", ms=3, color=colors[compound],
                label=f"{compound} observed (median)")
        slope = beta.get(compound)
        if slope is not None:
            ax.plot(life, slope * life, "--", color=colors[compound], lw=1.6,
                    label=f"{compound} fitted $\\beta$ = {slope:.4f} s/lap")
    ax.set(xlabel="Tyre life (laps)", ylabel="Pace loss vs stint best (s)",
           title="Model 3 — measured degradation curves against the fitted "
                 "$\\beta$ (fuel-corrected, SC laps and the out-lap excluded)")
    ax.set_ylim(-0.5, 8.0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "degradation_curves.png")


def cluster_embeddings() -> None:
    """2-D projection of the driver style latent space, coloured by cluster."""
    import polars as pl
    from sklearn.manifold import TSNE

    path = ROOT / "data/gold/driver_embeddings.parquet"
    if not path.exists():
        print("[plots] driver_embeddings.parquet unavailable — skipping "
              "cluster_embeddings")
        return
    emb = pl.read_parquet(path)
    latent = [c for c in emb.columns if c.startswith("lat_")]
    if not latent:
        print("[plots] no latent columns — skipping cluster_embeddings")
        return
    matrix = emb.select(latent).to_numpy().astype(float)
    if matrix.shape[0] < 5:
        print("[plots] too few driver-seasons — skipping cluster_embeddings")
        return
    coords = TSNE(n_components=2, perplexity=min(25, max(5, matrix.shape[0] // 6)),
                  init="pca", random_state=42).fit_transform(matrix)
    clusters = emb["cluster"].to_list()
    drivers = emb["driver"].to_list()
    seasons = emb["season"].to_list()

    metrics = _load_json("phase5_metrics.json") if (MODELS_DIR / "phase5_metrics.json").exists() else {}
    summary = metrics.get("cluster_summary", {})
    palette = {-1: "#7f7f7f", 0: "#1f77b4", 1: "#d62728", 2: "#2ca02c"}

    fig, ax = plt.subplots(figsize=(9.5, 7))
    for cluster in sorted({c for c in clusters if c is not None}):
        mask = np.array([c == cluster for c in clusters])
        entry = summary.get(str(cluster), {})
        deg = entry.get("deg_slope")
        label = f"cluster {cluster} (n={mask.sum()})"
        if deg is not None:
            label += f", $\\beta$ = {deg:.3f} s/lap"
        ax.scatter(coords[mask, 0], coords[mask, 1], s=70, alpha=.85,
                   color=palette.get(cluster, "#9467bd"), edgecolors="k",
                   linewidths=.4, label=label)
    # label the most extreme drivers so the archetypes are readable
    for i in np.argsort(-emb["lat_g_max_mean"].to_numpy() if "lat_g_max_mean" in emb.columns
                        else -np.arange(len(drivers)))[:6]:
        ax.annotate(f"{drivers[i]} {seasons[i]}", (coords[i, 0], coords[i, 1]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2",
           title="Model 2 — driving-style archetypes in the 16-D latent space")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, "cluster_embeddings.png")


def backtest_delta_histogram() -> None:
    """Distribution of the paired backtest deltas (actual - recommended)."""
    path = MODELS_DIR / "phase7_backtest.json"
    if not path.exists():
        print("[plots] phase7_backtest.json unavailable — skipping "
              "backtest_delta_histogram")
        return
    payload = json.loads(path.read_text())
    rows = payload.get("decisions", [])
    if not rows:
        print("[plots] no decisions in phase7_backtest.json — skipping "
              "backtest_delta_histogram")
        return
    deltas = np.array([float(r["delta"]) for r in rows], dtype=float)
    summary = payload.get("summary", {})

    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.hist(deltas, bins=40, color="#1f77b4", alpha=.75, edgecolor="k", linewidth=.4)
    median = float(np.median(deltas))
    mean = float(deltas.mean())
    ax.axvline(median, color="#d62728", lw=2, label=f"median {median:+.1f} s")
    ax.axvline(mean, color="#ff7f0e", lw=2, ls="--", label=f"mean {mean:+.1f} s")
    ax.axvline(0.0, color="k", lw=1, ls=":", label="no advantage")
    wins = float((deltas > 0).mean())
    ax.set(xlabel="$\\Delta$ = actual - recommended (s, positive favours the optimiser)",
           ylabel="Decisions",
           title=f"Backtest delta distribution — {len(deltas)} decisions, "
                 f"optimizer wins {wins:.1%}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, "backtest_delta_histogram.png")


# ---------------------------------------------------------------------------
FIGURES = {
    "scenario_multiplier_regression": scenario_multiplier_regression,
    "critical_zone_r2": critical_zone_r2,
    "rul_predicted_vs_actual": rul_predicted_vs_actual,
    "degradation_curves": degradation_curves,
    "cluster_embeddings": cluster_embeddings,
    "backtest_delta_histogram": backtest_delta_histogram,
    "mapreduce_shuffle": mapreduce_shuffle,
    "mongo_indexes": mongo_indexes,
    "three_way_parity": three_way_parity,
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
