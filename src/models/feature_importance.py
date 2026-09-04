"""Task 8.H.5 — Feature importance & interpretability for the RUL model.

Dumps XGBoost gain importance (sorted CSV + markdown) and generates SHAP
summary plots for the top-20 RUL predictors, then flags low-importance
features for future pruning.

    python -m src.models.feature_importance
    python -m src.models.feature_importance --max-bg 5000 --no-shap
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from src.ingest.config import ROOT

MODELS_DIR = ROOT / "data" / "models"
OUT_DIR = MODELS_DIR / "feature_importance"


def _load_features(meta: dict) -> list[str]:
    return list(meta["features"])


def _gain_importance(model, feats: list[str]) -> list[dict]:
    raw = model.get_score(importance_type="gain")
    rows = []
    for i, f in enumerate(feats):
        key = f"f{i}"
        rows.append({"feature": f, "gain": float(raw.get(key, 0.0))})
    rows.sort(key=lambda r: -r["gain"])
    total = sum(r["gain"] for r in rows) or 1.0
    for r in rows:
        r["share"] = round(r["gain"] / total, 6)
    return rows


def _markdown_table(rows: list[dict], top: int = 60) -> str:
    lines = ["| Rank | Feature | Gain | Share |",
             "|---:|:---|---:|---:|"]
    for i, r in enumerate(rows[:top], 1):
        lines.append(f"| {i} | `{r['feature']}` | {r['gain']:.1f} | "
                     f"{r['share']:.4f} |")
    return "\n".join(lines)


def _shap_importance(model, X: np.ndarray, feats: list[str],
                     max_bg: int) -> tuple[list[dict], object | None, object | None]:
    """Return (mean|SHAP| rows, shap_values, explainer). Uses XGBoost's native
    contribution API so no external shap dependency is required for the values;
    the shap package is used only for the summary plots."""
    import xgboost as xgb
    n = min(len(X), max_bg)
    Xs = X[:n]
    dmat = xgb.DMatrix(Xs, feature_names=feats)
    contrib = model.predict(dmat, pred_contribs=True)
    # last column is the bias (base margin); drop it
    shap_vals = contrib[:, :-1]
    mean_abs = np.abs(shap_vals).mean(axis=0)
    rows = [{"feature": f, "mean_abs_shap": float(v)}
            for f, v in zip(feats, mean_abs)]
    rows.sort(key=lambda r: -r["mean_abs_shap"])
    explainer = None
    return rows, shap_vals, Xs, explainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-bg", type=int, default=5000)
    ap.add_argument("--no-shap", action="store_true")
    args = ap.parse_args()

    import xgboost as xgb

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODELS_DIR / "feature_meta.json") as fh:
        meta = json.load(fh)
    feats = _load_features(meta)

    model = xgb.Booster()
    model.load_model(str(MODELS_DIR / "xgb_rul.json"))

    # --- 8.H.5.1: gain importance -> CSV + markdown ---
    gain = _gain_importance(model, feats)
    csv_path = OUT_DIR / "gain_importance.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "feature", "gain", "share"])
        for i, r in enumerate(gain, 1):
            w.writerow([i, r["feature"], r["gain"], r["share"]])
    md_path = OUT_DIR / "gain_importance.md"
    md_path.write_text(
        "# XGBoost RUL gain importance\n\n" + _markdown_table(gain) + "\n",
        encoding="utf-8")
    print(f"[importance] top-15 gain features:")
    for r in gain[:15]:
        print(f"  {r['feature']:26s} gain={r['gain']:12.1f} share={r['share']:.4f}")
    print(f"  -> saved {csv_path.name} and {md_path.name}")

    # --- 8.H.5.2: SHAP summary plots (top 20) ---
    if args.no_shap:
        print("[importance] --no-shap: skipping summary plots")
        shap_rows = []
    else:
        try:
            from src.models.rul_data import prepare
            print("[importance] preparing test data for SHAP...")
            tab, _, _, _, _ = prepare()
            Xte = tab["test"][0].to_pandas().to_numpy(
                dtype=np.float32, na_value=np.nan)
            shap_rows, shap_vals, Xs, _ = _shap_importance(
                model, Xte, feats, args.max_bg)
            top20 = [r["feature"] for r in shap_rows[:20]]
            idx = [feats.index(f) for f in top20]
            try:
                import shap
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                Xd = xgb.DMatrix(Xs, feature_names=feats)
                explainer = shap.TreeExplainer(model)
                sv = shap_vals[:, idx]

                fig = plt.figure(figsize=(8, 10))
                shap.summary_plot(sv, Xs[:, idx], feature_names=top20,
                                  show=False, plot_size=(8, 10))
                plt.title("Top 20 RUL predictors — SHAP (2025 test)")
                fig_path = OUT_DIR / "shap_summary_top20.png"
                plt.savefig(fig_path, bbox_inches="tight", dpi=110)
                plt.close(fig)

                fig2 = plt.figure(figsize=(8, 10))
                shap.summary_plot(sv, Xs[:, idx], feature_names=top20,
                                  plot_type="bar", show=False,
                                  plot_size=(8, 10))
                plt.title("Top 20 RUL predictors — mean |SHAP|")
                bar_path = OUT_DIR / "shap_bar_top20.png"
                plt.savefig(bar_path, bbox_inches="tight", dpi=110)
                plt.close(fig2)
                print(f"  -> saved {fig_path.name} and {bar_path.name}")
            except Exception as exc:  # shap plotting is best-effort
                print(f"  [warn] shap plotting failed: {type(exc).__name__}: {exc}")
        except Exception as exc:
            print(f"  [warn] SHAP data prep failed: {type(exc).__name__}: {exc}")

    # --- 8.H.5.3: low-importance review for pruning ---
    gain_by_feat = {r["feature"]: r["gain"] for r in gain}
    unused = [f for f in feats if gain_by_feat.get(f, 0.0) == 0.0]
    total = sum(gain_by_feat.values()) or 1.0
    cum = 0.0
    prune = []
    for r in gain[::-1]:  # ascending gain
        if r["gain"] > 0:
            break
    # bottom decile by gain share
    sorted_asc = sorted(gain, key=lambda r: r["gain"])
    bottom = sorted_asc[: max(1, len(sorted_asc) // 10)]
    review = {
        "n_features": len(feats),
        "zero_gain_features": unused,
        "bottom_decile_features": [r["feature"] for r in bottom],
        "top10_gain_share": round(sum(r["share"] for r in gain[:10]), 3),
        "note": ("Features with zero gain never produced a split; the bottom "
                 "decile contributes negligible importance. Candidates for "
                 "pruning in a future refactor."),
    }
    with open(OUT_DIR / "pruning_review.json", "w", encoding="utf-8") as fh:
        json.dump(review, fh, indent=2)
    print(f"\n[importance] pruning review:")
    print(f"  zero-gain features ({len(unused)}): {unused}")
    print(f"  top-10 gain share: {review['top10_gain_share']}")
    print(f"  -> saved {OUT_DIR / 'pruning_review.json'}")


if __name__ == "__main__":
    main()
