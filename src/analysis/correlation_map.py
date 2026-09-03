"""Correlation maps of the variables fed into the models (pre-ingest).

Reuses the exact feature matrices produced by the Phase 4 (RUL/cliff) and
Phase 5 (driving-style AE) data-prep modules, then renders Pearson correlation
heatmaps + feature-vs-target bar charts as PNGs.

Output folder: <repo-root>/correlation_maps/
  - rul_features_corr.png     heatmap of RUL/cliff numeric feature correlations
  - rul_target_corr.png       correlation of each feature with the two labels
  - style_features_corr.png   heatmap of style-AE continuous feature correlations

Usage:
    python -m src.analysis.correlation_map
    python -m src.analysis.correlation_map --only rul
    python -m src.analysis.correlation_map --only style
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.ingest.config import ROOT  # noqa: E402

OUT_DIR = ROOT / "correlation_maps"

sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   -> {path}")


def _drop_constant(X: pd.DataFrame) -> pd.DataFrame:
    """Drop non-numeric / constant / all-NaN columns (zero variance)."""
    X = X.select_dtypes(include=[np.number])
    keep = [c for c in X.columns
            if X[c].nunique(dropna=True) > 1
            and float(X[c].std(skipna=True)) > 0]
    return X[keep]


def _numeric_corr(X: pd.DataFrame) -> pd.DataFrame:
    """Drop zero-variance columns, then pairwise Pearson correlation."""
    X = _drop_constant(X)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = X.corr(method="pearson")
    return corr.fillna(0.0)


def _heatmap(corr: pd.DataFrame, title: str, name: str, annot_max: int = 28) -> None:
    n = corr.shape[0]
    fig, ax = plt.subplots(figsize=(max(10, n * 0.30), max(8, n * 0.30)))
    annot = n <= annot_max
    sns.heatmap(
        corr,
        ax=ax,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        square=True,
        annot=annot,
        fmt=".2f" if annot else "",
        annot_kws={"size": 5} if annot else None,
        cbar_kws={"label": "Pearson correlation", "shrink": 0.8},
        linewidths=0,
    )
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right",
                       fontsize=6 if n > 30 else 8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6 if n > 30 else 8)
    fig.tight_layout()
    _save(fig, name)


def _target_bars(tc: pd.DataFrame, title: str, name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(18, max(8, tc.shape[0] * 0.28)))
    for ax, col in zip(axes, tc.columns):
        s = tc[col].sort_values(key=lambda x: x.abs(), ascending=False)
        colors = ["#d62728" if v >= 0 else "#1f77b4" for v in s.values]
        ax.barh(s.index, s.values, color=colors)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"correlation with {col}", fontsize=12)
        ax.set_xlabel("Pearson r", fontsize=10)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlim(-1, 1)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    _save(fig, name)


# ---------------------------------------------------------------------------
# Phase 4 — RUL / cliff model features
# ---------------------------------------------------------------------------
def build_rul_maps() -> None:
    from src.models import rul_data as RD

    print("[rul] preparing model feature matrix (filtered stints, fuel-corrected, "
          "rolling)...")
    tab, _seq, feats, _codes, _norm = RD.prepare()

    num_feats = [f for f in feats if f not in RD.CATEGORICAL]
    frames, y_rul, y_cliff = [], [], []
    for sp in ("train", "val", "test"):
        X, r, c = tab[sp]
        frames.append(X.select(num_feats).to_pandas())
        y_rul.append(r)
        y_cliff.append(c)
    X = pd.concat(frames, ignore_index=True)
    y_rul = np.concatenate(y_rul)
    y_cliff = np.concatenate(y_cliff)

    X = _drop_constant(X)
    print(f"[rul] numeric features: {X.shape[1]}  rows: {X.shape[0]:,}")
    corr = _numeric_corr(X)
    _heatmap(corr, "RUL / cliff model inputs — feature correlation "
                    f"({corr.shape[0]} features)",
             "rul_features_corr.png")

    with np.errstate(divide="ignore", invalid="ignore"):
        tc = pd.DataFrame({
            "target_rul": X.corrwith(pd.Series(y_rul, index=X.index)),
            "cliff_within_3": X.corrwith(pd.Series(y_cliff, index=X.index)),
        }).fillna(0.0)
    _target_bars(tc, "RUL / cliff model inputs — feature vs. target",
                 "rul_target_corr.png")


# ---------------------------------------------------------------------------
# Phase 5 — driving-style AE features
# ---------------------------------------------------------------------------
def build_style_map() -> None:
    from src.models import style_data as SD

    print("[style] preparing style-AE feature matrix (micro-sectors + track "
          "evolution)...")
    splits, meta, _df = SD.prepare_style()
    feats = meta["features"]

    cont = [f for f in feats if not f.startswith("comp_")
            and not f.startswith("st_")]
    idx = [feats.index(f) for f in cont]
    X = np.vstack([splits["train"]["X"], splits["test"]["X"]])[:, idx]
    X = pd.DataFrame(X, columns=cont)

    print(f"[style] continuous features: {len(cont)}  rows: {X.shape[0]:,}")
    corr = _numeric_corr(X)
    _heatmap(corr, "Driving-style AE inputs — feature correlation "
                   f"({corr.shape[0]} features)",
             "style_features_corr.png")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate model-input correlation maps")
    ap.add_argument("--only", choices=["rul", "style"], default=None)
    args = ap.parse_args()

    print(f"Output folder: {OUT_DIR}")
    if args.only in (None, "rul"):
        build_rul_maps()
    if args.only in (None, "style"):
        build_style_map()
    print(f"Done. Figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
