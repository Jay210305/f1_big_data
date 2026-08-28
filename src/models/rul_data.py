"""Phase 4 data prep: labels, leakage-safe features, splits, sequences.

Temporal split (by season) to avoid same-GP leakage:
    train = 2021-2023   val = 2024   test = 2025

Labels (computed from full stint history — the label, not a feature):
    rul            = max_life_in_stint - life   (laps remaining in stint)
    cliff_within_3 = 1 if any of the next 3 laps in the stint has cliff_flag

Leakage-safe features: all lap_features columns EXCEPT identifiers
(season, gp, session, driver, lap) and the derived labels.
cliff_flag (current lap) is allowed because targets are strictly future.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from src.ingest.config import ROOT

GOLD = ROOT / "data" / "gold"
CACHE = GOLD / "model_data"
CACHE.mkdir(parents=True, exist_ok=True)

IDENTIFIERS = {"season", "gp", "session", "driver", "lap"}
LABELS = {"rul", "cliff_within_3"}
SPLIT = {"train": ["2021", "2022", "2023"], "val": ["2024"], "test": ["2025"]}
STINT_KEYS = ["season", "gp", "session", "driver", "stint"]
CATEGORICAL = ["compound", "session_type"]


def _session_type(name: str) -> str:
    n = (name or "").lower()
    if n.startswith("race"):
        return "Race"
    if n.startswith("sprint"):
        return "Sprint"
    if n.startswith("qualifying") or "shootout" in n:
        return "Qualifying"
    if n.startswith("practice"):
        return "Practice"
    return "Testing"


def load_lap_features() -> pl.DataFrame:
    return pl.read_parquet(GOLD / "lap_features.parquet")


def build_labels(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(STINT_KEYS + ["life"])
    df = df.with_columns(
        pl.col("life").max().over(STINT_KEYS).alias("max_life"))
    df = df.with_columns((pl.col("max_life") - pl.col("life")).alias("rul"))
    # forward cliff: any of next 3 laps within stint has cliff_flag
    for k in (1, 2, 3):
        df = df.with_columns(
            pl.col("cliff_flag").shift(-k).over(STINT_KEYS).alias(f"_cf{k}"))
    df = df.with_columns(
        (pl.col("_cf1").fill_null(False) | pl.col("_cf2").fill_null(False)
         | pl.col("_cf3").fill_null(False)).alias("cliff_within_3"))
    df = df.drop(["_cf1", "_cf2", "_cf3", "max_life"])
    return df


def feature_cols(df: pl.DataFrame) -> list[str]:
    excl = IDENTIFIERS | LABELS
    return [c for c in df.columns if c not in excl]


def make_tabular(df: pl.DataFrame):
    """Return dict split -> (X_df, y_rul, y_cliff) with categoricals encoded."""
    feats = feature_cols(df)
    codes: dict[str, dict] = {}
    for c in CATEGORICAL:
        df = df.with_columns(pl.col(c).fill_null("UNKNOWN").alias(c))
        mapping = {v: i for i, v in enumerate(sorted(df[c].unique().to_list()))}
        codes[c] = mapping
        df = df.with_columns(pl.col(c).replace(mapping).cast(pl.Int32).alias(c))
    out = {}
    for split, years in SPLIT.items():
        sub = df.filter(pl.col("season").is_in(years))
        X = sub.select(feats)
        out[split] = (X, sub["rul"].to_numpy().astype(np.float32),
                      sub["cliff_within_3"].to_numpy().astype(np.float32))
    return out, feats, codes


def make_sequences(df: pl.DataFrame, feats: list[str], codes: dict):
    """Build per-stint sequences for the LSTM. Returns dict split -> dict."""
    for c in CATEGORICAL:
        df = df.with_columns(pl.col(c).fill_null("UNKNOWN").alias(c))
        df = df.with_columns(
            pl.col(c).replace(codes[c]).cast(pl.Int32).alias(c))
    num_feats = [f for f in feats if f not in CATEGORICAL]
    train_df = df.filter(pl.col("season").is_in(SPLIT["train"]))
    means = {c: float(train_df[c].mean() or 0.0) for c in num_feats}
    stds = {c: float(train_df[c].std() or 1.0) or 1.0 for c in num_feats}

    def encode(sub: pl.DataFrame):
        sub = sub.sort(STINT_KEYS + ["life"])
        seqs, ruls, cliffs, sids = [], [], [], []
        for key, grp in sub.group_by(STINT_KEYS):
            mat = np.zeros((grp.height, len(feats)), dtype=np.float32)
            for j, f in enumerate(feats):
                if f in CATEGORICAL:
                    mat[:, j] = grp[f].to_numpy()
                else:
                    col = grp[f].to_numpy().astype(np.float32)
                    col = np.where(np.isfinite(col), col, 0.0)
                    mat[:, j] = (col - means[f]) / stds[f]
            seqs.append(mat)
            ruls.append(grp["rul"].to_numpy().astype(np.float32))
            cliffs.append(grp["cliff_within_3"].to_numpy().astype(np.float32))
            sids.append(key)
        return {"X": seqs, "y_rul": ruls, "y_cliff": cliffs, "sids": sids}

    return {sp: encode(df.filter(pl.col("season").is_in(yrs)))
            for sp, yrs in SPLIT.items()}, feats, (means, stds)


def prepare():
    df = load_lap_features()
    df = df.with_columns(pl.col("season").cast(pl.Utf8))
    df = df.with_columns(
        pl.col("session").map_elements(_session_type, return_dtype=pl.Utf8)
        .alias("session_type"))
    df = build_labels(df)
    # drop laps without a usable stint / life / rul
    df = df.filter(
        pl.col("life").is_not_null()
        & pl.col("stint").is_not_null()
        & pl.col("rul").is_not_null())
    tab, feats, codes = make_tabular(df)
    seq, feats, norm = make_sequences(df, feats, codes)
    return tab, seq, feats, codes, norm


if __name__ == "__main__":
    tab, seq, feats, codes, norm = prepare()
    for sp in ["train", "val", "test"]:
        X, yr, yc = tab[sp]
        print(f"{sp:5s} tabular: X={X.shape} rul mean={yr.mean():.2f} "
              f"cliff+={yc.mean():.3f}")
        sq = seq[sp]
        print(f"      seqs: {len(sq['X'])} stints, "
              f"len range {min(len(s) for s in sq['X'])}-"
              f"{max(len(s) for s in sq['X'])}")
    print(f"\nfeatures ({len(feats)}): {feats}")
    print(f"compound codes: {codes}")
