"""Materialize the common Phase 9 input, including the optimizer's style cluster."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import polars as pl

from src.ingest.config import ROOT

OUTPUT = ROOT / "data/gold/labeled_frame.parquet"
EMBEDDINGS = ROOT / "data/gold/driver_embeddings.parquet"
EXPECTED_ROWS = 107_606
REQUIRED = {"fuel_corr", "pace_above_best", "cluster", "is_sc_lap", "session_type"}


def _read_embeddings(embeddings=None):
    """The style clusters an export is built from (None when unavailable)."""
    if isinstance(embeddings, pl.DataFrame):
        return embeddings
    path = EMBEDDINGS if embeddings is None else embeddings
    try:
        return pl.read_parquet(path)
    except (FileNotFoundError, OSError):
        return None


def _build_frame(wet_context, builder, embeddings):
    """Materialize the labeled frame from the gold features and the clusters."""
    if builder is None:
        from .eval_rul import build_labeled_frame
        builder = build_labeled_frame
    frame = builder().with_columns(pl.col("season").cast(pl.String))
    emb = _read_embeddings(embeddings)
    if emb is None:
        raise FileNotFoundError(f"style embeddings are missing: {EMBEDDINGS}")
    emb = emb.with_columns(pl.col("season").cast(pl.String))
    frame = frame.join(emb.select("season", "driver", "cluster"),
                       on=["season", "driver"], how="left", validate="m:1")
    frame = frame.with_columns(pl.col("cluster").fill_null(-1).cast(pl.Int32))
    if wet_context:
        from .fit_degradation import _wet_context
        frame = _wet_context(frame)
    return frame


def _cluster_drift(frame, embeddings):
    """True when the exported clusters no longer match the style embeddings.

    A re-fitted style model rewrites driver_embeddings.parquet, so an
    exported frame can silently disagree with the source of its cluster
    column. The comparison is per driver-season, exactly the grain of the
    join that produced the column, and it uses the same convention: a driver
    without a cluster carries -1.
    """
    emb = _read_embeddings(embeddings)
    if emb is None or "cluster" not in frame.columns or "cluster" not in emb.columns:
        return False
    cached = (frame.select("season", "driver", "cluster")
              .group_by(["season", "driver"]).agg(pl.col("cluster").first())
              .with_columns(pl.col("season").cast(pl.String), pl.col("cluster").cast(pl.Int64)))
    current = (emb.select("season", "driver", "cluster")
               .with_columns(pl.col("season").cast(pl.String), pl.col("cluster").cast(pl.Int64)))
    expected = (cached.select("season", "driver")
                .join(current, on=["season", "driver"], how="left")
                .with_columns(pl.col("cluster").fill_null(-1).alias("expected")))
    joined = cached.join(expected.select("season", "driver", "expected"),
                          on=["season", "driver"], how="left")
    if joined.height == 0:
        return False
    return bool(joined.filter(pl.col("cluster") != pl.col("expected")).height)


def _validate(frame, expected_rows):
    if frame.height != expected_rows:
        raise ValueError(f"labeled row count {frame.height:,} != {expected_rows:,}; investigate upstream regression")
    if REQUIRED - set(frame.columns):
        raise ValueError(f"labeled frame missing {REQUIRED - set(frame.columns)}")


def export_labeled(output=OUTPUT, *, force=False, wet_context=False,
                   expected_rows=EXPECTED_ROWS, builder=None, embeddings=None):
    output = Path(output)
    write = force or not output.exists()
    if output.exists() and not force:
        frame = pl.read_parquet(output)
        if wet_context and "track_drying_proxy" not in frame.columns:
            raise ValueError("cached frame has no wet context; rebuild with --force --wet-context")
        _validate(frame, expected_rows)
        if _cluster_drift(frame, embeddings):
            # The style clusters were re-fitted: the cached labels are stale
            # and must never be served against the new embeddings.
            print("[labeled] the style embeddings changed; the cached frame is stale")
            frame = _build_frame(wet_context, builder, embeddings)
            _validate(frame, expected_rows)
            write = True
    else:
        frame = _build_frame(wet_context, builder, embeddings)
        _validate(frame, expected_rows)
    if write:
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_suffix(".tmp.parquet")
        frame.write_parquet(tmp, row_group_size=8192)
        os.replace(tmp, output)
    print(f"[labeled] {frame.height:,} rows x {frame.width} columns -> {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--wet-context", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    export_labeled(args.output, force=args.force, wet_context=args.wet_context)


if __name__ == "__main__":
    main()
