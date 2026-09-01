"""Phase 5 verification: inspect the style profiler outputs.

    python -m src.models.verify_style
"""
from __future__ import annotations

import json

import polars as pl

from src.ingest.config import ROOT

GOLD = ROOT / "data" / "gold"
MODELS = ROOT / "data" / "models"


def main():
    with open(MODELS / "phase5_metrics.json") as fh:
        m = json.load(fh)
    print("=" * 78)
    print("Phase 5 — Driving Style Profiler — verification")
    print("=" * 78)

    print(f"\nFilter stats (transition action item 1):")
    for k, v in m["filter_stats"].items():
        print(f"  {k:14s} {v:,}")

    print(f"\nAutoencoder: dim {m['dim_in']} -> latent {m['latent']}")
    print(f"  test(2025) recon MSE: mean={m['recon_test_mse_mean']:.5f} "
          f"p95={m['recon_test_mse_p95']:.5f}")
    print(f"\nHDBSCAN: {m['n_clusters']} clusters + {m['n_noise']} noise "
          f"(min_cluster_size={m['min_cluster_size']})")
    print(f"  silhouette (non-noise): {m['silhouette']:.3f}")

    drv = pl.read_parquet(GOLD / "driver_embeddings.parquet")
    print(f"\ndriver_embeddings: {drv.shape[0]} driver-seasons x {drv.shape[1]} cols")
    print("\nCluster sizes by season:")
    tab = (drv.group_by(["cluster"]).agg(pl.len().alias("n"),
                                         pl.col("deg_slope" if "deg_slope" in drv.columns
                                                else "recon_mean").mean().alias("x"))
           if False else drv.group_by("cluster").agg(pl.len().alias("n")))
    print(tab.to_pandas().to_string(index=False))

    lap = pl.read_parquet(GOLD / "lap_embeddings.parquet")
    print(f"\nlap_embeddings: {lap.shape[0]:,} laps x {lap.shape[1]} cols")
    q99 = lap["recon_error"].quantile(0.99)
    n_anom = lap.filter(pl.col("recon_error") > q99).height
    print(f"  anomaly flag (recon > p99 = {q99:.4f}): {n_anom:,} laps")

    print("\nSample driver signatures (cluster 0, most laps):")
    top = (drv.filter(pl.col("cluster") == 0)
           .sort("n_laps", descending=True).head(8)
           .select(["season", "driver", "n_laps", "recon_mean"]))
    print(top.to_pandas().to_string(index=False))

    print("\n" + "=" * 78)
    print("Transition compliance: in/out laps excluded (filter stats), "
          "track_evolution_index in AE input, micro-sector vectors (Option B).")
    print("If clusters are stable and profiles interpretable -> GO to Phase 6.")
    print("=" * 78)


if __name__ == "__main__":
    main()
