"""Phase 5 — Train the Driving Style Autoencoder + HDBSCAN profiler.

Pipeline (transition-spec compliant):
  1. Per-lap micro-sector vectors (Option B) + track_evolution_index covariate.
  2. MLP Autoencoder (GPU): 55 -> 128 -> 64 -> latent 16 -> 64 -> 128 -> 55.
     Trained on 2021-2024 (90/10 early stop), generalization recon on 2025.
  3. Latent embeddings for ALL filtered laps -> mean per (season, driver)
     = driving-style signature; per-lap recon error = anomaly flag.
  4. HDBSCAN on driver-season signatures -> style archetypes.
  5. Interpretation: cluster profiles from driver_style gold table +
     tyre-care impact (dry-compound stint degradation_slope per cluster).
  6. Export data/gold/driver_embeddings.parquet.

CLI:
    python -m src.models.train_style
    python -m src.models.train_style --latent 24 --min-cluster 10
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import polars as pl

from src.ingest.config import ROOT
from src.models.style_data import prepare_style

GOLD = ROOT / "data" / "gold"
MODELS_DIR = ROOT / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def train_autoencoder(Xtr, Xva, Xte, dim_in, latent, epochs, device):
    import torch
    import torch.nn as nn

    def t(a):
        return torch.from_numpy(a).to(device)

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Linear(dim_in, 128), nn.GELU(), nn.BatchNorm1d(128),
                nn.Linear(128, 64), nn.GELU(),
                nn.Linear(64, latent))
            self.dec = nn.Sequential(
                nn.Linear(latent, 64), nn.GELU(),
                nn.Linear(64, 128), nn.GELU(),
                nn.Linear(128, dim_in))

        def forward(self, x):
            return self.dec(self.enc(x))

    net = AE().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=2e-3, epochs=epochs, steps_per_epoch=int(np.ceil(len(Xtr) / 1024)))
    crit = nn.MSELoss()
    bs = 1024

    def epoch(X, train):
        net.train() if train else net.eval()
        idx = np.arange(len(X))
        if train:
            np.random.shuffle(idx)
        losses = []
        with torch.set_grad_enabled(train):
            for s in range(0, len(idx), bs):
                xb = t(X[idx[s:s + bs]])
                loss = crit(net(xb), xb)
                if train:
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    opt.step(); sched.step()
                losses.append(float(loss.detach()))
        return float(np.mean(losses))

    best, best_state, bad, patience = 1e9, None, 0, 8
    for e in range(epochs):
        tr = epoch(Xtr, True)
        va = epoch(Xva, False)
        if va < best - 1e-5:
            best, bad = va, 0
            best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if (e + 1) % 10 == 0 or e == 0:
            print(f"   ae epoch {e+1:3d}  tr={tr:.5f} va={va:.5f}")
        if bad >= patience:
            print(f"   ae early stop at epoch {e+1}")
            break
    net.load_state_dict(best_state)
    torch.save(best_state, str(MODELS_DIR / "style_ae.pt"))

    net.eval()
    with torch.no_grad():
        rec_te = ((net(t(Xte)) - t(Xte)) ** 2).mean(dim=1).cpu().numpy()
    return net, rec_te


def embed_all(net, X, device, bs=4096):
    import torch
    net.eval()
    Z, R = [], []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            xb = torch.from_numpy(X[s:s + bs]).to(device)
            z = net.enc(xb)
            r = ((net.dec(z) - xb) ** 2).mean(dim=1)
            Z.append(z.cpu().numpy())
            R.append(r.cpu().numpy())
    return np.vstack(Z), np.concatenate(R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--latent", type=int, default=16)
    ap.add_argument("--min-cluster", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()

    splits, meta, df = prepare_style()
    feats = meta["features"]
    dim_in = len(feats)
    X_all = np.vstack([splits["train"]["X"], splits["test"]["X"]])
    n_tr = len(splits["train"]["X"])

    rng = np.random.RandomState(0)
    idx = rng.permutation(n_tr)
    n_fit = int(0.9 * n_tr)
    Xtr, Xva = splits["train"]["X"][idx[:n_fit]], splits["train"]["X"][idx[n_fit:]]
    Xte = splits["test"]["X"]

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[ae] training on {device}: dim {dim_in} -> latent {args.latent}, "
          f"{len(Xtr):,} laps")
    net, rec_te = train_autoencoder(Xtr, Xva, Xte, dim_in, args.latent,
                                    args.epochs, device)
    print(f"[ae] test (2025) recon MSE: mean={rec_te.mean():.5f} "
          f"p95={np.quantile(rec_te, 0.95):.5f}")

    print("[ae] embedding all filtered laps...")
    Z, R = embed_all(net, X_all, device)
    keys = pl.concat([splits["train"]["keys"], splits["test"]["keys"]])
    lap_emb = keys.with_columns(
        [pl.Series(f"lat_{i}", Z[:, i]) for i in range(Z.shape[1])]
        + [pl.Series("recon_error", R)])

    print("[hdbscan] clustering driver-season signatures...")
    drv = (lap_emb.group_by(["season", "driver"])
           .agg([pl.col(f"lat_{i}").mean() for i in range(Z.shape[1])]
                + [pl.col("recon_error").mean().alias("recon_mean"),
                   pl.len().alias("n_laps")]))
    Zd = drv.select([f"lat_{i}" for i in range(Z.shape[1])]).to_numpy()
    import hdbscan
    clu = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster, metric="euclidean")
    labels = clu.fit_predict(Zd)
    drv = drv.with_columns(pl.Series("cluster", labels))
    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    print(f"[hdbscan] {n_clusters} clusters, {n_noise} noise driver-seasons "
          f"of {len(labels)}")

    sil = float("nan")
    if n_clusters >= 2:
        mask = labels != -1
        if mask.sum() > n_clusters:
            from sklearn.metrics import silhouette_score
            sil = float(silhouette_score(Zd[mask], labels[mask]))
    print(f"[hdbscan] silhouette (non-noise) = {sil:.3f}")

    # --- interpretation: mechanical profile + tyre-care impact per cluster ---
    style_tbl = GOLD / "driver_style.parquet"
    deg_tbl = GOLD / "stint_degradation.parquet"
    prof_cols = ["throttle_modulations_mean", "brake_event_count_mean",
                 "brake_intensity_mean", "lat_g_max_mean", "speed_max_mean",
                 "throttle_std_mean", "dirty_air_frac_mean"]
    prof = None
    if style_tbl.exists():
        st = pl.read_parquet(style_tbl).with_columns(
            pl.col("season").cast(pl.Utf8))
        have = [c for c in prof_cols if c in st.columns]
        prof = drv.select(["season", "driver", "cluster"]).join(
            st.select(["season", "driver"] + have),
            on=["season", "driver"], how="left")
    deg_slope = None
    if "laptime" in df.columns and "stint" in df.columns:
        dry = df.filter(pl.col("compound").is_in(["SOFT", "MEDIUM", "HARD"]))
        # fuel-correct so the slope reflects TYRE degradation, not fuel burn
        dry = dry.with_columns(
            pl.when(pl.col("session_type").is_in(["Race", "Sprint"]))
            .then(pl.col("laptime") + 0.03 * pl.col("lap"))
            .otherwise(pl.col("laptime")).alias("fc"))
        sd = (dry.sort(["season", "gp", "session", "driver", "stint", "life"])
              .group_by(["season", "gp", "session", "driver", "stint"])
              .agg(pl.col("fc").first().alias("p_start"),
                   pl.col("fc").last().alias("p_end"),
                   pl.col("lat_load_integral").mean().alias("stint_lat_load"),
                   pl.len().alias("n"))
              .filter(pl.col("n") >= 4)
              .with_columns(((pl.col("p_end") - pl.col("p_start"))
                             / (pl.col("n") - 1)).alias("deg_slope"))
              .group_by(["season", "driver"])
              .agg(pl.col("deg_slope").mean().alias("deg_slope"),
                   pl.col("stint_lat_load").mean().alias("stint_lat_load")))
        deg_slope = drv.select(["season", "driver", "cluster"]).join(
            sd, on=["season", "driver"], how="left")

    cluster_summary = {}
    for c in sorted(set(labels)):
        sub = drv.filter(pl.col("cluster") == c)
        entry = {"n": sub.height,
                 "drivers": sub["driver"].unique().to_list()[:12]}
        if prof is not None:
            psub = prof.filter(pl.col("cluster") == c)
            for f in have:
                entry[f] = round(float(psub[f].mean() or 0), 3)
        if deg_slope is not None:
            dsub = deg_slope.filter(pl.col("cluster") == c)
            entry["deg_slope"] = round(float(dsub["deg_slope"].mean() or 0), 4)
            entry["stint_lat_load"] = round(
                float(dsub["stint_lat_load"].mean() or 0), 1)
        cluster_summary[int(c)] = entry

    out = GOLD / "driver_embeddings.parquet"
    drv.write_parquet(out, compression="zstd")
    lap_emb.write_parquet(GOLD / "lap_embeddings.parquet", compression="zstd")
    with open(MODELS_DIR / "phase5_metrics.json", "w") as fh:
        json.dump({
            "n_laps": int(len(X_all)), "dim_in": dim_in,
            "latent": args.latent,
            "recon_test_mse_mean": float(rec_te.mean()),
            "recon_test_mse_p95": float(np.quantile(rec_te, 0.95)),
            "n_clusters": n_clusters, "n_noise": n_noise,
            "silhouette": sil, "min_cluster_size": args.min_cluster,
            "filter_stats": meta["filter_stats"],
            "features": feats,
            "cluster_summary": cluster_summary,
        }, fh, indent=2, default=str)

    print("\n" + "=" * 78)
    print("PHASE 5 — DRIVING STYLE PROFILER — RESULTS")
    print("=" * 78)
    print(f"  laps embedded      : {len(X_all):,} (hot laps, in/out/SC excluded)")
    print(f"  driver-seasons     : {drv.height}")
    print(f"  AE recon (2025)    : mean={rec_te.mean():.5f} "
          f"p95={np.quantile(rec_te, 0.95):.5f}")
    print(f"  clusters           : {n_clusters} (+{n_noise} noise)")
    print(f"  silhouette         : {sil:.3f}")
    print("\nCluster profiles:")
    for c, e in cluster_summary.items():
        prof_str = " ".join(f"{k}={v}" for k, v in e.items()
                            if k not in ("drivers", "n"))
        print(f"  [{c:>2}] n={e['n']:>3}  {prof_str}")
        print(f"        drivers: {', '.join(e['drivers'])}")
    print(f"\nExports: {out}")
    print(f"        {GOLD / 'lap_embeddings.parquet'}")
    print(f"Metrics: {MODELS_DIR / 'phase5_metrics.json'}")
    print(f"Total: {time.time()-t0:.0f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
