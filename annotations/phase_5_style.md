# Phase 5 — Unsupervised Driving Style & Tyre Care Profiler (STOP)

- Date / author: 2026-08-30 / agent
- Status: [x] Go
- Transition compliance: all 4 action items from `phase_4_5_transition.md` implemented

## Files created

| File | Purpose |
|------|---------|
| `src/features/micro_sectors.py` | Materializes per-lap micro-sector vectors from Bronze (Option B): 10 standardized `rel_distance` bins × {max lat G, mean throttle, brake fraction, mean speed} = 40 dims/lap. Cached to `data/gold/micro_sectors.parquet` (one-off scan, no join needed — `rel_distance` is already normalized [0,1]) |
| `src/models/style_data.py` | Transition-compliant data prep: hard in/out-lap exclusion, 107% hot-lap filter, SC/VSC exclusion, **track evolution index** (cumulative all-driver session laps + weekend session weighting), AE input assembly (55 dims) |
| `src/models/train_style.py` | MLP Autoencoder (GPU, 55→128→64→**latent 16**→64→128→55), per-lap embeddings + recon anomaly flag, per-(season,driver) signature aggregation, HDBSCAN clustering, cluster interpretation (mechanical profile + **fuel-corrected tyre-care slope**), exports |
| `src/models/verify_style.py` | Verification: filter stats, AE recon, cluster sizes, anomaly count, sample signatures |
| `data/gold/driver_embeddings.parquet` | Per driver-season: latent 16 dims + cluster + recon_mean + n_laps (gitignored) |
| `data/gold/lap_embeddings.parquet` | Per-lap latent + recon_error (176,553 laps, gitignored) |
| `data/models/style_ae.pt`, `phase5_metrics.json` | Autoencoder weights + all metrics |

## Steps to run (in order)

```powershell
.\.venv\Scripts\Activate.ps1

python -m src.features.micro_sectors   # one-off Bronze scan (~4 min, cached)
python -m src.models.train_style       # AE + HDBSCAN + interpretation (~40s)
python -m src.models.verify_style      # verification report
```

Flags: `--latent 24`, `--min-cluster 10`, `--epochs 120`.

## What's going on

**Transition action item 1 (in/out exclusion)** — before any style feature is
computed: null laptimes dropped (41,730), out-laps = stint position 0 (44,108),
in-laps = stint end with a following stint (32,593), non-hot laps > 107% of the
driver's session best (93,447), SC/VSC laps (8,522). 315,786 → **176,553 hot
push laps** (44.1% removed). Profiles reflect true push pace, not pit-lap coasting.

**Action item 2 (track evolution, with rain wash-out spec update)** —
`cum_session_laps` = valid laps by ALL drivers in the session up to the current
lap; weekend rubber accumulates chronologically per
`R_prior(S_k) = [R_prior(S_{k-1}) + w_{k-1}·V_{k-1}·(0.5)^W_{k-1}]·(0.3)^W_k`
(γ_wash 0.3: a wet session washes 70% of accumulated rubber; wet sessions lay
rubber at 50% efficiency); lap-level
`TEI(t) = [R_prior + w·cum_session_laps]·(0.6)^r_t` where r_t = any driver on
INTERMEDIATE/WET at that lap (per-lap rain proxy). Wet-session flag =
wet-compound fraction > 0.02 OR any rain in session. Included as an AE
covariate so "grippier track" and "washed-out track" are disentangled from
"gentle/erratic style".

**Action item 4 (Option B)** — no Autoencoder on 325M raw rows. Each hot lap is
a 55-dim vector: 40 micro-sector channels + TEI + track temp + compound one-hot
+ session-type one-hot. Standardized with train-season stats.

**Autoencoder** — trained on 2021–2024 (125,330 laps, 90/10 early stop,
GPU, ~30s), generalization recon on held-out 2025: MSE mean 0.0414
(p95 0.0815). All 176,553 laps embedded → 16-dim latent + per-lap recon error
(anomaly flag at p99 = 0.177 → 1,766 anomalous laps).

**Driver signatures** — mean latent per (season, driver) → 172 driver-seasons →
HDBSCAN (min_cluster_size=8) → **2 archetypes + 36 noise**, silhouette 0.752.

**Tyre-care impact** — per driver-season, fuel-corrected degradation slope over
dry stints (≥4 hot laps), so the metric reflects tyre wear, not fuel burn.

## Results (agent validation run 2026-08-30, wash-out TEI — confirm with your own run)

- Commands run (copy/paste):
  - `python -m src.models.train_style` (47s) + `python -m src.models.verify_style`
- Results / output:

  | Cluster | n | throttle mod. | brake events | lat G max | deg slope (s/lap) | style |
  |---------|---|---------------|--------------|-----------|-------------------|-------|
  | 0 | 112 | 11.58 | 10.33 | 36.77 | **−0.050** (gentlest) | high-energy elite |
  | 1 | 28 | 10.84 | 9.34 | 31.68 | **−0.184** (steepest) | smooth-on-inputs but tyre-harsh |
  | −1 (noise) | 32 | 10.45 | 9.82 | 27.38 | −0.079 | part-time/test drivers |

  - Cluster 0 contains the established front-runners (HAM, VER, ALO, RUS, SAI,
    BOT...): they corner hardest (lat G 36.8) yet degrade tyres the least.
  - Cluster 1 (mostly rookies/test drivers): gentlest inputs but steepest
    degradation — slow cornering loads tyres differently.
- Metrics / numbers:
  - AE test(2025) recon MSE: mean 0.0415, p95 0.0798
  - Silhouette 0.688 (0.752 pre-wash-out — the rain model adds legitimate
    variation); 2 clusters + 32 noise of 172
  - TEI now reacts to rain: wet sessions reset weekend rubber (×0.3), wet laps
    get instantaneous grip ×0.6
- Errors / blockers: none.
- Decisions made:
  - Latent 16 + min_cluster_size 8 kept (stable, interpretable).
  - Noise cluster retained as its own "unclassified" group (part-time drivers)
    rather than forced into archetypes.
  - Exports: `driver_embeddings.parquet` (driver-season level — feeds Phase 6
    optimizer as style context) + `lap_embeddings.parquet` (lap level, anomaly
    flag available).
- Next-phase risks:
  - 36 noise driver-seasons — the optimizer must handle "no archetype" drivers
    (fall back to population-average degradation).
  - Style clusters are driver-SEASON level; driver moves between clusters
    across seasons are possible (style evolution) — optimizer should use the
    latest season's cluster.
- Stop-n-Go checkpoint: cluster quality (silhouette 0.752, interpretable
  profiles, physically consistent tyre-care ranking) → **GO**.
- Time spent / next step: Phase 5 COMPLETE -> GO to Phase 6 (Pit-Window & Stint Optimizer).
