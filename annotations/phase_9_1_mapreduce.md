# Phase 9.1 — Real jobs and independent equivalence (STOP)

- Date / author: 2026-10-21 / Codex
- Status: [x] Go  [ ] No-Go  [ ] Rework

## Files created

`src/mapreduce/jobs/{lap_aggregate,stint_stats,driver_profile}.py`,
`src/mapreduce/evidence.py`, `tests/test_phase9_parity.py`.

## Steps to run (in order)

```powershell
python -m src.mapreduce.cli all
python -m src.mapreduce.cli run --job lap_aggregate --no-combiner --output data/mapreduce/output/lap_aggregate_plain
python -m src.mapreduce.evidence --silver
python -m src.eval.plots --only mapreduce_shuffle
```

## What's going on

Telemetry maps bounded Bronze batches to sufficient statistics; the combiner
merges counts, means and centered moments. The reducer reconstructs Silver
statistics and attaches raw lap-time metadata. Stint slopes use the production
warm-up/SC estimator and the model's nearest-quantile IQR/clipping rules.
Independent references use Polars rather than calling the production reducers.

## After you run it

Inspect `data/mapreduce/parity.json`, `output/*/metrics.json` and the full stint
trace. `data/models/phase9_tables.json` is a compact report snapshot.

## Results (filled after running)

| Experiment | Measured result |
|---|---|
| Bronze 2025 → Silver | 69,459 matched laps; zero differing rows at 1e-6 tolerance |
| Largest Silver numeric difference | 2.7000623958883807e-13 |
| Shuffle pairs, plain → combined | 103,244 → 69,459 |
| Serialized KV bytes, plain → combined | 96,823,560 → 65,750,633 (32.09% fewer) |
| Eligible stints, current estimator | 4,628 = 4,274 dry + 354 wet |
| MR vs independent Polars, maximum beta error | 3.552713678800501e-15 |
| Saved dry model beta/sigma | All three compounds pass 1e-6 tolerance |
| Driver/session profile | 70 groups; pace/dirty-air error below 6e-14 |
| Full trace / injected retry | Independent verification passes; final part bytes identical |

- The cached degradation model contains dry compounds only; no wet parameters
  were fabricated. Current MR/Polars/Mongo agreement includes wet stints.
- Gold's 44,298 raw stint slopes use a different target/filter: among 4,628
  matched eligible stints, mean absolute difference is 3.7098 and maximum 363.0841.
  This is a documented diagnostic comparison, not a false equivalence claim.
- `driver_style` has 178 season/driver aggregates over a different population;
  exact profile parity is checked on the shared labeled driver/session population.
- Identity/hash comparison artifacts are `stint_stats_identity` and
  `stint_stats_hash8`; identity sends all 4,981 keys to one partition (5.815 s).
  Hash across eight partitions sends 603–652 keys per partition (5.532 s).
  These small local timings are illustrative, not a multi-host scaling claim.
- Review regression: null pace requires pairwise covariance divided by marginal
  life variance, with separate sample-count factors, exactly as Polars computes.
- Next step: Mongo ingestion, indexing and third-engine comparison.
