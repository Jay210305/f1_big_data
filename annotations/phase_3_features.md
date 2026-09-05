# Phase 3 — Feature Engineering / Gold Feature Store (STOP)

- Date / author: 2026-08-28 / agent
- Status: [ ] Go  [ ] No-Go  [ ] Rework

## Files created

| File | Purpose |
|------|---------|
| `src/features/build_gold.py` | Builds the 3 Gold tables + manifest from Bronze/Silver (DuckDB for the heavy Bronze scan, Polars for degradation/style) |
| `src/features/verify_gold.py` | Verifies Gold: row counts, per-season, compound mix, null check, degradation + style samples |
| `src/features/FEATURE_DICTIONARY.md` | Human-readable feature dictionary (tracked in git) with leakage policy |
| `data/gold/feature_dictionary.json` | Machine-readable manifest (gitignored, local) |

## Steps to run (in order)

```powershell
.\.venv\Scripts\Activate.ps1

# Full run (all 5 years) — the Bronze scan is the slow part (~80s/year)
python -m src.features.build_gold
python -m src.features.verify_gold
```

Flags:
- `--years 2025`            build one season (quick test)
- `--skip-telemetry`        reuse existing `data/gold/telemetry_stats.parquet` (skip the heavy Bronze scan)

Expected wall-clock: ~6–8 min for the full run (Bronze groupby over 325M rows).

## What's going on

`build_gold.py` runs 5 steps:

1. **Rich per-lap telemetry stats (DuckDB, per season)** — scans Bronze
   (`data/bronze/telemetry/season=YYYY/**/*.parquet`) with a windowed subquery
   that counts **throttle rising-edge modulations** and **brake application
   events**, then groups by lap to produce: speed stats, throttle mean/std +
   active fraction + modulation count, brake fraction + event count + intensity,
   DRS fraction, gear/rpm stats, lateral/longitudinal/vertical G (mean/std/max),
   **dirty-air fraction** (<10 m), mean + p10 distance-to-car-ahead, lap distance,
   and **lat/long load integrals** (`avg|G|·distance`). One row per lap.
   → `data/gold/telemetry_stats.parquet`

2. **Environmental features** — aggregates `weather` per session:
   ambient/track temp mean/max/std/range, rain flag, wind speed/dir.
   → `data/gold/environmental.parquet`

3. **Master `lap_features`** — reads silver `stint_features` (stint/life/compound/
   laptime/sectors), adds **degradation kinetics** with Polars windows partitioned
   by stint (ordered by lap, using only current+past laps → **no leakage**):
   `laptime_delta_stint_start`, `laptime_delta_prev`, `sector_decay_21/32`,
   `cliff_flag` (>2 s vs stint start). Then left-joins telemetry_stats + environmental.
   → `data/gold/lap_features.parquet` (~317k laps × 58 cols)

4. **`driver_style`** — groups laps by `(season, driver)` → mean & std of all
   style/mechanical-stress features. ~100 driver-season rows. → `data/gold/driver_style.parquet`

5. **`stint_degradation`** — groups by stint: compound, stint_length, max_life,
   pace_start/end, pace_decay_end, had_cliff, mean_lap_delta, **degradation_slope**
   `(pace_end−pace_start)/(stint_length−1)`, stint env + load. ~50k stints.
   → `data/gold/stint_degradation.parquet`

**Leakage policy** (in `FEATURE_DICTIONARY.md`): for RUL prediction (Model 1),
use only `lap_features` OK columns; **exclude** `stint_length / max_life /
pace_end / degradation_slope` (those are stint outcomes / the label). `life`
(tyre age so far) is allowed. `stint_degradation` is for the optimizer/analysis,
not per-lap RUL features.

I validated on 2025: 69,459 laps × 58 cols, degradation curve rises correctly
(δ −5.47 → −0.58 over a stint), compounds distribute sensibly, ~14% null
laptimes (in/out laps, expected), 0 null speed_mean / track_temp_mean.

## After you run it

Paste into **Results** below:
- The `Phase 3 COMPLETE` summary (3 row counts + dims).
- The `verify_gold.py` blocks: per-season counts, compound distribution, the
  null check, the degradation-curve sample, and the driver-style sample.
- Total wall-clock time.

Key numbers I'll check:
- lap_features ≈ 317,264 laps × 58 cols (matches silver stint count).
- driver_style ≈ 100–180 driver-seasons.
- stint_degradation ≈ 44k–50k stints.
- null laptime ≈ 10–15% (in/out laps) — expected; null speed_mean should be ~0.

## Results (filled after running)

- Commands run (copy/paste):
- Results / output (from user):
- Metrics / numbers:
- Errors / blockers:
- Decisions made:
- Next-phase risks:
- Time spent / next step:
