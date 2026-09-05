# Phase 2 — Streaming ETL → Bronze/Silver Lakehouse (STOP)

- Date / author: 2026-08-27 / agent
- Status: [x] Go  [ ] No-Go  [ ] Rework

## Files created

| File | Purpose |
|------|---------|
| `src/ingest/config.py` | ETL configuration (years, session filter, paths, compression, workers) |
| `src/ingest/etl.py` | Main ETL: JSON → Bronze telemetry + Silver stint_features + race_events (multiprocessing, one worker per session, memory-bounded) |
| `src/ingest/duckdb_layer.py` | Opens a DuckDB connection with views (`telemetry`, `stint_features`, `weather`, `corners`, `rcm`) over the Hive-partitioned Parquet |
| `src/ingest/verify_lakehouse.py` | Post-ingest check: row counts per view & per season, on-disk footprint, sample query |

## Steps to run (in order)

I already validated on one session (2025 Belgian GP Sprint Qualifying): 365,531 bronze rows, 185 silver rows, weather/corners/rcm all written, 0 errors, DuckDB read confirmed. Now you run the real thing.

```powershell
.\.venv\Scripts\Activate.ps1

# --- Option A: quick test first (1 season, Race only) ---
python -m src.ingest.etl --years 2025 --sessions "Race" --workers 8
python -m src.ingest.verify_lakehouse

# --- Option B: FULL run (all 5 years, all sessions) ---
python -m src.ingest.etl --workers 8
python -m src.ingest.verify_lakehouse
```

Flags:
- `--years 2021 2023 2025`        restrict seasons
- `--sessions "Race,Sprint Qualifying"`   comma-separated session names (names with spaces OK)
- `--no-bronze`                   skip raw telemetry → silver only (much faster, ~no Bronze)
- `--max-sessions 3`              cap sessions (sanity check)
- `--workers N`                   parallel session workers (default = min(cpu, 8))

## What's going on

`etl.py` walks `year/Grand Prix/Session/` and schedules **one session per worker**
(spawn pool). Each `process_session`:

1. **Bronze `telemetry`** — for every `{lap}_tel.json`, unwraps the `{"tel": {...}}`
   envelope, builds a long-form DataFrame (one row per telemetry sample), normalizes
   columns to the canonical schema (`schema.py` dtypes, `"None"`→null via
   `cast(..., strict=False)`), adds `season/gp/session/driver/lap`, and writes one
   ZSTD Parquet file per session at
   `data/bronze/telemetry/season=YYYY/<gp>__<session>.parquet`.
2. **Silver `stint_features`** — from each lap's telemetry it computes per-lap
   aggregates (speed mean/max/min/std, throttle mean + active fraction, brake
   fraction, DRS fraction, lateral/longitudinal/vertical G means, **dirty-air
   fraction** = share of samples within 10 m of the car ahead, lap distance), then
   **left-joins** `session_laptimes.json` (or per-driver `laptimes.json` fallback) to
   attach `compound / stint / life / laptime / s1 / s2 / s3`. One row per lap.
3. **Silver `race_events`** — converts `weather.json`, `corners.json`, `rcm.json`
   into typed Parquet under `data/silver/race_events/{weather,corners,rcm}/`.

Memory stays bounded: only one session is ever in RAM per worker (a race ≈ 600k
rows ≈ ~100 MB). Output is Hive-partitioned by `season=YYYY` so DuckDB infers the
season column from the path. `duckdb_layer.open_lakehouse()` registers the 5 views;
`verify_lakehouse.py` reads them back and prints counts + footprint + a sample.

Known design choices:
- Raw telemetry is kept at full sample rate (no downsampling yet) — Bronze is the
  expensive part (~150–220M rows expected, ~10–12 GB ZSTD). Use `--no-bronze` for a
  fast silver-only pass if you want to reach modeling quickly.
- `dataKey` (file-level scalar) is dropped from Bronze to keep a uniform schema.

## After you run it

Paste into **Results** below:
- The ETL final summary block (`DONE in ...` + the counts: sessions, laps, bronze,
  silver, weather, corners, rcm, errors).
- The `verify_lakehouse.py` output (row counts per view + per season, the footprint
  lines, and the sample query).
- Total wall-clock time, and any `[ERR]`/`errors` lines.

Key numbers I'll check against the inventory:
- Telemetry rows: expect ~150–220M total.
- Stint features: expect ~300–500k laps total.
- On-disk Bronze: expect ~10–12 GB (well under the 513 GB budget).
- `errors` should be a handful at most.

If errors > ~1% of sessions, paste a few `[ERR]` lines and I'll patch the parser.

## Results (filled after running)

- Commands run (copy/paste):
  - `python -m src.ingest.etl --workers 8` (full ingest, 585 sessions)
  - `python -m src.ingest.verify_lakehouse`
  - (fix pass) `python -m src.ingest.etl --fix-failed --workers 6` + re-verify
- Results / output (from user):
  - Full run: DONE in 675s | sessions=585 laps=317,263 bronze=325,541,017 silver=310,943
    weather=59,599 corners=9,597 rcm=25,597 **errors=6**
  - 6 errors diagnosed & fixed (see below); fix pass: 6 sessions, 0 errors
- Metrics / numbers (FINAL, post-fix):
  - telemetry       = 325,541,022 rows  (16.69 GB)
  - stint_features  = 317,264 laps      (+6,321 recovered, 0.03 GB)
  - weather         = 59,599            corners = 9,597   rcm = 25,597
  - TOTAL data/     = 16.73 GB
  - Per-season telemetry: 2021=66.2M 2022=60.4M 2023=60.2M 2024=68.8M 2025=69.8M
  - Wall-clock full run: ~11 min (8 workers)
- Errors / blockers: 6 errors, all resolved:
  1. Pre-Season Testing x5 — `_load_laptimes` from_dicts schema inference (first 100 rows)
     hit `"None"` string later -> ComputeError. Fix: `infer_schema_length=len(rows)`.
  2. 2023 São Paulo Sprint Shootout, 1 lap (SAR/6_tel.json) — misaligned channel
     lengths (acc_y=9 vs time=5) -> ShapeError. Fix: truncate channels to min length.
  3. Defensive min-length alignment added to `_events_df` + session_laptimes path.
  4. Added `--fix-failed` flag to re-ingest the 6 sessions.
- Decisions made:
  - Misaligned telemetry channels are truncated to the shortest aligned subset
    (preserves per-sample alignment integrity for feature engineering).
  - Pre-Season Testing sessions retained (they have valid telemetry + laptimes).
- Next-phase risks:
  - Bronze is full sample-rate (~325M rows, 16.7 GB); Phase 3 features should
    prefer the silver per-lap aggregates + targeted Bronze scans to stay fast.
  - `dataKey` (file-level scalar) was dropped from Bronze — fine for features.
- Time spent / next step: Phase 2 COMPLETE -> GO to Phase 3 (Feature Engineering).
