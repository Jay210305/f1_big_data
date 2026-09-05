# Phase 1 — Data Exploration & Schema Validation (STOP)

- Date / author: 2026-08-27 / agent
- Status: [ ] Go  [ ] No-Go  [ ] Rework

## Files created

| File | Purpose |
|------|---------|
| `src/ingest/explore.py` | Walks 2021–2025 (dirs only) + bounded sample of file contents; reports structure, field presence, types, nulls |
| `src/ingest/schema.py` | Canonical schema manifest + normalizers (`to_float/to_int/to_str`, dtype map) used by Phase 2 ETL |

## Steps to run (in order)

```powershell
# venv already active from Phase 0 (re-activate if needed)
.\.venv\Scripts\Activate.ps1

python src/ingest/explore.py
```

The script writes its full report to the console and a machine-readable summary to
`data/bronze/phase1_schema_report.json`.

## What's going on

`explore.py` does **two passes, both bounded**:

1. **Structure pass (no file contents opened):** lists every `year/Grand Prix/Session/`
   directory and counts files from directory names only. This gives the per-year
   GP list, session types, driver-dir count, and total telemetry/laptimes file
   counts (expect ~367k telemetry files total) without touching 57 GB.

2. **Schema sample pass (bounded reads):** opens a *small* number of files per
   season — up to 2 GPs × 2 drivers × 2 laps per session type for telemetry, a few
   `laptimes.json`, and up to 12 of each session-level JSON (`weather`, `corners`,
   `rcm`, `session_laptimes`, `drivers`). For each file it records:
   - which fields exist per year (the `[YYYY.]` bitmask),
   - the value type + an example value,
   - how many `"None"` string values appear (the dataset uses the literal string
     `"None"` for nulls, not JSON `null`).

It also confirms the telemetry envelope: telemetry is nested under a `"tel"` key
(plus a file-level scalar `dataKey`), while `laptimes`/`weather`/`corners`/`rcm`
are flat dicts-of-lists and `drivers.json` wraps objects under `"drivers"`.

`schema.py` is the *provisional* schema manifest: the column lists per entity, a
dtype map, and `to_float`/`to_int`/`to_str` normalizers that treat `"None"`/`""`
as null. Phase 2 (ETL) will import these. **I finalize `schema.py` after your
run, so if `explore.py` shows a field I got wrong, it gets corrected here.**

## After you run it

Paste into the **Results** section below:

- The `File counts` block (total telemetry/laptimes per year + TOTAL).
- The `FIELD PRESENCE / TYPES / NULLS` block (or at least the `[telemetry]` and
  `[session_laptimes]` blocks).
- The `TELEMETRY ENVELOPE` block.
- Any `[ERR] ...` lines (corrupt/unreadable JSON).

Then answer the scope question in **Decisions made**.

## Results (filled after running)

- Commands run (copy/paste):
- Results / output (from user):
- Metrics / numbers:
- Errors / blockers:
- Decisions made: (scope: **all sessions** vs **Race-only** first pass?)
- Next-phase risks:
- Time spent / next step:
