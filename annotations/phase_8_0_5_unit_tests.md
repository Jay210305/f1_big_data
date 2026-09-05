# Phase 8.0.5 — Unit Tests for Pure and Leakage-Critical Functions (STOP)

- Date / author: 2026-09-04 / GitHub Copilot
- Status: [ ] Go  [ ] No-Go  [x] Rework

## Files created

| File | Purpose |
|------|---------|
| `tests/conftest.py` | Shared synthetic Polars fixtures with no dependency on the 57 GB dataset. |
| `tests/test_etl.py` | Tests telemetry normalization, dtype handling, truncation, and Silver aggregations. |
| `tests/test_rul_data.py` | Tests fuel correction, rolling-label preparation, cliff labels, and RUL clipping. |
| `tests/test_no_leakage.py` | Asserts that cliff-derived columns cannot enter model features. |
| `tests/test_style_data.py` | Tests TEI accumulation, rain wash-out, and the wet-lap factor. |
| `requirements.txt` | Consolidated runtime and development dependencies, including `pytest>=8.0`. |

## Steps to run (in order)

```powershell
# From the repository root.
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -v tests/
```

The test command must be run with the project virtual environment. Calling the
system `pytest` may select a Python installation that does not have Polars.

## What's going on

This phase adds a small regression safety net around the pure functions that
transform telemetry and build model inputs. The tests use synthetic in-memory
Polars frames, so they do not scan the raw JSON tree or the large `data/`
artifacts.

The initial failures were traced to test expectations and fixtures rather than
the production contracts:

- `throttle_active_frac` counts samples where `throttle > 0`.
- `dirty_air_frac` counts samples where `DistanceToDriverAhead < 10`.
- Race and Sprint fuel correction is `laptime + 0.03 * lap`.
- `compute_rolling_features` requires the sector columns `s1`, `s2`, and `s3`.
- `INSTANT_WET` applies to the accumulated TEI base for the current lap.
- The leakage test should assert the selector result, without a tautological
  set identity assertion.

`requirements-dev.txt` was removed after its `pytest` dependency was merged
into the single `requirements.txt` file.

## After you run it

Paste the complete output of `python -m pytest -v tests/` into the Results
section below. If any test fails, include the traceback and do not mark the
phase Go until the failing contract is understood.

## Results (filled after running)

- Commands run (copy/paste): Pending user execution.
- Results / output (from user): Pending.
- Metrics / numbers: Not available yet; no test command was run while preparing this annotation.
- Errors / blockers: Validation is pending in the project `.venv`.
- Decisions made: Consolidate dependencies in `requirements.txt`; keep production formulas unchanged and align tests with their documented contracts.
- Next-phase risks: The phase cannot be marked Go until the full suite passes in the user's environment. `requirements.txt` still installs the project ML/data stack, so dependency installation may require time and a compatible Python environment.
- Time spent / next step: Run the commands above, paste the output here, then update this annotation and mark Go, No-Go, or Rework.
