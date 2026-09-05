# Phase 8.2.1 — Empirical SC/VSC Hazard in the Optimizer

- Date: 2026-09-05
- Status: [x] Complete — Option A (probabilistic Monte Carlo)

## Implementation

`src/models/fit_degradation.py` now derives `sc_hazard` from corrected race-control data and race lap counts for 2021–2025. For each circuit (`gp`), the fit records the conditional onset hazard by lap:

- each race contributes exposure until its first SC/VSC-affected lap;
- the first affected lap is the onset event;
- contiguous affected laps provide the empirical SC/VSC window duration distribution.

The result is persisted in `data/models/degradation_model.json` as `sc_hazard`, including `lap_hazard`, `circuit_prior`, `duration_laps`, and fit metadata. The generated artifact contains 114 race exposures and 99 first-onset events.

`src/models/optimizer.py` adds:

- `_sample_sc_windows()` for sequential conditional-hazard sampling;
- circuit-specific hazard selection through `load_scenarios(..., circuit=...)`;
- common sampled windows reused for every candidate strategy in a scenario;
- discounted pit loss inside a sampled window, defaulting to `PIT_LOSS / 3`;
- `--sc-hazard` and `--pit-loss-sc` CLI arguments.

`src/models/backtest.py` supports `--sc-hazard` and passes identical sampled windows to actual and recommended strategies for paired comparisons.

## Validation

Commands:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
.\.venv\Scripts\python.exe -m src.models.fit_degradation
.\.venv\Scripts\python.exe -m src.models.optimizer --season 2025 --gp "Bahrain Grand Prix" --driver VER --lap 12 --n-sims 20 --sc-hazard
.\.venv\Scripts\python.exe -m src.models.backtest --sc-hazard
```

Results:

- Full test suite: `45 passed`; focused optimizer coverage: `8 passed`.
- End-to-end optimizer demo completed and produced the `sc_hazard` strategy scenario.
- 313 backtest decisions evaluated, 85 skipped.
- SC/VSC hazard mode: win rate `78.9%`, mean delta `+2.3 s`, median delta `+14.4 s`, p10/p90 `-75.2 / +36.8 s`.
- Average stops: actual `1.66`, recommended `1.42`.

The comparison remains subject to the existing selection-bias limitation because recommendations and evaluation use the same simulator.
