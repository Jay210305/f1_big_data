# Phase 7 — Integration, Backtest & Report (STOP)

- Date / author: 2026-08-31 / agent
- Status: [x] Go — PROJECT COMPLETE

## Files created

| File | Purpose |
|------|---------|
| `src/run_all.py` | End-to-end orchestrator: 6 stages with skip-if-cached logic (`--stage`, `--force`) — the executable runbook |
| `src/models/backtest.py` | Out-of-sample backtest: optimizer vs actual 2025 strategies at each driver's first pit window; paired MC with common random numbers; degradation params refit on 2021-24 (2025 held out); dry-race filter; ablations |
| `document/strategy_engine_report.tex` | Standalone LaTeX report (does not touch your `main.tex`) — abstract, architecture, methods, all metrics tables, limitations |
| `RUNBOOK.md` | Full runbook: stage table, commands, artifacts, wall-clocks, footprint, limitations |
| `data/models/phase7_backtest.json` | Backtest results + ablations |

## Steps to run (in order)

```powershell
.\.venv\Scripts\Activate.ps1

python -m src.run_all                 # verify chain (skips cached stages)
python -m src.models.backtest         # full backtest (~40s)
# LaTeX: compile document/strategy_engine_report.tex with your toolchain
```

## What's going on

**Backtest design (honest):** at each (2025 dry race, driver) decision point —
the lap where stint 1 actually ended (first real pit window) — the ACTUAL
remaining strategy (compounds + stint ends from raw lap data) and the
OPTIMIZER's best candidate are evaluated through the SAME Monte Carlo
simulator with common random numbers → paired Δ = mean(actual) −
mean(recommended). Degradation parameters are refit excluding 2025. Rain races
(wet-compound fraction > 2%: Australian, British, Belgian GPs) are excluded —
the dry model cannot score wet strategies. DNF/truncated strategies (actual
stints not reaching the race end) are skipped.

**Bugs found & fixed during the backtest build (documented for the record):**
1. σ blow-up: raw slope std (σ_MEDIUM = 8.05 s/lap from traffic-locked stints)
   + the β lower clip exploded E[β] ≈ 42× the fitted rate → phantom ±400-1100 s
   deltas. Fix: IQR-robust σ (capped [0.02, 0.5]) + physical β upper clip.
2. n_laps from the FILTERED labeled frame (short final stints dropped) — e.g.
   Bahrain read as 44 laps instead of 57 → all recs "ended" 13 laps early.
   Fix: race length from the raw lap table.
3. In-lap poisoning of the decision state (slow pace, false in_cliff_now).
   Fix: state references the last flying lap.
4. DNF artifact (actual strategy covering only 15/70 laps) → skip rule.
5. Cliff cost model too harsh (0.5 s/lap compounding forever) → one-time
   +1.5 s jump then +0.15 s/lap (3× base rate).

## Results (agent validation run 2026-08-31 — confirm with your own run)

- Commands run (copy/paste):
  - `python -m src.run_all` (all stages cached-skip ✓)
  - `python -m src.models.backtest` (38s)
- Results / output (313 decisions, 19 dry 2025 races, clean-air scenario):

  | Metric | Value |
  |--------|-------|
  | Decisions evaluated | 313 (85 skipped: DNF/short/wet) |
  | **Optimizer win rate** | **78.6%** |
  | Δ median (actual − recommended) | **+15.0 s** |
  | Δ mean / p10 / p90 | +2.1 / −76.2 / +37.9 s |
  | Avg stops: actual vs recommended | 1.66 vs 1.33 |

  Ablations (120-decision subsample): no-cliff changes 99/120 recommendations
  (the cliff model dominates strategy choice); no-pace-offsets 17/120;
  no-style-cluster 0/120 (correct — cluster offsets ≈ 0 in races).

- Metrics / numbers: `phase7_backtest.json` (summary + ablations + 200 decisions).
- Errors / blockers: none.
- Decisions made:
  - Backtest scope: dry races only, first-pit-window decisions, paired MC.
  - Report: standalone `document/strategy_engine_report.tex` (main.tex untouched).
  - RUNBOOK.md as the reproducibility deliverable.
- Known limitations (documented in report + runbook):
  1. Recommender and evaluator share the simulator (selection bias; true
     counterfactuals unknowable).
  2. Optimizer leans 1-stop vs the field (1.33 vs 1.66 stops).
  3. Pit loss circuit-constant (21 s); scenario multipliers are priors.
  4. Physical-target RUL R² ≈ 0.39 is the honest ceiling post-leakage-removal.
- Time spent / next step: Phase 7 COMPLETE. All 8 phases (0-7) done.
  PROJECT COMPLETE → optional future work: fit scenario multipliers from
  dirty-air-stratified slopes, per-circuit pit loss, wet-tyre model, safety-car
  windows in the MC.
