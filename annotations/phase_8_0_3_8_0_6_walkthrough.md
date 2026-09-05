# Phase 8.0.3–8.0.4–8.0.6.7 — Pace-Offset De-contamination, Scenario Multiplier Fit, Plot Diagnostics (STOP)

- Date / author: 2026-09-04 / Antigravity
- Status: [x] Go
- Options chosen: Task 8.0.3 Option A (continuous regression vs TEI) · Task 8.0.4 Option A (fitted β(dirty_air_frac)) · Task 8.0.6.7 conditional plot activated.

## Files created

| File | Purpose |
|------|---------|
| `src/models/fit_degradation.py` (edited) | `_fit_offset()` regresses pace gap vs ΔTEI; `_fit_dirty_air()` + `_theil_sen()` fit β(dirty_air_frac); persists `pace_offset`, `pace_offset_tei_fit`, `dirty_air_fit`, `scenario_beta_mult`, `scenario_notes`. |
| `src/models/optimizer.py` (edited) | `load_scenarios()` consumes fitted `scenario_beta_mult`; `PACE_PEN` split out; `main()` uses `load_scenarios(params)`. |
| `src/eval/plots.py` (new) | Headless `Agg` plotting module: `scenario_multiplier_regression()` + `critical_zone_r2()`, CLI `--only`. |
| `annotations/phase_8_0_3_hard_offset.md` (new) | 8.0.3 deliverable annotation. |
| `annotations/phase_8_0_4_scenario_multipliers.md` (new) | 8.0.4 deliverable annotation. |
| `annotations/phase_8_0_6_plots.md` (new) | 8.0.6.7 deliverable annotation. |

## Steps to run (in order)

```powershell
# 1. Re-fit the degradation model (writes pace_offset, scenario_beta_mult, etc.)
python -m src.models.fit_degradation

# 2. Regenerate the two diagnostic figures (Task 8.0.6.7)
python -m src.eval.plots

# 3. (Optional) verify the optimizer consumes the fitted multipliers
python -m src.models.optimizer --season 2025 --gp "Bahrain Grand Prix" --driver VER --lap 12

# 4. Safety net
python -m pytest tests/ -q
```

## What's going on

The strategy engine had two arbitrary heuristics masking physical reality:

1. **HARD pace offset** was faked with a symmetric prior `off_h = abs(off_s)` because the *measured* HARD−MEDIUM gap (−0.38 s) claimed the harder compound was *faster* — a track-evolution artifact (HARD stints run late on a rubbered track). Option A removes this by regressing, per driver-session, the pace gap between two compounds against the **ΔTEI** difference of their stints and taking the intercept at equal rubbering: `gap = β0 + β1·ΔTEI`.

2. **Scenario degradation multipliers** (1.00/1.15/1.25) were hand-set. Option A fits the per-stint degradation slope β against `dirty_air_frac` (share of lap within 10 m of the car ahead) via quantile-bin medians + robust Theil–Sen, normalizes to clean air, anchors on observed P90/P99, and caps at 2.5×.

3. **Plot module** renders the fitted β(dirty_air_frac) curve + empirical bins (validates no overfit of sparse traffic) and the critical-zone R² bar chart (RUL ≤ 10 vs > 10) per model.

Key discovery: `dirty_air_frac` is extremely sparse — P90 ≈ 0.05, max ≈ 0.26. Naive quadratic extrapolation to arbitrary 0.5/0.8 anchors blew up to ×16/×45; anchoring to observed percentiles + ceiling fixed it.

## After you run it

- Confirm `data/models/degradation_model.json` now contains `pace_offset` (de-contaminated), `pace_offset_tei_fit`, `dirty_air_fit`, `scenario_beta_mult`, `scenario_notes`.
- Confirm `eval_plots/scenario_multiplier_regression.png` and `eval_plots/critical_zone_r2.png` exist (non-empty PNGs).
- Paste back the `[deg]` console block (offsets + multipliers) and the `pytest` summary.

## Results (filled after running)

- Commands run (copy/paste):
  - `python -m src.models.fit_degradation`
  - `python -m src.eval.plots`
  - `python -m pytest tests/ -q`

- Results / output (from user): `[deg]` prints the TEI-decontaminated offsets and dirty-air-fitted scenario multipliers.

- Metrics / numbers:
  - Pace offsets (SOFT/MEDIUM/HARD): **−0.1964 / 0.0 / +0.1496** (was −0.188 / 0.0 / +0.188 symmetric prior).
  - TEI fit slope ≈ **−0.001 s/TEI** (consistent across SOFT n=403, R²=0.138 and HARD n=1002, R²=0.284).
  - scenario_beta_mult: **clean_air 1.00 · drs_train 1.2923 · backmarker 1.8161** (was 1.00/1.15/1.25).
  - dirty_air fit: 3,053 stints, 8 bins, Theil–Sen R²≈0.66; anchors P90=0.023, P99=0.071.
  - `pytest`: **24 passed**.

- Errors / blockers: Naive absolute-TEI→0 extrapolation drifted SOFT to +0.86 s (fixed → ΔTEI); quadratic dirty-air fit blew up ×16/×45 (fixed → Theil–Sen + percentile anchors + 2.5× ceiling).

- Decisions made: Selected Option A for both 8.0.3 and 8.0.4; activated conditional 8.0.6.7.

- Next-phase risks: dirty_air_frac dirtier anchors (0.5/0.8) are unobservable in dry telemetry — multipliers are floor-capped estimates; future wet/traffic-rich seasons may justify revisiting the ceiling.

- Time spent / next step: proceed to 8.0.5 unit-test expansion and remaining 8.0.6 figures.
