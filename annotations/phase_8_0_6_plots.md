# Phase 8.0.6.7 — Conditional Diagnostics ⟶ Scenario Multiplier Regression & Critical-Zone R² Charts

- Date / author: 2026-09-04 / Antigravity
- Status: [x] Complete — conditional visualization activated (headless `src/eval/plots.py`).
- Deliverable for: Task 8.0.6.7 (conditional) in `planning/tier_0_cheap_wins.md`

---

## Figures Generated (`eval_plots/`)

### 1. `scenario_multiplier_regression.png`
Visualizes the fitted `β(dirty_air_frac)` curve from Task 8.0.4 (Option A) overlaid with the empirical stint-bin medians and the clean/drs_train/backmarker anchors (observed P0/P90/P99). Purpose: **validate that the spline does not overfit sparse traffic extremes**, and confirm the resulting `scenario_beta_mult` are grounded in telemetry rather than hand-set scalars.

### 2. `critical_zone_r2.png`
Segmented remaining-useful-life $R^2$ bar chart — `R2_crit` ($RUL \le 10$) vs `R2_noncrit` ($RUL > 10$) per model (naive const/compound, LightGBM, XGBoost asym/sym, LSTM, piecewise blend), read from `data/models/phase4_metrics.json`. Purpose: **ensure degradation predictions remain robust right at the cliff threshold**.

## Implementation

- `src/eval/plots.py`: headless (`matplotlib.use("Agg")`) module with CLI `python -m src.eval.plots` (full run) and `--only <name>`; figures written to `eval_plots/` (already gitignored via `eval_plots/`).
- Critical-zone bars source their values from `phase4_metrics.json`'s `R2_crit`/`R2_noncrit` fields (added in Task 8.0.2.4), so the diagnostic stays in sync with the segmentation already validated in the optimizer blending work.

## Verification

```
python -m src.eval.plots
python -m src.eval.plots --only scenario_multiplier_regression
python -m src.eval.plots --only critical_zone_r2
```
All three paths render without error; outputs confirmed non-empty PNGs (66–110 kB).

## Observation (segment R² honesty)

Across engines the critical-zone $R^2$ is negative (best ≈ −0.12 LSTM / −0.19 XGBoost), i.e. the RUL regression has not yet learned to *rank* cliff-proximity laps within 10 laps — consistent with the 8.0.2 findings that only MAE/absolute error is trustworthy there. The bar chart surfaces this transparently rather than hiding it.
