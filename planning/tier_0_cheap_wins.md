# Tier 0 — Cheap Wins (Start Here)

## Goal Summary
Resolve quick, high-payoff gaps: fix report/code blending discrepancy, interrogate LSTM negative R², de-contaminate HARD pace offset with track evolution index, validate scenario multipliers with empirical dirty air data, build unit tests for critical paths, and implement an automated plotting module.

---

### Task 8.0.1: Fix the RUL model-blending claim (report ↔ code mismatch)
- [x] **8.0.1.1**: Audit `document/strategy_engine_report.tex` §4 and `document/main.tex` §IV against `src/models/optimizer.py::load_phase4()`.
- [x] **8.0.1.2**: Evaluate selection between Option A (update report text to document XGBoost as production & LSTM as research benchmark), Option B (implement piecewise blending $w(rul_{xgb})$ favoring LSTM in critical zone $\le 10$ laps in `optimizer.py`), and Option C (train meta-learner on validation split).
- [x] **8.0.1.3**: If Option B or C, update `src/models/optimizer.py::predict_laps()` and `load_phase4()` to load LSTM weights and sequence building pipeline.
- [x] **8.0.1.4**: If Option C, update `src/models/eval_rul.py` and produce `models/rul_stack.*`. *(B selected: piecewise blend directly integrated into optimizer)*.
- [x] **8.0.1.5**: Update `RUNBOOK.md` stage 4 table if new stack artifacts are produced.
- [x] **8.0.1.6**: Update LaTeX document files (`document/strategy_engine_report.tex`, `document/main.tex`) and abstract to align with chosen option.
- [x] **8.0.1.7**: Complete deliverable annotation `annotations/phase_8_0_1_blend_fix.md`.

---

### Task 8.0.2: Diagnose the LSTM's negative test R² before trusting it anywhere
- [x] **8.0.2.1**: Audit `src/models/train_rul.py` padding and masking in `pad_batch()` (`Y` filled with `NaN`, `mask = mb * torch.isfinite(yb)`).
- [x] **8.0.2.2**: Verify test-time sequence flattening in `train_lstm()` (`Yte[i, :L]` / `pt[i, :L]`) and assert `len(yt) == sum(unpadded_laps)`.
- [x] **8.0.2.3**: Inspect `OneCycleLR` schedule and log learning rate at `best_epoch` under early stopping.
- [x] **8.0.2.4**: Add segmented metrics `R2_crit` ($RUL \le 10$) and `R2_noncrit` ($RUL > 10$) to `reg_metrics()` in `src/models/train_rul.py`.
- [x] **8.0.2.5**: Execute `python -m src.models.train_rul` and record before/after validation/test metrics.
- [x] **8.0.2.6**: Produce diagnostic writeup and decision verdict in `annotations/phase_8_0_2_lstm_diagnostics.md`.

---

### Task 8.0.3: De-contaminate the HARD compound pace offset using `track_evolution_index`
- [ ] **8.0.3.1**: Inspect `fit_degradation.py`'s current symmetric prior fallback (`off_h = abs(off_s)`).
- [ ] **8.0.3.2**: Join `track_evolution_index` (from `style_data.py::_track_evolution`) onto the pairwise SOFT/MEDIUM/HARD stint comparison in `fit_degradation.py`.
- [ ] **8.0.3.3**: Select and implement de-contamination:
  - Option A: Regress measured pace gap against TEI at stint time and extrapolate to $TEI pprox 0$.
  - Option B: Discretize stints into TEI terciles and create a TEI-bucket-dependent pace offset table.
- [ ] **8.0.3.4**: Persist fitted parameters into `degradation_model.json` and update `params["pace_offset_notes"]`.
- [ ] **8.0.3.5**: If Option B, update `src/models/optimizer.py::simulate()` to consume bucketed offsets.
- [ ] **8.0.3.6**: Create deliverable annotation `annotations/phase_8_0_3_hard_offset.md` quantifying offset changes.

---

### Task 8.0.4: Replace hand-set scenario multipliers with a fitted `dirty_air_frac` relationship
- [ ] **8.0.4.1**: Query `dirty_air_frac` distribution and degradation slope $eta$ across filtered dry stints in `src/models/fit_degradation.py`.
- [ ] **8.0.4.2**: Choose and execute fitting path:
  - Option A: Fit $eta(dirty\_air\_frac)$ via binned regression or spline, and derive multipliers for `drs_train` and `backmarker`.
  - Option B: Compute and tabulate empirical mean $eta$ per `dirty_air_frac` bin alongside existing 1.00/1.15/1.25 constants for validation.
- [ ] **8.0.4.3**: If Option A, serialize fitted multipliers into `degradation_model.json` and update `src/models/optimizer.py` to read them dynamically.
- [ ] **8.0.4.4**: Document binned empirical comparison and findings in `annotations/phase_8_0_4_scenario_multipliers.md`.

---

### Task 8.0.5: Unit tests for pure/leakage-critical functions (safety net)
- [ ] **8.0.5.1**: Add `pytest` to `requirements.txt` and set up `tests/` directory structure.
- [ ] **8.0.5.2**: Create `tests/conftest.py` with synthetic Polars DataFrame fixtures (no dependency on 57 GB dataset).
- [ ] **8.0.5.3**: Implement `tests/test_etl.py` testing `_normalize_telemetry` array truncation/dtypes and `_silver_exprs` aggregations.
- [ ] **8.0.5.4**: Implement `tests/test_rul_data.py` testing `add_fuel_corr` session conditions and `compute_labels` edge cases (cliff detection, RUL_MAX clipping).
- [ ] **8.0.5.5**: Implement `tests/test_no_leakage.py::test_no_cliff_derived_features_in_X()` asserting `CLIFF_DERIVED & set(feats) == set()`.
- [ ] **8.0.5.6**: Implement `tests/test_style_data.py` testing `_track_evolution` reset on rain and `INSTANT_WET` factors.
- [ ] **8.0.5.7**: Run `pytest -v tests/` and record full test suite execution in `annotations/phase_8_0_5_unit_tests.md`.

---

### Task 8.0.6: Plotting module (figures for report & diagnostics)
- [ ] **8.0.6.1**: Create `src/eval/plots.py` with `matplotlib` / `seaborn` using headless `Agg` backend.
- [ ] **8.0.6.2**: Implement CLI interface supporting full run (`python -m src.eval.plots`) and single figure generation (`--only <name>`).
- [ ] **8.0.6.3**: Implement Figure 1: Predicted vs actual `target_rul` scatter on test 2025 split with $y=x$ reference line, compound coloring.
- [ ] **8.0.6.4**: Implement Figure 2: Fuel-corrected pace vs tyre life degradation curves overlaid with fitted compound $eta$.
- [ ] **8.0.6.5**: Implement Figure 3: 2D UMAP/t-SNE driver embeddings projection colored by HDBSCAN cluster labels.
- [ ] **8.0.6.6**: Implement Figure 4: Backtest strategy delta distribution histogram (313 paired decisions) with mean/median markers.
- [ ] **8.0.6.7**: (Optional/conditional) Implement critical-zone R² bar chart and scenario multiplier regression plots.
- [ ] **8.0.6.8**: Configure output directory `eval_plots/` and align gitignore configuration.
- [ ] **8.0.6.9**: Create deliverable annotation `annotations/phase_8_0_6_plots.md` linking generated figures.
