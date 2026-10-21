# Phase 8 — Feature Engineering & Correlation Corrections (STOP)

- Date / author: 2026-09-05 / Antigravity
- Status: [x] Go  [ ] No-Go  [ ] Rework
- Deliverable for: Correlation Map Audit & Physics/Mathematical Corrections ("Lo que limpiaría primero")

## Files created / modified

| File | Purpose |
|------|---------|
| `src/features/build_gold.py` | Added circular wind decomposition (sin, cos), calculated `telemetry_sample_rate_hz`, and updated `add_degradation()` to derive stint baselines strictly from clean flying racing laps (`pos >= 1`), masking out-laps (`pos == 0`) to delta = 0.0. |
| `src/features/FEATURE_DICTIONARY.md` | Updated Gold feature definitions to document `wind_dir_sin`, `wind_dir_cos`, `telemetry_sample_rate_hz`, and the flying-lap baseline degradation policy. |
| `src/models/rul_data.py` | Decimated overlapping rolling smoother windows; implemented out-lap robust fuel-corrected baseline and sector deltas; established `PRUNED_CORRELATIONS` in `EXCLUDED` to prune 11 categories of structural collinearities, spurious proxies, and formula duplicates. |
| `correlation_maps/DATA_DICTIONARY.md` | Updated data dictionary from 75 down to 50 well-conditioned numeric features, providing clear descriptions of the cleaned feature groups. |
| `correlation_maps/rul_features_corr.png` | Re-generated Pearson correlation heatmap free of artificial collinear identity blocks (r = +-1.0). |
| `correlation_maps/rul_target_corr.png` | Re-generated feature-vs-target correlation bar charts for `target_rul` and `cliff_within_3`. |
| `tests/test_rul_data.py` | Added unit tests verifying out-lap baseline protection (`test_out_lap_does_not_pollute_stint_baseline_delta`) and feature exclusion enforcement (`test_pruned_correlations_excluded_from_features`). |
| `annotations/phase_8_feature_corrections_walkthrough.md` | Complete walkthrough, mathematical rationale, empirical verification records, and before/after metrics. |

## Steps to run (in order)

```powershell
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Run unit test suite
python -m pytest tests/ -v

# 3. Rebuild Gold feature store with updated baseline and circular wind
python -m src.features.build_gold --skip-telemetry

# 4. Verify Gold tables and degradation curve sample
python -m src.features.verify_gold

# 5. Prepare RUL feature matrix and verify label distributions
python -m src.models.rul_data

# 6. Re-generate RUL Pearson correlation maps and feature-vs-target charts
python -m src.analysis.correlation_map --only rul

# 7. Retrain RUL and cliff models on clean features
python -m src.models.train_rul --epochs 40

# 8. Evaluate test metrics and verify stint prediction continuity
python -m src.models.eval_rul

# 9. Compute XGBoost gain importance on the cleaned feature set
python -m src.models.feature_importance --no-shap

# 10. Run strategy optimizer verification
python -m src.models.optimizer --season 2025 --gp "Bahrain Grand Prix" --driver VER --lap 12
```

## What's going on

An audit of the 75 numeric features in `correlation_maps/DATA_DICTIONARY.md` and `rul_features_corr.png` identified 11 methodological and mathematical flaws categorized across 3 priority levels in the table "Lo que limpiaría primero":

### 1. Stint Reference Baseline & Out-Lap Contamination (`stint_first_laptime` & deltas)
- **Problem**: Stint lap 0 is a pit exit or standing start (often 5–20s slower). In the initial implementation, `stint_first_laptime = pl.col("laptime").first().over(stint_keys)` used this out-lap as the reference baseline. As a result, flying laps produced large negative deltas (e.g., -5.474s for Max Verstappen in Bahrain 2025), and `cliff_flag = (laptime_delta_stint_start > 2.0)` failed to detect cliffs because pace never exceeded the out-lap by +2.0s.
- **Correction**: In `build_gold.py::add_degradation()` and `rul_data.py::_compute_rolling_features_raw()`, the baseline t_base is now computed strictly from the **first clean, flying green-flag racing lap** (`pos >= 1`, excluding standing starts, pit exits, and SC laps). For the out-lap itself (`pos == 0`), the delta is set to `0.000`s, completely eliminating negative degradation distortion.

### 2. Spurious Telemetry Duration Proxy (`n_samples`)
- **Problem**: Bronze telemetry is recorded at a nominal rate (delta_tau ~ 0.1 s or 10 Hz). Therefore, sample count N ~ f_s * t_lap is an affine proxy for lap duration, plus packet dropouts. It has zero physical relationship to tyre wear.
- **Correction**: Completely excluded `n_samples` from the model feature matrix X. Added `telemetry_sample_rate_hz = n_samples / laptime` to Gold tables strictly for data health monitoring.

### 3. Circular Wind Direction Discontinuity (`wind_dir_mean`)
- **Problem**: Wind direction is recorded in circular degrees theta in [0, 360). Taking an arithmetic average in SQL (`avg(wWD)`) commits the circular mean fallacy (e.g., average of 350 deg and 10 deg yields 180 deg South instead of 0 deg North). Moreover, 359 deg and 1 deg have a numerical difference of 358, invalidating linear models and Pearson correlation.
- **Correction**: Computed continuous circular coordinates on S^1:
  $$\text{wind\_dir\_sin} = \sin\left(\theta \cdot \frac{\pi}{180}\right), \quad \text{wind\_dir\_cos} = \cos\left(\theta \cdot \frac{\pi}{180}\right)$$
  using DuckDB's native trig functions in `build_gold.py`. Excluded raw `wind_dir_mean` from X.

### 4. Redundant Lap Pace Levels (`laptime` vs `fuel_corr`)
- **Problem**: `fuel_corr = laptime + 0.03*lap` is an exact linear combination of `laptime` and `lap`. Feeding both into tree and linear models introduces exact multicollinearity.
- **Correction**: Dropped raw `laptime` from X. Retained `fuel_corr` and its fuel-corrected degradation delta (`fuel_corr_delta_start`), which isolates tyre degradation kinetics from fuel burn-off.

### 5. Baseline + Delta Structural Collinearity (`stint_first_laptime` + deltas)
- **Problem**: Including both baseline t_0 and delta delta_t = t - t_0 forces the structural identity t = t_0 + delta_t, generating artificial correlation blocks with r ~ +-1.0.
- **Correction**: Excluded `stint_first_laptime` from X. Car and driver absolute pace levels are already captured by `driver` and `gp`; tyre degradation kinetics are relative.

### 6. Formula Redundancies & Cross-Sector Inconsistencies (`s1, s2, s3, sector_decay_*, s3_s1_gap`)
- **Problem**: s_2 - s_1 and s_3 - s_2 are dimensionally inconsistent across different tracks (a 25s sector compared to a 40s sector). Including raw sector times alongside inter-sector differences created formula duplicates.
- **Correction**: Excluded raw s1, s2, s3, s2 - s1, s3 - s2, and s3 - s1. Retained clean per-sector degradation from the flying baseline (`s1_delta_start`, `s2_delta_start`, `s3_delta_start`, and traction-limited `s3_delta_roll_5`).

### 7. Overlapping Rolling Smoother Decimation (`pace_roll_mean_3/5/8`, medians, EMA)
- **Problem**: 6 overlapping moving averages of `fuel_corr` had pairwise correlations > 0.98, inflating dimensionality with near-identical pace smoothers.
- **Correction**: Decimated the filter bank to 1 robust median (`pace_roll_med_3`), 1 smoothed mean (`pace_roll_mean_5`), 1 volatility (`pace_roll_std_5`), 1 min-max spread (`pace_roll_spread_5`), and 1 slope (`pace_slope_5`).

### 8. Gap-to-Best Window Pruning (`pace_above_best_roll3/5/8`)
- **Problem**: `pace_above_best` rolling windows across widths 3, 5, and 8 had mutual correlations > 0.95.
- **Correction**: Retained `pace_above_best_roll5` (smoothed degradation level relative to best) and `pace_above_best_slope5` (pace loss acceleration). Dropped raw `pace_above_best` and widths 3 and 8.

### 9. Collinear Environmental Temperatures (`mean, max, range`)
- **Problem**: range = max - mean is an exact linear combination, and T_max correlates with T_mean at r > 0.97.
- **Correction**: Retained `track_temp_mean`, `ambient_temp_mean`, and `track_temp_std` (thermal fluctuation). Excluded `track_temp_max`, `ambient_temp_max`, and `track_temp_range`.

### 10. Load Integrals vs Physical Tyre Energy Dissipation
- **Problem**: lat_load_integral = avg(|a_lat|) * lap_distance was a linear product with circuit length.
- **Correction**: Excluded `lat_load_integral` and `long_load_integral`. Retained true frictional work dissipation E_lat = mean(a_lat^2 * v) (`lat_energy_integral`, Grosch/Archard tyre wear) and peak cornering acceleration `lat_g_max`.

### 11. Circuit Identifier Leakage (`lap_distance`)
- **Problem**: Single-lap distance is a constant per Grand Prix layout (e.g. 5,412m vs 7,004m), allowing decision trees to use continuous distance splits as circuit memorization shortcuts.
- **Correction**: Excluded `lap_distance` from model inputs X. Circuit identity is cleanly handled by categorical `gp`.

## After you run it

1. Verify that `python -m src.features.verify_gold` prints positive progressive degradation deltas on flying laps (e.g. VER Bahrain 2025 stint 1).
2. Check `correlation_maps/rul_features_corr.png` to confirm that artificial r = 1.0 collinear blocks are eliminated and features show well-distributed correlations.
3. Review `data/models/feature_importance/gain_importance.md` to confirm that `fuel_corr_delta_start` and `pace_above_best_roll5` rank at the top of model gain share.

## Results (filled after running)

### Commands run (copy/paste):
```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
.venv\Scripts\python.exe -m src.features.build_gold --skip-telemetry
.venv\Scripts\python.exe -m src.features.verify_gold
.venv\Scripts\python.exe -m src.models.rul_data
.venv\Scripts\python.exe -m src.analysis.correlation_map --only rul
.venv\Scripts\python.exe -m src.models.train_rul --epochs 40
.venv\Scripts\python.exe -m src.models.eval_rul
.venv\Scripts\python.exe -m src.models.feature_importance --no-shap
.venv\Scripts\python.exe -m src.models.optimizer --season 2025 --gp "Bahrain Grand Prix" --driver VER --lap 12 --n-sims 100 --pit-loss 21
```

### Degradation Curve Verification (VER 2025 Bahrain Stint 1):
```text
 lap  life  laptime  laptime_delta_stint_start  cliff_flag  dirty_air_frac
   1     4  103.877                      0.000       False        0.775536  (Out-lap / grid start: masked)
   2     5   98.403                      0.000       False        0.019711  (Flying baseline established)
   3     6   98.475                      0.072       False        0.038926  (Clean progressive wear)
   4     7   98.549                      0.146       False        0.041667
   5     8   98.760                      0.357       False        0.059524
   6     9   98.593                      0.190       False        0.000000
   7    10   98.929                      0.526       False        0.000000
   8    11   99.385                      0.982       False        0.000000
   9    12   99.670                      1.267       False        0.000000
  10    13  103.293                      4.890        True        0.000000  (True Positive Cliff Detected!)
```

### Metrics / numbers:
- **Feature Count in RUL Matrix**: Pruned from 82 down to **55 features** (51 numeric + 4 categorical). Zero zero-gain features.
- **Top Feature Gains (`gain_importance.csv`)**:
  1. `fuel_corr_delta_start`: gain = **7756.3** (share = **7.59%**) — #1 predictor!
  2. `pace_above_best_roll5`: gain = **6430.9** (share = **6.30%**) — #2 predictor!
  3. `life`: gain = **6110.0** (share = **5.98%**)
  4. `compound`: gain = **5719.7** (share = **5.60%**)
  5. `ambient_temp_mean`: gain = **5372.0** (share = **5.26%**)
  6. `track_temp_mean`: gain = **4553.8** (share = **4.46%**)
  7. `pace_roll_med_3`: gain = **3750.1** (share = **3.67%**)
  8. `wind_dir_cos`: gain = **3142.8** (share = **3.08%**)
- **Test 2025 Performance**:
  - Cliff Classifier: ROC-AUC = **0.906**, F1 = **0.680** (threshold = 0.258).
  - XGBoost RUL: Test R^2 = **0.313** (0.359 symmetric reference), MAE = **5.202**, MAE_crit = **3.075**, NASA PHM = **32,774**.
  - LightGBM RUL: Test R^2 = **0.259**, MAE_crit = **2.670**, NASA PHM = **31,093**.
  - PyTorch LSTM: Test MAE_crit = **2.838**.
  - Piecewise Blend: Test MAE_crit = **3.038**.
- **Unit Tests**: **53/53 tests passed** in 1.57s.

### Errors / blockers:
- None. Downstream optimizer and evaluation scripts read metadata dynamically from `feature_meta.json` without breaking.

### Decisions made:
1. Retained `fuel_corr_delta_start` as the single primary degradation delta; excluded raw `laptime_delta_stint_start` to avoid redundant linear pairs.
2. Excluded `lat_load_integral` and `long_load_integral` in favor of E_lat = mean(a_lat^2 * v) (`lat_energy_integral`), aligning directly with Archard's abrasive wear physics.
3. Decimated overlapping moving-average windows (w=3, 5, 8) down to 5-lap filtered dynamics + robust 3-lap median.

### Next-phase risks:
- None. Model training, evaluation, optimizer demo, and unit tests all run cleanly and stably.

### Time spent / next step:
- Time spent: ~30 minutes.
- Next step: Ready for final user sign-off or integration into project documentation and thesis/report tables.
