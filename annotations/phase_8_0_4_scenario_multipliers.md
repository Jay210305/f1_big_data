# Phase 8.0.4 — Scenario Multipliers via Empirically Fitted β(dirty_air_frac)

- Date / author: 2026-09-04 / Antigravity
- Status: [x] Complete — Option A (empirically fitted β(dirty_air_frac) relationship) implemented, serialized, and consumed dynamically by the optimizer.
- Deliverable for: Task 8.0.4 in `planning/tier_0_cheap_wins.md`

---

## 1. Problem

The strategy optimizer (`src/models/optimizer.py`) hard-coded scenario degradation multipliers `{clean_air: 1.00, drs_train: 1.15, backmarker: 1.25}`. These are hand-set scalars with no grounding in the telemetry. Modern ground-effect cars lose front-axle downforce in wake turbulence, driving micro-sliding and thermal surface degradation — a non-linear penalty that static constants cannot capture.

## 2. Option Selection

| Option | Approach | Verdict |
|---|---|---|
| **A — Fit β(dirty_air_frac), derive multipliers** | Binned regression/spline over degradation slope vs dirty-air fraction | **Selected** |
| B — Tabulate empirical bins alongside constants | Keeps arbitrary constants in production | Rejected (leaves hand-set scalars in the engine) |

## 3. Implementation (`src/models/fit_degradation.py`)

Per-stint degradation slope `beta` (already fitted via cov/var on stint positions ≥3) is joined with the stint's mean `dirty_air_frac` (share of lap within 10 m of the car ahead). `β(dirty_air_frac)` is fitted with **quantile-bin medians + robust Theil–Sen linear regression**, then normalized by the clean-air slope so `clean_air ≡ 1.0`:

- `mult(daf) = clip( β(daf) / β(0), 1.0, ceiling )`, ceiling = 2.5.
- Scenario anchors are taken from the **observed** dirty-air distribution (P90 → `drs_train`, P99 → `backmarker`), not arbitrary 0.5/0.8.

Critical empirical finding: `dirty_air_frac` is **extremely sparse** — its 90th percentile is ~0.05 and its max is ~0.26 (drivers spend most laps in free air). The original 0.5/0.8 anchors lay far outside the observed range, which is why a naive quadratic extrapolation blows up (the first fit produced ×16.5 / ×45.6). Anchoring to observed percentiles + the ceiling keeps the spline honest at the cliff threshold.

## 4. Empirical Result (n = 3,053 dry stints)

| Scenario | dirty_air_frac anchor | Fitted ×mult (was) |
|---|---|---|
| clean_air | 0.0 | **1.00** (1.00) |
| drs_train | 0.023 (P90) | **1.29** (1.15) |
| backmarker | 0.071 (P99) | **1.82** (1.25) |

Bin-median fit: $R^2 \approx 0.66$ across 8 quantile bins; slope ≈ +0.52 s/lap per unit dirty-air fraction, intercept (clean-air slope) ≈ 0.085 s/lap.

## 5. Deliverables

- [x] `src/models/fit_degradation.py`: `_fit_dirty_air()` + `_theil_sen()`; persisted as `dirty_air_fit` + `scenario_beta_mult` + `scenario_notes`.
- [x] `data/models/degradation_model.json`: `scenario_beta_mult = {clean_air: 1.0, drs_train: 1.2923, backmarker: 1.8161}`.
- [x] `src/models/optimizer.py`: `load_scenarios()` merges fitted multipliers with the traffic `pace_pen`; `main()` consumes it (under-cut/over-cut evaluations now penalize releases into DRS trains with telemetry-derived weights).
- [x] `src/eval/plots.py`: `scenario_multiplier_regression()` visualizes the fitted curve + empirical bins + anchors (`eval_plots/scenario_multiplier_regression.png`).
- [x] Unit tests: `pytest tests/` → 24 passed.
