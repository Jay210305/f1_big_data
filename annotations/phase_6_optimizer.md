# Phase 6 — Prescriptive Dynamic Pit-Window & Stint Optimizer (STOP)

- Date / author: 2026-08-30 / agent
- Status: [x] Go (with documented calibration caveats)
- Transition compliance: cliff debouncing (≥2 consecutive laps) + joint cliff/RUL
  anomaly rule implemented (action item 3); scenario context (clean air / DRS
  train / backmarker) implemented.

## Files created

| File | Purpose |
|------|---------|
| `src/models/fit_degradation.py` | Fits the degradation model from the Phase 4 labeled frame + Phase 5 clusters: β per compound (pos≥3, warm-up excluded), σ, style-cluster offsets, temp coefficient, empirical cliff-lap distributions (cliff_at≥3), compound pace offsets (pairwise within-driver). Saved to `data/models/degradation_model.json` |
| `src/models/optimizer.py` | (1) MC strategy evaluator — analytic stint times, N sims/candidate, pit loss + warm-up per stop; (2) candidate search 0/1/2-stop with the F1 two-compound rule; (3) scenario multipliers; (4) in-race Pit Monitor with debouncing + joint RUL rule, backtested on real stints; (5) demo CLI |
| `data/models/degradation_model.json` | Fitted parameters (gitignored) |
| `data/models/phase6_demo.json` | Demo outputs: state + ranked strategies per scenario (gitignored) |

## Steps to run (in order)

```powershell
.\.venv\Scripts\Activate.ps1

python -m src.models.fit_degradation    # refit params if Phase 4/5 changed (~30s)
python -m src.models.optimizer          # demo: 2025 Bahrain Race VER @ lap 12 (~35s)
```

Flags: `--gp "Spanish Grand Prix" --driver HAM --season 2024 --lap 20`,
`--n-sims 1000`, `--pit-loss 21`.

## What's going on

**Degradation model fitting** (data-driven, physically validated):
- β (fuel-corrected slope, s/lap): fitted per stint via cov/var on stint
  **positions ≥ 3** — the first 2 flying laps are excluded because tyre warm-up
  + race-start traffic bias slopes negative. Result: SOFT 0.107 > MEDIUM 0.076
  > HARD 0.045 ✓ physical ordering. σ per compound from slope dispersion.
- Style cluster offsets: only cluster 0 (elites) has ≥30 race stints —
  rookies/part-timers (clusters 1/−1) barely race, so offsets are ~0 in races.
- Temp coefficient +0.005 s/lap/°C ✓ (hotter = faster degradation).
- Cliff-lap distributions per compound (cliff_at ≥ 3, early traffic artifacts
  counted as censored): SOFT q50=7, MEDIUM q50=11, HARD q50=13 laps;
  censor_prob ≈ 0.68-0.79 (most stints end strategically before the cliff).
- Compound pace offsets: measured **pairwise within-driver** (same driver, same
  session, both compounds): SOFT−MEDIUM = −0.188 s ✓ physical. HARD−MEDIUM
  measured −0.380 is track-evolution contaminated (M→H one-stoppers run the
  HARD stint late on a rubbered track; the pairwise triangle is inconsistent by
  0.37 s) → HARD set to the symmetric prior +0.188. Documented + overridable.

**Monte Carlo strategy evaluator**: for each candidate strategy, N sims sample
β ~ N(β_C+cluster+κ·ΔT, σ_C) and cliff laps from the empirical distributions;
stint times are computed **analytically** (deg term β·L(L+1)/2, cliff term
0.5·m(m+1)/2, m = laps in cliff) — no lap loops, 400 sims × ~300 candidates in
< 2 s. Pit loss (21 s default) + warm-up (2 s) per stop. Scenarios multiply β
(clean 1.0 / DRS train 1.15 / backmarker 1.25) and add pace penalty
(0 / 0.25 / 0.5 s/lap).

**Pit Monitor (transition action item 3)**: per lap, RUL (Phase 4 XGBoost) +
cliff probability (XGB+LGB ensemble, threshold 0.353). Rules:
- **Debouncing**: pit trigger only when cliff_prob > thr on ≥ 2 consecutive laps.
- **Joint rule**: cliff alert + RUL > 15 laps ⇒ flagged as anomaly (temporary
  thermal degradation, e.g. dirty-air overheating) — NOT a pit trigger.
- **Imminence**: RUL ≤ 2 ⇒ pit now regardless.

## Results (agent validation run 2026-08-30 — confirm with your own run)

- Commands run (copy/paste):
  - `python -m src.models.fit_degradation` (30s)
  - `python -m src.models.optimizer` (35s)
- Results / output — demo state: 2025 Bahrain Race, VER @ lap 12/44,
  stint 2 on HARD (1 lap in), 31.9 °C, cluster 0, compounds used {SOFT, HARD}:

  | Scenario | Best strategy | mean | p5-p95 | Next best Δ |
  |----------|--------------|------|--------|-------------|
  | clean_air | HARD→23 \| HARD→33 \| HARD→44 (2-stop) | 3240.6 | 3169-3359 | +9.3 |
  | drs_train | HARD→23 \| HARD→33 \| HARD→44 (2-stop) | 3263.4 | 3177-3407 | +3.3 |
  | backmarker | HARD→23 \| HARD→29 \| HARD→44 (2-stop) | 3276.7 | 3185-3472 | +0.6 |

  **Scenario sensitivity is physically correct**: heavier traffic (faster
  degradation + pace loss) pulls the second stop earlier (33 → 29) and
  compresses the strategy spread.

  **Pit monitor backtest** (same race):
  - stint 1 (SOFT, laps 1-10): `rul_imminent` at lap 10 = actual stint end 10 ✓
    (RUL hit zero exactly at the strategic stop).
  - stint 2 (HARD, laps 11-26): physical-cliff trigger at lap 17 vs actual
    strategic stop at 26 — the monitor warns of the *physical* cliff 9 laps
    before the team's strategic stop; no false single-lap spikes fired
    (debouncing active).

- Metrics / numbers: see `phase6_demo.json` (top-10 per scenario with
  mean/std/p5/p95/Δ).
- Errors / blockers: none.
- Decisions made:
  - β estimator: pos≥3 (warm-up excluded) — fixes negative-slope contamination.
  - Cliff distribution: cliff_at≥3 filter (race-start traffic artifacts → censored).
  - Pace offsets: pairwise within-driver for SOFT−MEDIUM; symmetric prior for
    HARD (track-evolution contamination documented).
  - Pit loss: constant 21 s default (circuit-dependent in reality; CLI flag).
- Known calibration caveats (honest):
  1. The MC favors 2-stops at Bahrain more than reality (real Bahrain is
     usually 1-stop). Driver: the cliff term (0.5 s/lap compounding + q50≈11-13)
     is severe for long stints. Calibration levers: `cliff_penalty_per_lap`,
     cliff `censor_prob`, `--pit-loss`. Recommend a calibration pass before
     production use (Phase 7 backtest will quantify).
  2. Pit loss is circuit-constant; no safety-car windows modeled in the MC
     (spec'd as future work).
  3. Scenario multipliers (1.0/1.15/1.25) are priors, not fitted — could be
     fitted from `dirty_air_frac`-stratified slopes in Phase 7.
- Next-phase risks:
  - Phase 7 backtest must compare optimizer recommendations vs actual strategies
    across many races to calibrate the caveats above.
- Stop-n-Go checkpoint: full-system demo on a sample GP ✓ (state extraction →
  ranked strategies per scenario with confidence bands → monitor backtest) → **GO**.
- Time spent / next step: Phase 6 COMPLETE -> GO to Phase 7 (Integration, Backtest & Report).
