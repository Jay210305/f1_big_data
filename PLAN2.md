# Plan: Phase 8 — Rigor, Robustness & Capability Hardening

**Status: NOT STARTED**
**Predecessor:** All of Phases 0–7 are complete (`PLAN.md`, status "ALL PHASES COMPLETE").
**Purpose of this file:** A checkpointed follow-on roadmap that closes the methodological,
engineering, and capability gaps identified in the post-mortem review of the F1 Prescriptive
Tyre & Stint Strategy System, before the report/backtest numbers are presented as final.

**Methodology:** Same Stop-n-Go discipline as `PLAN.md`. Each sub-phase below is a hard
checkpoint: implement → run → write `annotations/phase_8_X_*.md` with real results → Go/No-Go
decision → only then move to the next sub-phase. Do not reorder sub-phases within a priority
tier without a documented reason; the order encodes an effort-to-payoff ranking.

**Division of labor:** unchanged from `PLAN.md` — the agent writes code + run instructions +
explanation; the user runs it locally (32 GB RAM / RTX 5070 Ti) and reports results back into
the annotation file, which the agent uses to make the next Go/No-Go call.

---

## Priority tiers (why this order)

- **Tier 0 (do first, ~2–4 days total):** cheap, high-payoff, mostly reuse existing artifacts.
  Fixes a report/code mismatch, adds a safety net, closes two "documented limitation" gaps
  almost for free, and produces the visual material the report currently lacks entirely.
- **Tier 1 (moderate effort, real capability gains):** Safety Car/VSC correctness — this is
  flagged as a bigger blind spot than it first appears because it silently corrupts labels,
  degradation fits, *and* the optimizer's pit-loss assumption in three different ways.
- **Tier 2 (moderate, strengthens the headline claim):** stochastic SC/VSC modeling in the
  optimizer itself — directly relevant to the credibility of the "78.6% win rate" claim, since
  real 2025 strategists were reacting to SC information the simulator never sees.
- **Tier 3 (moderate-to-larger, new capability):** pre-race full-plan seeding and a live
  what-if comparator — natural extensions once Tier 1–2 land.
- **Tier 4 (larger, explicitly optional/stretch):** wet-weather modeling, live-timing adapter.
  Flag as future work unless scope allows.

---

## Detailed Planning Subdivisions

The execution breakdown for every group of tasks is decomposed into the `planning/` folder:
- [Tier 0 — Cheap Wins](planning/tier_0_cheap_wins.md)
- [Tier 1 — Safety Car / VSC Correctness](planning/tier_1_sc_vsc_correctness.md)
- [Tier 2 — Stochastic SC/VSC in Optimizer](planning/tier_2_stochastic_sc_vsc.md)
- [Tier 3 — Pre-Race Planning & Re-evaluation](planning/tier_3_pre_race_and_whatif.md)
- [Tier 4 — Stretch Goals](planning/tier_4_stretch_goals.md)
- [Engineering Hygiene](planning/engineering_hygiene.md)

---

## Tier 0 — Cheap wins (start here)

### 8.0.1 Fix the RUL model-blending claim (report ↔ code mismatch)

**Problem:** `document/strategy_engine_report.tex` and `main.tex` state "the pair is blended
in production," but `src/models/optimizer.py::load_phase4()` only loads `xgb_rul.json` — the
LSTM (`lstm_rul.pt`) is trained and evaluated in `train_rul.py` / `eval_rul.py` but never used
for inference downstream.

Tasks:
- [ ] Decide between:
  - **Option A (cheapest — do this if time-constrained):** Correct the report text — XGBoost
    is the production RUL model; LSTM is reported as a research comparison that wins in the
    critical zone (`MAE_crit=1.99` vs XGB `2.51`) but is not deployed. Update both `.tex`
    files and the abstract's methodology sentence accordingly.
  - **Option B (moderate, recommended if 8.0.2 confirms the negative-R² isn't a bug):**
    Actually blend at inference: `pred = w(rul_xgb) * xgb_pred + (1 - w(rul_xgb)) * lstm_pred`,
    where `w` shrinks toward 0 as `rul_xgb` approaches the critical zone (`<= CRIT_ZONE`,
    i.e. `<=10` laps), so the LSTM's critical-zone strength is actually used where it matters.
    Implement in `src/models/optimizer.py::predict_laps()` (parallel to the existing
    `cliff_x`/`cliff_l` 50/50 ensemble pattern) and thread the LSTM model + its per-lap
    sequence-building through `load_phase4()`.
  - **Option C (thorough, only if 8.0.2 shows the LSTM's non-critical predictions are stable):**
    Train a small meta-learner (linear or shallow tree) on `[xgb_pred, lstm_pred, compound,
    life] -> target_rul`, fit on the **validation** split only, saved as `models/rul_stack.*`.
- [ ] Update `RUNBOOK.md` stage 4 row if a new artifact (`rul_stack.*`) is produced.
- [ ] Update `document/strategy_engine_report.tex` §4 and `document/main.tex` §IV accordingly,
      whichever option is chosen.

**Files touched:** `src/models/optimizer.py`, `src/models/eval_rul.py` (if Option C),
`document/strategy_engine_report.tex`, `document/main.tex`.
**Deliverable:** `annotations/phase_8_0_1_blend_fix.md`.
**DoD:** the report's blending claim matches what `optimizer.py` actually does at inference.

### 8.0.2 Diagnose the LSTM's negative test R² before trusting it anywhere

**Problem:** `lstm_rul` has `R²=-0.01` on test 2025 despite winning `MAE_crit`. That's the kind
of result that should be interrogated, not just reported as "honest."

Tasks:
- [ ] In `src/models/train_rul.py` (or a new small diagnostic script
      `src/models/diagnose_lstm.py`), check:
  - Padding/masking: confirm `pad_batch()`'s zero-padding + `Y` filled with `NaN` truly excludes
    padded steps from both the training loss (`mask = mb * torch.isfinite(yb)`) **and** the
    test-time flattening in `train_lstm()` (`Yte[i, :L]` / `pt[i, :L]`) — add an assertion that
    `len(yt) == sum of real (unpadded) lap counts` exactly.
  - `OneCycleLR` schedule mismatch: `steps_per_epoch` is computed from the full `epochs` budget
    but early stopping can cut training short — confirm the scheduler doesn't leave the LR in a
    bad state at the epoch where `best_epoch` was captured (i.e., print the LR at `best_epoch`).
  - Add `reg_metrics()`-style **R² computed separately** for critical zone (`RUL<=10`) vs.
    non-critical zone, alongside the existing `MAE_crit`. This alone might rescue the optics
    (a model can have negative R² globally while still being well-calibrated where it matters)
    and is cheap since `MAE_crit` masking logic already exists in `reg_metrics()` — just add
    `R2_crit`/`R2_noncrit` fields.
- [ ] Re-run `python -m src.models.train_rul` and record before/after diagnostics.

**Files touched:** `src/models/train_rul.py`.
**Deliverable:** `annotations/phase_8_0_2_lstm_diagnostics.md` with the R2_crit/R2_noncrit
table and a written verdict on whether the negative R² is a bug or a genuine
noisy-non-critical-zone property.
**DoD:** either a fixed LSTM with non-negative overall R², or a documented, evidenced
explanation of why it's negative and why the critical-zone number is still trustworthy. This
result feeds directly into the 8.0.1 decision (A vs B vs C).

### 8.0.3 De-contaminate the HARD compound pace offset using `track_evolution_index`

**Problem:** `fit_degradation.py` falls back to a **symmetric prior**
(`off_h = abs(off_s)`) for the MEDIUM→HARD pace gap because the raw measured value (−0.38) is
confounded by late-race track rubbering — but `track_evolution_index` (Phase 5,
`style_data.py::_track_evolution`) already exists and was built for exactly this kind of
confound.

Tasks:
- [ ] Pick one:
  - **Option A (moderate, recommended):** Join `track_evolution_index` (computed per
    `season/gp/session/lap`) onto the pairwise SOFT/MEDIUM/HARD stint-best comparison in
    `fit_degradation.py`, regress the measured pace gap against TEI at stint time, and
    extrapolate to `TEI≈0` to get a de-contaminated compound offset. Replace the symmetric
    prior with this fitted, extrapolated value.
  - **Option B (cheap):** Bucket stints into TEI terciles and report/use a per-tercile offset
    directly in the optimizer (`params["pace_offset"]` becomes TEI-bucket-dependent instead of
    a single scalar) — simpler, still closes the gap, and is directly reusable as a
    time-of-race-dependent offset in `optimizer.py::simulate()`.
- [ ] Update `params["pace_offset_notes"]` in the saved `degradation_model.json` to describe
      the new method instead of "symmetric prior because contaminated."

**Files touched:** `src/models/fit_degradation.py`, `src/models/optimizer.py` (if Option B
requires a TEI-bucket lookup at simulate time).
**Deliverable:** `annotations/phase_8_0_3_hard_offset.md`.
**DoD:** the HARD compound offset is empirically derived rather than a symmetric assumption,
and the difference vs. the old constant is quantified.

### 8.0.4 Replace hand-set scenario multipliers with a fitted `dirty_air_frac` relationship

**Problem:** `SCENARIOS = {"drs_train": beta×1.15, "backmarker": beta×1.25}` in
`optimizer.py` are priors. `dirty_air_frac` is already a Gold feature
(`FEATURE_DICTIONARY.md` §1, built in `build_gold.py`'s `TELE_STATS_SQL`).

Tasks:
- [ ] Pick one:
  - **Option A (moderate, high payoff — do this if time allows):** Fit degradation slope
    `beta` as a function of `dirty_air_frac` (binned regression or a simple GAM/spline) using
    the same filtered dry-stint frame `fit_degradation.py` already builds. Replace
    `SCENARIOS["drs_train"]["beta_mult"]` / `["backmarker"]["beta_mult"]` with values read off
    the fitted curve at representative `dirty_air_frac` levels for each scenario.
  - **Option B (cheap, minimum bar):** Sanity-check only — bin stints by `dirty_air_frac` and
    report empirical mean beta per bin next to the current 1.00/1.15/1.25 multipliers, stating
    whether they over- or under-shoot, without changing the constants.
- [ ] If Option A: persist the fitted curve/coefficients into `degradation_model.json` so
      `optimizer.py` reads fitted multipliers instead of the `SCENARIOS` dict literal.

**Files touched:** `src/models/fit_degradation.py`, `src/models/optimizer.py`.
**Deliverable:** `annotations/phase_8_0_4_scenario_multipliers.md` with the binned empirical
table either way.
**DoD:** scenario multipliers are either fitted from data or explicitly validated against data,
turning a stated limitation into a resolved (or at least evidenced) one.

### 8.0.5 Unit tests for the pure/leakage-critical functions (safety net)

**Problem:** 300+ features, several non-trivial Polars pipelines (rolling windows, leakage
exclusion, the TEI chronological fold), zero automated tests. A silent regression is easy to
introduce and currently only caught by eyeballing printed diagnostics.

Tasks:
- [ ] Create `tests/` with `pytest` covering, at minimum:
  - `src/ingest/etl.py::_normalize_telemetry` — synthetic `tel` dict with mismatched array
    lengths → assert truncation to `min_len` and correct dtype casts.
  - `src/ingest/etl.py::_silver_exprs` — synthetic lap DataFrame → assert aggregate columns
    (`speed_mean`, `dirty_air_frac`, etc.) compute correctly and null-fill when a channel is
    absent.
  - `src/models/rul_data.py::add_fuel_corr` — assert `fuel_corr == laptime + 0.03*lap` only for
    Race/Sprint session types, unchanged otherwise.
  - `src/models/rul_data.py::compute_labels` — feed a small synthetic stint (e.g. 10 laps with
    a clear, hand-computed pace jump at lap 6) and assert `target_rul`/`cliff_within_3` match a
    manually worked-out expectation, including the piecewise RUL_MAX clipping per compound.
  - **Leakage assertion as a real test, not a printed diagnostic:** convert the existing
    `CLIFF_DERIVED & set(feats)` check at the bottom of `rul_data.py`'s `__main__` block into
    `tests/test_no_leakage.py::test_no_cliff_derived_features_in_X()` that asserts the
    intersection is the empty set and fails loudly (not just prints) if violated.
  - `src/models/style_data.py::_track_evolution` — synthetic 2-session weekend with one wet
    session, assert the `gamma_wash` reset and `INSTANT_WET` per-lap factor apply as specified.
- [ ] Add a `tests/conftest.py` with small synthetic Polars DataFrames as fixtures (no
      dependency on the 57 GB dataset).

**Files touched:** new `tests/` directory, `pytest` added to `requirements.txt`.
**Deliverable:** `annotations/phase_8_0_5_unit_tests.md` with `pytest -v` output pasted in.
**DoD:** `pytest tests/` passes locally in the venv with zero dependency on `data/` or the raw
JSON tree; the leakage check is enforced by an assertion, not a print statement.

### 8.0.6 Plotting module (currently zero plots exist anywhere in the pipeline)

**Problem:** `PLAN.md` mentions notebooks but none exist in the repo; the report has tables
but no figures generated from the actual artifacts. This is the single highest-leverage
addition for both the report and day-to-day debugging.

Tasks:
- [ ] Create `src/eval/plots.py` (mirrors `src/analysis/correlation_map.py`'s
      matplotlib/seaborn + `Agg` backend + save-to-folder pattern) producing, at minimum:
  - Predicted-vs-actual `target_rul` scatter (XGBoost, and LSTM if kept) on test 2025, colored
    by compound, with the `y=x` reference line.
  - Degradation curves per compound: sampled real stints' `fuel_corr` vs `life`, with the
    fitted `beta[compound]` line overlaid from `degradation_model.json`.
  - Cluster embeddings: 2D UMAP or t-SNE of `driver_embeddings.parquet` latent vectors, colored
    by HDBSCAN `cluster` label (reuse the `cluster_summary` from `phase5_metrics.json` for
    legend labels).
  - Backtest delta distribution: histogram of the 313 paired `delta` values from
    `phase7_backtest.json["decisions"]`, with a vertical line at the median and mean.
  - (If 8.0.2/8.0.4 land) critical-vs-noncritical R² bar chart; empirical vs. fitted scenario
    multiplier curve.
- [ ] Output folder: `<repo-root>/eval_plots/` (add to `.gitignore` alongside
      `correlation_maps/`... actually check: `correlation_maps/` is **not** gitignored
      currently — confirm intended visibility and mirror that choice for `eval_plots/`).
- [ ] CLI: `python -m src.eval.plots` (all figures) and `--only <name>` per figure, matching
      `correlation_map.py`'s `--only` pattern.

**Files touched:** new `src/eval/plots.py`.
**Deliverable:** `annotations/phase_8_0_6_plots.md` + the generated PNGs referenced in it.
**DoD:** four figures listed above exist as PNGs generated from real artifacts (not
placeholders), and are ready to drop into `document/strategy_engine_report.tex` /
`document/main.tex` as `\includegraphics` figures.

---

## Tier 1 — Safety Car / VSC correctness (bigger blind spot than it looks)

### 8.1.1 Verify SC/VSC detection coverage in the raw data (do this before anything else in Tier 1)

**Problem:** `load_sc_laps()` in `rul_data.py` only flags `cat = 'SafetyCar'` or
`flag IN ('DOUBLE YELLOW', 'RED')`. It is **not verified** that VSC actually surfaces under
`cat='SafetyCar'` in `rcm.json`, or under a different, currently-uncaught category/flag value.
If VSC isn't being caught, every downstream fix in this tier is built on a wrong premise.

Tasks:
- [ ] Run a DuckDB query against the silver `rcm` table (via
      `src.ingest.duckdb_layer.open_lakehouse()`) for a race with a known VSC period (pick one
      from memory/public record — e.g. a 2023 or 2024 race with a documented VSC) and dump the
      distinct `(cat, flag, msg)` combinations for that race's `rcm` rows.
- [ ] Confirm whether VSC has its own token (e.g. `cat='VirtualSafetyCar'` or a `msg` substring
      like `"VIRTUAL SAFETY CAR"`) that the current filter misses entirely.
- [ ] Extend `SC/VSC` detection in `load_sc_laps()` (and `src/ingest/etl.py`'s RCM handling if
      the schema manifest `src/ingest/schema.py::RCM_COLUMNS` needs a note) to cover whatever
      token(s) are found — this is expected to be close to a one-line filter-list change once
      the token is confirmed.
- [ ] Re-run `python -m src.models.rul_data` and note the change in `sc_laps.height` before vs.
      after (should increase materially if VSC was previously missed).

**Files touched:** `src/models/rul_data.py::load_sc_laps`, possibly
`src/ingest/schema.py` (documentation only).
**Deliverable:** `annotations/phase_8_1_1_sc_detection_audit.md` with the actual distinct
`(cat, flag)` tokens found and the before/after SC-lap counts.
**DoD:** SC and VSC are both provably caught by the filter, backed by a query against real
`rcm.json` data for at least one race with a known VSC period.

### 8.1.2 Exclude mid-stint SC/VSC laps from rolling features and cliff-crossing detection

**Problem:** `filter_stints()` only drops a stint if SC/VSC hits the **last 2 laps**
(`_sc_end`). Mid-stint SC/VSC laps (which can be 20–40s slower than green-flag pace) still flow
into `compute_rolling_features()`'s `pace_roll_mean/std/med/ema` and into
`compute_labels()`'s cliff-crossing rule (`pace_roll_med_3 > stint_best_pace + 1.5s` sustained 3
laps). A 3-lap SC period is a highly plausible false-positive physical cliff. The same
contamination reaches `fit_degradation.py`'s `cov(life, fuel_corr)/var(life)` slope regression,
since it isn't currently masking SC laps out.

Tasks:
- [ ] Pick one:
  - **Option A (moderate, recommended — correctness fix):** In `rul_data.py`, compute
    `pace_roll_med_3` and the cliff-crossing rule (`_above`/`_cross` in `compute_labels()`)
    over **non-SC laps only** within the stint — same pattern as the existing out-lap guard
    (`_pos >= 2`), but gated on `_is_sc` too (skip-and-continue through the rolling window
    rather than including SC laps in it). This likely means restructuring the rolling-window
    calls in `compute_rolling_features()` to operate on an SC-filtered view, then re-joining
    back onto the full per-lap frame for the features that don't need this masking (e.g. raw
    telemetry stats can keep SC laps, since tyre wear continues).
  - **Option B (bigger, more physically correct):** Keep SC laps in the sequence (tyre still
    ages under SC), but mask them out of pace-based label logic specifically, **and** add an
    explicit `is_sc_lap` feature so RUL/cliff models can learn "ignore this lap's pace, but tyre
    life still ticks forward." Requires adding `is_sc_lap` to the feature set in `rul_data.py`
    and confirming it's not accidentally added to `CLIFF_DERIVED`/`EXCLUDED`.
  - **Option C (cheapest, partial — minimum bar):** At minimum, exclude SC-affected laps from
    the `fit_degradation.py` slope regression (the `_pos >= 3` dry-stint filter already there)
    since a contaminated β estimate propagates into *every* downstream Monte Carlo simulation,
    not just one label.
- [ ] Re-run `python -m src.models.train_rul` and `python -m src.models.fit_degradation`;
      compare `cliff_within_3` positive rate and `beta[compound]` values before vs. after.

**Files touched:** `src/models/rul_data.py`, `src/models/fit_degradation.py`.
**Deliverable:** `annotations/phase_8_1_2_sc_contamination_fix.md` with before/after metrics
(`cliff_within_3` rate, `xgb_cliff` F1/AUC, `beta[compound]` values).
**DoD:** SC/VSC laps no longer trigger false physical-cliff labels or bias the degradation
slope fit; the change is quantified, not just asserted.

---

## Tier 2 — Stochastic SC/VSC in the optimizer (strengthens the 78.6% headline claim)

### 8.2.1 Empirical SC/VSC hazard model + discounted pit-loss scenario

**Problem:** `optimizer.py` has `PIT_LOSS = 21.0` as a hardcoded constant with zero SC-state
dependency, and `enumerate_strategies()`/`simulate()` have no concept of "probability an SC/VSC
occurs in the next N laps." Real strategists frequently pit *because* an SC is out (pit loss
drops to roughly a third of green-flag pit loss), not purely because the tyre is done. This is
flagged as the single biggest strategic omission relative to real F1 strategy, and it directly
undermines the credibility of the backtest's win-rate claim, since real 2025 strategists were
reacting to SC information the simulator never sees.

Tasks (requires 8.1.1 done first — hazard rates are meaningless if detection is incomplete):
- [ ] Pick one:
  - **Option A (moderate, good payoff — recommended):** Fit an empirical per-lap SC/VSC hazard
    rate `P(SC onset | circuit, lap number)` from the corrected `rcm`-derived SC/VSC laps
    across 2021–2025 (expect strong circuit effects — e.g. Baku/Singapore/Jeddah vs. Barcelona).
    In the Monte Carlo (`optimizer.py::simulate()`), for each simulated stint, sample whether
    an SC/VSC period occurs (reusing the existing per-simulation RNG pattern already used for
    `beta`/`cliff_rem`), and apply a discounted pit-loss (`PIT_LOSS / 3` or similar, tunable)
    for strategies that pit during that sampled window. This is "one more sampled variable" in
    an MC framework that already samples β and cliff timing per simulation — not a structural
    rewrite.
  - **Option B (simpler first pass, ~80% of the decision value for far less work):** Add a new
    scenario alongside `clean_air`/`drs_train`/`backmarker` — e.g. `SCENARIOS["sc_window"]`
    with a discounted `pit_loss_mult` and a degradation reset, so a user can directly compare
    "if an SC comes now" vs. "if it doesn't" without full probabilistic integration. No new
    per-lap hazard fitting required.
- [ ] Update `RUNBOOK.md`'s in-race recommendation section and the optimizer CLI `--help` text
      to document the new scenario/flag.
- [ ] Re-run `python -m src.models.backtest` — even a partial SC-awareness may change the
      win-rate / delta numbers; report the new numbers next to the old ones honestly (don't
      cherry-pick).

**Files touched:** `src/models/optimizer.py`, `src/models/fit_degradation.py` (if Option A
needs the hazard table persisted into `degradation_model.json`), `RUNBOOK.md`.
**Deliverable:** `annotations/phase_8_2_1_sc_hazard.md` with the fitted hazard table (Option A)
or the new scenario definition (Option B), and updated backtest numbers.
**DoD:** the optimizer has *some* explicit representation of SC/VSC probability and its effect
on pit-loss economics — no longer a documented-but-unaddressed limitation.

*(If descoped: explicitly keep as a documented limitation in the report rather than silently
dropping it — do not let the 78.6% claim stand unqualified if this tier is skipped.)*

---

## Tier 3 — New capability: pre-race planning + live re-evaluation

### 8.3.1 Pre-race (lap-0) strategy seeding

**Problem:** `get_state()` requires a lap the driver has already run (`fuel_corr` from that
lap's row, `laps_into_stint` from actual driven laps) — there is no lap-0 state, so the
optimizer cannot currently produce a full plan before lights out, even though
`enumerate_strategies()`/`simulate()` don't structurally require live data — they just need a
starting pace and compound.

Tasks:
- [ ] Pick one:
  - **Option A (moderate, most useful — recommended):** Seed pre-race `base_pace` from the
    driver's best representative **long-run FP2 pace** (not qualifying — different fuel/tyre
    mode than a race stint) per starting compound: pull the driver's median fuel-corrected pace
    over their longest FP2 run per compound from the already-ingested Practice sessions, then
    call `enumerate_strategies`/`optimize` with `cur_lap = 1`.
  - **Option B (cheaper, rougher):** Use the same driver's same-circuit pace from the prior
    year as a prior (only viable where the circuit repeats year-over-year) — simpler but
    ignores current-year car performance.
- [ ] Add a `--pre-race` CLI flag to `src/models/optimizer.py` that triggers the new seeding
      path instead of `get_state()`'s live-lap lookup.
- [ ] (Only after 8.2.1 lands) Consider Option C from the source review: output branch points
      instead of one linear plan ("optimal 1-stop if the race stays green" vs. "what changes if
      an SC appears in laps 15–25") — i.e., a small decision tree rather than a single
      recommendation. Treat as a stretch sub-task within 8.3.1, not a blocker.

**Files touched:** `src/models/optimizer.py`, `RUNBOOK.md`.
**Deliverable:** `annotations/phase_8_3_1_pre_race_planning.md` with a worked example (one real
2025 race, comparing the FP2-seeded pre-race plan against the driver's actual executed
strategy).
**DoD:** the optimizer can be invoked with no live race data and produce a ranked pre-race
strategy from practice-session pace alone.

### 8.3.2 "What-if" comparator mode (cheap, reuses existing `simulate()`)

**Problem:** The CLI already supports re-running the optimizer at any lap
(`optimizer.py --lap N`) using real Gold-store state — functionally a "re-plan given what's
happened so far." What's missing is a direct paired comparison between two user-specified
strategies (e.g. "pit now for MEDIUM" vs. "stay out 5 more laps then HARD").

Tasks:
- [ ] Add a `--compare "SOFT->18,MEDIUM->end" "SOFT->23,HARD->end"`-style CLI argument (exact
      syntax TBD by agent, but should parse into the same `[(compound, end_lap, is_current)]`
      structure `simulate()` already consumes) that runs both strategies through the same
      Monte Carlo with common random numbers and prints the paired delta + confidence interval
      — this is a thin UI layer over `simulate()`, not new modeling.
- [ ] Add this mode to `RUNBOOK.md`'s "In-race recommendation" section with an example command.

**Files touched:** `src/models/optimizer.py`, `RUNBOOK.md`.
**Deliverable:** `annotations/phase_8_3_2_whatif_comparator.md` with an example run.
**DoD:** a user can compare two explicit named strategies head-to-head from the CLI without
reading raw `phase6_demo.json`.

---

## Tier 4 — Explicit stretch goals (flag as future work unless scope allows)

### 8.4.1 Wet-weather modeling (INTERMEDIATE/WET compounds)

**Problem:** Wet races are fully excluded from `fit_degradation.py` and `backtest.py` — a real
strategic gap, since F1 strategy is often decided in wet conditions. `rain_flag` is already
tagged from `weather.json`, so most of the ETL/Gold plumbing exists; this is primarily a
modeling extension, not a data-engineering one.

Tasks (only if scope allows — otherwise explicitly document as future work, do not silently
drop):
- [ ] Extend `fit_degradation()`'s `DRY = ["SOFT","MEDIUM","HARD"]` list handling to a parallel
      wet-compound path (`INTERMEDIATE`, `WET`), reusing the same β/σ/cliff-distribution
      machinery but with rain-state-dependent features (rain intensity trend, track-drying
      proxy from `track_evolution_index`'s existing wet-session handling).
  - [ ] Note the interaction with `style_data.py`'s existing `WET_EFFICIENCY`/`INSTANT_WET`
      constants — these were built for the TEI rain wash-out and may be directly reusable here.
- [ ] Extend `backtest.py` to no longer categorically exclude `wet_races` (currently anything
      with `wet_frac > 0.02`), instead scoring wet-affected races with the new wet degradation
      model.

**Files touched:** `src/models/fit_degradation.py`, `src/models/optimizer.py`,
`src/models/backtest.py`.
**Deliverable:** `annotations/phase_8_4_1_wet_weather.md`.
**DoD:** either a working wet-compound degradation/backtest path, or an explicit, scoped
"future work" note in the report replacing the current silent exclusion.

### 8.4.2 Live-timing adapter + dashboard (largest lift — stretch only)

**Problem:** There is no path from "live timing right now" to a queryable optimizer state
during an actual session — `Gold/lap_features` is a batch artifact rebuilt from the full
lakehouse.

Tasks (only pursue if "live tool" is an explicit stated deliverable — otherwise skip):
- [ ] Build a lightweight incremental state updater that computes just the handful of derived
      fields `get_state()` needs (`fuel_corr`, `pace_above_best`, `track_temp_mean`, `cluster`)
      from the last few laps' raw telemetry/laptimes, without touching the 325M-row Bronze
      table — this is the prerequisite for anything "live," independent of the data source.
- [ ] Only after that: an actual live-timing adapter (e.g. OpenF1 API) feeding the incremental
      updater, wrapped in a small Streamlit dashboard (ties into 8.0.6's plotting module) that
      re-runs the optimizer every lap and shows the recommendation evolving in real time.

**Files touched:** new `src/live/` package (state updater, optional adapter), optional
`app/streamlit_app.py`.
**Deliverable:** `annotations/phase_8_4_2_live_tool.md` (even if the conclusion is "descoped").
**DoD:** either a working incremental updater + dashboard, or an explicit scope decision
recorded in the annotation.

---

## Engineering hygiene (interleave with the tiers above, not a separate phase)

These are small, ongoing, and should be picked up alongside whichever tier is active rather
than blocking any of it:

- [ ] **8.H.1 — Structured error logging in ETL.** `src/ingest/etl.py`'s
      `except Exception: stats["errors"] += 1` pattern silently swallows exceptions (this
      already required a bespoke reactive script, `diagnose_errors.py`, for 6 known failures).
      Log `type(exc).__name__: str(exc)` into the stats dict (or a per-session log file) instead
      of just incrementing a counter, so future failures are diagnosable without writing a new
      one-off diagnostic script each time.
- [ ] **8.H.2 — CI workflow.** Add a GitHub Actions workflow that runs `pytest tests/` (from
      8.0.5) on push/PR. No dependency on the 57 GB dataset, so this is cheap once 8.0.5 exists.
- [ ] **8.H.3 — Dependency lockfile.** `requirements.txt` uses `>=` with no lockfile. Move to
      `uv` or `poetry` with a lockfile so the exact package versions behind the reported numbers
      (R²=0.35, F1=0.747, etc.) are reproducible later.
- [ ] **8.H.4 — Schema-drift / data-quality check on real Bronze output.** `explore.py` only
      samples a bounded subset. Add `src/ingest/verify_schema.py` that runs against the **full**
      Bronze output (not a sample) and asserts null rates / column presence per season, so a
      repeat of the 6-session failure class (or a new one from an added season) is caught
      automatically rather than requiring another bespoke diagnostic pass.
- [ ] **8.H.5 — Feature importance / interpretability.** Cheap first pass:
      `xgb_rul.get_score(importance_type='gain')` dumped to a ranked table (a few lines,
      useful both for the report and for pruning noisy features from the 300+ feature set).
      SHAP summary/dependence plots are a nice-to-have moderate-cost follow-up, best combined
      with 8.0.6's plotting module.
- [ ] **8.H.6 — Weight the cliff ensemble instead of a fixed 50/50 mean.**
      `optimizer.py::predict_laps()` and `train_rul.py::train_xgb_cliff()` currently do
      `0.5 * (xgb_prob + lgb_prob)`. Cheap fix: weight by each model's validation AUC instead of
      a fixed 0.5/0.5; moderate fix: a 2-feature logistic stacker trained on val-set
      `[xgb_prob, lgb_prob] -> label`.
- [ ] **8.H.7 — Hyperparameter search on the flagship RUL regressor.** Before investing here,
      run a cheap 1-trial-budget sanity check: is `R²≈0.39` plausibly a physical noise ceiling
      (tyre wear has real stint-to-stint variance not observable from telemetry alone)? If so,
      an Optuna sweep (30–50 trials, existing val split as objective) is unlikely to move the
      needle much and effort is better spent on Tiers 0–3. Only extend to the cliff classifier
      if the RUL sweep shows meaningful gains.
- [ ] **8.H.8 — Backtest selection-bias partial mitigation.** Two cheap/moderate options,
      independent of Tier 2:
      1. *(cheap, data already exists)* Stratify the 313-decision win rate by how much the
         recommended strategy's stop count diverges from the actual one — if wins concentrate
         where recommendation ≈ actual, that's worth surfacing rather than only reporting the
         aggregate 78.6%.
      2. *(moderate)* For the subset of 2025 races where the actual second stint's realized
         pace is fully known, score both the actual and recommended strategy using the
         **realized empirical pace** instead of the fitted degradation model — breaking the
         shared-simulator bias for at least the actual-vs-actual leg of the comparison.
- [ ] **8.H.9 — Cliff threshold / RUL cap sensitivity table (lowest priority — only if time
      remains).** Grid `CLIFF_THR ∈ {1.0, 1.5, 2.0}` × sustain-window `∈ {2, 3, 4}` laps in
      `rul_data.py::compute_labels()`, report how cliff-classifier F1/AUC move. Strengthens the
      "physical, not arbitrary" framing of the 1.5s/3-lap constants but is a modeling nicety,
      not a correctness issue — skip first if time-constrained.

---

## Suggested execution order (single recommended path if not parallelizing)

1. 8.0.2 (LSTM diagnostics) → 8.0.1 (blend fix, now correctly informed)
2. 8.0.5 (unit tests) — do this early so every subsequent change has a safety net
3. 8.0.3 + 8.0.4 (HARD offset, scenario multipliers) — cheap, reuse existing features
4. 8.0.6 (plots) — needed for the report regardless of what else ships
5. 8.1.1 → 8.1.2 (SC/VSC correctness) — must precede Tier 2
6. 8.2.1 (SC/VSC hazard in the optimizer) — directly strengthens the backtest headline
7. 8.3.1 → 8.3.2 (pre-race seeding, what-if comparator)
8. 8.4.x only if scope/time allows; otherwise write the future-work note and stop here.

Interleave the 8.H.* engineering-hygiene items opportunistically (8.H.1–8.H.4 are nearly free
and should not wait for a dedicated tier).

## Annotation template (reuse from `PLAN.md`)

```markdown
# Phase 8.X — <name> (STOP)

- Date / author:
- Status: [ ] Go  [ ] No-Go  [ ] Rework
- Commands run by user (copy/paste):
- Results / output (from user):
- Metrics / numbers (before vs. after, where applicable):
- Errors / blockers:
- Decisions made:
- Next-phase risks:
- Time spent / next step:
```
