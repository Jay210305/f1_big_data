# Tier 3 — Pre-Race Planning & Live Re-evaluation

## Goal Summary
Expand system capabilities beyond in-race mid-stint optimization by enabling pre-race strategy generation at Lap 0 (seeded from practice data) and pairwise "what-if" comparative simulation from the CLI.

---

### Task 8.3.1: Pre-race (lap-0) strategy seeding
- [ ] **8.3.1.1**: Inspect `src/models/optimizer.py::get_state()` dependencies on driven race laps.
- [ ] **8.3.1.2**: Select seeding method:
  - **Option A (FP2 Long-Run Pace — Recommended)**: Extract median fuel-corrected lap times per compound during longest continuous FP2 stints from ingested Practice sessions.
  - **Option B (Prior-Year Historical Race Pace)**: Pull driver pace on the same circuit from previous season.
- [ ] **8.3.1.3**: Implement state-seeding function `get_prerace_state(season, gp, driver, starting_compound)` in `src/models/optimizer.py`.
- [ ] **8.3.1.4**: Add `--pre-race` flag to `optimizer.py` CLI supporting Lap 0 strategy optimization without requiring race telemetry.
- [ ] **8.3.1.5**: (Stretch) If Tier 2 is complete, add branch point outputs (e.g., green-flag vs SC window plan).
- [ ] **8.3.1.6**: Update `RUNBOOK.md` with pre-race invocation instructions.
- [ ] **8.3.1.7**: Execute validation on a 2025 race and record comparative results against actual driver execution in `annotations/phase_8_3_1_pre_race_planning.md`.

---

### Task 8.3.2: "What-if" comparator mode
- [ ] **8.3.2.1**: Define CLI argument syntax for paired strategies (e.g. `--compare "SOFT->18,MEDIUM->end" "SOFT->23,HARD->end"`).
- [ ] **8.3.2.2**: Implement string parsing into `[(compound, end_lap, is_current)]` tuples consumed by `optimizer.py::simulate()`.
- [ ] **8.3.2.3**: Execute Monte Carlo simulations for both strategies using shared random seeds / common random numbers.
- [ ] **8.3.2.4**: Format output to display paired time delta, confidence intervals, win probability of Strategy A vs B, and stint degradation profiles.
- [ ] **8.3.2.5**: Add usage examples and CLI command references to `RUNBOOK.md`.
- [ ] **8.3.2.6**: Test comparator CLI against actual race decisions and capture transcript in `annotations/phase_8_3_2_whatif_comparator.md`.
