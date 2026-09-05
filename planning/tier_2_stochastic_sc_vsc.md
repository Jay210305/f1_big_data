# Tier 2 — Stochastic SC/VSC in the Optimizer

## Goal Summary
Incorporate Safety Car / VSC probabilities and discounted pit loss into strategy evaluation. Address the fundamental gap that real strategists react to SC deployment where pit stop delta is drastically reduced (~1/3 of green flag pit loss).

---

### Task 8.2.1: Empirical SC/VSC hazard model + discounted pit-loss scenario
*Requires 8.1.1 completed.*

- [x] **8.2.1.1**: Choose implementation route: **Option A selected — Probabilistic Monte Carlo**.
  - **Option A (Probabilistic Monte Carlo — Selected)**:
    - Compute empirical hazard rate $P(\text{SC onset} \mid \text{circuit}, \text{lap})$ across 2021–2025 corrected RCM data.
    - Persist circuit-lap hazard lookup in `degradation_model.json`.
    - In `src/models/optimizer.py::simulate()`, sample SC onset per simulated stint using common random numbers.
    - Apply discounted pit loss ($\text{PIT\_LOSS} / 3$ or parameter `PIT_LOSS_SC`) when a stop falls inside the sampled SC window.
  - **Option B (Deterministic Scenario Branching — Rapid pass, not selected)**:
    - Introduce `SCENARIOS["sc_window"]` into `optimizer.py` with discounted pit loss multiplier (`pit_loss_mult ~ 0.4`) and degradation reset.
- [x] **8.2.1.2**: If Option A, update `src/models/fit_degradation.py` to fit and export the circuit hazard table.
- [x] **8.2.1.3**: Update `src/models/optimizer.py` simulation loop and argument parser to support SC hazard sampling / scenario flag.
- [x] **8.2.1.4**: Update CLI documentation and `--help` text in `optimizer.py`.
- [x] **8.2.1.5**: Update `RUNBOOK.md` in-race recommendation section.
- [x] **8.2.1.6**: Run `python -m src.models.backtest --sc-hazard` to evaluate win rate and average delta impact across the 313 decisions.
- [x] **8.2.1.7**: Document fitted hazard distributions (or scenario specs) and backtest metrics in `annotations/phase_8_2_1_sc_hazard.md`.
