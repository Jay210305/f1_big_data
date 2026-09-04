# Tier 1 — Safety Car / VSC Correctness

## Goal Summary
Eliminate silent data contamination caused by missed or unhandled Safety Car (SC) and Virtual Safety Car (VSC) periods. Verify detection coverage across race control messages (RCM), and exclude SC/VSC laps from rolling features, cliff-crossing detection, and degradation regression.

---

### Task 8.1.1: Verify SC/VSC detection coverage in raw data
*Prerequisite for all Tier 1 and Tier 2 tasks.*

- [ ] **8.1.1.1**: Query DuckDB silver `rcm` table via `src.ingest.duckdb_layer.open_lakehouse()` for a known VSC race (e.g. 2023 or 2024 GP with documented VSC).
- [ ] **8.1.1.2**: Extract and audit distinct `(cat, flag, msg)` tuples to identify exact VSC event signatures (e.g. `cat='VirtualSafetyCar'`, `msg LIKE '%VIRTUAL SAFETY CAR%'`).
- [ ] **8.1.1.3**: Audit current filter `cat = 'SafetyCar' OR flag IN ('DOUBLE YELLOW', 'RED')` in `src/models/rul_data.py::load_sc_laps()`.
- [ ] **8.1.1.4**: Extend filter logic in `load_sc_laps()` to include all verified VSC tokens and message patterns.
- [ ] **8.1.1.5**: Check if `src/ingest/schema.py::RCM_COLUMNS` requires documentation updates.
- [ ] **8.1.1.6**: Execute `python -m src.models.rul_data` and verify `sc_laps.height` before vs. after.
- [ ] **8.1.1.7**: Document distinct tokens and before/after counts in `annotations/phase_8_1_1_sc_detection_audit.md`.

---

### Task 8.1.2: Exclude mid-stint SC/VSC laps from rolling features and cliff detection
*Mid-stint SC laps (20–40s off-pace) artificially trigger cliff crossings and contaminate $\beta$ degradation fits.*

- [ ] **8.1.2.1**: Audit `filter_stints()` in `rul_data.py` regarding the `_sc_end` guard (last 2 laps).
- [ ] **8.1.2.2**: Choose and implement exclusion strategy:
  - **Option A (Recommended correctness fix)**: In `rul_data.py`, compute rolling pace statistics (`pace_roll_med_3`, `pace_roll_mean`, etc.) and cliff flags (`_above`, `_cross` in `compute_labels()`) across non-SC laps only, skipping SC laps in rolling windows. Re-join onto full per-lap frame.
  - **Option B (Physically nuanced)**: Keep SC laps in sequence for tyre aging, mask pace-based cliff rules, and introduce an explicit `is_sc_lap` feature in `rul_data.py` (ensuring exclusion from `CLIFF_DERIVED` and `EXCLUDED`).
  - **Option C (Minimum bar)**: Mask SC laps out of `fit_degradation.py` slope regression ($\beta$).
- [ ] **8.1.2.3**: Update `fit_degradation.py` regression filtering to mask out any lap tagged as SC/VSC.
- [ ] **8.1.2.4**: Run `python -m src.models.train_rul` and `python -m src.models.fit_degradation`.
- [ ] **8.1.2.5**: Compare before/after metrics: `cliff_within_3` positive prevalence rate, XGBoost cliff F1/AUC, and fitted compound $\beta$ slopes.
- [ ] **8.1.2.6**: Record results and verification in `annotations/phase_8_1_2_sc_contamination_fix.md`.
