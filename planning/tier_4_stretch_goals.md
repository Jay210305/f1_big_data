# Tier 4 — Explicit Stretch Goals

## Goal Summary
Larger architectural and modeling extensions: wet-weather compound modeling (Intermediate/Wet) and live-timing incremental update architecture with interactive dashboard. If time/scope does not permit, formulate clear "future work" documentation.

---

### Task 8.4.1: Wet-weather modeling (INTERMEDIATE/WET compounds)
- [ ] **8.4.1.1**: Audit `fit_degradation.py` and `backtest.py` exclusions where `wet_frac > 0.02` or compound not in `["SOFT", "MEDIUM", "HARD"]`.
- [ ] **8.4.1.2**: Extend `fit_degradation.py` to support `INTERMEDIATE` and `WET` compounds using rainfall intensity and track-drying proxy features.
- [ ] **8.4.1.3**: Leverage `WET_EFFICIENCY` and `INSTANT_WET` interaction mechanisms from `style_data.py`.
- [ ] **8.4.1.4**: Update `optimizer.py` compound parameters with wet tyre degradation characteristics.
- [ ] **8.4.1.5**: Update `backtest.py` to evaluate wet-affected sessions instead of dropping them.
- [ ] **8.4.1.6**: Complete deliverable writeup or scoped future-work statement in `annotations/phase_8_4_1_wet_weather.md`.

---

### Task 8.4.2: Live-timing adapter + dashboard
- [ ] **8.4.2.1**: Design lightweight incremental state calculation for `get_state()` features (`fuel_corr`, `pace_above_best`, `track_temp_mean`, `cluster`) without scanning full lakehouse.
- [ ] **8.4.2.2**: Implement state updater in `src/live/state_updater.py`.
- [ ] **8.4.2.3**: Build optional adapter client for live telemetry stream (e.g. OpenF1 API).
- [ ] **8.4.2.4**: Create interactive Streamlit dashboard (`app/streamlit_app.py`) visualizing real-time lap progression, degradation tracking, and live recommendations.
- [ ] **8.4.2.5**: Record implementation or explicit descope decision in `annotations/phase_8_4_2_live_tool.md`.
