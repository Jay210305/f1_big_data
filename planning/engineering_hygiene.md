# Engineering Hygiene (Interleaved Tasks)

## Goal Summary
Cross-cutting quality, reproducibility, and reliability enhancements interleaved opportunistically across all execution tiers.

---

### Task 8.H.1: Structured error logging in ETL
- [ ] **8.H.1.1**: Identify silent `except Exception: stats["errors"] += 1` blocks across `src/ingest/etl.py`.
- [ ] **8.H.1.2**: Replace bare counter increments with structured exception logging capturing `type(exc).__name__` and `str(exc)`.
- [ ] **8.H.1.3**: Add session-level error log aggregation to output summary.

---

### Task 8.H.2: CI workflow
- [ ] **8.H.2.1**: Create `.github/workflows/ci.yml`.
- [ ] **8.H.2.2**: Configure Python virtual environment, dependencies, and automated test execution with `pytest tests/`.
- [ ] **8.H.2.3**: Ensure CI job runs cleanly without downloading the full 57 GB dataset.

---

### Task 8.H.3: Dependency lockfile
- [ ] **8.H.3.1**: Benchmark current environment dependencies against `requirements.txt`.
- [ ] **8.H.3.2**: Initialize lockfile tooling using `uv.lock` or `poetry.lock`.
- [ ] **8.H.3.3**: Verify reproducible install inside a clean environment.

---

### Task 8.H.4: Schema-drift & data-quality validation on Bronze
- [ ] **8.H.4.1**: Create `src/ingest/verify_schema.py`.
- [ ] **8.H.4.2**: Implement verification across full Bronze output (column availability, unexpected null rates, type checks across seasons 2021–2025).
- [ ] **8.H.4.3**: Integrate check into standard ingestion pipeline.

---

### Task 8.H.5: Feature importance & interpretability
- [ ] **8.H.5.1**: Dump XGBoost gain importance via `xgb_rul.get_score(importance_type='gain')` to sorted CSV/markdown table.
- [ ] **8.H.5.2**: Generate SHAP summary plots for top 20 RUL predictors.
- [ ] **8.H.5.3**: Review low-importance features for future feature pruning.

---

### Task 8.H.6: Weighted cliff ensemble
- [ ] **8.H.6.1**: Evaluate replacement of fixed 50/50 mean (`0.5 * xgb + 0.5 * lgb`) in `optimizer.py::predict_laps()` and `train_rul.py`.
- [ ] **8.H.6.2**: Implement weighting proportional to validation ROC-AUC, or fit a 2-feature logistic meta-learner.
- [ ] **8.H.6.3**: Compare validation cliff F1/AUC before and after weighting.

---

### Task 8.H.7: Hyperparameter search on flagship RUL regressor
- [ ] **8.H.7.1**: Run a 1-trial budget sanity check on residual error vs telemetry noise ceiling.
- [ ] **8.H.7.2**: If justified, configure Optuna study (30–50 trials) optimizing validation RMSE on `src/models/train_rul.py`.
- [ ] **8.H.7.3**: Save best hyperparameters to model configuration.

---

### Task 8.H.8: Backtest selection-bias partial mitigation
- [ ] **8.H.8.1**: Stratify the 313 backtest decisions by stop-count divergence (recommended stops == actual stops vs different).
- [ ] **8.H.8.2**: For 2025 races with known realization, evaluate actual vs recommended strategy using realized lap times rather than model simulation.
- [ ] **8.H.8.3**: Document stratified win rates and realized-pace deltas in report.

---

### Task 8.H.9: Cliff threshold & RUL cap sensitivity analysis
- [ ] **8.H.9.1**: Configure grid: `CLIFF_THR` $\in [1.0, 1.5, 2.0]$s $\times$ sustain-window $\in [2, 3, 4]$ laps.
- [ ] **8.H.9.2**: Compute label generation sensitivity and cliff classifier performance metrics.
- [ ] **8.H.9.3**: Format results into sensitivity table in report.
