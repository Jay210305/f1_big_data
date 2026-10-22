# Phase 9.3 — System integration (STOP)

- Date / author: 2026-10-21 / Codex
- Status: [x] Go  [ ] No-Go  [ ] Rework

## Files created / updated

Optimizer repository source, two-tab Streamlit app, three registered plotting
functions, both LaTeX reports, `src/run_all.py`, CI Mongo service, runbook,
`data/models/phase9_tables.json` exporter and integration tests.

## Steps to run (in order)

```powershell
python -m src.mapreduce.evidence --silver --mongo
python -m src.models.optimizer --source mongo --season 2025 --gp "Bahrain Grand Prix" --driver VER --lap 12 --n-sims 50
streamlit run app/streamlit_app.py
python -m src.eval.plots --only mapreduce_shuffle
python -m src.eval.plots --only mongo_indexes
python -m src.eval.plots --only three_way_parity
python -m src.run_all
python -m pytest -q
```

## What's going on

The optimizer accepts Mongo history and the raw session length through the same
state builder. It logs the chosen source and falls back to Parquet only on a
connection failure. A connected but missing dataset is an ingestion error.
The history tab renders even without a live upload; filters, pagination,
histogram grouping and collection statistics run against Mongo.

## After you run it

Review `document/main.tex`, `document/strategy_engine_report.tex`, the three
PNG figures in `eval_plots/`, and the JSON table snapshot. Generated figures,
data and PDFs remain ignored; source and regeneration commands are retained.

## Results (filled after running)

- Five real 2025 driver/lap states match field-for-field between Mongo and
  Parquet, including unfiltered race length; their seeded recommendations match.
- The exact Bahrain/VER/lap-12 Mongo CLI completes, including the Phase 4 pit
  monitor and saved demo. Its ranked strategy and pit-monitor lines are byte-
  identical to the matching Parquet CLI run. The selected source is logged.
- Streamlit AppTest: both Spanish tabs render with no uploaded file, no
  exceptions; live Mongo table renders. Server-blocked run shows the graceful
  unavailable banner and the live tab remains usable.
- Cached quick pipeline: **36 seconds** with live idempotent Mongo ingest;
  **4 seconds** with unavailable Mongo gracefully skipped. Backtest stays manual.
- Final full suite: **86 passed**, including live Mongo tests; compileall passes.
- Native LaTeX compiles both reports with no unresolved references. The app
  compiler timed out, so approved native MiKTeX execution supplied the evidence.
- Reports reconcile current Phase 4 metrics, 78.9% backtest win rate, 14.4-second
  median gain, and the 313 evaluated / 200 persisted distinction.
- Next step: defense and traceability notes.
