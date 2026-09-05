# Tier 4.2 — Live timing tool

Implemented components:

- `src/live/state_updater.py`: an incremental state machine that consumes one
  normalized lap at a time and exposes the fields expected by the optimizer.
  Fuel correction, pace-above-best, running track temperature, compound/stint
  transitions, cluster, and an online degradation slope require no lakehouse
  scan.
- `src/live/openf1.py`: optional standard-library HTTP adapter for OpenF1 lap
  records. It is deliberately separate from the updater so offline replay and
  tests do not require network access.
- `app/streamlit_app.py`: interactive dashboard for uploaded JSON lap streams,
  showing lap progression, pace above best, degradation slope, temperature, and
  compounds used.

Run locally with:

```powershell
streamlit run app/streamlit_app.py
```

The dashboard currently visualizes and updates state; strategy recommendation
execution remains an explicit integration point because it requires a fitted
`data/models/degradation_model.json` and the selected model artifacts. This
keeps the live path usable with partial timing data and prevents an accidental
full-lakehouse load on every update.
