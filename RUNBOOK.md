# RUNBOOK — F1 Prescriptive Tyre & Stint Strategy Engine

End-to-end pipeline: raw JSON telemetry (2021–2025, ~57 GB) → Bronze/Silver
lakehouse → Gold feature store → RUL/Cliff models → Driving-style profiler →
Pit-window optimizer → backtest.

All commands run from the repo root with the venv active
(`.\.venv\Scripts\Activate.ps1`).

## Quick path (skip-if-cached)

```powershell
python -m src.run_all              # runs only stages with missing artifacts
python -m src.run_all --stage etl  # single stage
python -m src.run_all --force      # full rebuild from scratch
```

## Stage-by-stage

| # | Stage | Command | Artifacts | Wall-clock |
|---|-------|---------|-----------|------------|
| 0 | Environment | `python src/verify_env.py` | venv + CUDA check | ~2 min (once) |
| 1 | Schema check | `python -m src.ingest.explore` | `data/bronze/phase1_schema_report.json` | ~1 min |
| 2 | ETL → lakehouse | `python -m src.ingest.etl --workers 8` | `data/bronze/telemetry/` (16.7 GB), `data/silver/` | ~11 min |
| 2b | Verify lakehouse | `python -m src.ingest.verify_lakehouse` | console report | ~30 s |
| 3 | Gold features | `python -m src.features.build_gold` | `data/gold/*.parquet` | ~6–8 min |
| 3b | Verify gold | `python -m src.features.verify_gold` | console report | ~10 s |
| 4 | RUL + cliff models | `python -m src.models.train_rul --epochs 40` | `data/models/{xgb_rul,xgb_rul_sym,lgb_rul,xgb_cliff,lgb_cliff,lstm_rul}.*` + metrics + `piecewise_blend` | ~30 s |
| 4b | Eval | `python -m src.models.eval_rul` | console + per-stint sample (XGB vs LSTM vs Blend) | ~30 s |
| 5 | Style profiler | `python -m src.features.micro_sectors` then `python -m src.models.train_style` | `data/gold/micro_sectors.parquet`, `driver_embeddings.parquet`, `lap_embeddings.parquet`, `style_ae.pt` | ~4 min + ~40 s |
| 5b | Verify | `python -m src.models.verify_style` | console report | ~5 s |
| 6 | Degradation fit | `python -m src.models.fit_degradation` | `data/models/degradation_model.json` | ~30 s |
| 6b | Optimizer demo | `python -m src.models.optimizer` | `data/models/phase6_demo.json` | ~35 s |
| 7 | Backtest | `python -m src.models.backtest` | `data/models/phase7_backtest.json` | ~40 s |
| 8.2.1 | SC/VSC hazard fit + optimizer | `python -m src.models.fit_degradation` then `python -m src.models.optimizer --sc-hazard` | `data/models/degradation_model.json` (`sc_hazard`) + `phase6_demo.json` | ~30 s + demo |

## Engineering-hygiene tools (cross-cutting)

| Task | Command | Artifacts |
|------|---------|-----------|
| Schema-drift validation | `python -m src.ingest.verify_schema` | `data/bronze/phase2_schema_verify.json` |
| Feature importance + SHAP | `python -m src.models.feature_importance` | `data/models/feature_importance/*` |
| Hyperparameter sanity/study | `python -m src.models.tune_rul [--study N]` | `data/models/tune_*.json` |
| Cliff-threshold sensitivity | `python -m src.models.sensitivity_cliff` | `data/models/cliff_sensitivity.{json,md}` |
| Realized-pace backtest | `python -m src.models.backtest_realized` | `data/models/phase7_backtest_realized.json` |
| Dependency benchmark | `python -m src.verify_deps` | console report |
| Tests + CI | `python -m pytest tests/` | `.github/workflows/ci.yml` |

## In-race recommendation (single query)

```powershell
python -m src.models.optimizer --season 2025 --gp "Bahrain Grand Prix" `
    --driver VER --lap 12 --n-sims 500 --pit-loss 21

# Include empirical SC/VSC windows and discounted pit loss (default: 21 / 3 s)
python -m src.models.optimizer --season 2025 --gp "Bahrain Grand Prix" `
  --driver VER --lap 12 --n-sims 500 --pit-loss 21 --sc-hazard
```

Outputs: ranked strategies per scenario (clean air / DRS train / backmarker;
optionally empirical SC/VSC hazard) with p5–p95 confidence bands, plus the pit-monitor backtest for that driver
(debounced cliff alerts + joint cliff/RUL anomaly rule).

## Tier 3 — pre-race and what-if

Pre-race planning uses the driver's longest available Practice 2 stints as
the pace prior (falling back to another Practice session when FP2 is absent).
The state starts at lap 0 and does not build the race-only labelled frame:

```powershell
python -m src.models.optimizer --pre-race --season 2025 `
  --gp "Bahrain Grand Prix" --driver VER --starting-compound MEDIUM --n-sims 500
```

Add the Tier 2 stochastic branch point to print separate green-flag and
SC/VSC-window plans, including p5-p95 bands and the SC-minus-green delta:

```powershell
python -m src.models.optimizer --pre-race --sc-hazard --season 2025 `
  --gp "Bahrain Grand Prix" --driver VER --starting-compound MEDIUM --n-sims 500
```
Compare two explicit plans with common random numbers. The first segment must
match the current compound and both plans must finish at `end`:

```powershell
python -m src.models.optimizer --season 2025 --gp "Bahrain Grand Prix" `
  --driver VER --lap 12 --compare `
  "HARD->18,MEDIUM->end" "HARD->23,SOFT->end" --n-sims 1000
```

The comparator prints the paired A minus B time delta, its 90% simulation
interval, and the probability that strategy A is faster. `laptime` is used as
the Practice pace proxy because `fuel_corr` is generated only for race labels.

## Tier 4 — wet weather and live timing

Refit the degradation model to include `INTERMEDIATE` and `WET` stints, then
run the wet-aware backtest:

```powershell
python -m src.models.fit_degradation
python -m src.models.backtest --n-sims 250
```

The live updater consumes one normalized lap at a time. An offline JSON replay
can be inspected with the dashboard; OpenF1 is an optional source adapter:

```powershell
streamlit run app/streamlit_app.py
```

See `annotations/phase_8_4_1_wet_weather.md` and
`annotations/phase_8_4_2_live_tool.md` for model assumptions and payload shape.

## Data footprint (after full run)

| Path | Size | Content |
|------|------|---------|
| `data/bronze/telemetry/` | 16.7 GB | 325.5M telemetry rows (ZSTD Parquet) |
| `data/silver/` | 0.04 GB | stint_features + race_events |
| `data/gold/` | ~0.5 GB | lap_features, micro_sectors, embeddings, etc. |
| `data/models/` | < 50 MB | all model artifacts + metrics |

## Reproducibility notes

- Every stage is idempotent (overwrites its outputs).
- Monte Carlo uses fixed seeds (optimizer rng 42; backtest paired rng 7).
- Model metrics are logged to `data/models/phase{4,5,6,7}_*.json`.
- Local phase annotations live in `annotations/` (gitignored).

## Known limitations (see annotations for details)

- Optimizer leans 1-stop vs actual field behaviour (1.33 vs 1.66 avg stops).
- Pit loss is circuit-constant (default 21 s).
- Scenario multipliers (clean air / DRS / backmarker) are priors, not fitted.
- Wet compounds use first-order rain and track-drying proxies; detailed
  rainfall intensity remains a future data-source improvement.
- Backtest comparison shares the simulator between recommender and evaluator
  (selection bias documented in `annotations/phase_7_report.md`).
