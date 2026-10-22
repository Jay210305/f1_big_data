# F1 tyre strategy engine

An F1 telemetry lakehouse and tyre strategy engine: JSON → Bronze/Silver Parquet
→ Gold features → RUL/style/degradation models → Monte Carlo recommendations.
Phase 9 adds a local MapReduce runtime and a MongoDB operational store, with
independent parity checks against the columnar pipeline.

Use Python 3.11 or newer. Install `requirements-dev.txt` for CPU development,
or follow [RUNBOOK.md](RUNBOOK.md) for the local CUDA setup and model training.

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
python -m src.run_all
```

The ignored `2021/`–`2025/` raw datasets and `data/` artifacts must be supplied
locally; tests use small synthetic fixtures and do not need the lakehouse.

## Documentation

| Document | What it covers |
|---|---|
| [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) | The whole project end to end, against the current artifacts |
| [MAPREDUCE_Y_MONGODB.md](MAPREDUCE_Y_MONGODB.md) | MapReduce y MongoDB en detalle (ES) — the two Big Data layers, with the measured evidence |
| [WORK_LOG.md](WORK_LOG.md) | What was done, in order, with the numbers measured at each step |
| [RUNBOOK.md](RUNBOOK.md) | Every command, stage by stage |
| [PLAN.md](PLAN.md) · [PLAN2.md](PLAN2.md) · [PLAN3.md](PLAN3.md) | The three roadmaps |
| [F1_Data_Inventory.md](F1_Data_Inventory.md) | Every artifact and collection, with row counts |
| [src/mapreduce/README.md](src/mapreduce/README.md) · [src/nosql/README.md](src/nosql/README.md) | One-page guides for each layer |
| [document/phase9_traceability.md](document/phase9_traceability.md) | Requirement → implementation → evidence matrix |

## Verification

```powershell
python -m pytest -q                              # 157 passed
python -m src.models.calibrate                   # simulator vs the realized lap times
python -m src.mapreduce.evidence --silver --mongo # three-engine parity
```

See [PLAN3.md](PLAN3.md), [the runtime guide](src/mapreduce/README.md), and
[the Mongo guide](src/nosql/README.md) for the Phase 9 commands and evidence.
