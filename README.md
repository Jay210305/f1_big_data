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
See [PLAN3.md](PLAN3.md), [the runtime guide](src/mapreduce/README.md), and
[the Mongo guide](src/nosql/README.md) for the Phase 9 commands and evidence.
