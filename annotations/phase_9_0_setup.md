# Phase 9.0 — Environment and shared input (STOP)

- Date / author: 2026-10-21 / Codex
- Status: [x] Go  [ ] No-Go  [ ] Rework

## Files created

| File | Purpose |
|---|---|
| `src/models/export_labeled.py` | Shared materialized input with cluster join and row-count guard |
| `src/mapreduce/`, `src/nosql/` | Laboratory packages |
| `README.md`, `requirements-dev.txt` | Fix missing project metadata and CI dependency file |
| `pyproject.toml`, `requirements.txt`, `uv.lock` | Reproducible PyMongo dependency |

## Steps to run (in order)

```powershell
docker run -d --name f1-mongo -p 127.0.0.1:27017:27017 -v f1mongo:/data/db mongo:7
python -m src.models.export_labeled
python -m src.nosql.cli health
python -m src.mapreduce.cli --help
```

## What's going on

Both laboratories consume the same labeled Parquet export. Cluster joins are
many-to-one on season/driver, with missing clusters filled with Int32 -1. Cached
exports are checked too; upstream row-count changes fail rather than silently
changing the experiments. `--force` rebuilds; `--wet-context` adds optional context.

## After you run it

Use the shared export for MR stint/profile jobs and Mongo laps. No additional
ignore rules are necessary: `data/` covers both new generated directories.

## Results (filled after running)

- Baseline: 53 tests passed before implementation.
- Python 3.14.7 at `.venv/Scripts/python.exe`; Polars 1.44.1; Arrow 25.0.1;
  PyMongo 4.19.0.
- Docker MongoDB 7.0.45, local single node. Image digest:
  `sha256:c53852c00c23ca9f23fc39690350ffa9a80e3642fe8f96879726d8da0abe80a7`.
- Mongo's shell is provided by `docker exec f1-mongo mongosh`; no native MSI needed.
- Export: **107,606 rows × 85 columns**, from 138,449 candidate laps.
- Health: `OK MongoDB 7.0.45 database=f1_tyre`.
- CLI help runs. `uv lock` resolves the project after adding the missing README.
- Decisions: preserve existing Phase 8 changes in commit `2589372`; implement on
  `codex/phase9-mapreduce-mongodb`. Data/venv stay in the existing checkout.
- Environment: sandbox blocks localhost sockets and default temporary paths;
  live checks ran with approved execution access, tests use workspace basetemp.
- Next step: runtime gate, followed by real-data parity.
