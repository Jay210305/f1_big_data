# Phase 9 requirement → implementation → evidence

All commands are run from the repository root with the virtual environment
active. Data and figures are regenerable local evidence. The report's measured
scope is one computer and one MongoDB node.

| Requirement | Implementation | Verification / evidence | Figure / report |
|---|---|---|---|
| Shared labeled input | `src/models/export_labeled.py` | `python -m src.models.export_labeled`; 107,606 × 85; `tests/test_phase9_integration.py` | report architecture |
| Typed MapReduce stages | `src/mapreduce/{types,runner,trace,io}.py` | `tests/test_mapreduce.py`; complete stint trace | `mapreduce_shuffle.png` |
| Stable hash/range/identity partitioning | `src/mapreduce/partitioner.py` | partition tests, output metrics, per-partition counts | report runtime |
| Associative combiner | `src/mapreduce/jobs/lap_aggregate.py` | plain/combined Silver outputs; metrics: 32.09% fewer bytes | `mapreduce_shuffle.png` |
| Task retry and atomic commit | `src/mapreduce/runner.py` | injected map-3 first-attempt fault; four final hashes equal baseline | defense slide 5 |
| Three real jobs | `src/mapreduce/jobs/` | `python -m src.mapreduce.cli all`; 69,459 Silver laps, 4,628 eligible stints, 70 driver/session groups | report verification |
| Full trace conservation | `src/mapreduce/verify.py` | 107,606 inputs / 106,735 emissions / 4,981 keys / four nonempty partitions | report verification |
| Independent MR/Polars parity | `src/mapreduce/evidence.py` | `python -m src.mapreduce.evidence --silver`; `data/mapreduce/parity.json` | compound table; `three_way_parity.png` |
| Embedded document model + strict validators | `src/nosql/{schemas,ingest}.py` | `tests/test_nosql.py`; `python -m src.nosql.cli quality` | report Mongo model |
| CRUD, filters and projection | `src/nosql/queries.py` | `python -m src.nosql.cli crud`; `queries`; live tests | defense slide 7 |
| Idempotent batched upserts | `src/nosql/ingest.py` | first/second ingest manifests; same counts, second modified/upserted zero | report Mongo ingest |
| Index effectiveness and explain | `src/nosql/indexes.py` | `python -m src.nosql.cli explain`; docs/keys examined and plan per query | explain table; `mongo_indexes.png` |
| Cursor streaming + synthetic insertion | `src/nosql/{synthetic,benchmark}.py` | `benchmark --n 100000`; timings, memory, collection/index sizes | storage table; defense slide 8 |
| Server-side grouping | `src/nosql/aggregation.py` | `aggregate`; pipeline tests; `evidence --mongo --silver` | `three_way_parity.png` |
| Mongo optimizer source | `src/nosql/repository.py`, `src/models/optimizer.py` | `tests/test_phase9_integration.py`; >=5 state-query parity cases; logged fallback | report integration |
| History without upload | `app/streamlit_app.py` | Spanish tabs, server-side histogram, pagination and unavailable-server behavior | defense slide 9 |
| Optional orchestration | `src/run_all.py` | cached stages; unavailable Mongo skips; backtest stays manual | `RUNBOOK.md` |
| Pure + live CI coverage | `.github/workflows/ci.yml`, `pyproject.toml`, `tests/` | Mongo service and marker; pure tests always run; live tests need server | annotations |
| Evidence figures and report tables | `src/eval/plots.py`, `src/mapreduce/evidence.py` | three PNGs and local JSON evidence; report compile | Spanish/English reports |
| Defense, reproducibility and status | `RUNBOOK.md`, `F1_Data_Inventory.md`, `PLAN2.md`, defense notes | commands/schema/counts checked against artifacts | ten-slide outline |

## Interpretation boundaries

- `stint_degradation.parquet` has 44,298 raw Gold slopes. Its population and
  target differ from filtered fuel-corrected MR slopes; their differences are
  diagnostic, not failed equivalence. The current driver-style table's 178
  season/driver rows likewise differ from 70 labeled driver/session groups.
- Saved dry-compound beta/sigma match. Missing saved wet parameters mean global
  saved-model equality is false; compare wet MR/Mongo/Polars on shared inputs.
- Full trace verification allows zero emissions after filtering. Distinct key
  count does not equal the number of reducers. Sampled traces are illustrative.
- Existing backtest summary: 313 decisions, 21 races, `sc_hazard`, 78.9% wins and
  +14.4 s median. Stored history: 200. No attempt is made to invent the other 113.
- Replication, sharding and multi-host throughput/failover are discussion only.
  Index bytes, collection bytes, latency and client-memory observations apply
  to the measured local experiment.
