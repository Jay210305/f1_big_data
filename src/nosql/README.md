# MongoDB operational layer

Use `.venv/Scripts/python` on Windows. The default connection is
`mongodb://localhost:27017`, database `f1_tyre`; override with `MONGO_URI` /
`MONGO_DB` or the CLI's global `--uri` / `--db` options.

```powershell
python -m src.nosql.cli health
python -m src.models.export_labeled
python -m src.nosql.cli ingest
python -m src.nosql.cli indexes
python -m src.nosql.cli explain
python -m src.nosql.cli quality
python -m src.nosql.cli aggregate
python -m src.nosql.cli crud
python -m src.nosql.cli queries
python -m src.nosql.cli benchmark
```

Run ingest twice: counts stay fixed and the second JSON reports zero modified
and zero inserted documents. All bulk upserts use deterministic natural `_id`s,
5,000 records per call and `$set` excluding `_id`. `--season`, `--gp`, `--session`
scope ingestion to one session. The ingest retains every labeled-frame column;
`race_sessions.n_laps` comes from raw Gold, including laps excluded by model
filtering. `_pos` is calculated in life order before excluding SC laps.

`race_sessions` embeds Gold weather summaries, original Silver weather samples,
static circuit corners and raw-channel race control messages. `sc_laps` explicitly
records its derived source. `stints` uses Gold's actual schema (`cliff_laps` is a
count in the current artifact). Historic backtest decisions inherit season 2025,
scenario and simulation count from their run metadata, with compounds extracted
from the recommended strategy. The archived 200 rows are preserved without
inventing the 113 decisions omitted by its original writer.

MongoDB Server (`mongod`) owns storage, indexes and execution. Start it with
`docker run -d --name f1-mongo -p 127.0.0.1:27017:27017 -v f1mongo:/data/db mongo:7`.
Compass is an optional GUI client: connect to the URI and inspect `f1_tyre.laps`.
`mongosh` is a shell client; PyMongo is this package's Python client library.
For the same query shown by `python -m src.nosql.cli queries`, use:

```powershell
docker exec f1-mongo mongosh f1_tyre --quiet --eval 'db.laps.find({season:"2025",gp:"Bahrain Grand Prix",driver:"VER"},{_id:0,lap:1,laptime:1}).sort({lap:1}).limit(5).toArray()'
```

The query cookbook has eight real filters, projections, ranges and pagination.
CRUD runs in an isolated temporary collection and demonstrates insert, update,
find-and-update, `$inc`, `$setOnInsert`, BSON dates and precise delete. Destructive
commands require `reset --force` or `delete --collection ... --season ... --gp ...
--force`; delete prints its dry-run count without `--force`.

`explain` compares identical queries using `$natural` (collection scan) and an
explicit named index, preserving existing indexes. Six real executionStats
records include returned rows, examined documents/keys, time and winning plan.
The benchmark measures write throughput and collStats with/without three extra
indexes on dedicated temporary collections, then removes those collections.
It uses 100,000 deterministic documents, records their SHA-256, compares list
materialization with cursor batches 1/100/1000/10000, and reports a growth sweep.
Memory figures are peak Python allocations measured by tracemalloc, **not RSS**;
the measured 16 MiB budget and bounded-growth flags are reported truthfully.
An existing `bench_laps` cache must have exactly the requested `--n` count;
otherwise the command fails explicitly. To change its size, clear only the
synthetic collection using `docker exec f1-mongo mongosh f1_tyre --quiet --eval
'db.bench_laps.deleteMany({})'`, then rerun the benchmark with the new `--n`.
This deliberately leaves the four real lakehouse collections intact.

MongoDB can represent BSON NaN/Infinity; analytical ingestion deliberately maps
non-finite values to null so regression and serialization stay reproducible.
Canonical primitive coercion is reused from `src.ingest.schema`. Validators run
strictly, and `quality_report.json` counts validator violations, missing optional
fields and nonnumeric lap times; ingest JSON records all coercions.

`$group` implements the same key grouping/reduction as MapReduce in the storage
engine. The stint pipeline applies `_pos >= 3`, `is_sc_lap == 0`, minimum four
laps and a centered cov/var slope. The shared Phase 9 evidence command checks
that pipeline against MapReduce and Polars, on identical inputs.

This deployment is a local single node. Replication would add availability and
read scaling, plus majority-write latency and replica-set configuration.
Sharding on `{gp: "hashed"}` would distribute sessions and require reviewing
shard-key routing and uniqueness constraints; it is not deployed. Use the real
collection counts and collStats in the generated artifacts to assess whether
that cost is justified at this dataset's scale.

Tests: `python -m pytest tests/test_nosql.py`; live tests carry `mongo` and skip
when the server is unreachable. No mocked Mongo engine is used for evidence.
