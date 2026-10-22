# Practice 5 — Document-Oriented NoSQL Databases with MongoDB & Python

> Sub-plan of `PLAN3.md` (Phase 9, Tier 2). Every checkbox below is a task inside the
> `annotations/phase_9_2_mongodb.md` checkpoint.

## Goal Summary

Add a **document-oriented operational store** to the strategy engine: the semi-structured parts of
the F1 data that are currently flattened into rigid Parquet columns (race control messages, weather
samples, corner geometry, per-lap facts, optimizer decisions) are re-expressed as MongoDB
**documents** and exercised with the full CRUD + query-analysis + aggregation toolkit the guide
names — then cross-validated against the MapReduce implementation from Practice 3.

**Design constraint:** the real `mongod` server is required (the guide's `explain("executionStats")`,
index behaviour and BSON types cannot be emulated). `pymongo` is the only new dependency.

---

## 0. Environment

- [x] **P5.0** MongoDB running on `localhost:27017` (Docker container `f1-mongo` or native service).
- [x] **P5.1** `pymongo>=4.8` added to `requirements.txt` and `pyproject.toml`.
- [x] **P5.2** `python -m src.nosql.cli health` -> `OK` (server ping + version + db name).
- [x] **P5.3** Record the server version and deployment mode (docker/native) in the annotation —
  it goes into the report's reproducibility appendix.

---

## 1. Tooling distinction (the guide explicitly asks for this)

Document the four tools in `src/nosql/README.md` and in the report, each with a command the reader
can run:

| Tool | Role | Command used in this project |
|---|---|---|
| **MongoDB Server** (`mongod`) | the storage daemon/engine that owns the data files, the query planner and the oplog | `docker run ... mongo:7` / the Windows service |
| **Compass** | GUI for browsing collections, building indexes and running aggregation pipelines | optional, for the demo screenshots |
| **`mongosh`** | interactive CLI shell | `mongosh f1_tyre --eval "db.laps.explain('executionStats').executionStats"` |
| **PyMongo** | the Python driver — the application's data-access layer | `src/nosql/*.py` |

Show, with real output, the same query executed through `mongosh` and through PyMongo, and note
that Compass and `mongosh` are *clients* while `mongod` is the *server*.

---

## 2. Document model & collection design

### 2.1 Collections — exact sources and row counts

| Collection | Grain | `_id` (deterministic natural key) | Source artifact | Docs |
|---|---|---|---|---|
| `race_sessions` | one document per (season, gp, session) | `"<season>\|<gp>\|<session>"` | Silver `race_events/{weather,corners,rcm}` + Gold `environmental.parquet` | 576 |
| `stints` | one document per stint | `"<season>\|<gp>\|<session>\|<driver>\|<stint>"` | Gold `stint_degradation.parquet` | 44,298 |
| `laps` | one document per lap (flat fact table) | `"<season>\|<gp>\|<session>\|<driver>\|<lap>"` | Gold `labeled_frame.parquet` | 107,606 |
| `strategy_decisions` | one document per backtest decision | `"<gp>\|<driver>\|<scenario>"` | `data/models/phase7_backtest.json::decisions` | 200 (**verify** — see below) |
| `bench_laps` | synthetic benchmark documents | `"bench-<i>"` | `src/nosql/synthetic.py` | 100,000 |

> **Pre-existing discrepancy to verify before sizing anything:** `phase7_backtest.json` reports
> `summary.n_decisions = 313` but its `decisions` array holds **200** entries, and
> `src/eval/plots.py`'s Figure-4 docstring also says "313". Resolve this in Phase 9 and record the
> true number; the Mongo collection must hold whatever the array actually contains.

### 2.2 Document shapes

`race_sessions` — the flagship *embedded* document (this is where the document model beats
relational flattening: one race session is one coherent unit with nested sub-documents):

```json
{
  "_id": "2025|Bahrain Grand Prix|Race",
  "season": "2025", "gp": "Bahrain Grand Prix", "session": "Race",
  "session_type": "Race",
  "weather": { "track_temp_mean": 34.1, "ambient_temp_mean": 29.8, "rain_flag": 0,
               "wind_speed_mean": 14.2, "wind_dir_sin": -0.31, "wind_dir_cos": 0.95,
               "weather_samples": 412 },
  "corners": [ { "CornerNumber": 1, "X": 412.3, "Y": -118.0, "Angle": 0.87,
                 "Distance": 118.4, "Rotation": 0.62 }, ... ],
  "rcm": [ { "lap": 12, "time": 183422.5, "cat": "SafetyCar", "flag": "DOUBLE YELLOW",
             "msg": "YELLOW IN TRACK 1-4", "status": "CLOSED", "scope": "Track",
             "sector": null, "dNum": null }, ... ],
  "sc_laps": [3, 4, 5],
  "n_drivers": 20, "n_laps": 57
}
```

Field-name notes (these come from `src/ingest/schema.py` and must be exact):
- `corners` sub-documents use the **raw `corners.json` channel names**
  (`CORNERS_COLUMNS = ["CornerNumber", "X", "Y", "Angle", "Distance", "Rotation"]`) — they are
  **static circuit geometry only**; there is no per-corner speed / lateral-G / brake energy in the
  raw data.
- `rcm` sub-documents use the **raw `rcm.json` channel names**
  (`RCM_COLUMNS = ["time", "cat", "msg", "status", "flag", "scope", "sector", "dNum", "lap"]`).
- `weather` comes from the Gold `environmental.parquet` row for that session (it already carries
  `track_temp_mean`, `ambient_temp_mean`, `rain_flag`, `wind_speed_mean`, `wind_dir_sin`,
  `wind_dir_cos`, `weather_samples`).
- `sc_laps` is **derived**, not raw: compute it with `src/models/rul_data.py::load_sc_laps()`
  (the SC/VSC expansion logic). Label it as derived in the schema.

`laps` — flat, query-optimised fact document (source: `labeled_frame.parquet`):

```json
{ "_id": "2025|Bahrain Grand Prix|Race|VER|12",
  "season": "2025", "gp": "Bahrain Grand Prix", "session": "Race", "session_type": "Race",
  "driver": "VER", "lap": 12, "stint": 1, "life": 12, "compound": "SOFT",
  "laptime": 98.404, "fuel_corr": 98.764, "pace_above_best": 0.982,
  "speed_mean": 281.4, "throttle_mean": 62.1, "dirty_air_frac": 0.0197,
  "is_sc_lap": 0, "cluster": 0, "track_temp_mean": 34.1,
  "target_rul": 11, "cliff_within_3": 0 }
```

Field-name notes:
- The SC flag column is called **`is_sc_lap`** (int 0/1), not `is_sc` — it is added by
  `rul_data.py::filter_stints()`.
- `cluster` is **not** in `labeled_frame.parquet`; it is joined from
  `data/gold/driver_embeddings.parquet` (values `-1`, `0`, `1`). The export step (task 9.0.3)
  adds it, so the Mongo document can carry it.
- `fuel_corr`, `pace_above_best`, `target_rul`, `cliff_within_3` exist only in the labeled frame —
  sourcing `laps` from `lap_features.parquet` would break the `--source mongo` parity.

`stints` — one document per stint (source: `stint_degradation.parquet`, 44,298 rows):

```json
{ "_id": "2025|Bahrain Grand Prix|Race|VER|1",
  "season": "2025", "gp": "Bahrain Grand Prix", "session": "Race", "driver": "VER",
  "stint": 1, "compound": "SOFT", "stint_length": 14, "max_life": 14,
  "pace_start": 98.40, "pace_end": 99.67, "pace_decay_end": 1.27,
  "had_cliff": true, "cliff_laps": [10, 11, 12, 13], "mean_lap_delta": 0.091,
  "stint_track_temp": 34.1, "stint_rain": 0, "stint_lat_load": 32966.2,
  "stint_long_load": 18220.4, "degradation_slope": 0.0841 }
```

`strategy_decisions` — the decision record (source: `phase7_backtest.json::decisions`; the real
field names are listed below — do **not** invent `p5_s`/`confidence`, they do not exist):

```json
{ "_id": "Abu Dhabi Grand Prix|GAS|sc_hazard",
  "gp": "Abu Dhabi Grand Prix", "driver": "GAS",
  "actual_stops": 2, "rec_stops": 2, "stop_match": true,
  "actual_strategy": "MEDIUM->13 | HARD->40 | MEDIUM->57",
  "rec_strategy": "MEDIUM->24 | HARD->38 | HARD->58",
  "actual_mean": 4165.7, "rec_mean": 4241.4, "delta": -75.7,
  "scenario": "sc_hazard", "n_sims": 250 }
```

### 2.3 Schema validation & data hygiene

- [x] **P5.4** `src/nosql/schemas.py` — per-collection JSON-Schema used as a MongoDB
  `$jsonSchema` validator with `validationLevel: "strict"` and `validationAction: "error"`.
- [x] **P5.5** `src/nosql/cleaning.py` — a coercion layer applied **before** insert that
  **reuses the project's existing normalizers** (`src/ingest/schema.py::to_float / to_int /
  to_str` and its `NULL_TOKENS = {"", "None", "none", "null", "N/A", "nan"}`) instead of writing a
  second one. Additionally:
  - booleans stored as real BSON booleans, not `0/1`;
  - `NaN`/`Infinity` -> `null` (BSON has no NaN — a classic MongoDB trap worth calling out in the
    report; `fuel_corr` etc. can contain them after arithmetic);
  - dates stored as BSON `Date`, not strings.
- [x] **P5.6** **Data-quality report**: documents rejected by the validator, documents with missing
  optional fields, and the count of each coercion type. Save as
  `data/mongo/quality_report.json`.

---

## 3. CRUD

### 3.1 Create

- [x] **P5.7** `insert_one` for a single document (e.g. a manual `strategy_decisions` entry).
- [x] **P5.8** `insert_many` with **batched** writes (5,000 per call) for the bulk ingest — the
  guide's "batch ingestion" requirement. Measure per-batch wall-clock and docs/sec.

### 3.2 Read

- [x] **P5.9** `find_one` and `find` with **filters**: `$gt`, `$gte`, `$lt`, `$in`, `$ne`, `$exists`,
  plus compound conditions.
- [x] **P5.10** **Projections**: `{ "laptime": 1, "compound": 1, "_id": 0 }`.
- [x] **P5.11** `sort`, `limit`, `skip` usage.
- [x] **P5.12** A "query cookbook" of >= 8 realistic questions the app asks the store:
  - "all VER laps in the 2025 Bahrain race" -> filter,
  - "laps of a stint with `life` between 10 and 20" -> range,
  - "all SOFT stints with more than 15 laps" -> filter + `$gt`,
  - "SC-affected laps in 2024" -> `is_sc_lap`,
  - "drivers who pitted more than twice" -> aggregation,
  - etc.

### 3.3 Update

- [x] **P5.13** `update_one` with `$set` (e.g. attach the realised outcome to a
  `strategy_decisions` document after the backtest completes).
- [x] **P5.14** `upsert: True` — the idempotency mechanism. **Idempotency proof:** run the full
  ingest twice and assert (a) `count_documents` is unchanged, (b) the second run reports
  `modifiedCount == 0` and `upsertedCount == 0`, (c) a re-run of a single-session ingest updates
  only that session's documents.
- [x] **P5.15** `$inc`, `$setOnInsert`, `$currentDate` demonstrated on a counter document.

### 3.4 Delete

- [x] **P5.16** `delete_one` with a precise filter (delete one strategy decision).
- [x] **P5.17** A guarded bulk delete (`delete_many` with an explicit, narrow filter) behind a
  `--force` flag, with a dry-run count printed first — the guide's "safe deletions" requirement.
- [x] **P5.18** A `reset` command that drops the collections and recreates the indexes (used by the
  CI job and by a clean re-run).

---

## 4. Scale, indexing & query analysis

### 4.1 The synthetic benchmark dataset

- [x] **P5.19** `src/nosql/synthetic.py` — a **reproducible** generator: fixed seed
  (`random.Random(42)` / `numpy.random.default_rng(42)`), **100,000 documents** with a realistic F1
  flavour (drivers, compounds, stint structure, plausible lap times with degradation and noise).
- [x] **P5.20** Verify reproducibility: generate twice -> identical SHA-256 of the canonical JSON
  dump (record the hash in the annotation).
- [x] **P5.21** Load with `insert_many` in batches; report throughput (docs/s) and total time.

### 4.2 Streaming cursors vs in-memory loads

- [x] **P5.22** `src/nosql/stream.py` — iterate with `find(...).batch_size(n)` and a bounded-memory
  accumulator (never materialising the full result set).
- [x] **P5.23** **Demonstrate the failure mode**: a naive `list(cursor.find(...))` on the full
  collection is shown to blow the memory budget, whereas the batched cursor stays flat. Report peak
  RSS for both (use `tracemalloc` / `psutil`) on the 100k benchmark collection.
- [x] **P5.24** `batch_size` sweep (1 / 100 / 1,000 / 10,000) with wall-clock and memory per
  setting.

### 4.3 Indexes

- [x] **P5.25** `src/nosql/indexes.py` — `create_index` calls:
  | Collection | Index | Serves |
  |---|---|---|
  | `laps` | `{season:1, gp:1, driver:1, lap:1}` | per-driver lap history |
  | `laps` | `{compound:1, life:1}` | compound x tyre-age aggregations |
  | `laps` | `{laptime:1}` | slowest-lap queries |
  | `stints` | `{season:1, gp:1, driver:1}` | stint history |
  | `strategy_decisions` | `{gp:1, driver:1}` | decision history |
- [x] **P5.26** `explain_query()` — runs `.explain("executionStats")` and extracts
  `nReturned`, `totalDocsExamined`, `totalKeysExamined`, `executionTimeMillis`, and the winning plan
  stage (`COLLSCAN` / `IXSCAN` / `FETCH`).
- [x] **P5.27** **Before/after table** for >= 6 representative queries: capture the explain() output
  with the index dropped and with the index created; report the stage change and the
  `totalDocsExamined` reduction factor.
- [x] **P5.28** **Index cost**: `db.command("collStats", "laps")` -> `size`, `totalIndexSize`,
  `nindexes`; measure insert throughput and collection growth with vs without indexes -> the
  write-amplification / storage trade-off, with real numbers.

### 4.4 Scalability discussion (evidence-backed)

- [x] **P5.29** Measure and document: **local server** (single node, what we run) vs
  **replication** (a replica set gives HA + read scaling; not deployed here — explain the trade-off
  and what would change in the ingest path, e.g. write concern `w: "majority"`) vs **sharding**
  (horizontal partitioning by `gp` or by a hashed `driver`; would require the `_id` to be
  shard-key-aware and the `laps` collection to be sharded on `{gp: "hashed"}`).
- [x] **P5.30** State explicitly in the report which of the three the project uses (local) and why:
  the total document count is ~152k (576 sessions + 44,298 stints + 107,606 laps) plus a 100k
  synthetic benchmark collection — well within a single node's comfortable working set. Quote the
  measured document count and data size as the justification.

---

## 5. Aggregation pipelines

- [x] **P5.31** `src/nosql/aggregation.py` — the pipeline toolkit:
  | Stage | Operator | Purpose in this project |
  |---|---|---|
  | filter | `$match` | season/GP/driver/session scoping |
  | group | `$group` | key grouping + `$sum`/`$avg`/`$min`/`$max`/`$stdDevSamp` |
  | sort | `$sort` | ordering by a computed metric |
  | project | `$project` | shaping the output, `$cond`, computed fields |
  | unwind | `$unwind` | flattening `race_sessions.rcm[]` |
  | limit / skip | | pagination |
  | facet | `$facet` | multi-branch aggregation in one pass |
- [x] **P5.32** Pipeline 1 — per-compound mean lap time and degradation per season:
  `$match` -> `$group` by `{season, compound}` -> `$avg`/`$stdDevSamp` -> `$sort` -> `$project`.
- [x] **P5.33** Pipeline 2 — per-stint degradation slope, computed in-database (the
  MapReduce-equivalent job): `$match` -> `$group` by stint with `$push` of `{life, fuel_corr}` ->
  `$project` with `$reduce`/`$map` to run the slope inside the server
  (`cov(life, fuel_corr)/var(life)`, i.e. the OLS slope on `life`).
  **It must apply the same filters as Practice 3 job 2** (`_pos >= 3`, `is_sc_lap == 0`,
  `n_laps >= 4`, slope = `cov(life, fuel_corr)/var(life)`), otherwise the parity proof is
  vacuous.
- [x] **P5.34** Pipeline 3 — race-control message analytics via `$unwind` on `race_sessions.rcm`:
  count of SC/VSC periods per GP per season.
- [x] **P5.35** Pipeline 4 — `$facet` producing "worst 5 laps" and "average pace per compound" in a
  single pass.

### 5.1 Aggregation == MapReduce (the guide's explicit equivalence point)

- [x] **P5.36** Run the same per-stint aggregation through **three** implementations:
  (a) MapReduce job 2 (`src/mapreduce/jobs/stint_stats.py`),
  (b) MongoDB `$match`/`$group`/`$project`,
  (c) the existing Polars pipeline (`src/models/fit_degradation.py`).
  Assert all three agree within tolerance and report the parity table.
- [x] **P5.37** Write the explanatory paragraph (for the report): `$group` fulfils the *same
  conceptual role* as MapReduce's shuffle+reduce — values are grouped by key and reduced into one
  output per key — but the grouping is executed by the storage engine's own pipeline planner on a
  single (or sharded) server, not by a distributed compute cluster with map tasks, combiners and
  network shuffles. Where MapReduce wins: elastic parallelism over heterogeneous data; where the
  aggregation framework wins: in-place execution over indexed data, no network shuffle, single
  round-trip.

---

## 6. Deliverables & evidence table

| Guide requirement | Where it is demonstrated | Artifact |
|---|---|---|
| Document model / BSON / collections / `_id` | `src/nosql/schemas.py`, collection design §2 | Compass screenshot / `mongosh` listing |
| Server vs Compass vs `mongosh` vs PyMongo | §1 | commands + output |
| CRUD: create | `insert_one` / `insert_many` (batched) | ingest log |
| CRUD: read | `find` + filters + projections | query cookbook output |
| CRUD: update | `update_one` + `$set` + `upsert` | idempotency proof |
| CRUD: delete | `delete_one` / guarded `delete_many` | dry-run output |
| Batch ingestion, 100k synthetic, fixed seed | `src/nosql/synthetic.py` | SHA-256 + throughput |
| Streaming cursors vs in-memory | `src/nosql/stream.py` | memory table |
| Indexes (`createIndex`) | `src/nosql/indexes.py` | `getIndexes()` output |
| `.explain("executionStats")` | `explain_query()` | before/after table |
| Scalability trade-offs | §4.4 | `collStats` numbers |
| Aggregation pipelines | `src/nosql/aggregation.py` | pipeline outputs |
| Aggregation == MapReduce | §5.1 | three-way parity table |
| Data quality / hygiene | `src/nosql/cleaning.py` | quality report |
| Independent validation | §5.1 + `verify` tests | parity assertions |

---

## 7. Files to touch

| File | Action |
|---|---|
| `src/nosql/**` | **new** |
| `requirements.txt`, `pyproject.toml` | add `pymongo>=4.8` |
| `src/models/optimizer.py` | add `--source mongo` (Tier 3) |
| `app/streamlit_app.py` | add the strategy-history tab (Tier 3) |
| `src/eval/plots.py` | add the explain()/parity figures (Tier 3) |
| `tests/test_nosql.py` | **new** — marked `@pytest.mark.mongo` |
| `pyproject.toml` | register the `mongo` pytest marker |
| `.github/workflows/ci.yml` | add a `mongodb` service container (Tier 3) |
| `RUNBOOK.md` | add the MongoDB stage rows (Tier 3) |
| `F1_Data_Inventory.md` | add the collection inventory (Tier 4) |
