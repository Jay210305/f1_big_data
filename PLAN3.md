# Plan: Phase 9 — Distributed Batch Processing (MapReduce) & Document-Oriented NoSQL (MongoDB)

**Status: IMPLEMENTED — locally verified on 2026-10-21; all 25 audited defects closed and the reports reconciled on 2026-10-21**
**Evidence:** `annotations/phase_9_0_setup.md` through `phase_9_4_polish.md`;
see `annotations/phase_9_ledger.md` for specification rulings and review fixes.
The CI workflow is configured; a hosted run awaits a future push. Memory evidence
uses `tracemalloc` as permitted by the Practice 5 subplan, rather than claiming RSS.
The saved model contains 4,274 dry stints; the current shared-input estimator also
retains 354 wet stints. Historical decision history contains 200 of 313 evaluated
decisions; future writes now preserve every decision.
**Predecessor:** Phases 0–7 complete (`PLAN.md`) and Phase 8 complete (`PLAN2.md`, all tiers).
**Purpose of this file:** Master roadmap for the two laboratory guides the professor asked to be
integrated into the project — **Practice 3 (Distributed Batch Processing with MapReduce)** and
**Practice 5 (Document-Oriented NoSQL Databases with MongoDB & Python)** — implemented *on top of
the real F1 telemetry pipeline*, not as isolated toy demos, so they become defensible, gradeable
contributions to the existing thesis system.

**Methodology:** Same Stop-n-Go discipline as `PLAN.md` / `PLAN2.md`. Every sub-phase is a hard
checkpoint: implement → run → write `annotations/phase_9_X_*.md` with real results → Go/No-Go
decision → only then advance. Tiers are ordered by effort-to-payoff.

**Division of labor:** unchanged — the agent writes code + run instructions + explanation; the user
runs it locally (32 GB RAM / RTX 5070 Ti) and pastes results into the annotation file, which drives
the next Go/No-Go call.

---

## Why this is a real contribution (not decoration)

Both guides map onto genuine weaknesses of the current system, so the work is additive rather than
bolted-on:

| Guide topic | Real gap in this repo | Deliverable that closes it |
|---|---|---|
| MapReduce (k/v, map, shuffle&sort, reduce) | The ETL is *parallel* but never *expressed* in the k/v paradigm; there is no explicit shuffle/sort/partition stage to reason about | `src/mapreduce/` runtime + 3 real jobs re-implemented from the lakehouse |
| Combiners / partitioners | No local pre-aggregation; the Silver stage ships every telemetry row | Combiner-enabled job that cuts shuffled bytes by a measured % |
| Fault tolerance | ETL failures are handled by a bespoke `diagnose_errors.py` + manual `--fix-failed` re-run | Task-level retry + speculative-safe output commit in the MR runner, with a *demonstrated* injected-fault run |
| Trace & independent verification | Numbers in the report are self-reported by the same code that produced them | MR trace (`map→combine→shuffle→reduce`) **plus** an equivalence check against the existing Polars/DuckDB pipeline |
| MongoDB document model | `rcm.json` / `weather.json` / `corners.json` are semi-structured JSON flattened into rigid Parquet columns | `race_sessions` documents with embedded weather/corner/rcm sub-documents |
| CRUD + upsert idempotency | Re-running the ETL rewrites Parquet files wholesale; there is no row-level upsert | Idempotent `updateOne(..., upsert=True)` ingest keyed by a deterministic natural `_id` |
| Indexing + `.explain("executionStats")` | Parquet/DuckDB has no index-layer story; query cost is never measured | Indexed `laps` collection with a before/after `nReturned` / `totalDocsExamined` / `totalKeysExamined` table |
| Streaming cursors vs in-memory loads | Gold store is loaded whole into Polars in several places | `batch_size` cursor streaming demonstrated on a 100k-document collection with a memory ceiling |
| `$group` ≡ MapReduce reduce | No second implementation to cross-check the aggregation logic | Parity proof: Mongo `$match/$group/$project` **vs** MapReduce reduce **vs** Polars baseline — three implementations, one number |
| Scalability trade-offs | Only documented, never measured | Measured index write-overhead + collection size, replication/sharding discussion with real `collStats` |

**Thesis angle:** the final report (`document/main.tex`, Spanish) gains a section
*"§5 Arquitecturas Big Data: MapReduce y MongoDB"* presenting the two paradigms as the
*operational* layer of the strategy engine, with the parity proofs and the measured index/scaling
tables as its evidence.

---

## Verified facts about the current codebase (read this before implementing)

These were measured against the actual repo on the working tree — do not re-derive them, and do
not assume anything else about the schemas.

| Artifact | Shape | Role in Phase 9 |
|---|---|---|
| `data/gold/lap_features.parquet` | **317,264 rows × 61 cols** | Raw Gold facts. Has `laptime`, `compound`, `stint`, `life`, `dirty_air_frac`, `track_temp_mean`, `wind_dir_sin/cos`. **Does NOT have** `fuel_corr`, `pace_above_best`, `cluster`, `is_sc_lap`, `session_type`. |
| labeled frame (`build_labeled_frame()`) | **107,606 rows × 84 cols** | `Gold` + `session_type` + `lat_energy_integral` + `is_sc_lap` + `fuel_corr` + the `pace_roll_*` / `pace_above_best*` family + `target_rul` / `cliff_within_3`. **This is the frame `optimizer.get_state()` actually consumes.** Does NOT have `cluster` (joined from `driver_embeddings.parquet` separately). |
| stint filtering | 138,449 Race/Sprint laps → **107,606** (22.3 % removed) | The SC/VSC-contaminated stints are dropped upstream (Phase 8.1.2). |
| `data/gold/stint_degradation.parquet` | **44,298 rows** | Per-stint aggregates, **already contains `degradation_slope`** → the ready-made baseline for the MapReduce parity proof. |
| `data/gold/environmental.parquet` | **576 rows** (one per season/gp/session) | Source for the embedded `weather` sub-document. |
| `data/gold/driver_embeddings.parquet` | 172 rows | Source of `cluster` (−1 / 0 / 1). |
| `data/models/degradation_model.json` | `beta{SOFT,MEDIUM,HARD}`, `sigma`, `n_stints=4274`, `beta_global` | Parity target for MapReduce job 2. |
| `src/ingest/schema.py` | `NULL_TOKENS = {"", "None", "none", "null", "N/A", "nan"}`, `RCM_COLUMNS`, `CORNERS_COLUMNS`, `WEATHER_COLUMNS` | **Reuse** `to_float` / `to_int` / `to_str` — do not write a second cleaning layer. |
| `src/eval/plots.py` | `FIGURES` dict + `--only` CLI, `Agg` backend → `eval_plots/` | House style for the new figures. |
| `src/run_all.py` | `STAGES = [etl, verify_schema, gold, rul, style, degradation, optimizer]` | `backtest` is deliberately **not** a stage. New stages must skip gracefully. |
| `app/streamlit_app.py` | single page, sidebar, **early `return`** when no JSON is uploaded, **UI text is in Spanish** | The Mongo tab requires restructuring to `st.tabs(...)` — see `planning/practice3_5_integration.md` §2. |

> **Open discrepancy to resolve in Phase 9 (pre-existing, not introduced here):**
> `data/models/phase7_backtest.json` reports `summary.n_decisions = 313` but its `decisions` array
> holds **200** entries. `src/eval/plots.py`'s Figure-4 docstring also says "313". Verify before
> the Mongo `strategy_decisions` collection is sized or the report quotes the count.

---

## Prerequisites (Phase 9.0 — must be satisfied first)

| Requirement | Current state on this machine | Action |
|---|---|---|
| MongoDB server (`mongod`) | **Not installed** | Install (9.0.1) |
| MongoDB shell (`mongosh`) | **Not installed** | Install (9.0.1) |
| Docker | CLI present at `C:\Program Files\Docker\Docker\resources\bin\docker.exe`, **daemon not running** | Start Docker Desktop (9.0.1 Option B) |
| `pymongo` | **Not in venv** | `pip install pymongo` (9.0.1) |
| Python ≥ 3.11 venv | ✅ present (`.venv`) | — |
| Existing lakehouse (`data/bronze`, `data/silver`, `data/gold`) | ✅ present (16.7 GB bronze / 325.5 M rows, 317 k Gold laps, 107.6 k labeled-frame laps) | — |

> The MapReduce runtime has **zero new dependencies** — it is pure-Python stdlib + the existing
> `polars`/`pyarrow`, so it runs on Windows with no Hadoop/WSL. Only MongoDB needs a server.

---

## Priority tiers (why this order)

- **Tier 0 (setup, ~1 h):** install the MongoDB server, add `pymongo`, scaffold both packages,
  prove connectivity. Cheap, and everything else is blocked on it.
- **Tier 1 (Practice 3, ~4–6 h):** the MapReduce runtime and three real jobs. This is the harder
  and more valuable half — it is what a MapReduce lab is actually graded on.
- **Tier 2 (Practice 5, ~4–6 h):** MongoDB document model, CRUD, indexing + explain, aggregation
  pipelines, streaming cursors, synthetic benchmark.
- **Tier 3 (integration, ~3–4 h):** wire Mongo into the optimizer/Streamlit, add the parity
  proofs, generate the figures, write the report section, extend CI and the runbook.
- **Tier 4 (polish, ~1–2 h):** defense materials, final walkthrough, CI hardening.

---

## Tier 0 — Environment & scaffolding

### 9.0.1 MongoDB server + `pymongo`
- [x] **9.0.1.1** Install MongoDB. Two supported paths:
  - **Option A (Docker, recommended — CLI already present):** start Docker Desktop, then
    ```powershell
    docker run -d --name f1-mongo -p 27017:27017 -v f1mongo:/data/db mongo:7
    docker exec -it f1-mongo mongosh --eval "db.runCommand({ping:1})"
    ```
  - **Option B (native MSI):** MongoDB Community Server for Windows
    (`https://www.mongodb.com/try/download/community`) + MongoDB Shell (`mongosh`).
    Service starts automatically as `MongoDB Server` on `localhost:27017`.
- [x] **9.0.1.2** Add `pymongo>=4.8` to `requirements.txt` **and** `pyproject.toml` `dependencies`.
- [x] **9.0.1.3** Verify from Python: `MongoClient("mongodb://localhost:27017").admin.command("ping")`.
- [x] **9.0.1.4** Record the exact install path + version in the annotation (needed for the report's
  reproducibility appendix).

### 9.0.2 Package scaffolding
- [x] **9.0.2.1** Create `src/mapreduce/` (engine) and `src/nosql/` (Mongo layer) packages with
  `__init__.py` and docstrings.
- [x] **9.0.2.2** Create `src/mapreduce/jobs/` and `src/nosql/` sub-packages.
- [x] **9.0.2.3** Create `annotations/phase_9_0_setup.md` from `annotations/_TEMPLATE.md`.
- [x] **9.0.2.4** Add `data/mapreduce/` and `data/mongo/` to `.gitignore` (already covered by the
  `data/` rule — confirm).

### 9.0.3 Materialize the labeled frame (shared input for both labs)
The MapReduce jobs and the Mongo `laps` collection both need the **labeled frame** (107,606 rows
× 84 cols), which today only exists in memory inside `build_labeled_frame()`. Materialize it once so
both subsystems read the *same* file and the parity proof is meaningful.

- [x] **9.0.3.1** New `src/models/export_labeled.py` — writes
  `data/gold/labeled_frame.parquet`:
  - `build_labeled_frame()` (from `src/models/eval_rul.py`);
  - join `cluster` from `driver_embeddings.parquet` exactly as `fit_degradation.py` and
    `optimizer.main()` do (`fill_null(-1).cast(Int32)`);
  - optionally attach `rain_intensity` / `track_drying_proxy` from
    `fit_degradation.py::_wet_context` so the wet path is reproducible too.
- [x] **9.0.3.2** Idempotent: skip-if-cached, `--force` to rebuild; print the row count and assert
  it equals **107,606** (a regression here would silently change every downstream number).
- [x] **9.0.3.3** Wire it as a `run_all.py` stage `labeled` placed **before** any Phase 9 stage.

**Deliverable:** `annotations/phase_9_0_setup.md`
**DoD:** `python -m src.nosql.cli health` prints `OK` against a live server;
`python -m src.mapreduce.cli --help` runs; `data/gold/labeled_frame.parquet` exists with
107,606 rows × 85 cols.

---

## Tier 1 — Practice 3: Distributed Batch Processing with MapReduce

> Full task breakdown: [`planning/practice3_mapreduce.md`](planning/practice3_mapreduce.md)

### 9.1.1 The MapReduce runtime (`src/mapreduce/`)
- [x] **9.1.1.1** `types.py` — `KV` alias (`tuple[str, Any]`), `Mapper`/`Reducer`/`Combiner`/
  `Partitioner` protocols, `JobSpec` dataclass (name, mapper, reducer, combiner, partitioner,
  num_partitions, input_splitter).
- [x] **9.1.1.2** `partitioner.py` — `HashPartitioner` (stable hash → bucket),
  `RangePartitioner` (sorted key ranges → bucket), `IdentityPartitioner`.
- [x] **9.1.1.3** `runner.py` — `LocalJobRunner`: splits input into map tasks, runs map tasks in a
  `multiprocessing.Pool`, applies the combiner per partition, performs shuffle + sort, runs reduce
  tasks, writes output through a **commit protocol** (write to `_tmp/`, atomic rename on success).
- [x] **9.1.1.4** Fault tolerance in the runner — per-task retry with a max attempt count, error
  propagation that fails only the affected task, and a `--inject-fault` switch used to *prove*
  re-execution.
- [x] **9.1.1.5** `trace.py` — JSONL trace writer emitting one record per emitted KV at every stage
  (`map`, `combine`, `shuffle`, `reduce`) with the task id and partition id.
- [x] **9.1.1.6** `io.py` — input splitters for a Parquet dataset (by row-group / by file) and a
  JSONL sink.
- [x] **9.1.1.7** `cli.py` — `python -m src.mapreduce.cli run --job <name> [--input ...] [--trace]`.

**Deliverable:** `annotations/phase_9_1_mapreduce_runtime.md`
**DoD:** the runtime runs a canonical word-count-style job over a synthetic input, produces a
`map→combine→shuffle→reduce` trace, and survives an injected fault by retrying the task.

### 9.1.2 MapReduce job 1 — lap telemetry aggregation (Bronze → Silver)
- [x] **9.1.2.1** Mapper: one input record = one lap's telemetry arrays → emits
  `key = "season|gp|session|driver|lap"`, `value = {n_samples, speed_mean, throttle_mean, …}` —
  a **direct port of `src/ingest/etl.py::_silver_exprs`**. This job runs on the **raw JSON /
  Bronze** side, so it is independent of the labeled frame and exercises the full
  map→combine→shuffle→reduce path on 325M telemetry rows.
- [x] **9.1.2.2** Reducer: merge the per-lap values for a key into the Silver stint-feature row.
- [x] **9.1.2.3** Combiner: optional local pre-aggregation of the numeric stats per partition;
  measure and report the reduction in shuffled bytes.
- [x] **9.1.2.4** Run on ≥ 1 season and diff the output against the existing Silver parquet
  (`data/silver/stint_features/season=YYYY/*.parquet`).

### 9.1.3 MapReduce job 2 — per-stint degradation statistics
- [x] **9.1.3.1** Mapper: lap → `key = "season|gp|session|driver|stint"`,
  `value = {life, fuel_corr, compound, dirty_air_frac, is_sc_lap, track_temp_mean, cluster}`.
  Input = `data/gold/labeled_frame.parquet` (from 9.0.3), **not** `lap_features.parquet` —
  `fuel_corr` and `is_sc_lap` exist only in the labeled frame.
- [x] **9.1.3.2** Reducer: per stint compute the **exact** estimator used by
  `src/models/fit_degradation.py` (lines 278–298) so the parity is meaningful:
  - drop laps with stint position `_pos < 3` (out-lap + first two flying laps — warm-up and
    race-start traffic bias the slope negative and trigger false cliffs);
  - drop laps with `is_sc_lap != 0`;
  - `beta = cov(life, fuel_corr) / var(life)` over the surviving laps;
  - keep the stint only if `n_laps >= 4`;
  - also emit `best_pace` (`stint_best_pace`), `mean_dirty_air_frac`, `track_temp_mean`.
- [x] **9.1.3.3** Model-level roll-up: `beta[compound] = median(per-stint beta)` and
  `sigma[compound] = (Q3 − Q1) / 1.349` (IQR-robust), matching `degradation_model.json`.
  This is what makes the guide's content real: the MapReduce reduce output **is** the degradation
  model that drives the optimizer.
- [x] **9.1.3.4** Parity target: `degradation_model.json["beta"][compound]` and
  `["sigma"][compound]` — expected to match to floating-point tolerance because the estimator is
  identical.

### 9.1.4 MapReduce job 3 — driver/session profile
- [x] **9.1.4.1** Mapper: lap → `key = "driver|session_type"`.
- [x] **9.1.4.2** Reducer: aggregate `n_laps`, mean/median pace, degradation slope, SC-lap share,
  dirty-air share → compare with `src/models/style_data.py`'s driver-level aggregates
  (`data/gold/driver_style.parquet`, 178 rows).

### 9.1.5 Trace verification & independent equivalence proof
- [x] **9.1.5.1** `src/mapreduce/verify.py` — compute and assert, on the trace:
  - map input records == sum of map output KV counts;
  - distinct shuffle keys == number of non-empty reduce partitions;
  - values fed to each reducer == values emitted by the mappers for those keys (**no omitted or
    duplicated records**);
  - per-key reduce output == an independent Python reference reduction of the same key's values.
- [x] **9.1.5.2** Cross-implementation parity table: MapReduce β vs the Polars/DuckDB baseline
  (`fit_degradation.py`'s frame) vs `degradation_model.json` β, per compound, with max absolute
  difference; plus a per-stint comparison against the pre-computed
  `data/gold/stint_degradation.parquet::degradation_slope` (44,298 rows).
- [x] **9.1.5.3** Fault-tolerance demonstration run: inject a failure in map task *k*, show the
  retry, and show the final output is byte-identical to the fault-free run.

**Deliverable:** `annotations/phase_9_1_mapreduce.md`
**DoD:** three real jobs run on real telemetry; the trace verification passes all four assertions;
the three-way parity table is within tolerance; the injected-fault run produces identical output.

---

## Tier 2 — Practice 5: Document-Oriented NoSQL with MongoDB

> Full task breakdown: [`planning/practice5_mongodb.md`](planning/practice5_mongodb.md)

### 9.2.1 Document model & schema validation
- [x] **9.2.1.1** Collection design (see `planning/practice5_mongodb.md` §2), with the **exact**
  source artifact and row count for each:
  | Collection | Source | Docs |
  |---|---|---|
  | `race_sessions` | Silver `race_events/{weather,corners,rcm}` + Gold `environmental.parquet` | 576 |
  | `stints` | Gold `stint_degradation.parquet` | 44,298 |
  | `laps` | Gold `labeled_frame.parquet` (from 9.0.3) | 107,606 |
  | `strategy_decisions` | `data/models/phase7_backtest.json::decisions` | 200 (verify — summary claims 313) |
  | `bench_laps` | `src/nosql/synthetic.py` | 100,000 |
- [x] **9.2.1.2** `src/nosql/schemas.py` — JSON-Schema validators per collection
  (`$jsonSchema` + `validationLevel: strict`) and a coercion layer that **reuses**
  `src/ingest/schema.py::to_float / to_int / to_str` and its `NULL_TOKENS`
  (`{"", "None", "none", "null", "N/A", "nan"}`) — do not write a second normalizer.
- [x] **9.2.1.3** Deterministic natural `_id` for every collection (e.g.
  `f"{season}|{gp}|{session}|{driver}|{lap}"`) so re-ingestion is idempotent.

### 9.2.2 CRUD + idempotent upsert ingest
- [x] **9.2.2.1** `src/nosql/ingest.py` — read the lakehouse (Silver + Gold) and upsert into Mongo
  in batches of 5,000 with `update_one(filter, {"$set": doc}, upsert=True)`.
- [x] **9.2.2.2** `src/nosql/queries.py` — `find` with filters (`$gt`, `$gte`, `$in`, `$ne`) and
  projections; `updateOne` with `$set`; `deleteOne`; a "find one and update" flow.
- [x] **9.2.2.3** Idempotency test: run the ingest twice, assert `collection.count_documents()` is
  unchanged and `modifiedCount == 0` on the second run.
- [x] **9.2.2.4** Data-quality sweep: count documents violating the validator, missing optional
  fields, and non-numeric `laptime`; report the numbers.

### 9.2.3 Indexing & `.explain("executionStats")`
- [x] **9.2.3.1** `src/nosql/indexes.py` — create the index set:
  `laps`: `{season:1, gp:1, driver:1, lap:1}`, `{compound:1, life:1}`, `{laptime:1}`;
  `strategy_decisions`: `{season:1, gp:1, driver:1}`; `stints`: `{season:1, gp:1, driver:1}`.
- [x] **9.2.3.2** `explain_query()` helper that runs `.explain("executionStats")` and extracts
  `nReturned`, `totalDocsExamined`, `totalKeysExamined`, `executionTimeMillis`, and the winning
  stage (`COLLSCAN` vs `IXSCAN`).
- [x] **9.2.3.3** Before/after table for ≥ 5 representative queries (COLLSCAN → IXSCAN).
- [x] **9.2.3.4** Measure index overhead with `collStats`: collection size and index size with and
  without indexes; report the write/storage cost.

### 9.2.4 Aggregation pipelines, streaming cursors, synthetic benchmark
- [x] **9.2.4.1** `src/nosql/aggregation.py` — `$match` → `$group` → `$sort` → `$project` pipelines
  reproducing: mean lap time per compound per season, per-stint degradation slope, SC-lap share.
- [x] **9.2.4.2** `src/nosql/stream.py` — `find(...).batch_size(n)` streaming with a bounded-memory
  accumulator; assert peak RSS stays flat as the collection grows.
- [x] **9.2.4.3** `src/nosql/synthetic.py` — reproducible synthetic generator (fixed seed,
  100,000 documents) with a realistic F1 flavour (driver/compound/lap/time), loaded with
  `insert_many` in batches.
- [x] **9.2.4.4** Benchmark: batched-insert throughput, cursor vs in-memory load wall-clock and
  peak memory.

### 9.2.5 Aggregation ≡ MapReduce parity proof
- [x] **9.2.5.1** Run the same per-stint aggregation through (a) MapReduce job 2, (b) MongoDB
  `$group`, (c) the existing Polars pipeline (`fit_degradation.py`); assert all three agree within
  tolerance. Use the **same** estimator definition as 9.1.3.2 — a Mongo `$group` that computes a
  different slope would make the parity proof vacuous.
- [x] **9.2.5.2** Write the explanatory note: `$group` is the engine-native equivalent of
  shuffle+reduce — same k/v semantics, but executed by a specialized storage engine rather than a
  distributed compute cluster.

**Deliverable:** `annotations/phase_9_2_mongodb.md`
**DoD:** MongoDB holds the lakehouse in document form; CRUD + upsert idempotency proven; the
explain() before/after table shows real index usage; the three-way parity proof passes.

---

## Tier 3 — Integration into the existing system

> Full task breakdown: [`planning/practice3_5_integration.md`](planning/practice3_5_integration.md)

### 9.3.1 MongoDB as an operational state source for the optimizer
- [x] **9.3.1.1** `src/models/optimizer.py --source mongo` — `get_state()` currently takes the
  **labeled frame** (`df`) plus `lapf` (Gold `lap_features`, used only to compute `n_laps`). The
  Mongo path must therefore serve the *labeled-frame* columns, not the raw Gold ones: `fuel_corr`,
  `pace_above_best`, `compound`, `stint`, `life`, `track_temp_mean`, `cluster`, plus `session`,
  `driver`, `lap`, `season`, `gp`. `cluster` is joined from `driver_embeddings.parquet` — it must
  be present in the Mongo document too (the export in 9.0.3 adds it).
- [x] **9.3.1.2** `--source` flag plumbed through `main()` and documented in `RUNBOOK.md`.

### 9.3.2 Streamlit dashboard tab
- [x] **9.3.2.1** `app/streamlit_app.py` — the app is currently a **single page** with an **early
  `return`** when no JSON is uploaded, and all UI strings are **in Spanish**. Restructure to
  `st.tabs(["Vivo", "Historial MongoDB"])`, move the early return *inside* the first tab, and add
  the Mongo tab (paginated `strategy_decisions` documents, a compound filter, a delta histogram
  computed server-side by an aggregation pipeline, and a `collStats` panel). Graceful banner when
  the server is down.

### 9.3.3 Figures for the report
- [x] **9.3.3.1** `src/eval/plots.py` — three new figures (registered in the existing `FIGURES`
  dict, `--only` CLI, output to `eval_plots/`):
  (a) MapReduce shuffle/sort trace: KV count per reducer partition (with and without combiner);
  (b) MongoDB `explain()` before/after index: `totalDocsExamined` bar chart;
  (c) Three-way parity plot: MapReduce β vs Polars β vs Mongo `$group` β.
- [x] **9.3.3.2** Output to `eval_plots/` (already gitignored) — confirm with the same convention as
  `correlation_maps/`.

### 9.3.4 Report + runbook + `run_all.py` + CI
- [x] **9.3.4.1** `document/main.tex` — new Spanish section *"§5 Arquitecturas Big Data: MapReduce y
  MongoDB"*: architecture diagram, the parity table, the explain() table, the index-overhead table,
  and the figures.
- [x] **9.3.4.2** `document/strategy_engine_report.tex` — a short English subsection with the same
  evidence.
- [x] **9.3.4.3** `RUNBOOK.md` — new stage rows for the MapReduce and MongoDB stages.
- [x] **9.3.4.4** `src/run_all.py` — add the `labeled`, `mapreduce` and `mongo` stages. The `mongo`
  stage must **skip gracefully** when the server is unreachable so the parquet pipeline never
  breaks because MongoDB is off. Do **not** add `backtest` (deliberately manual).
- [x] **9.3.4.5** `.github/workflows/ci.yml` — add a `mongodb` service container; mark the
  Mongo-dependent tests `@pytest.mark.mongo` so they skip cleanly when the server is absent.
- [x] **9.3.4.6** `tests/` — new test files `test_mapreduce.py`, `test_nosql.py`.

**Deliverable:** `annotations/phase_9_3_integration.md`
**DoD:** `--source mongo` produces the same recommendation as the parquet path; CI is green; the
report contains the new section with real numbers.

---

## Tier 4 — Polish & defense

### 9.4.1 Documentation & reproducibility
- [x] **9.4.1.1** `src/mapreduce/README.md` and `src/nosql/README.md` — one-page guides.
- [x] **9.4.1.2** `F1_Data_Inventory.md` — add the MongoDB collection inventory.
- [x] **9.4.1.3** Verify the full `python -m src.run_all` quick path still passes after the changes.

### 9.4.2 Defense materials
- [x] **9.4.2.1** `document/phase9_defense_notes.md` — 10-slide outline mapping each guide
  requirement to the exact file/figure that demonstrates it (this is the sheet to present to the
  professor).
- [x] **9.4.2.2** A requirements → implementation traceability matrix (guide bullet → file → test →
  figure).

**Deliverable:** `annotations/phase_9_4_polish.md`
**DoD:** a fresh clone + the runbook reproduces every number shown to the professor.

---

## Suggested execution order

1. **9.0** environment + scaffolding + **export the labeled frame** (blocks everything else —
   both labs read `data/gold/labeled_frame.parquet`).
2. **9.1** MapReduce runtime → jobs 1–3 → trace verification + parity (Practice 3 is graded on the
   *runtime*, so do not start the Mongo half until the runtime and its verification pass).
3. **9.2** MongoDB: schema → CRUD/upsert → indexes/explain → aggregation/cursors → parity.
4. **9.3** integration (optimizer `--source mongo`, Streamlit, plots, report, `run_all.py`, CI).
5. **9.4** polish + defense notes.

Do not reorder the tiers without a documented reason — the parity proofs in 9.1.5 and 9.2.5 are the
highest-value single artifacts for both the labs and the thesis.

---

## Carried-over open issues (fix during Phase 9, do not let them silently drift)

- [x] **9.X.1** `data/models/phase7_backtest.json` reports `summary.n_decisions = 313` but its
  `decisions` array holds **200** entries; `src/eval/plots.py`'s Figure-4 docstring also says
  "313". Resolve before sizing the Mongo `strategy_decisions` collection or quoting the count in
  the report.
- [x] **9.X.2** `pyproject.toml` declares `readme = "README.md"` but **no `README.md` exists at the
  repo root** — `uv sync` / any build will fail. Create it (the project has no README at all).
- [x] **9.X.3** `PLAN2.md` still says *"Status: NOT STARTED"* at the top although every tier
  shipped. Update it (and add a Phase 9 status line) so the roadmap is not self-contradicting.
- [x] **9.X.4** The Phase 8 feature-correction change set (`src/features/build_gold.py`,
  `src/models/rul_data.py`, `src/features/FEATURE_DICTIONARY.md`, `tests/test_rul_data.py`,
  `correlation_maps/*`) plus `annotations/phase_8_feature_corrections_walkthrough.md` are still
  **uncommitted**. Commit them before starting Phase 9.
- [x] **9.X.5** `document/main.tex` and `document/strategy_engine_report.tex` still quote the
  pre-correction numbers (cliff F1 = 0.747 / AUC = 0.889, XGB R² = 0.35, MAE_crit = 2.51, win rate
  78.6 %) while the current artifacts say F1 = 0.680 / AUC = 0.906, R² = 0.313 (sym 0.359),
  MAE_crit = 3.075 (LGB 2.670, LSTM 2.838), win rate 78.9 % (`sc_hazard`). Reconcile before Phase 9
  adds a third set of numbers to the report.

---

## Annotation template (reuse from `PLAN.md`)

```markdown
# Phase 9.X — <name> (STOP)

- Date / author:
- Status: [ ] Go  [ ] No-Go  [ ] Rework

## Files created

| File | Purpose |
|------|---------|
| ... | ... |

## Steps to run (in order)

```powershell
# ...
```

## What's going on

<!-- plain-language explanation of what the code does and why -->

## After you run it

<!-- what to do with the output, what to paste back -->

## Results (filled after running)

- Commands run (copy/paste):
- Results / output (from user):
- Metrics / numbers:
- Errors / blockers:
- Decisions made:
- Next-phase risks:
- Time spent / next step:
```
