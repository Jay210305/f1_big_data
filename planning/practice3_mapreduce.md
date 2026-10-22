# Practice 3 — Distributed Batch Processing with MapReduce

> Sub-plan of `PLAN3.md` (Phase 9, Tier 1). Every checkbox below is a task inside the
> `annotations/phase_9_1_mapreduce*.md` checkpoints.

## Goal Summary

Build a **from-scratch, dependency-free MapReduce runtime** in `src/mapreduce/` that implements every
concept the guide names — key/value abstraction, map, shuffle & sort, reduce, combiners,
partitioners, fault tolerance, parallelism, trace simulation and independent verification — and
then run **three real jobs over the actual F1 telemetry lakehouse** so the lab content is
demonstrated on the project's own data rather than on a toy text corpus.

**Design constraint:** no Hadoop, no WSL, no `mrjob`. Pure Python + the existing `polars`/`pyarrow`
stack so it runs natively on Windows and is unit-testable without the 57 GB dataset.

---

## 0. Inputs available (measured, do not re-derive)

| Input | Shape | Used by |
|---|---|---|
| `data/bronze/telemetry/season=YYYY/*.parquet` | 325.5 M rows, 16.7 GB | Job 1 |
| `data/silver/stint_features/season=YYYY/*.parquet` | per-session Silver features | Job 1 (parity target) |
| `data/gold/labeled_frame.parquet` | **107,606 rows x 85 cols** (built by task 9.0.3) | Job 2, Job 3 |
| `data/gold/stint_degradation.parquet` | 44,298 rows, has `degradation_slope` | Job 2 (parity target) |
| `data/gold/driver_style.parquet` | 178 rows | Job 3 (parity target) |
| `data/models/degradation_model.json` | `beta{SOFT,MEDIUM,HARD}`, `sigma`, `n_stints=4274` | Job 2 (parity target) |

The labeled frame (`build_labeled_frame()`) is the **only** place where `fuel_corr`,
`pace_above_best`, `is_sc_lap` and `session_type` exist — `lap_features.parquet` (317,264 x 61)
does not have them. Job 2 and Job 3 therefore read `labeled_frame.parquet`, never
`lap_features.parquet`.

---

## 1. The runtime — file-by-file contract

```
src/mapreduce/
├── __init__.py          # public API: JobSpec, run_job
├── types.py             # KV, Mapper, Reducer, Combiner, Partitioner protocols, JobSpec
├── partitioner.py       # HashPartitioner / RangePartitioner / IdentityPartitioner
├── runner.py            # LocalJobRunner: scheduling, retries, shuffle, commit
├── trace.py             # JSONL trace writer/reader
├── io.py                # input splitters (parquet / jsonl) + output sinks
├── verify.py            # trace assertions + cross-implementation parity
├── cli.py               # python -m src.mapreduce.cli
└── jobs/
    ├── lap_aggregate.py     # Job 1: telemetry lap  -> silver stint features
    ├── stint_stats.py       # Job 2: laps          -> per-stint degradation stats
    └── driver_profile.py    # Job 3: laps          -> per-driver/session profile
```

### 1.1 Core abstractions (`types.py`)

```python
KV = tuple[str, Any]          # key must be hashable + orderable (str)

class Mapper(Protocol):
    def __call__(self, record: dict, ctx: "MapContext") -> list[KV]: ...

class Reducer(Protocol):
    def __call__(self, key: str, values: list[Any]) -> list[KV]: ...

class Combiner(Protocol):
    """Same shape as the Mapper, but operates on already-emitted values."""
    def __call__(self, key: str, value: Any) -> Any: ...

class Partitioner(Protocol):
    def __call__(self, key: str, num_partitions: int) -> int: ...
```

`JobSpec` dataclass:

| Field | Meaning |
|---|---|
| `name` | job id, used for output/trace directories |
| `mapper` / `reducer` | callables |
| `combiner` | optional `Combiner` (local pre-aggregation) |
| `partitioner` | default `HashPartitioner` |
| `num_partitions` | number of reduce tasks |
| `input_splitter` | callable `(path) -> list[TaskInput]` |
| `output_sink` | callable `(path, records) -> None` |
| `max_attempts` | per-task retry budget (default 3) |

**Key design point for the lab:** the *mapper* emits zero, one, or many KV pairs per input record
(mandated by the guide), and the *reducer* receives `(key, [v1, v2, ...])` — never a pre-grouped
dict. That keeps the shuffle stage meaningful and lets the trace prove the grouping happened in the
shuffle, not in the mapper.

### 1.2 Partitioners (`partitioner.py`)

- `HashPartitioner` — `crc32(key.encode()) % num_partitions` (stable across runs and processes;
  **not** Python's randomized `hash()`).
- `RangePartitioner` — takes a sorted sample of keys and assigns contiguous ranges; used to show
  that a range partitioner gives *balanced, ordered* reduce partitions.
- `IdentityPartitioner` — all keys to partition 0; used to demonstrate the pathological case that
  motivates partitioning.

Each partitioner must expose `partition_bounds()` so the runner can report which key ranges landed
where (needed for the shuffle figure).

### 1.3 The runner (`runner.py`) — stage by stage

```
input --> [SPLIT]  --> [MAP] --> [COMBINE] --> [SHUFFLE+SORT] --> [REDUCE] --> [COMMIT]
             |            |          |              |                  |           |
        task boundaries  per-task   per-partition  group by key    one task per   atomic
        (parallel)       retry      pre-aggregate   + sort keys     partition      rename
```

Required behaviours:

- [x] **P3.1** Split the input into *map tasks*. Default: one task per Parquet file or per row
  group, so task granularity is data-driven, not arbitrary.
- [x] **P3.2** Run map tasks in a `multiprocessing.Pool` (`spawn` context, exactly like
  `src/ingest/etl.py::main()`) with a configurable worker count.
- [x] **P3.3** **Combiners**: apply the combiner per map task (or per partition chunk) *before* the
  shuffle, so the number of KV pairs crossing the network boundary drops. Record
  `kv_before_combine` vs `kv_after_combine` and the estimated bytes saved.
- [x] **P3.4** **Shuffle & sort**: group emitted KVs by key and sort the keys (deterministic
  order -> deterministic output -> diffable runs).
- [x] **P3.5** **Reduce**: one task per partition; each reducer receives the key + its grouped
  values.
- [x] **P3.6** **Fault tolerance**:
  - a map or reduce task that raises is retried up to `max_attempts`;
  - other tasks are **not** affected;
  - a task that exhausts its retries fails the whole job with a structured error;
  - `--inject-fault map_task=<i> attempt=<n>` forces a synthetic failure so the retry path can be
    *demonstrated*, and the resulting output must be byte-identical to the clean run.
- [x] **P3.7** **Output commit protocol**: output is written to `data/mapreduce/output/<job>/_tmp/`
  and atomically renamed to the final directory only after every task succeeds (Hadoop's
  `FileOutputCommitter` pattern). A crashed run therefore never leaves a half-written result that a
  later stage could read.
- [x] **P3.8** **Parallelism report**: tasks, workers, wall-clock, and the slowest task's duration
  (the classic MapReduce skew indicator).

### 1.4 Trace (`trace.py`)

Every stage writes JSONL to `data/mapreduce/traces/<job>/<stage>.jsonl`:

```json
{"stage": "map",    "task": "map-0007",   "partition": 3, "key": "2025|Bahrain_Grand_Prix|Race|VER|12", "value": {"laptime": 98.4}}
{"stage": "combine","task": "map-0007",   "partition": 3, "key": "...", "value": {...}}
{"stage": "shuffle","task": "shuffle-0",  "partition": 1, "key": "...", "value": {...}}
{"stage": "reduce", "task": "reduce-01",  "partition": 1, "key": "...", "value": {...}}
```

CLI flags: `--trace` (on), `--trace-sample N` (cap the number of records per stage file — the
Silver run emits millions of KVs), `--trace-dir`.

### 1.5 Verification (`verify.py`) — the guide's "independent verification"

Four assertions computed from the trace alone (no access to the reducer's internal state):

- [x] **P3.9** **No omitted records**: the number of *values* emitted by the map stage equals the
  number of values fed to the reducers (accounting for combiner merges by counting values, not
  pairs).
- [x] **P3.10** **No duplicated records**: every `(key, value)` pair appears in the shuffle stage at
  most as many times as it was emitted by the map stage; the multiset difference must be empty.
- [x] **P3.11** **Partition coverage**: the number of distinct keys in the shuffle stage equals the
  number of non-empty reduce partitions, and every key appears in exactly one partition.
- [x] **P3.12** **Independent reference reduction**: recompute the expected output for a sample of
  keys with a plain Python loop (no shared code with the reducer) and assert equality.

These become `tests/test_mapreduce.py` assertions so they run in CI.

---

## 2. Job 1 — Lap telemetry aggregation (Bronze -> Silver)

**Guide concept exercised:** map over raw records -> shuffle by key -> reduce into per-lap
aggregates; combiner to pre-aggregate.

- [x] **P3.13** Mapper: one input record = one Bronze telemetry row-group (or one raw
  `*_tel.json` lap). Emits `key = "<season>|<gp>|<session>|<driver>|<lap>"`, value = the telemetry
  aggregate dict (`n_samples`, `speed_mean/max/min/std`, `throttle_mean/std`,
  `throttle_active_frac`, `brake_frac`, `drs_frac`, `lat_g_mean`, `long_g_mean`, `vert_g_mean`,
  `dirty_air_frac`, `lap_distance`).
  **Port the expressions verbatim from `src/ingest/etl.py::_silver_exprs`** so the comparison in
  P3.17 is meaningful.
- [x] **P3.14** Combiner: a per-key numeric accumulator that merges two partial aggregate dicts
  (counts + sums -> means), so a lap split across several input chunks merges correctly.
- [x] **P3.15** Reducer: emits the final per-lap Silver row, joined with the lap-time metadata
  (`laptime`, `compound`, `stint`, `life`, `s1/s2/s3`).
- [x] **P3.16** Run it on one season (2025) with `--trace-sample`; save the trace.
- [x] **P3.17** **Parity vs the existing pipeline**: diff the MapReduce Silver output against
  `data/silver/stint_features/season=2025/*.parquet`. Report: rows matched, rows differing, max
  absolute difference per column (expect 0.0 within float tolerance).

**Acceptance:** identical Silver rows for 2025; the combiner demonstrably shrinks the shuffled KV
count.

---

## 3. Job 2 — Per-stint degradation statistics

**Guide concept exercised:** the canonical "group by key, reduce to an aggregate" job — this is the
job the guide's shuffle/reduce narrative is really about.

**Input:** `data/gold/labeled_frame.parquet` (107,606 rows x 85 cols, from task 9.0.3).
**Not** `lap_features.parquet` — `fuel_corr` and `is_sc_lap` exist only in the labeled frame.

- [x] **P3.18** Mapper: lap -> `key = "<season>|<gp>|<session>|<driver>|<stint>"`,
  value = `{life, fuel_corr, compound, dirty_air_frac, is_sc_lap, track_temp_mean, cluster}`.
  The mapper also computes the **stint position** `_pos = int_range(len).over(stint_keys)` so the
  reducer can apply the warm-up exclusion without re-sorting.
- [x] **P3.19** Reducer per stint — replicate `src/models/fit_degradation.py` (lines 278-298)
  **exactly**:
  1. keep only laps with `_pos >= 3` (drops the out-lap + the first two flying laps; warm-up and
     race-start traffic bias the slope negative and trigger false cliffs);
  2. keep only laps with `is_sc_lap == 0` (Task 8.1.2.3 SC/VSC mask);
  3. `beta = cov(life, fuel_corr) / var(life)` over the surviving laps;
  4. keep the stint only if `n_laps >= 4`;
  5. emit `{beta, n_laps, best_pace (= stint_best_pace), mean_dirty_air_frac, track_temp_mean,
     compound, cluster, life0, t_end_life}`.
- [x] **P3.20** Model-level roll-up (a second reduce — a legitimate two-stage MapReduce pattern):
  `beta[compound] = median(per-stint beta)`, `sigma[compound] = (Q3 - Q1) / 1.349` (IQR-robust —
  raw std is destroyed by traffic-locked outliers).
- [x] **P3.21** **Partitioner matters here**: run with `IdentityPartitioner` (all keys -> 1 reducer)
  and with `HashPartitioner(8)`; report the wall-clock and the per-partition key-count histogram.
  The skew is the pedagogical payload: it motivates partitioning.
- [x] **P3.22** Run over 2021-2025; persist the output as
  `data/mapreduce/output/stint_stats/part-*.jsonl`.
- [x] **P3.23** **Three-way parity**: MapReduce beta vs the Polars/DuckDB baseline
  (`src/models/fit_degradation.py`'s frame) vs `degradation_model.json` beta per compound. Report
  max/mean absolute difference and the number of stints compared (expect ~4,274 after the
  `n_laps >= 4` filter; `degradation_model.json["n_stints"] = 4274`).
- [x] **P3.24** **Per-stint parity** against the pre-computed
  `data/gold/stint_degradation.parquet::degradation_slope` (44,298 rows) — this table is the
  strongest single artifact in the whole lab.

**Acceptance:** parity within `1e-6 s/lap` for the beta values, or a written explanation of any
difference (e.g. `stint_degradation.parquet` applies a slightly different lap filter).

---

## 4. Job 3 — Driver / session profile

**Guide concept exercised:** a second, differently-keyed aggregation to show the runtime is generic
(not a one-off word count).

- [x] **P3.25** Mapper: lap -> `key = "<driver>|<session_type>"`, value =
  `{laptime, fuel_corr, life, dirty_air_frac, is_sc_lap, speed_mean, throttle_mean, lat_g_max}`.
- [x] **P3.26** Reducer: `n_laps`, mean/median lap time, mean degradation, SC-lap share,
  dirty-air share, mean speed, mean throttle modulation.
- [x] **P3.27** Compare with the driver-level aggregates in `src/models/style_data.py` and with
  `data/gold/driver_style.parquet` (178 rows). Report the parity table.

**Acceptance:** parity with the Polars baseline for `n_laps`, mean pace and dirty-air share.

---

## 5. Fault-tolerance demonstration

- [x] **P3.28** Inject a failure in map task 3 of job 2; capture the log showing the retry.
- [x] **P3.29** Diff the output of the fault-injected run against the clean run -> must be
  **byte-identical**.
- [x] **P3.30** Inject a failure that exceeds `max_attempts` -> the job must fail cleanly, leave
  **no** output directory (commit protocol), and print a structured error (no silent partial
  result).
- [x] **P3.31** Add these three behaviours as pytest cases in `tests/test_mapreduce.py`.

---

## 6. Deliverables & evidence table

| Guide requirement | Where it is demonstrated | Artifact |
|---|---|---|
| Key-value abstraction | `src/mapreduce/types.py` (`KV`, protocols) | — |
| Map stage (0..n pairs, parallel) | `runner.py` map phase + job mappers | trace `map` |
| Shuffle & sort | `runner.py` shuffle phase | trace `shuffle` |
| Reduce (key + grouped values) | `runner.py` reduce phase + job reducers | trace `reduce` |
| Combiners (local pre-aggregation) | `runner.py` combine phase | `kv_before/after_combine` counters |
| Partitioners (which reducer gets a key) | `src/mapreduce/partitioner.py` | partition-boundary report |
| Fault tolerance & parallelism | `runner.py` retry + `--inject-fault` | fault demo log |
| Trace simulation | `src/mapreduce/trace.py` | `data/mapreduce/traces/**` |
| Independent verification | `src/mapreduce/verify.py` | parity tables |

---

## 7. Files to touch

| File | Action |
|---|---|
| `src/mapreduce/**` | **new** |
| `src/models/export_labeled.py` | **new** — materializes the shared input (task 9.0.3) |
| `src/ingest/etl.py` | read-only reference for `_silver_exprs` (do **not** modify the production ETL) |
| `src/models/fit_degradation.py` | read-only reference for the beta estimator (lines 278-298) |
| `src/models/style_data.py` | read-only reference for driver aggregates |
| `src/eval/plots.py` | add the shuffle + parity figures (Tier 3) |
| `tests/test_mapreduce.py` | **new** |
| `requirements.txt` / `pyproject.toml` | no new dependency (stdlib only) |
| `.github/workflows/ci.yml` | no change needed (pure-Python tests) |
| `RUNBOOK.md` | add the MapReduce stage rows (Tier 3) |
