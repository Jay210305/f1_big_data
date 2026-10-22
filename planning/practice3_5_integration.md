# Practice 3 + Practice 5 — Integration into the F1 Strategy Engine

> Sub-plan of `PLAN3.md` (Phase 9, Tier 3). Covers wiring the two new subsystems into the existing
> system so they are *load-bearing* rather than side projects, and producing the report evidence.

## Goal Summary

The two labs only count as "implemented into the project" if the rest of the pipeline can consume
them and the thesis can cite them. This sub-plan covers five integration surfaces:

1. **A shared materialized input** (`labeled_frame.parquet`) so both subsystems read the same file.
2. **MongoDB -> the optimizer** (the strategy engine can read its live state from the document store).
3. **MongoDB -> the Streamlit dashboard** (decision history as documents).
4. **MapReduce + MongoDB -> the figures** (evidence for the report).
5. **Everything -> the report, the runbook, `run_all.py` and CI** (reproducibility).

---

## 1. The shared input (task 9.0.3, prerequisite for everything else)

- [x] **I.1** `src/models/export_labeled.py` — materialize the labeled frame:
  `build_labeled_frame()` (from `src/models/eval_rul.py`) + the `cluster` join from
  `data/gold/driver_embeddings.parquet` (`fill_null(-1).cast(Int32)`, exactly as
  `fit_degradation.py` and `optimizer.main()` do) -> `data/gold/labeled_frame.parquet`.
  - Expected shape: **107,606 rows** (138,449 Race/Sprint laps minus the 22.3 % of
    SC-contaminated stints removed by `rul_data.py::filter_stints`) x **85 columns**.
  - Add a row-count assertion so a silent regression in `filter_stints` / `add_fuel_corr` cannot
    quietly change every downstream number.
- [x] **I.2** Optional extras, behind a flag: `rain_intensity` and `track_drying_proxy` from
  `fit_degradation.py::_wet_context` so the wet path is reproducible too.
- [x] **I.3** Skip-if-cached + `--force`; print the shape on every run.

**Why this matters:** `optimizer.get_state()` consumes the **labeled frame**, not
`lap_features.parquet`. If the Mongo `laps` collection were sourced from Gold, the
`--source mongo` path could not reproduce `fuel_corr` / `pace_above_best` and the parity proof
would be meaningless. One file, one source of truth.

---

## 2. MongoDB as an operational state source (`--source mongo`)

Today `src/models/optimizer.py::get_state(df, season, gp, session, driver, lap, lapf=None)` reads
from the labeled frame (`df`) and uses `lapf` (Gold `lap_features`) only to compute `n_laps`. The
fields it needs are: `stint`, `compound`, `life`, `lap`, `fuel_corr`, `pace_above_best`,
`track_temp_mean`, `cluster` — plus `session`, `driver`, `season`, `gp`.

- [x] **I.4** `src/nosql/repository.py` — a thin read API over the `laps` collection:
  `lap_history(season, gp, session, driver) -> list[dict]`,
  `stint_for(season, gp, session, driver, lap) -> dict`,
  `session_lap_count(season, gp, session) -> int`,
  `latest_decisions(gp, driver) -> list[dict]`.
- [x] **I.5** `src/models/optimizer.py` — add `--source {parquet,mongo}` (default `parquet`, so
  existing behaviour is unchanged) and `--mongo-uri` / `--mongo-db`.
- [x] **I.6** `get_state()` gains an optional `repo` parameter; when `repo` is supplied it uses the
  repository instead of the parquet path. Keep the parquet path as the fallback when Mongo is
  unreachable, and **log which source was used** (never silently switch).
- [x] **I.7** **Parity test**: for >= 5 (season, gp, driver, lap) combinations, assert the Mongo
  path and the parquet path return the same `fuel_corr`, `pace_above_best`, `track_temp_mean` and
  `cluster`. Record the table in the annotation.
- [ ] **I.8** `src/models/backtest.py` — optional `--source mongo` for the decision reader.
- [x] **I.9** `RUNBOOK.md` — document the flag and the prerequisite (the server must be running).

**Acceptance:** `python -m src.models.optimizer --source mongo --season 2025 --gp "Bahrain Grand
Prix" --driver VER --lap 12` prints the same ranked strategies as the parquet path.

---

## 3. Streamlit dashboard tab

`app/streamlit_app.py` is currently a **single page** (sidebar + metrics + a line chart) that
**returns early** when no JSON has been uploaded, and all its UI strings are **in Spanish**. Adding
a tab therefore requires a small restructuring, not just an appended block.

- [x] **I.10** Restructure to `st.tabs(["Vivo", "Historial MongoDB"])`:
  - move the sidebar + `new_state(...)` + the existing upload/replay/metrics/recommendation block
    into the first tab ("Vivo");
  - move the current early `return` (`st.info("Carga un JSON de vueltas ...")`) **inside** the
    first tab so the second tab still renders when no file is uploaded;
  - keep `st.json(current)` at the end of the first tab.
- [x] **I.11** New tab **"Historial MongoDB"** (Spanish labels, matching the existing app):
  - a filter row (temporada, GP, piloto) -> an aggregation pipeline executed server-side;
  - a paginated table of `strategy_decisions` documents;
  - a delta histogram computed by a Mongo aggregation pipeline (not in Python) — this demonstrates
    the "push computation to the database" advantage;
  - an "estado del índice" panel (`collStats`) showing document count and index size.
- [x] **I.12** Graceful degradation: if the server is down, show an informative banner
  ("MongoDB no disponible — inicia el servidor o usa `python -m src.nosql.cli health`") instead of
  a stack trace.
- [x] **I.13** Keep the existing offline-replay behaviour untouched.

---

## 4. Figures for the report

`src/eval/plots.py` already establishes the house style (matplotlib/seaborn, `Agg` backend, a
`FIGURES` dict, `--only <name>` CLI, output to `eval_plots/`). Add three figures following it.

- [x] **I.14** **Figure A — MapReduce shuffle/sort trace.**
  Plot the number of keys and values per reduce partition, side by side with and without the
  combiner, annotated with the byte reduction. This is the visual proof of the shuffle stage and
  the combiner's value.
- [x] **I.15** **Figure B — MongoDB index impact.**
  Grouped bar chart: `totalDocsExamined` (and `totalKeysExamined`) per query, before vs after
  index creation, with the winning-plan stage (`COLLSCAN` -> `IXSCAN`) annotated. A second panel
  shows the index size overhead vs the document size.
- [x] **I.16** **Figure C — three-way parity.**
  Scatter of MapReduce beta vs Polars beta (one point per stint) with the `y = x` line, plus the
  Mongo `$group` beta as a third series; annotate the max absolute difference.
- [x] **I.17** Register all three in the `FIGURES` dict in `src/eval/plots.py` and update the
  module docstring (which currently lists only the two Phase-8 figures).
- [x] **I.18** Confirm `eval_plots/` gitignore status and mirror the `correlation_maps/` decision
  (those 4 files are currently tracked despite being listed in `.gitignore` — decide deliberately
  whether the new figures are committed).

---

## 5. Report, runbook, `run_all.py`, CI

### 5.1 `document/main.tex` (the Spanish thesis) — new section

- [x] **I.19** Insert **"§5 Arquitecturas Big Data: MapReduce y MongoDB"** after the current
  methodology section and renumber the following sections.
- [x] **I.20** Content:
  1. **§5.1 Arquitectura MapReduce** — the stage diagram
     (split -> map -> combine -> shuffle/sort -> reduce -> commit), the three jobs, the partitioner
     and combiner roles, the fault-tolerance behaviour and the injected-fault demonstration.
  2. **§5.2 Implementación del runtime** — the class/function map, the CLI, the trace format.
  3. **§5.3 Verificación independiente** — the four trace assertions, the three-way parity table.
  4. **§5.4 MongoDB: modelo documental** — the collection list, the embedded `race_sessions`
     document, the schema validators, the `_id` strategy and idempotency.
  5. **§5.5 Indexación y análisis de consultas** — the explain() before/after table, the index
     cost table, the collection-scan -> index-scan discussion.
  6. **§5.6 Agregaciones y su equivalencia con MapReduce** — the pipelines, the equivalence
     argument, the parity table.
  7. **§5.7 Escalabilidad** — local vs replication vs sharding, with the measured justification.
  8. **§5.8 Limitaciones** — single-node server, ~152k-document scale, the Mongo path not yet used
     in the backtest loop.
- [x] **I.21** Insert Figures A, B and C with `\includegraphics` and reference them in the text.
- [x] **I.22** Add the two tables (parity, explain) as `booktabs` tables.
- [x] **I.23** Keep the numbers **generated from the artifacts**, not typed by hand — add a small
  script that dumps the report tables to `data/models/phase9_tables.json` so the `.tex` numbers can
  be checked against the run output.
- [x] **I.24** Rebuild the PDF and check the references resolve (no `??` in the log).

### 5.2 `document/strategy_engine_report.tex` (the English report)

- [x] **I.25** Add a shorter English subsection with the same parity and explain() tables and two
  of the three figures.

### 5.3 `RUNBOOK.md`

- [x] **I.26** Add stage rows:

  | # | Stage | Command | Artifacts | Wall-clock |
  |---|-------|---------|-----------|------------|
  | 9.0 | MongoDB health | `python -m src.nosql.cli health` | console | ~1 s |
  | 9.0b | Export labeled frame | `python -m src.models.export_labeled` | `data/gold/labeled_frame.parquet` | ~30 s |
  | 9.0c | MongoDB ingest | `python -m src.nosql.cli ingest --scope laps` | `data/mongo/` | ~3-5 min |
  | 9.1 | MapReduce job 2 | `python -m src.mapreduce.cli run --job stint_stats` | `data/mapreduce/output/` | ~2-5 min |
  | 9.1b | MR verification | `python -m src.mapreduce.cli verify --job stint_stats` | `data/mapreduce/verify/*.json` | ~1 min |
  | 9.2 | Indexes + explain | `python -m src.nosql.cli index --explain` | `data/mongo/explain/*.json` | ~2 min |
  | 9.2b | Aggregation parity | `python -m src.nosql.cli parity` | `data/mongo/parity.json` | ~2 min |

- [x] **I.27** Add a "MongoDB prerequisites" note (server must be running; how to start it).
- [x] **I.28** Add the `optimizer --source mongo` variant to the "In-race recommendation" section.

### 5.4 `src/run_all.py`

The current `STAGES = ["etl", "verify_schema", "gold", "rul", "style", "degradation", "optimizer"]`
(note: `backtest` is deliberately **not** a stage — it is printed as a manual follow-up).

- [x] **I.29** Add `labeled` (task 9.0.3) as a stage **before** any Phase 9 stage.
- [x] **I.30** Add `mapreduce` as a stage (pure Python, safe to run unconditionally) with a
  skip-if-cached check on `data/mapreduce/output/stint_stats/`.
- [x] **I.31** Add `mongo` as a stage that **skips gracefully** when the server is unreachable
  (do not fail the whole pipeline because MongoDB is off) and a skip-if-cached check on the
  collection counts.
- [x] **I.32** Do **not** add `backtest` — leave the existing behaviour alone.

### 5.5 CI

- [x] **I.33** `.github/workflows/ci.yml` — add a `mongodb` service container:

  ```yaml
  services:
    mongodb:
      image: mongo:7
      ports: ["27017:27017"]
      options: >-
        --health-cmd "mongosh --eval 'db.runCommand({ping:1})'"
        --health-interval 10s --health-timeout 5s --health-retries 5
  ```

- [x] **I.34** `pyproject.toml` — register the marker:
  `[tool.pytest.ini_options] markers = ["mongo: requires a running MongoDB server"]`.
- [x] **I.35** `tests/test_nosql.py` — every test that needs the server is marked
  `@pytest.mark.mongo` and skips cleanly when `MONGO_URI` is unreachable; the pure-logic tests
  (cleaning, schema validation, pipeline *builders*) always run.
- [x] **I.36** `tests/test_mapreduce.py` — pure Python, always runs in CI. Include a tiny synthetic
  job so the CI runtime stays under a few seconds.
- [ ] **I.37** Confirm the whole CI suite is green and under the 30-minute timeout.

---

## 6. Cross-cutting checks before the phase closes

- [x] **I.38** `python -m src.run_all` still completes (the parquet pipeline must be untouched by
  the Mongo work).
- [x] **I.39** The existing 53 tests still pass — no regression.
- [x] **I.40** `.gitignore` — confirm `data/mongo/`, `data/mapreduce/` are ignored (the `data/`
  rule covers them) and that no large Mongo dump or trace file is committed.
- [x] **I.41** `git status` clean of accidental artefacts (Parquet exports, `.bson` dumps).

---

## 7. Traceability matrix (fill this in as the work lands)

This table is the single artefact the professor can read to verify every guide bullet is covered.

| Guide bullet | Implementation file | Test | Figure / table | Annotation |
|---|---|---|---|---|
| k/v abstraction | `src/mapreduce/types.py` | `test_mapreduce.py::test_kv_contract` | — | phase_9_1 |
| Map stage | `src/mapreduce/runner.py` | `test_mapreduce.py::test_map_emits_zero_many` | Fig. A | phase_9_1 |
| Shuffle & sort | `src/mapreduce/runner.py` | `test_mapreduce.py::test_shuffle_groups_and_sorts` | Fig. A | phase_9_1 |
| Reduce | `src/mapreduce/runner.py` | `test_mapreduce.py::test_reduce_receives_grouped_values` | — | phase_9_1 |
| Combiner | `src/mapreduce/runner.py` | `test_mapreduce.py::test_combiner_reduces_shuffle_volume` | Fig. A | phase_9_1 |
| Partitioner | `src/mapreduce/partitioner.py` | `test_mapreduce.py::test_partitioner_distribution` | Fig. A | phase_9_1 |
| Fault tolerance | `src/mapreduce/runner.py` | `test_mapreduce.py::test_task_retry_and_commit` | — | phase_9_1 |
| Trace + verification | `src/mapreduce/trace.py`, `verify.py` | `test_mapreduce.py::test_trace_assertions` | Tab. parity | phase_9_1 |
| Document model | `src/nosql/schemas.py` | `test_nosql.py::test_schema_validation` | Compass shot | phase_9_2 |
| CRUD | `src/nosql/queries.py` | `test_nosql.py::test_crud_cycle` | — | phase_9_2 |
| Batch ingest 100k | `src/nosql/synthetic.py` | `test_nosql.py::test_synthetic_reproducible` | — | phase_9_2 |
| Streaming cursors | `src/nosql/stream.py` | `test_nosql.py::test_cursor_batch_size` | memory table | phase_9_2 |
| Indexes + explain | `src/nosql/indexes.py` | `test_nosql.py::test_explain_uses_index` | Fig. B, Tab. explain | phase_9_2 |
| Scalability trade-offs | `src/nosql/indexes.py` (`collStats`) | — | Tab. index cost | phase_9_2 |
| Aggregation pipelines | `src/nosql/aggregation.py` | `test_nosql.py::test_pipeline_results` | — | phase_9_2 |
| Aggregation == MapReduce | `src/nosql/cli.py parity` | `test_nosql.py::test_parity_with_mapreduce` | Fig. C, Tab. parity | phase_9_2 |
| Data quality | `src/nosql/cleaning.py` | `test_nosql.py::test_cleaning_edge_cases` | quality report | phase_9_2 |
| Independent validation | `src/mapreduce/verify.py` | both | Tab. parity | phase_9_1/9_2 |

---

## 8. Open issue carried into Phase 9 (pre-existing, not introduced here)

- [x] **I.43** Resolve the `phase7_backtest.json` inconsistency: `summary.n_decisions = 313` vs a
  `decisions` array of **200** entries (and `src/eval/plots.py`'s Figure-4 docstring says "313").
  Either the summary or the array is wrong. Fix it, then size the Mongo `strategy_decisions`
  collection from the corrected artifact and update the report if it quotes the count.
