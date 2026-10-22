# Phase 9.2 — MongoDB laboratory (STOP)

- Date / author: 2026-10-21 / Codex
- Status: [x] Go  [ ] No-Go  [ ] Rework

## Files created

`src/nosql/` contains schemas, canonical cleaning, batched ingestion, CRUD/query
cookbook, indexes/explain, aggregation, streaming, synthetic data, benchmark,
repository, CLI and guide. `tests/test_nosql.py` exercises pure and live contracts.

## Steps to run (in order)

```powershell
python -m src.nosql.cli health
python -m src.nosql.cli ingest --output data/mongo/ingest_first.json
python -m src.nosql.cli ingest --output data/mongo/ingest_second.json
python -m src.nosql.cli indexes
python -m src.nosql.cli explain
python -m src.nosql.cli quality
python -m src.nosql.cli crud
python -m src.nosql.cli queries
python -m src.nosql.cli aggregate
python -m src.nosql.cli benchmark --n 100000
python -m src.mapreduce.evidence --silver --mongo
```

## What's going on

Natural IDs make 5,000-record upserts idempotent. Embedded session documents
retain weather, corner geometry, race-control messages and derived SC laps.
Race length comes from unfiltered Gold, preserving optimizer state semantics.
Strict validators allow actual nullable analytical fields; cleaning reuses the
existing ingest normalizers. BSON supports NaN/Infinity, but analytical input
deliberately maps nonfinite values to null.

## After you run it

All evidence is under `data/mongo/`. Index comparisons use `$natural` and named
index hints on the same data; they measure controlled access paths without
dropping user indexes. They do not claim an unhinted planner always picks IXSCAN.

## Results (filled after running)

- Collections: **576 sessions; 44,298 stints; 107,606 laps; 200 decisions;
  100,000 synthetic laps**. All validators report zero violating documents.
- Second full ingest: zero modifications and upserts in every real collection.
  Scoped Bahrain Race rerun: 1 session, 62 stints, 384 laps, 19 decisions; zero
  modifications/upserts and unchanged total counts.
- Six explain queries: each full scan examines 107,606 documents; indexed scans
  examine **30 / 21,179 / 5,016 / 42,763 / 4,601 / 26,659** respectively.
- Synthetic SHA-256:
  `7492c493268307632361bcfd50a4aa832bfec076de6bfe125473e5a37ff49353`.
- Insert throughput: **66,366/s** with `_id` only; **43,053/s** with secondary
  indexes; write-time ratio **1.5415**. Index bytes: **610,304 → 2,301,952**;
  logical document bytes: **21,111,910** in both cases.
- Cursor peak Python allocations: **1,649,106 bytes**, versus **141,560,662**
  for full materialization (**85.84×**). Streaming fits a 16 MiB allocation budget;
  the list exceeds it. Growth from 1k to 100k documents remains bounded.
  This metric is `tracemalloc`, not total process RSS; batch sweep is saved.
- Mongo/Polars maximum beta error: **1.7763568394002505e-15**, over **4,628 stints**.
- mongosh and PyMongo return the same first Bahrain VER laps:
  `(1, 103.877), (2, 98.403), (3, 98.475)`.
- Decisions: 313 evaluated in the historical backtest summary but only 200
  persisted by its old truncation. Future writes now retain all decisions.
- Deployment: one local node; replication and sharding are discussed, not deployed.
- Next step: optimizer/dashboard/report integration.
