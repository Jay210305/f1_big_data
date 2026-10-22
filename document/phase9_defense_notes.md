# Phase 9 defense notes — ten-slide outline

Use the report tables and generated evidence for the final numbers. This is a
local multi-process MapReduce runtime plus one local MongoDB server, not a
multi-node deployment. The following outline is a presentation script, not a PPTX.

1. **Problem and contribution.** Real F1 telemetry already supports a strategy
   engine. Practice 3 makes batch stages and task recovery explicit; Practice 5
   adds embedded sessions, indexed queries and operational history. Show the
   report architecture diagram.
2. **Shared data contract.** Gold raw laps: 317,264 × 61. Labeled input:
   107,606 × 85 with fuel-corrected pace, labels and cluster. Both MR and Mongo
   consume that file; equal populations are necessary for meaningful parity.
3. **MapReduce stages.** Split → map → optional combine → shuffle/sort → reduce
   → atomic commit. Show one `season|gp|session|driver|stint` key. CRC32 is
   deterministic; range and identity illustrate partitioning alternatives.
4. **Three real jobs.** Silver 2025: 69,459 output laps with maximum numeric
   error 2.71e-13. Stint estimator: 4,628 outputs, maximum MR/Polars error
   3.56e-15. Driver/session profile: 70 groups, maximum error 5.69e-14.
5. **Combiner and fault tolerance.** Show `mapreduce_shuffle.png`: 103,244 →
   69,459 pairs, 96,823,560 → 65,750,633 bytes (32.09%). Map task 3 fails on
   attempt 1, retries once, and all four final partitions match baseline bytes.
   Trace/no-trace timings are not a controlled speedup comparison.
6. **Independent verification.** Full trace: 107,606 inputs, 106,735 emitted
   pairs, 4,981 distinct keys, four nonempty reducers. Check emissions,
   assignment, multiset conservation and independent per-key reduction.
   Explain why filtered inputs may emit no pair and keys need not equal reducers.
7. **Document model and CRUD.** Show `race_sessions` with embedded weather,
   corners and RCM; `laps`, `stints`, `strategy_decisions`. Natural `_id`, strict
   `$jsonSchema`, reused coercions and upserts establish the data contract.
   Compare `ingest_first.json` and `ingest_second.json`; cite zero changes only
   when the actual second-pass manifest confirms them.
8. **Index and streaming evidence.** Show `mongo_indexes.png` and explain table:
   returned rows, docs/keys examined and COLLSCAN/IXSCAN plans. Cite measured
   collection/index bytes and write overhead from `benchmark.json`. A streaming
   cursor bounds client batches; it does not prove total server memory is bounded.
9. **Three-way parity and integration.** Show `three_way_parity.png`. Discuss
   identical warm-up/SC/minimum-lap filters and population. Demonstrate optimizer
   `--source mongo` plus Streamlit's Spanish history tab without a replay upload.
   Turn off the server only in an isolated demonstration to show clear fallback.
10. **Limits and conclusion.** Dry saved-model parity passes; saved wet entries
    are absent. Raw Gold slope (mean difference 3.7098, max 363.0841) and driver
    style use different populations. One node does not prove horizontal scaling.
    The 313-decision backtest summary has only 200 stored history rows; its
    78.9% win rate is simulator-based, not a causal real-world guarantee.

## Reproducible demonstration

Run from the repository root with the virtual environment active:

```powershell
python -m src.models.export_labeled
python -m src.mapreduce.cli all
python -m src.nosql.cli health
python -m src.nosql.cli ingest --scope all --output data/mongo/ingest_first.json
python -m src.nosql.cli ingest --scope all --output data/mongo/ingest_second.json
python -m src.nosql.cli quality
python -m src.nosql.cli explain
python -m src.nosql.cli benchmark --n 100000
python -m src.mapreduce.evidence --mongo --silver
python -m src.eval.plots --only mapreduce_shuffle
python -m src.eval.plots --only mongo_indexes
python -m src.eval.plots --only three_way_parity
python -m pytest tests/test_mapreduce.py tests/test_nosql.py tests/test_phase9_parity.py tests/test_phase9_integration.py -q
```

For a short defense, retain the existing complete trace instead of regenerating
all telemetry. A sampled trace can illustrate stages but cannot replace the
full conservation proof. Do not run reset or broad deletion during the defense.

## Likely questions

- **Is this Hadoop?** No. It is an inspectable single-host runtime expressing
  MapReduce with local worker processes; network shuffle is not measured.
- **Why MongoDB alongside Parquet?** Embedded session context and indexed
  operational queries complement batch analytics. No universal speed advantage
  is asserted.
- **What runs MongoDB?** `mongod` is the server, Compass a GUI, `mongosh` a shell,
  PyMongo the Python client. Compass and mongosh are optional.
- **Why does saved-model match say false?** The saved model contains only dry
  compounds; MR/Polars include wet statistics. Report dry equality separately.
- **Why 200 versus 313?** The old writer intentionally retained `decisions[:200]`.
  The aggregate summary covers 313; the fixed writer preserves full future runs.
- **What would sharding require?** A measured access pattern, shard-key design,
  multiple servers and failure/load experiments. Those are future deployment work.

## Evidence checklist for the presenter

The current measured snapshot confirms all four second-pass ingest modification
and upsert counts are zero. Six explain queries change COLLSCAN to FETCH/IXSCAN;
driver history examines 30 documents instead of 107,606. The 100,000-document
benchmark inserts in 1.507 s with only `_id`, versus 2.323 s with four indexes
(1.541 times); index bytes grow from 610,304 to 2,301,952 while logical document
bytes remain 21,111,910. At cursor batch size 1,000, peak Python allocations are
1,649,106 bytes versus 141,560,662 for a list. This is `tracemalloc`, not RSS.
Three-way parity covers 4,628 stints with maximum Mongo/Polars error 1.78e-15.

Keep `data/mapreduce/parity.json`, each job's `metrics.json`, full trace,
four-partition retry comparison, `data/mongo/` manifests and three figures
available. The report reads numbers derived from these files. Server-down tests
and pure tests are distinct from live-server evidence; skipped Mongo tests are
not passing live tests. Read `phase9_traceability.md` for the requirement map.
