# Phase 9.1 — MapReduce runtime gate (STOP)

- Date / author: 2026-10-21 / Codex
- Status: [x] Go  [ ] No-Go  [ ] Rework

## Files created

`src/mapreduce/{types,partitioner,runner,io,trace,verify,cli}.py` and
`tests/test_mapreduce.py` provide the runtime and executable checks.

## Steps to run (in order)

```powershell
python -m pytest tests/test_mapreduce.py -q
python -m src.mapreduce.cli run --job stint_stats --trace
python -m src.mapreduce.cli run --job stint_stats --inject-fault map 3 1 --output data/mapreduce/output/stint_stats_retry
```

## What's going on

Spawn workers run map and reduce attempts. The parent spills shuffle to SQLite
and sorts by partition/key. Each reducer receives one key's grouped values.
Successful-attempt files alone enter the trace and committed output. Exhausted
tasks fail the job, preserving an earlier committed result on an attempted rerun.
Hash partitioning uses CRC32; range and identity partitioners are also available.

## After you run it

Compare part-file bytes, not metrics timestamps. Full traces prove conservation;
sampled traces are rejected by the verifier as incomplete evidence.

## Results (filled after running)

- Tests cover one-to-many/zero-emission word count, combiner reduction, stable
  partitions, map retry, reducer retry, exhausted attempts, preservation of an
  existing commit, duplicate shuffle corruption, sampling, null-life ordering,
  Silver statistics and labeled export regression checks.
- Real fault: map task **3**, first attempt failed; the retry completed. All four
  final part files are **byte-identical** to the fault-free run.
- Full stint trace: **107,606 input records**, **106,735 mapper emissions**,
  **4,981 keys**, **4 nonempty reducer partitions**; independent reduction passes.
- Ruling: input records need not equal emitted KVs; distinct keys need not equal
  reducer partitions. The contradictory plan equalities were replaced with
  conservation, exact multiset, key ownership and reference-result assertions.
- Ruling: overwrite has rollback, but is not a transactional snapshot for
  concurrent readers; use separate output directories when readers are active.
- Review found and fixed missing-Silver-column and saved-model-checkpoint holes;
  regression tests first demonstrated both failures.
- Next step: real-data job/equivalence gate.
