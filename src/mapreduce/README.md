# Local MapReduce laboratory

`LocalJobRunner` uses a Windows-compatible `spawn` process pool for map and
reduce tasks. The parent spills shuffle KVs to SQLite, sorts by partition/key,
and feeds one key's values to each reducer. It does not need Hadoop or WSL.
Workers use only the stdlib and the project's existing Arrow/Polars stack.

```powershell
python -m src.models.export_labeled
python -m src.mapreduce.cli all
python -m src.mapreduce.cli run --job stint_stats --trace --workers 4 --force
python -m src.mapreduce.cli run --job lap_aggregate --trace --trace-sample 1000 --force
python -m src.mapreduce.cli run --job lap_aggregate --no-combiner --output data/mapreduce/output/lap_aggregate_plain
python -m src.mapreduce.cli run --job stint_stats --inject-fault map 3 1 --output data/mapreduce/output/stint_stats_retry
python -m src.mapreduce.evidence --silver
python -m src.mapreduce.evidence --silver --mongo
python -m pytest tests/test_mapreduce.py tests/test_phase9_parity.py
```

The default telemetry input is the complete 2025 Bronze season; the other
jobs share the exported labeled frame. Use `--input` for a different Parquet
file or dataset directory, `--partitions`, `--workers`, and `--partitioner
{hash,identity,range}` to compare task skew. Range partitioning takes exactly
`partitions - 1` ordered `--range-boundaries` strings. Hash uses stable CRC32.

Mappers emit zero or more string-keyed pairs. Combiners must be associative
and preserve the value type; `merge` combines telemetry counts, means, and
centered second moments without biasing sample standard deviations. The
Silver reducer reuses the raw lap metadata loader, not the Silver baseline.

Each task attempt has private files. Only successful attempts enter shuffle
and trace files. A retry re-executes the affected task; exhausted attempts
raise a structured error. Final part files and traces are prepared in a
temporary sibling directory and renamed into place after every task succeeds.
`--force` replaces an existing result with rollback if the final rename fails;
an exhausted map/reduce task preserves the previous result. Avoid concurrent
writers to the same output directory. The published directory is never renamed
away: `--force` renames the new files into it one at a time and keeps the
previous files in the staging area until the swap completes, so a concurrent
reader always finds the output directory present and readable.

Complete `map/combine/shuffle/reduce` JSONL traces live under the committed
output's `trace/`. `verify_trace` compares multisets, key ownership, combiner
reference results, and independently supplied reference reductions. A sampled
trace is a visualization aid and is explicitly rejected as a complete proof.
Input records and emitted pairs are separate counters: filtering and one-to-many
mappers make them different. Keys and reducer partitions are also different.

`data/mapreduce/parity.json` and `data/models/phase9_tables.json` contain the
measured results. The independent Polars reference matches the current
degradation estimator (warm-up positions ≥3, green laps, ≥4 surviving laps).
The existing saved model contains dry compounds only: all 4,274 dry stints
match its β and σ; the current estimator additionally yields 354 wet stints.
Raw Gold `degradation_slope` and `driver_style` use different populations, so
their differences are reported explicitly rather than called equivalence.
