"""Bounded Parquet row-group readers and deterministic JSONL sinks."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class TaskInput:
    path: str
    row_group: int | None = None
    telemetry: bool = False


def parquet_splits(path, *, by_file=False, telemetry=False):
    import pyarrow.parquet as pq
    path = Path(path)
    files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    return [TaskInput(str(f), rg, telemetry) for f in files
            for rg in ([None] if by_file else range(pq.ParquetFile(f).num_row_groups))]


def telemetry_splits(path):
    return parquet_splits(path, telemetry=True)


def records(split):
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(split.path)
    groups = None if split.row_group is None else [split.row_group]
    for batch in parquet.iter_batches(batch_size=2048 if split.telemetry else 8192,
                                      row_groups=groups):
        if split.telemetry:
            import polars as pl
            from .jobs.lap_aggregate import partials
            yield from partials(pl.from_arrow(batch))
        else:
            yield from batch.to_pylist()


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(canonical(row) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def read_output(path):
    for part in sorted(Path(path).glob("part-*.jsonl")):
        yield from read_jsonl(part)
