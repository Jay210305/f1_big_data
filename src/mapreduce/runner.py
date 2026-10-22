"""Spawn map/reduce tasks; SQLite spills shuffle to disk instead of holding it in RAM."""
from __future__ import annotations

import itertools
from contextlib import closing
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sqlite3
import time
import uuid

from .io import canonical, parquet_splits, read_jsonl, records, write_jsonl
from .trace import TraceWriter
from .types import MapContext


def _rename(src: Path, dst: Path, attempts: int = 8) -> None:
    """Rename a path, retrying a transient Windows file lock.

    Windows returns `PermissionError` (WinError 5) when a directory that was
    just written is renamed while an indexer or a virus scanner still holds a
    handle on one of its files. The rename is atomic and part of the commit
    protocol, so a short retry is the right response; the job must not fail
    because of a scanner.
    """
    for attempt in range(attempts):
        try:
            src.rename(dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def _remove_tree(path: Path) -> None:
    """Remove a directory tree, tolerating transient Windows file locks.

    A worker process can still hold a handle on the shuffle database or on a
    per-attempt file when the pool shuts down, and on Windows that turns the
    cleanup into a PermissionError. Failing the whole job because a scratch
    directory could not be deleted would mask the real result, so retry a few
    times and then give up quietly: the scratch trees carry a unique name and
    are never read again.
    """
    for attempt in range(5):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if attempt == 4:
                return
            time.sleep(0.2 * (attempt + 1))


def _bucket(spec, key):
    """Reducer bucket of a key; partitioners must return a real integer bucket."""
    partition = spec.partitioner(key, spec.num_partitions)
    if isinstance(partition, bool) or not isinstance(partition, int):
        raise TypeError(f"partitioner returned a non-integer bucket {partition!r} "
                        f"for key {key!r}")
    if not 0 <= partition < spec.num_partitions:
        raise ValueError(f"partitioner returned invalid bucket {partition} "
                         f"for key {key!r}")
    return partition


def _map_task(spec, split, task, attempt, path, trace, sample):
    writer = TraceWriter(path, trace, sample)
    groups = {}
    before = after = inputs = bytes_before = bytes_after = 0
    out = Path(path) / "values.jsonl"
    try:
        with out.open("w", encoding="utf-8", newline="\n") as fh:
            for record in records(split):
                inputs += 1
                for key, value in spec.mapper(record, MapContext(task, attempt)):
                    if not isinstance(key, str):
                        raise TypeError("MapReduce keys must be strings")
                    partition = _bucket(spec, key)
                    before += 1
                    bytes_before += len(canonical([key, value]).encode())
                    writer.emit("map", task, partition, key, value)
                    if spec.combiner:
                        # Incremental two-value combining bounds memory to one value per key/task.
                        groups[key] = (spec.combiner(key, [groups[key], value])[0][1]
                                       if key in groups else value)
                    else:
                        fh.write(canonical([partition, key, value]) + "\n")
                        after += 1
                        bytes_after += len(canonical([key, value]).encode())
            if spec.combiner:
                for key, value in sorted(groups.items()):
                    partition = _bucket(spec, key)
                    writer.emit("combine", task, partition, key, value)
                    fh.write(canonical([partition, key, value]) + "\n")
                    after += 1
                    bytes_after += len(canonical([key, value]).encode())
    finally:
        writer.close()
    return dict(input_records=inputs, kv_before_combine=before, kv_after_combine=after,
                bytes_before_combine=bytes_before, bytes_after_combine=bytes_after)


def _reduce_task(spec, db_path, task, path, trace, sample):
    writer = TraceWriter(path, trace, sample)
    connection = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
    cursor = connection.execute("SELECT key,value FROM kv WHERE partition=? ORDER BY key,seq", (task,))
    def output():
        for key, group in itertools.groupby(cursor, key=lambda row: row[0]):
            values = []
            for _, encoded in group:
                value = json.loads(encoded)
                writer.emit("shuffle", task, task, key, value)
                values.append(value)
            for out_key, value in spec.reducer(key, values):
                writer.emit("reduce", task, task, out_key, value)
                yield out_key, value
    try:
        sink = spec.output_sink or write_jsonl
        sink(Path(path) / f"part-{task:05d}.jsonl", output())
    finally:
        writer.close()
        connection.close()
    return {}


def _attempt(payload):
    spec, stage, item, task, attempt, path, trace, sample, fault = payload
    start = time.perf_counter()
    Path(path).mkdir(parents=True, exist_ok=True)
    try:
        if fault and stage == fault[0] and task == fault[1] and attempt <= fault[2]:
            raise RuntimeError(f"injected fault {stage}-{task} attempt {attempt}")
        metrics = (_map_task(spec, item, task, attempt, path, trace, sample)
                   if stage == "map" else _reduce_task(spec, item, task, path, trace, sample))
        return dict(ok=True, task=task, attempt=attempt, path=str(path),
                    duration=time.perf_counter() - start, **metrics)
    except Exception as exc:
        return dict(ok=False, task=task, attempt=attempt, error=f"{type(exc).__name__}: {exc}")


class LocalJobRunner:
    def __init__(self, workers=4):
        if workers < 1:
            raise ValueError("workers must be positive")
        self.workers = workers

    def run(self, spec, input_path, output, *, trace=False, trace_sample=None,
            inject_fault=None, overwrite=False):
        if spec.num_partitions < 1 or spec.max_attempts < 1:
            raise ValueError("partitions and attempts must be positive")
        if trace_sample is not None and trace_sample < 1:
            raise ValueError("trace_sample must be positive")
        output = Path(output).resolve()
        if output.exists() and not overwrite:
            raise FileExistsError(f"{output} already exists; use --force")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.parent / ("_tmp-" + uuid.uuid4().hex)
        temporary.mkdir()
        start = time.perf_counter()
        retries, events = 0, []
        def stage(pool, name, items):
            nonlocal retries
            pending = list(enumerate(items))
            successes = {}
            for attempt in range(1, spec.max_attempts + 1):
                tasks = [(spec, name, item, task, attempt,
                          str(temporary / f"{name}-{task}-{attempt}"), trace, trace_sample,
                          inject_fault) for task, item in pending]
                results = pool.map(_attempt, tasks)
                failed = []
                item_by_task = dict(pending)
                for result in results:
                    if result["ok"]:
                        successes[result["task"]] = result
                    else:
                        events.append(dict(stage=name, **result))
                        if attempt == spec.max_attempts:
                            raise RuntimeError(canonical(dict(stage=name, **result)))
                        print(f"[retry] {name}-{result['task']} attempt {attempt}: {result['error']}", flush=True)
                        retries += 1
                        failed.append((result["task"], item_by_task[result["task"]]))
                pending = failed
                if not pending:
                    break
            return [successes[k] for k in sorted(successes)]
        try:
            splits = (spec.input_splitter or parquet_splits)(input_path)
            if not splits:
                raise ValueError("no input splits")
            # Set before spawning, so workers do not each create a full Polars thread pool.
            old_threads = os.environ.get("POLARS_MAX_THREADS")
            os.environ["POLARS_MAX_THREADS"] = "1"
            try:
                with mp.get_context("spawn").Pool(self.workers) as pool:
                    maps = stage(pool, "map", splits)
                    db = temporary / "shuffle.sqlite"
                    with closing(sqlite3.connect(db)) as connection:
                        connection.execute("CREATE TABLE kv(seq INTEGER PRIMARY KEY, partition INTEGER,key TEXT,value TEXT)")
                        for task in maps:
                            connection.executemany("INSERT INTO kv(partition,key,value) VALUES (?,?,?)",
                                ((p, k, canonical(v)) for p, k, v in read_jsonl(Path(task["path"]) / "values.jsonl")))
                        connection.execute("CREATE INDEX ordered ON kv(partition,key,seq)")
                        partitions = [dict(partition=p, keys=n, values=v) for p, n, v in
                            connection.execute("SELECT partition,COUNT(DISTINCT key),COUNT(*) FROM kv GROUP BY partition")]
                        connection.commit()
                    reduces = stage(pool, "reduce", [str(db)] * spec.num_partitions)
            finally:
                if old_threads is None:
                    os.environ.pop("POLARS_MAX_THREADS", None)
                else:
                    os.environ["POLARS_MAX_THREADS"] = old_threads
            committed = temporary / "committed"
            committed.mkdir()
            for task in reduces:
                for part in Path(task["path"]).glob("part-*"):
                    shutil.copyfile(part, committed / part.name)
            trace_dir = committed / "trace"
            if trace:
                trace_dir.mkdir()
                for name in ("map", "combine", "shuffle", "reduce"):
                    count = 0
                    with (trace_dir / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
                        for task in maps + reduces:
                            source = Path(task["path"]) / f"{name}.jsonl"
                            if source.exists():
                                for row in read_jsonl(source):
                                    if trace_sample is None or count < trace_sample:
                                        fh.write(canonical(row) + "\n")
                                        count += 1
            report = dict(job=spec.name, output=str(output), trace=str(output / "trace") if trace else None,
                          sampled=trace_sample is not None, combiner=bool(spec.combiner),
                          map_tasks=len(maps), reduce_tasks=len(reduces), workers=self.workers,
                          retries=retries, retry_events=events, partitions=partitions,
                          partitioner=spec.partitioner.partition_bounds(),
                          wall_seconds=time.perf_counter() - start,
                          slowest_task_seconds=max(t["duration"] for t in maps + reduces))
            for metric in ("input_records", "kv_before_combine", "kv_after_combine",
                           "bytes_before_combine", "bytes_after_combine"):
                report[metric] = sum(t[metric] for t in maps)
            (committed / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            if trace:
                (trace_dir / "manifest.json").write_text(json.dumps(report), encoding="utf-8")
            # Publish the committed result.
            #
            # The published path is never renamed away: a concurrent reader must
            # always find it present and readable. The committed directory is
            # renamed onto its staged name first, so the promotion is a single
            # rename that fails before the published path is touched. The staged
            # files are then renamed into the published directory one at a time,
            # each rename atomic, while the previous files are kept in the
            # staging area, so a failed publication restores them.
            staged = temporary / "staged"
            _rename(committed, staged)
            previous = temporary / "previous"
            if output.exists():
                # The published path is renamed onto itself: the rename is a
                # no-op on every supported platform, so the path stays present
                # and readable for concurrent readers, and it fails loudly when
                # the published path is missing.
                _rename(output, output)
                previous.mkdir()
                names = {child.name for child in staged.iterdir()}
                swapped = set()
                try:
                    for child in sorted(staged.iterdir()):
                        stale = output / child.name
                        if stale.exists():
                            _rename(stale, previous / child.name)
                        _rename(child, stale)
                        swapped.add(child.name)
                    for child in sorted(output.iterdir()):
                        if child.name not in names:
                            _rename(child, previous / child.name)
                except BaseException:
                    for child in sorted(output.iterdir()):
                        if child.name in swapped:
                            if child.is_dir():
                                _remove_tree(child)
                            else:
                                child.unlink()
                    for child in sorted(previous.iterdir()):
                        _rename(child, output / child.name)
                    raise
                _remove_tree(previous)
            else:
                staged.rename(output)
            return report
        finally:
            _remove_tree(temporary)


def run_job(spec, input_path, output, **kwargs):
    return LocalJobRunner(kwargs.pop("workers", 4)).run(spec, input_path, output, **kwargs)
