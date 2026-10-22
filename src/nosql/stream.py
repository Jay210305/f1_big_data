"""Cursor streaming and measured Python allocation budgets (not process RSS)."""
import time
import tracemalloc


def scan_laps(collection, batch_size=1000, materialize=False, limit=0):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    tracemalloc.start()
    start = time.perf_counter()
    count, total, numeric = 0, 0., 0
    try:
        with collection.find({}).batch_size(batch_size).limit(limit) as cursor:
            records = list(cursor) if materialize else cursor
            for row in records:
                count += 1
                if isinstance(row.get("laptime"), (int, float)):
                    total += row["laptime"]
                    numeric += 1
        _, peak = tracemalloc.get_traced_memory()
        return {"count": count, "mean_laptime": total/numeric if numeric else None,
                "seconds": time.perf_counter()-start, "peak_python_bytes": peak,
                "batch_size": batch_size, "materialize": materialize, "limit": limit,
                "memory_metric": "tracemalloc peak Python allocations, not total process RSS"}
    finally:
        tracemalloc.stop()


def memory_benchmark(collection):
    sweep = [scan_laps(collection, batch) for batch in (1, 100, 1000, 10000)]
    streaming = scan_laps(collection, 1000)
    loaded = scan_laps(collection, 1000, materialize=True)
    growth = [scan_laps(collection, 1000, limit=n) for n in (1000, 10000, 100000)]
    return {"sweep": sweep, "streaming": streaming, "materialized": loaded, "growth": growth,
            "memory_ratio": loaded["peak_python_bytes"]/max(1, streaming["peak_python_bytes"]),
            "bounded_growth": growth[-1]["peak_python_bytes"] <= 4*max(1, growth[0]["peak_python_bytes"])}
