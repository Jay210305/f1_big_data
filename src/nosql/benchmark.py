"""Measured write/storage and cursor costs on dedicated synthetic collections."""
from uuid import uuid4
from .synthetic import load_synthetic, synthetic_laps, canonical_hash, cached_hash
from .stream import memory_benchmark


def _bench_laps(db, documents, expected, existing, batch_size):
    """Fill the shared benchmark collection, validating any cached content.

    A cache is only reused when its documents still hash to the expected
    value. The same count with different content is a corrupted cache: it is
    rebuilt instead of being reported as a reproducible population.
    """
    if existing == 0:
        measured = load_synthetic(db.bench_laps, len(documents), batch_size=batch_size)
        measured["cached"] = False
        return measured
    cached = cached_hash(db.bench_laps, documents)
    if cached == expected:
        return {"count": existing, "cached": True, "sha256": cached}
    db.bench_laps.delete_many({})
    measured = load_synthetic(db.bench_laps, len(documents), batch_size=batch_size)
    measured["cached"] = False
    measured["cache_rebuilt"] = True
    return measured


def benchmark(db, n=100_000, batch_sizes=(5000,)):
    if n < 1:
        raise ValueError("n must be positive")
    token = uuid4().hex
    collections = [db["phase9_benchmark_"+token+suffix] for suffix in ("_plain", "_indexed")]
    documents = list(synthetic_laps(n))
    expected = canonical_hash(documents)
    result = {"n": n, "seed": 42,
              "reproducible": expected == canonical_hash(synthetic_laps(n))}
    existing = db.bench_laps.count_documents({})
    if existing not in (0, n):
        raise ValueError(f"bench_laps contains {existing} documents but benchmark requests {n}; "
                         "clear only bench_laps explicitly before changing the benchmark size")
    result["bench_laps"] = _bench_laps(db, documents, expected, existing, batch_sizes[0])
    result["reproducible"] = bool(result["reproducible"]) and result["bench_laps"]["sha256"] == expected
    try:
        for name, collection, indexed in zip(("without_indexes", "with_indexes"), collections, (False, True)):
            db.create_collection(collection.name)
            if indexed:
                for label, keys in (("driver_laps", [("season", 1), ("gp", 1), ("driver", 1), ("lap", 1)]),
                                    ("compound_life", [("compound", 1), ("life", 1)]), ("laptime", [("laptime", 1)])):
                    collection.create_index(keys, name=label)
            measured = load_synthetic(collection, n, batch_size=batch_sizes[0])
            stats = db.command("collStats", collection.name)
            measured["stats"] = {k: stats[k] for k in ("count", "size", "storageSize", "totalIndexSize", "nindexes")}
            result[name] = measured
        result["memory"] = memory_benchmark(collections[0])
        result["write_cost_ratio"] = result["with_indexes"]["seconds"]/result["without_indexes"]["seconds"]
        # Budget is measured, not asserted to have been exceeded by a specific host.
        result["memory"]["budget_bytes"] = 16*1024*1024
        for kind in ("streaming", "materialized"):
            result["memory"][kind]["within_budget"] = result["memory"][kind]["peak_python_bytes"] <= result["memory"]["budget_bytes"]
        return result
    finally:
        for collection in collections:
            collection.drop()
