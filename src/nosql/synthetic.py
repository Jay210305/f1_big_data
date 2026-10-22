"""Deterministic F1-flavoured documents and batched synthetic loading."""
import hashlib
import json
import random
import time
from itertools import islice


def synthetic_laps(n=100_000, seed=42):
    rng = random.Random(seed)
    drivers = ["VER", "HAM", "LEC", "NOR", "ALO", "GAS", "SAI", "RUS", "PIA", "TSU"]
    for i in range(n):
        life = (i // len(drivers)) % 25 + 1
        compound = ["SOFT", "MEDIUM", "HARD"][(i // 250) % 3]
        yield {"_id": f"bench-{i}", "season": str(2021 + i % 5), "gp": f"Benchmark GP {i % 23}",
               "session": "Race", "driver": drivers[i % len(drivers)], "lap": i // 10 % 60 + 1,
               "stint": i // 250 % 3 + 1, "life": life, "_pos": life-1, "compound": compound,
               "laptime": round(90 + life * .08 + rng.gauss(0, .4), 5),
               "fuel_corr": round(92 + life * .1 + rng.gauss(0, .3), 5), "is_sc_lap": int(rng.random() < .03)}


def canonical_hash(documents):
    digest = hashlib.sha256()
    for doc in documents:
        digest.update((json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n").encode())
    return digest.hexdigest()


def cached_hash(collection, documents):
    """Canonical hash of the documents a collection already holds.

    The stored documents are hashed in the generator's canonical order, so a
    cache is only comparable with a freshly generated population. ``None`` is
    returned when the collection holds other identifiers than the generator:
    a count match alone must never be mistaken for a content match.
    """
    stored = {doc["_id"]: doc for doc in collection.find({})}
    if set(stored) != {doc["_id"] for doc in documents}:
        return None
    return canonical_hash(stored[doc["_id"]] for doc in documents)


def load_synthetic(collection, n=100_000, batch_size=5000):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    documents = iter(synthetic_laps(n))
    batches = []
    loaded = 0
    start = time.perf_counter()
    while batch := list(islice(documents, batch_size)):
        tick = time.perf_counter()
        collection.insert_many(batch, ordered=False)
        elapsed = time.perf_counter()-tick
        loaded += len(batch)
        batches.append({"count": len(batch), "seconds": elapsed, "docs_per_second": len(batch)/elapsed})
    elapsed = time.perf_counter()-start
    return {"count": loaded, "seconds": elapsed, "docs_per_second": loaded/elapsed, "batches": batches,
            "sha256": canonical_hash(synthetic_laps(n))}
