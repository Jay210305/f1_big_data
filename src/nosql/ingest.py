"""Idempotent, bounded-batch lakehouse ingestion with source-derived metadata."""
import json
import time
from collections import Counter
from itertools import islice
from pathlib import Path
import polars as pl
from pymongo import UpdateOne
from .cleaning import clean_document
from .schemas import ensure_collections, SCHEMAS, validator

ROOT = Path(__file__).resolve().parents[2]
KEYS = ["season", "gp", "session", "driver", "stint"]
# The Phase 7 backtest only produced decisions for the Grand Prix race.
RACE_SESSION = "Race"
# Collections that carry the style cluster of a driver-season.
CLUSTER_COLLECTIONS = ("laps", "stints")


def sc_lap_numbers(values):
    return sorted({int(value) for value in values if value is not None})


def upsert_documents(collection, documents, batch_size=5000, prune=False):
    """Upsert documents by natural key, optionally pruning stale ones.

    `prune` removes the documents of the collection that the source no
    longer produces. It is only safe for a FULL ingest: a scoped ingest
    covers part of the collection and would delete the rest. Without it a
    re-ingest after the source changed (a new backtest scenario, a dropped
    race) leaves the previous documents behind, so the collection stops
    being a faithful mirror of the source.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    counts = Counter()
    source = iter(documents)
    stats = {"processed": 0, "modified": 0, "upserted": 0, "matched": 0, "batches": []}
    started = time.perf_counter()
    live_ids = set() if prune else None
    while raw := list(islice(source, batch_size)):
        docs = [clean_document(doc, counts) for doc in raw]
        if live_ids is not None:
            live_ids.update(doc["_id"] for doc in docs)
        tick = time.perf_counter()
        result = collection.bulk_write([
            UpdateOne({"_id": doc["_id"]}, {"$set": {k: v for k, v in doc.items() if k != "_id"}}, upsert=True)
            for doc in docs], ordered=False)
        elapsed = time.perf_counter()-tick
        stats["processed"] += len(docs)
        for name in ("modified", "upserted", "matched"):
            stats[name] += getattr(result, name+"_count")
        stats["batches"].append({"count": len(docs), "seconds": elapsed, "docs_per_second": len(docs)/elapsed})
    stats["seconds"] = time.perf_counter()-started
    stats["coercions"] = dict(counts)
    if live_ids is not None:
        stale = [doc["_id"] for doc in collection.find({}, {"_id": 1})
                 if doc["_id"] not in live_ids]
        if stale:
            for start in range(0, len(stale), 1000):
                collection.delete_many({"_id": {"$in": stale[start:start + 1000]}})
        stats["pruned"] = len(stale)
    stats["count"] = collection.count_documents({})
    return stats


def _scoped(frame, season=None, gp=None, session=None):
    frame = frame.with_columns(pl.col("season").cast(pl.String))
    for key, value in (("season", season), ("gp", gp), ("session", session)):
        if value is not None:
            frame = frame.filter(pl.col(key) == str(value))
    return frame


def lap_documents(root=ROOT, **scope):
    frame = _scoped(pl.read_parquet(root / "data/gold/labeled_frame.parquet"), **scope)
    # Match fit_degradation exactly: life sort, zero-based position, then SC removal
    # happens in the consuming aggregation, never before this position assignment.
    from .aggregation import COMPOUNDS
    if "_pos" in frame.columns:
        frame = frame.drop("_pos")
    frame = frame.with_row_index("__row")
    eligible = (frame.filter(pl.col("compound").is_in(COMPOUNDS)).sort(KEYS+["life"])
                .with_columns(pl.int_range(pl.len()).over(KEYS).alias("_pos"))
                .select("__row", "_pos"))
    frame = (frame.join(eligible, on="__row", how="left")
             .with_columns(pl.col("_pos").fill_null(-1)).drop("__row"))
    for row in frame.iter_rows(named=True):
        row["_id"] = "|".join(str(row[k]) for k in ["season", "gp", "session", "driver", "lap"])
        yield row


def stint_documents(root=ROOT, **scope):
    frame = _scoped(pl.read_parquet(root / "data/gold/stint_degradation.parquet"), **scope)
    for row in frame.iter_rows(named=True):
        row["_id"] = "|".join(str(row[k]) for k in KEYS)
        yield row


def session_documents(root=ROOT, **scope):
    from src.models.rul_data import load_sc_laps
    env = _scoped(pl.read_parquet(root / "data/gold/environmental.parquet"), **scope)
    raw = _scoped(pl.read_parquet(root / "data/gold/lap_features.parquet"), **scope)
    summary = {tuple(row[k] for k in ["season", "gp", "session"]): row for row in raw.group_by(
        ["season", "gp", "session"]).agg(pl.col("lap").max().alias("n_laps"),
        pl.col("driver").n_unique().alias("n_drivers")).iter_rows(named=True)}
    sc = {}
    for row in _scoped(load_sc_laps(), **scope).iter_rows(named=True):
        sc.setdefault(tuple(row[k] for k in ["season", "gp", "session"]), set()).add(row["lap"])
    events = {}
    # Read each small Silver partition once, preserving the raw channel names.
    for name in ("weather", "corners", "rcm"):
        groups = {}
        for path in (root / "data/silver/race_events" / name).rglob("*.parquet"):
            frame = _scoped(pl.read_parquet(path), **scope)
            for row in frame.iter_rows(named=True):
                key = tuple(row[k] for k in ["season", "gp", "session"])
                groups.setdefault(key, []).append({k: v for k, v in row.items() if k not in ["season", "gp", "session"]})
        events[name] = groups
    for row in env.iter_rows(named=True):
        key = tuple(row[k] for k in ["season", "gp", "session"])
        meta = summary.get(key, {})
        yield dict(_id="|".join(key), **{k: row[k] for k in ["season", "gp", "session"]},
                   session_type=row["session"], weather={k: v for k, v in row.items() if k not in ["season", "gp", "session"]},
                   weather_samples=events["weather"].get(key, []), corners=events["corners"].get(key, []),
                   rcm=events["rcm"].get(key, []), sc_laps=sc_lap_numbers(sc.get(key, set())),
                   sc_laps_source="rul_data.load_sc_laps: derived SC/VSC interval expansion",
                   n_laps=int(meta.get("n_laps") or 0), n_drivers=int(meta.get("n_drivers") or 0))


def decision_documents(root=ROOT, **scope):
    # The decisions are race-level: a scoped non-Race session has no race
    # decision of its own and must not inherit the Grand Prix ones.
    session = scope.get("session")
    if session is not None and str(session) != RACE_SESSION:
        return
    payload = json.loads((root / "data/models/phase7_backtest.json").read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    # The historic Phase 7 run is explicitly 2025. Future writers preserve season.
    season = str(payload.get("season", summary.get("season", 2025)))
    scenario = summary.get("scenario", "unknown")
    import re
    for row in payload["decisions"]:
        row = dict(row)
        row.setdefault("season", season)
        row.setdefault("scenario", scenario)
        row.setdefault("n_sims", summary.get("n_sims"))
        if any(scope.get(k) is not None and str(row.get(k)) != str(scope[k]) for k in ("season", "gp")):
            continue
        row["compounds"] = sorted(set(re.findall(r"\b(SOFT|MEDIUM|HARD|INTERMEDIATE|WET)\b", row.get("rec_strategy", ""))))
        row["_id"] = "|".join(str(row[k]) for k in ("season", "gp", "driver", "scenario"))
        yield row


def clear_cluster_cache(db, names):
    """Invalidate the stored style cluster before a re-ingest.

    The cluster label is re-derived from the labeled frame, so the previous
    assignment is dropped first: a document that the frame no longer produces
    keeps no stale cluster.
    """
    for name in names:
        if name in CLUSTER_COLLECTIONS:
            db[name].update_many({}, {"$unset": {"cluster": ""}})


def ingest(db, scope="all", root=ROOT, season=None, gp=None, session=None,
           refresh_clusters=False):
    ensure_collections(db)
    sources = {"race_sessions": session_documents, "stints": stint_documents,
               "laps": lap_documents, "strategy_decisions": decision_documents}
    names = list(sources) if scope == "all" else [scope]
    if refresh_clusters:
        clear_cluster_cache(db, names)
    # A full ingest reconciles the collection with the source: documents the
    # source no longer produces are pruned. A scoped ingest only covers part
    # of the collection, so it must never delete the rest.
    prune = scope == "all" and season is None and gp is None and session is None
    return {name: upsert_documents(db[name],
                                   sources[name](root, season=season, gp=gp, session=session),
                                   prune=prune) for name in names}


def quality_report(db):
    result = {}
    for name, schema in SCHEMAS.items():
        optional = set(schema["properties"]) - set(schema["required"])
        result[name] = {
            "count": db[name].count_documents({}),
            "violating_validator": db[name].count_documents({"$nor": [validator(name)]}),
            "missing_optional_fields": {field: db[name].count_documents({field: {"$exists": False}}) for field in sorted(optional)},
        }
    result["laps"]["nonnumeric_laptime"] = db.laps.count_documents({"laptime": {"$ne": None}, "$nor": [{"laptime": {"$type": "number"}}]})
    return result
