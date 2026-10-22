"""Mergeable sufficient statistics matching ingest.etl._silver_exprs exactly."""
from copy import deepcopy
from functools import lru_cache
import math

import polars as pl

from ..io import telemetry_splits
from ..types import JobSpec

KEYS = ["season", "gp", "session", "driver", "lap"]
NUMERIC = {"speed": "speed", "throttle": "throttle", "acc_x": "lat_g",
           "acc_y": "long_g", "acc_z": "vert_g", "distance": "distance"}
FRACTIONS = {"throttle": "throttle_active_frac", "brake": "brake_frac",
             "drs": "drs_frac", "DistanceToDriverAhead": "dirty_air_frac"}


@lru_cache(maxsize=8)
def _lap_metadata(season, gp, session):
    from src.ingest.config import ROOT
    from src.ingest.etl import _load_laptimes
    path = ROOT / season / gp / session
    if not path.is_dir():
        return {}
    df = _load_laptimes(path)
    return {(r["driver"], r["lap"]): r for r in df.to_dicts()} if df is not None else {}


def partials(df):
    expressions = [pl.len().alias("n_samples")]
    for channel in NUMERIC:
        if channel not in df.columns:
            continue
        col = pl.col(channel)
        if channel.startswith("acc_"):
            col = col.abs()
        expressions.extend([col.count().alias(channel + "_n"),
                            col.sum().alias(channel + "_sum"),
                            col.mean().alias(channel + "_mean"),
                            (col.var(ddof=0) * col.count()).alias(channel + "_m2"),
                            col.min().alias(channel + "_min"), col.max().alias(channel + "_max")])
    for channel, alias in FRACTIONS.items():
        if channel in df.columns:
            predicate = pl.col(channel) < 10 if channel == "DistanceToDriverAhead" else pl.col(channel) > 0
            expressions.append(predicate.fill_null(False).sum().alias(alias + "_count"))
    for row in df.group_by(KEYS, maintain_order=True).agg(expressions).to_dicts():
        value = dict(n_samples=row["n_samples"], channels={}, fractions={})
        for channel in NUMERIC:
            if channel + "_n" in row:
                value["channels"][channel] = {field: row[channel + "_" + field]
                                              for field in ("n", "sum", "mean", "m2", "min", "max")}
        for alias in FRACTIONS.values():
            if alias + "_count" in row:
                value["fractions"][alias] = row[alias + "_count"]
        yield {**{k: row[k] for k in KEYS}, "_partial": value}


def map_lap(record, ctx):
    return [("|".join(str(record[k]) for k in KEYS), record["_partial"])]


def merge(key, values):
    result = deepcopy(values[0])
    for value in values[1:]:
        result["n_samples"] += value["n_samples"]
        for channel, b in value["channels"].items():
            if channel not in result["channels"]:
                result["channels"][channel] = deepcopy(b)
                continue
            a = result["channels"][channel]
            n = a["n"] + b["n"]
            delta = (b["mean"] or 0.) - (a["mean"] or 0.)
            a["m2"] = (a["m2"] or 0.) + (b["m2"] or 0.) + (delta * delta * a["n"] * b["n"] / n if n else 0.)
            a["sum"] += b["sum"]
            a["mean"] = a["sum"] / n if n else None
            a["n"] = n
            for field, fn in (("min", min), ("max", max)):
                xs = [x for x in [a[field], b[field]] if x is not None]
                a[field] = fn(xs) if xs else None
        for alias, count in value["fractions"].items():
            result["fractions"][alias] = result["fractions"].get(alias, 0) + count
    return [(key, result)]


def reduce_lap(key, values):
    acc = merge(key, values)[0][1]
    row = dict(zip(KEYS, key.split("|")))
    row["lap"] = int(row["lap"])
    row["n_samples"] = acc["n_samples"]
    for channel, prefix in NUMERIC.items():
        stats = acc["channels"].get(channel, {})
        if channel == "distance":
            row["lap_distance"] = stats.get("max")
        else:
            row[prefix + "_mean"] = stats.get("mean")
        if channel == "speed":
            row.update(speed_max=stats.get("max"), speed_min=stats.get("min"),
                speed_std=math.sqrt(max(0., stats["m2"] or 0.) / (stats["n"] - 1))
                          if stats.get("n", 0) > 1 else None)
    for alias in FRACTIONS.values():
        row[alias] = (acc["fractions"][alias] / acc["n_samples"]
                      if alias in acc["fractions"] else None)
    metadata = _lap_metadata(row["season"], row["gp"], row["session"])
    row.update({k: v for k, v in metadata.get((row["driver"], row["lap"]), {}).items()
                if k not in KEYS})
    return [(key, row)]


def job(combine=True, **kwargs):
    return JobSpec("lap_aggregate", map_lap, reduce_lap,
                   combiner=merge if combine else None, input_splitter=telemetry_splits, **kwargs)
