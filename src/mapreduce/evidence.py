"""Reproducible cross-engine parity evidence; never equate different populations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from src.ingest.config import ROOT
from .io import read_output
from .jobs.stint_stats import COMPOUNDS, KEYS, rollup
from .verify import verify_trace

BASE = ROOT / "data/mapreduce/output"


def polars_stints(frame):
    """Independent columnar implementation of fit_degradation's estimator."""
    # Same tie policy as the production reducer: null lives first, then life,
    # then lap, so both implementations select the same warm-up population.
    ordered = frame.filter(pl.col("compound").is_in(COMPOUNDS)).sort(KEYS + ["life", "lap"],
                                                                    nulls_last=False)
    ordered = ordered.with_columns(pl.int_range(pl.len()).over(KEYS).alias("_pos"))
    eligible = ordered.filter((pl.col("_pos") >= 3) & (pl.col("is_sc_lap") == 0))
    return (eligible.group_by(KEYS).agg(
        (pl.cov("life", "fuel_corr") / pl.var("life")).alias("beta"),
        pl.len().alias("n_laps"), pl.col("compound").first(),
        pl.col("cluster").first(), pl.col("fuel_corr").min().alias("best_pace"),
        pl.col("dirty_air_frac").mean().alias("mean_dirty_air_frac"),
        pl.col("track_temp_mean").mean(), pl.col("life").min().alias("life0"))
        .filter((pl.col("n_laps") >= 4) & pl.col("beta").is_finite()))


def reference_stint(key, values):
    """Trace reference deliberately uses Polars, never the production reducer."""
    frame = pl.DataFrame(values, infer_schema_length=None).sort(["life", "lap"]).slice(3)
    frame = frame.filter(pl.col("is_sc_lap") == 0)
    if frame.height < 4:
        return []
    beta = frame.select(pl.cov("life", "fuel_corr") / pl.var("life")).item()
    if beta is None or not __import__("math").isfinite(beta):
        return []
    return [(key, dict(beta=beta, n_laps=frame.height, compound=frame["compound"][0],
        cluster=frame["cluster"][0], best_pace=frame["stint_best_pace"][0],
        mean_dirty_air_frac=frame["dirty_air_frac"].mean(),
        track_temp_mean=frame["track_temp_mean"].mean(), life0=frame["life"].min(),
        t_end_life=frame["t_end_life"][0]))]


def stint_parity(frame, output=BASE / "stint_stats"):
    reference = polars_stints(frame).with_columns(
        pl.concat_str([pl.col(k).cast(pl.String) for k in KEYS], separator="|").alias("key"))
    mr = pl.DataFrame([dict(key=k, **v) for k, v in read_output(output)])
    joined = reference.select("key", pl.col("beta").alias("polars"), "n_laps").join(
        mr.select("key", pl.col("beta").alias("mapreduce"), pl.col("n_laps").alias("mr_n_laps")),
        on="key", how="full", coalesce=True)
    assert joined.height == reference.height == mr.height
    assert sum(joined.null_count().row(0)) == 0
    assert joined.filter(pl.col("n_laps") != pl.col("mr_n_laps")).is_empty()
    diff = (joined["polars"] - joined["mapreduce"]).abs()
    assert diff.max() < 1e-6, f"MR/Polars maximum difference {diff.max()}"
    mr_model = rollup(mr.to_dicts())
    saved = json.loads((ROOT / "data/models/degradation_model.json").read_text())
    model_table = []
    for compound in COMPOUNDS:
        sub = reference.filter(pl.col("compound") == compound)
        polars_beta = sub["beta"].median() if sub.height >= 5 else reference["beta"].median()
        row = dict(compound=compound, mapreduce=mr_model["beta"][compound], polars=polars_beta,
                   saved=saved["beta"].get(compound), sigma=mr_model["sigma"][compound],
                   saved_sigma=saved["sigma"].get(compound))
        row["beta_abs_diff"] = abs(row["mapreduce"] - row["saved"]) if row["saved"] is not None else None
        row["sigma_abs_diff"] = abs(row["sigma"] - row["saved_sigma"]) if row["saved_sigma"] is not None else None
        model_table.append(row)
    assert all(r["beta_abs_diff"] is not None and r["beta_abs_diff"] < 1e-6
               and r["sigma_abs_diff"] is not None and r["sigma_abs_diff"] < 1e-6
               for r in model_table if r["compound"] in COMPOUNDS[:3]), "saved dry model beta/sigma parity failed"
    gold = pl.read_parquet(ROOT / "data/gold/stint_degradation.parquet").with_columns(pl.col("season").cast(pl.String))
    raw = reference.join(gold.select(KEYS + ["degradation_slope"]), on=KEYS)
    raw_diff = (raw["beta"] - raw["degradation_slope"]).abs()
    return dict(n_stints=mr.height, max_abs_diff=diff.max(), mean_abs_diff=diff.mean(),
        model_table=model_table, saved_model_matches_dry=all(
            r["beta_abs_diff"] is not None and r["beta_abs_diff"] < 1e-6 and
            r["sigma_abs_diff"] is not None and r["sigma_abs_diff"] < 1e-6
            for r in model_table if r["compound"] in COMPOUNDS[:3]),
        dry_stints=reference.filter(pl.col("compound").is_in(COMPOUNDS[:3])).height,
        missing_saved_compounds=[r["compound"] for r in model_table if r["saved"] is None],
        raw_gold_comparison=dict(n_matched=raw.height, max_abs_diff=raw_diff.max(),
            mean_abs_diff=raw_diff.mean(), note="Gold uses raw laptime without this fuel correction/warmup/SC population; diagnostic, not an equivalence claim"),
        stints=joined.to_dicts())


def driver_parity(frame, output=BASE / "driver_profile"):
    baseline = frame.group_by(["driver", "session_type"]).agg(
        pl.len().alias("n_laps"), pl.col("laptime").mean().alias("mean_pace"),
        pl.col("dirty_air_frac").mean().alias("dirty_air_share"))
    baseline = baseline.with_columns(pl.concat_str("driver", "session_type", separator="|").alias("key"))
    actual = {k: v for k, v in read_output(output)}
    assert set(actual) == set(baseline["key"])
    differences = []
    for row in baseline.to_dicts():
        for field in ("n_laps", "mean_pace", "dirty_air_share"):
            error = abs(row[field] - actual[row["key"]][field])
            assert error < 1e-6
            differences.append(error)
    return dict(groups=len(actual), max_abs_diff=max(differences),
        existing_driver_style_rows=pl.read_parquet(ROOT / "data/gold/driver_style.parquet").height,
        note="driver_style groups season/driver over all sessions and different lap filters; not the labeled driver/session population")


def silver_parity(output=BASE / "lap_aggregate", season="2025"):
    actual = pl.DataFrame([dict(value) for _, value in read_output(output)], infer_schema_length=None)
    baseline = pl.read_parquet(ROOT / f"data/silver/stint_features/season={season}/*.parquet")
    assert not (set(baseline.columns) - set(actual.columns)), "missing Silver columns"
    keys = ["season", "gp", "session", "driver", "lap"]
    common = [c for c in baseline.columns if c in actual.columns and c not in keys]
    joined = baseline.join(actual, on=keys, suffix="_mr", how="full", coalesce=True)
    assert baseline.height == actual.height == joined.height
    errors = {}
    for column in common:
        a, b = joined[column], joined[column + "_mr"]
        assert a.is_null().equals(b.is_null()), f"null mismatch {column}"
        if a.dtype.is_numeric():
            error = (a - b).abs().max()
            assert error is None or error < 1e-6, f"Silver mismatch {column}: {error}"
            errors[column] = error
        else:
            assert a.equals(b), column
    return dict(season=season, rows_matched=actual.height, rows_differing=0, max_abs_diff=errors)


def optimizer_parity(db, frame):
    from src.nosql.repository import Repository
    from src.models.optimizer import get_state, optimize, load_scenarios
    repo = Repository(db)
    gold = pl.read_parquet(ROOT / "data/gold/lap_features.parquet")
    params = json.loads((ROOT / "data/models/degradation_model.json").read_text())
    samples = frame.filter((pl.col("season") == "2025") & (pl.col("session") == "Race"))
    samples = samples.sort(["gp", "driver", "lap"]).group_by(["gp", "driver"], maintain_order=True).first().head(5)
    table = []
    for row in samples.to_dicts():
        args = ("2025", row["gp"], "Race", row["driver"], row["lap"])
        parquet = get_state(frame, *args, lapf=gold)
        mongo = get_state(None, *args, repo=repo)
        assert parquet == mongo, f"state differs for {args}"
        scenarios = load_scenarios(params)
        a = optimize(parquet, params, 30, scenarios)["clean_air"][0]
        b = optimize(mongo, params, 30, scenarios)["clean_air"][0]
        assert a["strategy"] == b["strategy"] and a["mean"] == b["mean"]
        table.append(dict(gp=row["gp"], driver=row["driver"], lap=row["lap"], n_laps=mongo["n_laps"],
                          state_matches=True, recommendation_matches=True, strategy=a["strategy"]))
    assert len(table) == 5
    return table


def mongo_parity(db, reference):
    from src.nosql.aggregation import stint_pipeline
    docs = list(db.laps.aggregate(stint_pipeline(), allowDiskUse=True))
    actual = {r["key"]: r for r in docs}
    assert set(actual) == {r["key"] for r in reference}
    compared = []
    for row in reference:
        mongo = actual[row["key"]]
        assert row["n_laps"] == mongo["n_laps"]
        difference = abs(row["polars"] - mongo["beta"])
        assert difference < 1e-6, f"Mongo beta differs for {row['key']}: {difference}"
        compared.append({**row, "mongo": mongo["beta"], "mongo_abs_diff": difference})
    return dict(n_stints=len(docs), max_abs_diff=max(r["mongo_abs_diff"] for r in compared),
                stints=compared)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver", action="store_true")
    parser.add_argument("--mongo", action="store_true")
    parser.add_argument("--mongo-uri")
    parser.add_argument("--mongo-db")
    parser.add_argument("--output", type=Path, default=ROOT / "data/mapreduce/parity.json")
    args = parser.parse_args(argv)
    frame = pl.read_parquet(ROOT / "data/gold/labeled_frame.parquet")
    result = dict(stint=stint_parity(frame), driver=driver_parity(frame))
    trace_path = BASE / "stint_stats/trace"
    if trace_path.exists():
        result["trace"] = verify_trace(trace_path, reference_stint)
    if args.silver:
        result["silver"] = silver_parity()
    if args.mongo:
        from src.nosql.connection import get_database
        db = get_database(args.mongo_uri, args.mongo_db)
        try:
            result["mongo"] = mongo_parity(db, result["stint"]["stints"])
            result["optimizer"] = optimizer_parity(db, frame)
        finally:
            db.client.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    tables = {k: ({field: value for field, value in data.items() if field != "stints"}
                  if isinstance(data, dict) else data) for k, data in result.items()}
    for name, path in (("explain", ROOT / "data/mongo/explain.json"),
                       ("benchmark", ROOT / "data/mongo/benchmark.json"),
                       ("quality", ROOT / "data/mongo/quality_report.json"),
                       ("ingest_first", ROOT / "data/mongo/ingest_first.json"),
                       ("ingest_second", ROOT / "data/mongo/ingest_second.json")):
        if path.exists():
            tables[name] = json.loads(path.read_text())
    (ROOT / "data/models/phase9_tables.json").write_text(json.dumps(tables, indent=2), encoding="utf-8")
    print(json.dumps({k: ({x: y for x, y in v.items() if x != "stints"} if isinstance(v, dict) else len(v))
                      for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
