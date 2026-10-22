"""Mongo tests use a disposable database; pure data contracts always run."""
import uuid
import pytest


def lap(i, **extra):
    return dict(_id=f"2025|Test|Race|VER|{i}", season="2025", gp="Test", session="Race",
                driver="VER", lap=i, stint=1, life=i, compound="SOFT", laptime=90+i*.2,
                fuel_corr=90+i*.2, is_sc_lap=0, _pos=i-1, cluster=0, **extra)


def test_cleaning_edge_cases():
    from src.nosql.cleaning import clean_document
    d = clean_document(dict(season=2025, lap="12", laptime="None", fuel_corr=float("inf"),
                            had_cliff=1, nested={"value": float("nan")}))
    assert d == dict(season="2025", lap=12, laptime=None, fuel_corr=None,
                     had_cliff=True, nested={"value": None})


def test_session_sc_laps_exclude_unassigned_messages():
    from src.nosql.ingest import sc_lap_numbers
    assert sc_lap_numbers([None, 1, 1, 3]) == [1, 3]


def test_lap_position_after_compound_before_sc_filter(tmp_path):
    import polars as pl
    from src.nosql.ingest import lap_documents
    path = tmp_path/"data/gold"
    path.mkdir(parents=True)
    rows = [lap(i) for i in range(1, 8)]
    rows[1]["compound"] = "UNKNOWN"
    rows[2]["is_sc_lap"] = 1
    pl.DataFrame(rows).write_parquet(path/"labeled_frame.parquet")
    result = {row["lap"]: row for row in lap_documents(tmp_path)}
    assert result[4]["_pos"] == 2
    assert result[5]["_pos"] == 3
    assert len(result) == 7


def test_synthetic_reproducible():
    from src.nosql.synthetic import synthetic_laps, canonical_hash
    a, b = list(synthetic_laps(123)), list(synthetic_laps(123))
    assert a == b
    assert len({d["_id"] for d in a}) == 123
    assert canonical_hash(a) == canonical_hash(b)


def test_cli_help_and_reset_guard():
    from src.nosql.cli import build_parser
    parser = build_parser()
    assert parser.parse_args(["ingest", "--skip-if-unavailable"]).skip_if_unavailable
    assert parser.parse_args(["reset"]).force is False


def test_cli_parity_forwards_connection_and_output(monkeypatch, tmp_path):
    from src.nosql import cli
    from src.mapreduce import evidence
    # Only the external proof-command dispatch is intercepted. No fake Mongo
    # engine is used; the CLI connection can be represented by a closeable owner.
    from types import SimpleNamespace
    closed = []
    monkeypatch.setattr(cli, "get_database", lambda uri, db_name: SimpleNamespace(
        client=SimpleNamespace(close=lambda: closed.append(True))))
    received = []
    monkeypatch.setattr(evidence, "main", lambda argv: received.extend(argv) or 0)
    output = tmp_path/"proof.json"
    assert cli.main(["--uri", "mongodb://example.test:27018", "--db", "custom_f1",
                     "parity", "--output", str(output)]) == 0
    assert received == ["--mongo", "--silver", "--mongo-uri", "mongodb://example.test:27018",
                        "--mongo-db", "custom_f1", "--output", str(output)]
    assert closed == [True]


@pytest.mark.mongo
def test_benchmark_and_quality(db):
    from src.nosql.benchmark import benchmark
    from src.nosql.ingest import quality_report
    from src.nosql.schemas import ensure_collections
    ensure_collections(db)
    result = benchmark(db, n=100, batch_sizes=(10,))
    assert result["without_indexes"]["count"] == result["with_indexes"]["count"] == 100
    assert result["with_indexes"]["stats"]["nindexes"] > result["without_indexes"]["stats"]["nindexes"]
    assert result["reproducible"]
    assert db.bench_laps.count_documents({}) == 100
    assert quality_report(db)["laps"]["violating_validator"] == 0


@pytest.mark.mongo
def test_benchmark_rejects_cached_count_mismatch(db):
    from src.nosql.benchmark import benchmark
    benchmark(db, n=10)
    with pytest.raises(ValueError, match="bench_laps.*10.*100"):
        benchmark(db, n=100)
    assert db.bench_laps.count_documents({}) == 10


@pytest.fixture
def db():
    from src.nosql.connection import get_database
    from pymongo.errors import PyMongoError
    try:
        database = get_database(db_name="test_phase9_" + uuid.uuid4().hex)
    except PyMongoError as exc:
        pytest.skip(f"Mongo unavailable: {exc}")
    yield database
    database.client.drop_database(database.name)
    database.client.close()


@pytest.mark.mongo
def test_schema_validation(db):
    from src.nosql.schemas import ensure_collections
    from pymongo.errors import WriteError
    ensure_collections(db)
    db.laps.insert_one(lap(1))
    nullable = lap(2)
    nullable.update(compound=None, life=None, stint=None)
    db.laps.insert_one(nullable)
    # Gold has 17 real pre-season aggregates with no known stint/compound.
    db.stints.insert_one(dict(_id="2023|Test|Practice|TSU|None", season="2023", gp="Test",
                              session="Practice", driver="TSU", stint=None, compound=None))
    with pytest.raises(WriteError):
        db.laps.insert_one({"_id": "bad", "lap": "not an integer"})


@pytest.mark.mongo
def test_ingest_idempotent(db):
    from src.nosql.schemas import ensure_collections
    from src.nosql.ingest import upsert_documents
    ensure_collections(db)
    a = upsert_documents(db.laps, (lap(i) for i in range(1, 12)), batch_size=3)
    b = upsert_documents(db.laps, (lap(i) for i in range(1, 12)), batch_size=3)
    assert a["upserted"] == 11
    assert b["modified"] == b["upserted"] == 0
    assert db.laps.count_documents({}) == 11


@pytest.mark.mongo
def test_pipeline_results(db):
    from src.nosql.aggregation import stint_slopes, compound_pace, safety_car_share
    db.laps.insert_many([lap(i) for i in range(1, 10)])
    # Outliers in warm-up and SC rows must not enter the estimator.
    db.laps.update_one({"lap": 1}, {"$set": {"fuel_corr": 200}})
    db.laps.update_one({"lap": 7}, {"$set": {"fuel_corr": 400, "is_sc_lap": 1}})
    result = list(db.laps.aggregate(stint_slopes()))
    assert len(result) == 1
    assert result[0]["beta"] == pytest.approx(.2, abs=1e-10)
    assert result[0]["n_laps"] == 5
    assert result[0]["key"] == "2025|Test|Race|VER|1"
    assert list(db.laps.aggregate(compound_pace()))[0]["n_laps"] == 9
    assert list(db.laps.aggregate(safety_car_share()))[0]["sc_share"] == pytest.approx(1/9)


@pytest.mark.mongo
def test_slope_counts_null_life_rows_like_polars(db):
    import polars as pl
    from src.nosql.aggregation import stint_slopes
    rows = [lap(i) for i in range(4, 8)]
    rows[0]["life"] = None
    rows[0]["fuel_corr"] = 200
    db.laps.insert_many(rows)
    frame = pl.DataFrame(rows)
    expected = frame.select(pl.cov("life", "fuel_corr")/pl.var("life")).item()
    result = list(db.laps.aggregate(stint_slopes()))
    assert len(result) == 1
    assert result[0]["n_laps"] == 4
    assert result[0]["beta"] == pytest.approx(expected)


@pytest.mark.mongo
def test_explain_uses_index(db):
    from src.nosql.indexes import create_indexes, explain_query
    db.laps.insert_many([lap(i) for i in range(1, 100)])
    create_indexes(db)
    filt = {"compound": "SOFT", "life": {"$gte": 90}}
    before = explain_query(db.laps, filt, hint={"$natural": 1})
    after = explain_query(db.laps, filt, hint="compound_life")
    assert before["stage"] == "COLLSCAN"
    assert after["stage"] == "IXSCAN"
    assert before["nReturned"] == after["nReturned"] == 10
    assert after["totalDocsExamined"] < before["totalDocsExamined"]


@pytest.mark.mongo
def test_repository_history_and_histogram(db):
    from src.nosql.repository import Repository
    db.laps.insert_many([lap(i) for i in range(1, 9)])
    db.race_sessions.insert_one({"_id": "2025|Test|Race", "n_laps": 60})
    db.strategy_decisions.insert_many([dict(_id=str(i), season="2025", gp="Test", driver="VER",
                                          compounds=["SOFT"], delta=i*10) for i in range(5)])
    r = Repository(db)
    assert len(r.lap_history("2025", "Test", "Race", "VER")) == 8
    assert r.session_lap_count("2025", "Test", "Race") == 60
    assert r.stint_for("2025", "Test", "Race", "VER", 5)["lap"] == 5
    assert len(r.decisions(compound="SOFT", page=1, page_size=2)) == 2
    assert sum(v["count"] for v in r.delta_histogram(compound="SOFT")) == 5


@pytest.mark.mongo
def test_crud_cycle(db):
    from src.nosql.queries import crud_demo, delete_matching
    result = crud_demo(db)
    assert result["created"] and result["updated"] and result["deleted"]
    db.strategy_decisions.insert_one({"_id": "x", "gp": "Test"})
    assert delete_matching(db.strategy_decisions, {"gp": "Test"})["deleted"] == 0
    assert delete_matching(db.strategy_decisions, {"gp": "Test"}, force=True)["deleted"] == 1
    with pytest.raises(ValueError):
        delete_matching(db.strategy_decisions, {}, force=True)


@pytest.mark.mongo
def test_cursor_batch_size(db):
    from src.nosql.stream import scan_laps
    db.bench_laps.insert_many([lap(i) for i in range(1, 24)])
    a, b = scan_laps(db.bench_laps, 1), scan_laps(db.bench_laps, 10)
    assert a["count"] == b["count"] == 23
    assert a["mean_laptime"] == pytest.approx(b["mean_laptime"])
