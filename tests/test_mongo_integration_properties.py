"""Independent Mongo/integration audit properties; all writes use UUID test DBs."""
import json
import uuid

import polars as pl
import pytest


def document(i, **overrides):
    row = dict(_id=f"2025|Audit GP|Race|VER|{i}", season="2025", gp="Audit GP",
               session="Race", driver="VER", lap=i, life=i, stint=1,
               compound="SOFT", fuel_corr=100 + .2*i, laptime=100 + .2*i,
               is_sc_lap=0, _pos=i-1, cluster=-1, pace_above_best=.2*i,
               track_temp_mean=30., dirty_air_frac=.1)
    return {**row, **overrides}


@pytest.fixture
def audit_db():
    from src.nosql.connection import get_database
    from pymongo.errors import PyMongoError
    name = "test_phase9_audit_" + uuid.uuid4().hex
    try:
        database = get_database(db_name=name)
    except PyMongoError as exc:
        pytest.skip(f"Mongo unavailable: {exc}")
    try:
        yield database
    finally:
        assert database.name == name and name.startswith("test_phase9_audit_")
        database.client.drop_database(name)
        database.client.close()


@pytest.mark.mongo
def test_actual_bson_nulls_validator_and_required_keys(audit_db):
    from src.nosql.schemas import ensure_collections
    from src.nosql.ingest import upsert_documents, quality_report
    from pymongo.errors import WriteError
    ensure_collections(audit_db)
    values = ("None", "none", "null", "N/A", "nan", "", float("nan"), float("inf"))
    upsert_documents(audit_db.laps, [document(i, laptime=v, fuel_corr=v)
                                    for i, v in enumerate(values, 1)], 3)
    assert audit_db.laps.count_documents({"laptime": {"$type": "null"}}) == len(values)
    assert audit_db.laps.count_documents({"fuel_corr": {"$type": "null"}}) == len(values)
    with pytest.raises(WriteError):
        audit_db.laps.insert_one(document(100, lap="100"))
    invalid = document(101)
    del invalid["driver"]
    with pytest.raises(WriteError):
        audit_db.laps.insert_one(invalid)
    assert quality_report(audit_db)["laps"]["violating_validator"] == 0


@pytest.mark.mongo
def test_insert_reingest_update_and_set_preserves_omitted_fields(audit_db):
    from src.nosql.ingest import upsert_documents
    from pymongo.errors import DuplicateKeyError
    original = document(1, old_source_field=1)
    initial = upsert_documents(audit_db.laps, [original], 1)
    repeated = upsert_documents(audit_db.laps, [original], 1)
    changed = upsert_documents(audit_db.laps, [document(1, fuel_corr=123.)], 1)
    assert (initial["upserted"], repeated["modified"], changed["modified"]) == (1, 0, 1)
    assert audit_db.laps.count_documents({}) == 1
    assert audit_db.laps.find_one()["fuel_corr"] == 123.
    # Upsert is intentionally $set, not replacement or deletion synchronization.
    assert audit_db.laps.find_one()["old_source_field"] == 1
    with pytest.raises(DuplicateKeyError):
        audit_db.laps.insert_one(document(1))


@pytest.mark.mongo
@pytest.mark.parametrize("null_x,null_y", [(None, None), (4, None), (None, 4), (4, 5), (5, 5)])
def test_nullable_aggregation_matches_independent_polars(audit_db, null_x, null_y):
    from src.nosql.aggregation import stint_slopes, compound_pace, safety_car_share
    rows = [document(i, life=None if i == null_x else i,
                     fuel_corr=None if i == null_y else 100+.2*i)
            for i in range(4, 10)]
    audit_db.laps.insert_many(rows)
    expected = pl.DataFrame(rows).select(pl.cov("life", "fuel_corr")/pl.var("life")).item()
    result = list(audit_db.laps.aggregate(stint_slopes()))
    assert len(result) == 1
    assert result[0]["n_laps"] == len(rows)
    assert result[0]["beta"] == pytest.approx(expected, abs=1e-11)
    assert list(audit_db.laps.aggregate(compound_pace()))[0]["n_laps"] == len(rows)
    assert list(audit_db.laps.aggregate(safety_car_share()))[0]["sc_share"] == 0


@pytest.mark.mongo
def test_history_filter_page_union_and_histogram_conservation(audit_db):
    from src.nosql.repository import Repository
    rows = [dict(_id=str(i), season="2025", gp="Audit GP", driver="VER", scenario=f"s{i:03}",
                 compounds=["SOFT", "HARD"], delta=None if i == 8 else (i-4)*10)
            for i in range(9)]
    audit_db.strategy_decisions.insert_many(rows)
    audit_db.strategy_decisions.insert_one({**rows[0], "_id": "excluded", "driver": "HAM"})
    repo = Repository(audit_db)
    filters = dict(season=2025, gp="Audit GP", driver="VER", compound="HARD")
    pages = [repo.decisions(**filters, page=i, page_size=2) for i in range(6)]
    flattened = [row for page in pages for row in page]
    assert len(flattened) == repo.decision_count(**filters) == 9
    assert len({row["scenario"] for row in flattened}) == 9
    assert [row["scenario"] for row in flattened] == sorted(row["scenario"] for row in flattened)
    bins = repo.delta_histogram(**filters, width=20)
    assert sum(row["count"] for row in bins) == 8  # null is excluded, negative bins use floor
    assert [(b["bin_start"], b["count"]) for b in bins] == [(-40, 2), (-20, 2), (0, 2), (20, 2)]
    for kwargs in ({"page": -1}, {"page_size": 0}, {"page_size": 1001}):
        with pytest.raises(ValueError):
            repo.decisions(**kwargs)
    for width in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            repo.delta_histogram(width=width)


@pytest.mark.mongo
def test_hinted_index_and_stream_conserve_population(audit_db):
    from src.nosql.indexes import create_indexes, explain_query
    from src.nosql.stream import scan_laps
    audit_db.laps.insert_many([document(i) for i in range(1, 101)])
    create_indexes(audit_db)
    query = dict(compound="SOFT", life={"$gte": 90})
    before = explain_query(audit_db.laps, query, hint={"$natural": 1})
    after = explain_query(audit_db.laps, query, hint="compound_life")
    assert before["nReturned"] == after["nReturned"] == audit_db.laps.count_documents(query) == 11
    assert before["stage"] == "COLLSCAN" and after["stage"] == "IXSCAN"
    assert after["totalDocsExamined"] <= before["totalDocsExamined"]
    scans = [scan_laps(audit_db.laps, n, materialize=m)
             for n, m in ((1, False), (17, False), (17, True))]
    assert {s["count"] for s in scans} == {100}
    assert all(s["mean_laptime"] == pytest.approx(110.1) for s in scans)
    assert all("not total process RSS" in s["memory_metric"] for s in scans)


def test_scoped_sources_filter_and_assign_stable_ids(tmp_path):
    from src.nosql.ingest import lap_documents, stint_documents, decision_documents
    gold, models = tmp_path/"data/gold", tmp_path/"data/models"
    gold.mkdir(parents=True)
    models.mkdir(parents=True)
    rows = [document(i) for i in range(1, 9)] + [document(1, season="2024"), document(1, gp="Other")]
    pl.DataFrame(rows).write_parquet(gold/"labeled_frame.parquet")
    pl.DataFrame([document(1), document(1, season="2024")]).write_parquet(gold/"stint_degradation.parquet")
    (models/"phase7_backtest.json").write_text(json.dumps(dict(season=2025,
        summary=dict(scenario="clean_air"), decisions=[dict(gp="Audit GP", driver="VER", delta=1.,
        rec_strategy="SOFT->18,MEDIUM->end"), dict(gp="Other", driver="VER", delta=1.)])))
    scope = dict(season="2025", gp="Audit GP", session="Race")
    laps = list(lap_documents(tmp_path, **scope))
    assert len(laps) == len({r["_id"] for r in laps}) == 8
    assert [r["_pos"] for r in laps] == list(range(8))
    assert len(list(stint_documents(tmp_path, **scope))) == 1
    decisions = list(decision_documents(tmp_path, **scope))
    assert len(decisions) == 1 and decisions[0]["compounds"] == ["MEDIUM", "SOFT"]


def test_mongo_connection_failure_falls_back_with_explicit_message(capsys):
    from src.models.optimizer import get_state
    from pymongo.errors import ConnectionFailure
    class Unavailable:
        def lap_history(self, *args):
            raise ConnectionFailure("audit disconnection")
    rows = pl.DataFrame([document(i) for i in range(1, 9)])
    lapf = pl.DataFrame([dict(season="2025", gp="Audit GP", session="Race", lap=57)])
    expected = get_state(rows, "2025", "Audit GP", "Race", "VER", 4, lapf=lapf)
    actual = get_state(rows, "2025", "Audit GP", "Race", "VER", 4, lapf=lapf, repo=Unavailable())
    assert actual == expected and actual["n_laps"] == 57
    assert "using parquet" in capsys.readouterr().out


def test_mongo_missing_history_fails_actionably():
    from src.models.optimizer import get_state
    class Empty:
        def lap_history(self, *args):
            return []
    with pytest.raises(ValueError, match="run ingest"):
        get_state(None, "2025", "Audit GP", "Race", "VER", 4, repo=Empty())


def test_offline_cli_skip_and_error_status(monkeypatch, capsys):
    from src.nosql import cli
    from pymongo.errors import ServerSelectionTimeoutError
    def unavailable(*args):
        raise ServerSelectionTimeoutError("audit unavailable")
    monkeypatch.setattr(cli, "get_database", unavailable)
    assert cli.main(["health"]) == 1
    assert cli.main(["ingest", "--skip-if-unavailable"]) == 0
    assert "SKIP MongoDB unavailable" in capsys.readouterr().out


@pytest.mark.mongo
# Audit M01 (fixed): zero batch size reported n loaded while inserting none.
def test_synthetic_loader_rejects_zero_batch_size(audit_db):
    from src.nosql.synthetic import load_synthetic
    with pytest.raises(ValueError, match="batch_size"):
        load_synthetic(audit_db.bench_laps, n=5, batch_size=0)


@pytest.mark.mongo
# Audit M02 (fixed): the count-only cache accepted corrupted documents.
def test_benchmark_reproducibility_validates_existing_cache_content(audit_db):
    from src.nosql.benchmark import benchmark
    from src.nosql.synthetic import synthetic_laps, canonical_hash
    audit_db.bench_laps.insert_many(list(synthetic_laps(10)))
    audit_db.bench_laps.update_one({"_id": "bench-0"}, {"$set": {"laptime": -999.}})
    result = benchmark(audit_db, n=10, batch_sizes=(5,))
    stored = list(audit_db.bench_laps.find().sort("_id", 1))
    assert not result["reproducible"] or canonical_hash(stored) == canonical_hash(synthetic_laps(10))


@pytest.mark.mongo
# Audit I01 (fixed): a canonical BSON null compound made get_state sorting fail.
def test_mongo_state_tolerates_unknown_compound_in_prior_history(audit_db):
    from src.nosql.repository import Repository
    from src.nosql.ingest import upsert_documents
    from src.models.optimizer import get_state
    rows = [document(1, compound="None"), document(2, compound="SOFT")]
    upsert_documents(audit_db.laps, rows)
    audit_db.race_sessions.insert_one(dict(_id="2025|Audit GP|Race", n_laps=57))
    state = get_state(None, "2025", "Audit GP", "Race", "VER", 2, repo=Repository(audit_db))
    assert state["compound"] == "SOFT" and state["n_laps"] == 57


# Audit I02 (fixed): a rebuilt style artifact left the labeled/MR caches stale.
def test_pipeline_invalidates_labeled_after_missing_style_stage_reruns(tmp_path, monkeypatch):
    from src import run_all
    monkeypatch.setattr(run_all, "ROOT", tmp_path)
    monkeypatch.setattr(run_all, "DATA", tmp_path/"data")
    monkeypatch.setattr(run_all, "BRONZE", tmp_path/"data/bronze/telemetry")
    monkeypatch.setattr(run_all, "GOLD", tmp_path/"data/gold")
    monkeypatch.setattr(run_all, "MODELS", tmp_path/"data/models")
    paths = ["data/bronze/telemetry/fixture.parquet", "data/bronze/phase2_schema_verify.json",
             "data/gold/lap_features.parquet", "data/models/phase4_metrics.json",
             "data/gold/labeled_frame.parquet", "data/models/degradation_model.json",
             "data/models/phase6_demo.json", "data/mapreduce/parity.json"]
    for name in paths:
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x"*1001)
    commands = []
    def fake_run(command):
        commands.append(command)
        if "src.models.train_style" in command:
            # Stage really ran: its new cluster artifact can differ from cached labels.
            (tmp_path/"data/models/phase5_metrics.json").write_bytes(b"new"*400)
    monkeypatch.setattr(run_all, "_run", fake_run)
    monkeypatch.setattr("sys.argv", ["run_all"])
    run_all.main()
    assert any("src.models.train_style" in command for command in commands)
    assert any("src.models.export_labeled" in command for command in commands)


# Audit M03 (fixed): the decision source ignored the non-Race session scope.
def test_scoped_ingest_does_not_include_race_decisions_in_practice_scope(tmp_path):
    from src.nosql.ingest import decision_documents
    models = tmp_path/"data/models"
    models.mkdir(parents=True)
    (models/"phase7_backtest.json").write_text(json.dumps(dict(season=2025,
        summary=dict(scenario="clean_air"), decisions=[dict(gp="Audit GP", driver="VER", delta=1.)])))
    assert list(decision_documents(tmp_path, season="2025", gp="Audit GP", session="Practice 1")) == []
