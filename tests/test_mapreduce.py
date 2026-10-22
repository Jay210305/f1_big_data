"""Independent runtime and estimator checks, using small real files."""
import json
from pathlib import Path

import polars as pl
import pytest


def words(record, ctx):
    return [(word, 1) for word in record["text"].split()]


def total(key, values):
    return [(key, sum(values))]


def reference(key, values):
    count = 0
    for value in values:
        count += value
    return [(key, count)]


def test_wordcount_trace_retry_and_commit(tmp_path):
    from src.mapreduce import JobSpec, LocalJobRunner
    from src.mapreduce.verify import verify_trace
    from src.mapreduce.io import read_output

    source = tmp_path / "words.parquet"
    pl.DataFrame({"text": ["a b a a", "b c", ""]}).write_parquet(source, row_group_size=1)
    spec = JobSpec("words", words, total, combiner=total, num_partitions=2)
    clean = LocalJobRunner(workers=2).run(spec, source, tmp_path / "clean", trace=True)
    retried = LocalJobRunner(workers=2).run(
        spec, source, tmp_path / "retried", trace=True, inject_fault=("map", 1, 1))
    assert dict(read_output(clean["output"])) == {"a": 3, "b": 2, "c": 1}
    assert retried["retries"] == 1
    assert retried["kv_after_combine"] < retried["kv_before_combine"]
    assert verify_trace(clean["trace"], reference, combiner=reference)["verified"]
    for p, q in zip(sorted(Path(clean["output"]).glob("part-*")),
                    sorted(Path(retried["output"]).glob("part-*"))):
        assert p.read_bytes() == q.read_bytes()


def test_exhausted_task_never_commits(tmp_path):
    from src.mapreduce import JobSpec, LocalJobRunner
    source = tmp_path / "data.parquet"
    pl.DataFrame({"text": ["a"]}).write_parquet(source)
    with pytest.raises(RuntimeError, match="map-0"):
        LocalJobRunner(workers=1).run(JobSpec("fail", words, total, max_attempts=2),
            source, tmp_path / "output", inject_fault=("map", 0, 3))
    assert not (tmp_path / "output").exists()


def test_reduce_retry_and_failed_overwrite_preserves_commit(tmp_path):
    from src.mapreduce import JobSpec, LocalJobRunner
    from src.mapreduce.io import read_output
    source = tmp_path / "data.parquet"
    pl.DataFrame({"text": ["a b a"]}).write_parquet(source)
    spec = JobSpec("words", words, total, max_attempts=2)
    output = tmp_path / "output"
    runner = LocalJobRunner(workers=1)
    result = runner.run(spec, source, output, inject_fault=("reduce", 0, 1))
    assert result["retries"] == 1
    before = {p.name: p.read_bytes() for p in output.glob("part-*")}
    with pytest.raises(RuntimeError, match="reduce-0"):
        runner.run(spec, source, output, overwrite=True, inject_fault=("reduce", 0, 2))
    assert before == {p.name: p.read_bytes() for p in output.glob("part-*")}
    assert dict(read_output(output)) == {"a": 2, "b": 1}


def test_trace_detects_corruption_and_refuses_sampling(tmp_path):
    from src.mapreduce import JobSpec, LocalJobRunner
    from src.mapreduce.verify import verify_trace
    source = tmp_path / "data.parquet"
    pl.DataFrame({"text": ["a a b"]}).write_parquet(source)
    result = LocalJobRunner(workers=1).run(JobSpec("words", words, total), source,
                                         tmp_path / "output", trace=True)
    shuffle = Path(result["trace"]) / "shuffle.jsonl"
    shuffle.write_text(shuffle.read_text() + shuffle.read_text().splitlines()[0] + "\n")
    with pytest.raises(AssertionError):
        verify_trace(result["trace"], reference)
    sampled = LocalJobRunner(workers=1).run(JobSpec("words", words, total), source,
            tmp_path / "sampled", trace=True, trace_sample=1)
    with pytest.raises(ValueError, match="sample"):
        verify_trace(sampled["trace"], reference)


def test_partitioners_stable_and_cover_ranges():
    from src.mapreduce.partitioner import HashPartitioner, RangePartitioner, IdentityPartitioner
    assert HashPartitioner()("VER", 8) == HashPartitioner()("VER", 8)
    p = RangePartitioner(["b", "d"])
    assert [p(k, 3) for k in ["a", "b", "c", "d", "z"]] == [0, 1, 1, 2, 2]
    assert IdentityPartitioner()("any", 4) == 0
    with pytest.raises(ValueError):
        HashPartitioner()("x", 0)


def test_stint_warmup_sc_and_model_rollup():
    from src.mapreduce.jobs.stint_stats import reduce_stint, rollup
    rows = [{"life": i, "lap": i + 1, "fuel_corr": 100 + i * .1,
             "compound": "SOFT", "is_sc_lap": 0, "cluster": -1,
             "dirty_air_frac": .2, "track_temp_mean": 30} for i in range(8)]
    rows[0]["fuel_corr"] = 999
    rows[7].update(is_sc_lap=1, fuel_corr=999)
    result = reduce_stint("2025|Test|Race|VER|1", rows)
    assert result[0][1]["beta"] == pytest.approx(.1)
    assert result[0][1]["n_laps"] == 4
    assert rollup([result[0][1]])["beta"]["SOFT"] == pytest.approx(.1)
    assert reduce_stint("short", rows[:6]) == []


def test_stint_null_life_sorts_before_warmup_like_polars():
    from src.mapreduce.jobs.stint_stats import reduce_stint
    rows = [{"life": i, "lap": i+1, "fuel_corr": 100+i*.1, "compound": "SOFT"}
             for i in range(8)]
    rows[0]["life"] = None
    assert reduce_stint("test", rows)[0][1]["beta"] == pytest.approx(.1)


def test_telemetry_partial_merge_matches_silver():
    from src.ingest.etl import _silver_exprs
    from src.mapreduce.jobs.lap_aggregate import partials, reduce_lap
    meta = dict(season="2025", gp="Test", session="Race", driver="VER", lap=1)
    df = pl.DataFrame({"speed": [100., None, 200., 300.], "throttle": [0., 50., None, 100.],
                       "brake": [0., 1., None, 0.], "drs": [0, 1, 0, None],
                       "acc_x": [-1., 2., None, 3.], "distance": [0., 100., 200., 300.],
                       "DistanceToDriverAhead": [None, 5., 20., 0.],
                       **{k: [v] * 4 for k, v in meta.items()}})
    parts = [next(partials(df.slice(i, 2)))["_partial"] for i in (0, 2)]
    actual = reduce_lap("2025|Test|Race|VER|1", parts)[0][1]
    expected = df.select(_silver_exprs(df.columns, meta)).row(0, named=True)
    for key, value in expected.items():
        if isinstance(value, float):
            assert actual[key] == pytest.approx(value), key
        else:
            assert actual[key] == value, key


def test_labeled_export_cluster_and_cached_regression(tmp_path):
    from src.models.export_labeled import export_labeled
    frame = pl.DataFrame(dict(season=["2025", "2025"], driver=["VER", "HAM"],
        fuel_corr=[100., 101.], pace_above_best=[0., 1.], is_sc_lap=[0, 0],
        session_type=["Race", "Race"]))
    emb = pl.DataFrame(dict(season=["2025"], driver=["VER"], cluster=[1]))
    output = export_labeled(tmp_path / "labeled.parquet", expected_rows=2,
                            builder=lambda: frame, embeddings=emb)
    assert pl.read_parquet(output)["cluster"].to_list() == [1, -1]
    with pytest.raises(ValueError, match="row count"):
        export_labeled(output, expected_rows=3)


def test_driver_degradation_averages_stints_without_mixing_pace_offsets():
    from src.mapreduce.jobs.driver_profile import map_lap, reduce_driver
    rows = [dict(season="2025", gp="Test", session="Race", session_type="Race", driver="VER",
        stint=stint, lap=i+1+(stint-1)*8, life=i+(stint-1)*10, compound="SOFT",
        fuel_corr=100*stint+i*beta, laptime=100*stint+i*beta, is_sc_lap=0,
        dirty_air_frac=.1, speed_mean=200., throttle_mean=80.)
        for stint, beta in ((1, 1.), (2, 3.)) for i in range(8)]
    values = [map_lap(row, None)[0][1] for row in rows]
    profile = reduce_driver("VER|Race", values)[0][1]
    assert profile["degradation_slope"] == pytest.approx(2.)
