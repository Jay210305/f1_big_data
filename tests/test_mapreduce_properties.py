"""Seeded audit properties; small files only, no Mongo or lakehouse mutation."""
from collections import Counter
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

import polars as pl
import pytest

from src.mapreduce import JobSpec, LocalJobRunner
from src.mapreduce.io import read_jsonl, read_output, write_jsonl
from src.mapreduce.partitioner import HashPartitioner, IdentityPartitioner, RangePartitioner
from src.mapreduce.verify import verify_trace


def emit_words(record, ctx):
    return [(word, 1) for word in record["text"].split()]


def sum_values(key, values):
    return [(key, sum(values))]


def independent_count(key, values):
    answer = 0
    for number in values:
        answer += number
    return [(key, answer)]


def partial_mapper_failure(record, ctx):
    if ctx.attempt == 1 and record["text"] == "fail":
        raise RuntimeError("failure after prior emitted values")
    return emit_words(record, ctx)


def partial_sink_failure(path, rows):
    write_jsonl(path, rows)
    if path.parent.name.endswith("-1"):
        raise RuntimeError("failure after writing a complete provisional part")


def invalid_fractional_partition(key, count):
    return 0.5


invalid_fractional_partition.partition_bounds = lambda: {"method": "invalid"}


def output_bytes(path):
    return {p.name: p.read_bytes() for p in Path(path).glob("part-*")}


@pytest.mark.parametrize("partitioner,partitions", [
    (HashPartitioner(), 7), (RangePartitioner(("d", "m", "t")), 4),
    (IdentityPartitioner(), 5),
])
@pytest.mark.parametrize("combine", [False, True])
def test_seeded_zero_to_many_shuffle_and_partition_conservation(tmp_path, partitioner, partitions, combine):
    rng = random.Random(917)
    texts = [" ".join(rng.choice(["a", "b", "d", "m", "z", "ñ", "😀"])
                      for _ in range(rng.randrange(8))) for _ in range(101)]
    texts[0] = ""  # Explicit zero-emission record.
    source = tmp_path / "input.parquet"
    pl.DataFrame({"text": texts}).write_parquet(source, row_group_size=13)
    expected = Counter(word for text in texts for word in text.split())
    spec = JobSpec("audit", emit_words, sum_values, num_partitions=partitions,
                   partitioner=partitioner, combiner=sum_values if combine else None)
    report = LocalJobRunner(2).run(spec, source, tmp_path / "out", trace=True)
    assert dict(read_output(report["output"])) == expected
    assert report["input_records"] == len(texts)
    assert report["kv_before_combine"] == sum(expected.values())
    assert report["reduce_tasks"] == partitions
    assert len(output_bytes(report["output"])) == partitions
    assert verify_trace(report["trace"], independent_count,
                        combiner=independent_count if combine else None)["verified"]
    for stage in ("map", "combine", "shuffle", "reduce"):
        for row in read_jsonl(Path(report["trace"]) / (stage + ".jsonl")):
            assert row["partition"] == partitioner(row["key"], partitions)
            if stage in ("shuffle", "reduce"):
                assert row["task"] == row["partition"]


def test_integer_reduction_is_byte_stable_across_workers_splits_and_combining(tmp_path):
    texts = ["a b b c" * (i % 4) for i in range(57)]
    snapshots = []
    for workers, group_size, combine in [(1, 57, False), (3, 7, False), (2, 11, True)]:
        source = tmp_path / f"in-{workers}.parquet"
        pl.DataFrame({"text": texts}).write_parquet(source, row_group_size=group_size)
        spec = JobSpec("stable", emit_words, sum_values,
                       combiner=sum_values if combine else None)
        result = LocalJobRunner(workers).run(spec, source, tmp_path / f"out-{workers}")
        snapshots.append(output_bytes(result["output"]))
    assert snapshots[0] == snapshots[1] == snapshots[2]


def test_builtin_hash_is_independent_of_python_hash_seed():
    program = ("import json; from src.mapreduce.partitioner import HashPartitioner; "
               "print(json.dumps([HashPartitioner()(k,n) for k in ['', 'VER', 'ñ', '😀'] "
               "for n in [1,2,7,31]]))")
    import os
    outputs = [subprocess.check_output([sys.executable, "-c", program],
               env={**os.environ, "PYTHONHASHSEED": seed}) for seed in ("1", "8142")]
    assert outputs[0] == outputs[1]
    rng = random.Random(123)
    p = RangePartitioner(("t", "d", "m"))
    keys = sorted("".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(3))
                  for _ in range(100))
    buckets = [p(k, 4) for k in keys]
    assert buckets == sorted(buckets)
    assert [p(k, 4) for k in ("d", "m", "t")] == [1, 2, 3]
    with pytest.raises(ValueError):
        p("a", 3)


def test_all_records_filtered_produces_empty_complete_part_set(tmp_path):
    source = tmp_path / "in.parquet"
    pl.DataFrame({"text": ["", " ", ""]}).write_parquet(source, row_group_size=1)
    result = LocalJobRunner(1).run(JobSpec("zero", emit_words, sum_values), source,
                                  tmp_path / "out", trace=True)
    assert result["input_records"] == 3
    assert result["kv_before_combine"] == result["kv_after_combine"] == 0
    assert list(read_output(result["output"])) == []
    assert len(output_bytes(result["output"])) == 4
    assert verify_trace(result["trace"], independent_count)["keys"] == 0


def test_absent_splits_fail_cleanly_before_publication(tmp_path):
    source = tmp_path / "no-input"
    source.mkdir()
    with pytest.raises(ValueError, match="no input splits"):
        LocalJobRunner(1).run(JobSpec("none", emit_words, sum_values), source, tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob("_tmp-*"))


def test_trace_sampling_is_globally_capped_and_refuses_proof(tmp_path):
    source = tmp_path / "in.parquet"
    pl.DataFrame({"text": ["a a b"] * 9}).write_parquet(source, row_group_size=1)
    result = LocalJobRunner(2).run(JobSpec("sample", emit_words, sum_values,
                                         combiner=sum_values), source,
                                  tmp_path / "out", trace=True, trace_sample=2)
    for stage in ("map", "combine", "shuffle", "reduce"):
        assert len(list(read_jsonl(Path(result["trace"]) / f"{stage}.jsonl"))) <= 2
    assert result["kv_before_combine"] == 27
    with pytest.raises(ValueError, match="sampled"):
        verify_trace(result["trace"], independent_count, combiner=independent_count)


@pytest.mark.parametrize("stage", ["map", "reduce"])
def test_partial_attempt_outputs_do_not_survive_retries(tmp_path, stage):
    source = tmp_path / "in.parquet"
    pl.DataFrame({"text": ["a a", "fail", "b"]}).write_parquet(source)
    clean = LocalJobRunner(1).run(JobSpec("clean", emit_words, sum_values),
                                 source, tmp_path / "clean", trace=True)
    spec = JobSpec("retry", partial_mapper_failure if stage == "map" else emit_words,
                   sum_values, output_sink=partial_sink_failure if stage == "reduce" else None)
    report = LocalJobRunner(1).run(spec, source, tmp_path / "retry", trace=True)
    assert report["retries"] == (1 if stage == "map" else spec.num_partitions)
    assert output_bytes(report["output"]) == output_bytes(clean["output"])
    assert verify_trace(report["trace"], independent_count)["verified"]
    assert not list(tmp_path.glob("_tmp-*"))


@pytest.mark.parametrize("stage", ["map", "reduce"])
def test_exhaustion_preserves_previous_output_and_cleans_temporary(tmp_path, stage):
    source = tmp_path / "in.parquet"
    pl.DataFrame({"text": ["a"]}).write_parquet(source)
    output = tmp_path / "out"
    runner = LocalJobRunner(1)
    spec = JobSpec("fail", emit_words, sum_values, max_attempts=2)
    runner.run(spec, source, output)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(RuntimeError, match=f"{stage}-0"):
        runner.run(spec, source, output, overwrite=True, inject_fault=(stage, 0, 2))
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
    assert not list(tmp_path.glob("_tmp-*"))


@pytest.fixture(scope="module")
def trace_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("trace-source")
    source = root / "in.parquet"
    pl.DataFrame({"text": ["a a b", "c", ""]}).write_parquet(source, row_group_size=1)
    result = LocalJobRunner(1).run(JobSpec("trace", emit_words, sum_values),
                                  source, root / "out", trace=True)
    return Path(result["trace"])


@pytest.mark.parametrize("corruption", ["lost", "duplicate", "changed_value", "sampled"])
def test_trace_rejects_conservation_and_sampling_corruption(tmp_path, trace_source, corruption):
    trace = tmp_path / "trace"
    shutil.copytree(trace_source, trace)
    shuffle = trace / "shuffle.jsonl"
    rows = list(read_jsonl(shuffle))
    if corruption == "lost":
        rows.pop()
    elif corruption == "duplicate":
        rows.append(rows[0])
    elif corruption == "changed_value":
        rows[0]["value"] += 1
    else:
        manifest = json.loads((trace / "manifest.json").read_text())
        manifest["sampled"] = True
        (trace / "manifest.json").write_text(json.dumps(manifest))
    write_jsonl(shuffle, rows)
    with pytest.raises((AssertionError, ValueError)):
        verify_trace(trace, independent_count)


@pytest.mark.parametrize("stage", ["shuffle", "reduce"])
def test_trace_rejects_wrong_reducer_task_identity(tmp_path, trace_source, stage):
    """Intended integrity property: trace task must be its recorded reducer bucket."""
    trace = tmp_path / "trace"
    shutil.copytree(trace_source, trace)
    path = trace / f"{stage}.jsonl"
    rows = list(read_jsonl(path))
    rows[0]["task"] = rows[0]["partition"] + 1000
    write_jsonl(path, rows)
    with pytest.raises((AssertionError, ValueError)):
        verify_trace(trace, independent_count)


def test_trace_rejects_partition_outside_manifest_range(tmp_path, trace_source):
    trace = tmp_path / "trace"
    shutil.copytree(trace_source, trace)
    for name in ("map", "shuffle", "reduce"):
        rows = list(read_jsonl(trace / f"{name}.jsonl"))
        for row in rows:
            row["partition"] = 9999
            if name in ("shuffle", "reduce"):
                row["task"] = 9999
        write_jsonl(trace / f"{name}.jsonl", rows)
    with pytest.raises((AssertionError, ValueError)):
        verify_trace(trace, independent_count)


def test_commit_promotion_exception_restores_previous_result(tmp_path, monkeypatch):
    source = tmp_path / "in.parquet"
    pl.DataFrame({"text": ["a"]}).write_parquet(source)
    output = tmp_path / "out"
    spec = JobSpec("commit", emit_words, sum_values)
    runner = LocalJobRunner(1)
    runner.run(spec, source, output)
    previous = output_bytes(output)
    rename = Path.rename
    def fail_promotion(path, target):
        if path.name == "committed":
            raise PermissionError("audit promotion failure")
        return rename(path, target)
    monkeypatch.setattr(Path, "rename", fail_promotion)
    with pytest.raises(PermissionError, match="audit promotion"):
        runner.run(spec, source, output, overwrite=True)
    assert output_bytes(output) == previous


def test_overwrite_keeps_output_visible_at_commit_boundary(tmp_path, monkeypatch):
    """The stronger concurrent-reader atomicity claim requires continuous visibility."""
    source = tmp_path / "in.parquet"
    pl.DataFrame({"text": ["a"]}).write_parquet(source)
    output = tmp_path / "out"
    spec = JobSpec("commit", emit_words, sum_values)
    runner = LocalJobRunner(1)
    runner.run(spec, source, output)
    visibility = []
    rename = Path.rename
    def observe_boundary(path, target):
        answer = rename(path, target)
        if path == output:
            visibility.append(output.exists())
        return answer
    monkeypatch.setattr(Path, "rename", observe_boundary)
    runner.run(spec, source, output, overwrite=True)
    assert visibility == [True], "old output is absent before replacement promotion"


def test_rejects_fractional_custom_partition_instead_of_silently_losing_values(tmp_path):
    source = tmp_path / "in.parquet"
    pl.DataFrame({"text": ["a"]}).write_parquet(source)
    spec = JobSpec("invalid", emit_words, sum_values, partitioner=invalid_fractional_partition,
                   max_attempts=1)
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        LocalJobRunner(1).run(spec, source, tmp_path / "out")


def test_seeded_telemetry_sufficient_statistics_and_null_parity(monkeypatch):
    from src.ingest.etl import _silver_exprs
    from src.mapreduce.jobs import lap_aggregate as lap
    monkeypatch.setattr(lap, "_lap_metadata", lambda *args: {})
    rng = random.Random(915)
    meta = dict(season="2099", gp="Audit", session="Race", driver="TST", lap=1)
    for case in range(30):
        n = rng.randrange(1, 35)
        columns = {k: [v] * n for k, v in meta.items()}
        for channel in [*lap.NUMERIC, "brake", "drs", "DistanceToDriverAhead"]:
            if rng.random() < .25:
                continue  # Missing channels must remain null in final schema.
            columns[channel] = pl.Series(channel,
                [None if case == 0 or rng.random() < .4 else rng.uniform(-10, 300)
                 for _ in range(n)], dtype=pl.Float64)
        frame = pl.DataFrame(columns)
        parts = [next(lap.partials(frame.slice(i, 3)))["_partial"] for i in range(0, n, 3)]
        rng.shuffle(parts)
        actual = lap.reduce_lap("2099|Audit|Race|TST|1", parts)[0][1]
        expected = frame.select(_silver_exprs(frame.columns, meta)).row(0, named=True)
        assert actual == pytest.approx(expected, rel=1e-10, abs=1e-10), f"case {case}"


def test_seeded_stint_estimator_nulls_safety_car_and_rollup_match_polars():
    from src.mapreduce.evidence import polars_stints
    from src.mapreduce.jobs.stint_stats import reduce_stint, rollup, COMPOUNDS
    rng = random.Random(272)
    all_results = []
    for stint in range(100):
        rows = [dict(season="2099", gp="Audit", session="Race", driver="TST", stint=stint,
                     life=i if rng.random() > .15 else None, lap=i + 1,
                     fuel_corr=100 + i * rng.random() if rng.random() > .15 else None,
                     compound=COMPOUNDS[stint % 5], is_sc_lap=int(rng.random() < .15),
                     cluster=-1, dirty_air_frac=.1, track_temp_mean=30.)
                for i in range(rng.randrange(4, 23))]
        # Both implementations agree on null ordering, but reference sort has no
        # deterministic tie-breaker. Test its defined (untied) population here;
        # the separate AuditMR04 regression demonstrates the tie-policy gap.
        missing_life = [r for r in rows if r["life"] is None]
        for row in missing_life[1:]:
            row["life"] = row["lap"] - 1
        # Explicit float types also exercise all-null short populations safely.
        frame = pl.DataFrame(rows).with_columns(pl.col("life").cast(pl.Float64),
                                                pl.col("fuel_corr").cast(pl.Float64))
        reference = polars_stints(frame)
        actual = reduce_stint(str(stint), rows)
        assert bool(actual) == bool(reference.height), f"stint {stint}"
        if actual:
            value = actual[0][1]
            assert value["beta"] == pytest.approx(reference["beta"][0], abs=1e-10)
            assert value["n_laps"] == reference["n_laps"][0]
            all_results.append(value)
    for population in [all_results, all_results[:1], all_results[:3]]:
        result = rollup(population)
        values = pl.DataFrame(population)
        assert result["beta_global"] == values["beta"].median()
        for compound in COMPOUNDS:
            subset = values.filter(pl.col("compound") == compound)["beta"]
            expected_beta = subset.median() if len(subset) >= 5 else values["beta"].median()
            expected_sigma = (max(.02, min(.5, (subset.quantile(.75) - subset.quantile(.25)) / 1.349))
                              if len(subset) >= 5 else (values["beta"].std() or .05))
            assert result["beta"][compound] == pytest.approx(expected_beta)
            assert result["sigma"][compound] == pytest.approx(expected_sigma)


def test_multiple_null_life_ties_preserve_estimator_population_parity():
    from src.mapreduce.evidence import polars_stints
    from src.mapreduce.jobs.stint_stats import reduce_stint
    rows = [dict(season="2099", gp="Audit", session="Race", driver="TST", stint=13,
                 life=None if i in (11, 14, 15, 16) else i, lap=i + 1,
                 fuel_corr=100 + i * .2, compound="INTERMEDIATE",
                 is_sc_lap=int(i in (5, 11, 14)), cluster=-1,
                 dirty_air_frac=.1, track_temp_mean=30.) for i in range(21)]
    expected = polars_stints(pl.DataFrame(rows)).row(0, named=True)
    actual = reduce_stint("13", rows)[0][1]
    assert actual["n_laps"] == expected["n_laps"]
