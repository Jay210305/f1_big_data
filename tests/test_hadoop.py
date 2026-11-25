"""Exercise the packaged Streaming processes without requiring a Docker daemon."""
import os
import subprocess
import sys

import pytest

from src.hadoop.cli import demo_frame, index_stints, verify_stints, worker_archive
from src.hadoop.streaming import read_pairs
from src.live.replay import bahrain_frame
from src.mapreduce.io import canonical, read_output
from src.mapreduce.jobs import stint_stats
from src.mapreduce.runner import LocalJobRunner


@pytest.mark.parametrize("load_frame, lap_counts", [(demo_frame, [5, 5]),
                                                    (bahrain_frame, [7, 13, 24])])
def test_packaged_streaming_matches_local_and_polars(tmp_path, load_frame, lap_counts):
    frame = load_frame()
    archive = tmp_path / "worker.zip"
    worker_archive(archive)

    def worker(stage, payload):
        return subprocess.run(
            [sys.executable, "-S", "-m", "src.hadoop.streaming", stage],
            cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(archive)},
            input=payload, capture_output=True, text=True, check=True,
        ).stdout

    # Split the source across mappers and sort only their emitted keys, as Hadoop does.
    first = "".join(canonical(row) + "\n" for row in frame[::2].to_dicts())
    second = "".join(canonical(row) + "\n" for row in frame[1::2].to_dicts())
    mapped = (worker("map", first) + worker("map", second)).splitlines()
    shuffled = "\n".join(sorted(mapped, key=lambda line: line.partition("\t")[0])) + "\n"
    actual = index_stints(read_pairs(worker("reduce", shuffled).splitlines()))
    assert sorted(row["n_laps"] for row in actual.values()) == lap_counts

    source = tmp_path / "laps.parquet"
    frame.write_parquet(source)
    output = tmp_path / "local"
    LocalJobRunner(workers=2).run(stint_stats.job(num_partitions=2), source, output)
    errors = verify_stints(frame, actual, read_output(output))
    assert errors["local_mapreduce"] == 0
    assert errors["polars"] < 1e-6

    next(iter(actual.values()))["beta"] += .01
    with pytest.raises(ValueError, match="beta error"):
        verify_stints(frame, actual, read_output(output))
