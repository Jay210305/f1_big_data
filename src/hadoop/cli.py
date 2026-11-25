"""Run F1 stint degradation on HDFS/YARN and verify the output locally."""
import argparse
import json
import math
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import polars as pl

from src.ingest.config import ROOT
from src.mapreduce.evidence import polars_stints
from src.mapreduce.io import read_output, write_jsonl
from src.mapreduce.jobs import stint_stats
from src.mapreduce.runner import LocalJobRunner

from .streaming import read_pairs

COMPOSE = ("docker", "compose", "-f", str(ROOT / "compose.hadoop.yaml"))
HDFS_URL = "hdfs://namenode:8020"
DEFAULT_INPUT = ROOT / "data/gold/labeled_frame.parquet"


def demo_frame():
    return pl.DataFrame([
        dict(season="2025", gp="Demo Grand Prix", session="Race", driver=driver,
             stint=1, life=life, lap=life + 1, fuel_corr=90 + slope * life,
             compound=compound, is_sc_lap=int(life == 5), cluster=0,
             stint_best_pace=90., dirty_air_frac=.1, track_temp_mean=30., t_end_life=9)
        for driver, compound, slope in [("VER", "SOFT", .2), ("NOR", "MEDIUM", .3)]
        for life in range(9)
    ])


def worker_archive(path):
    files = [ROOT / "src/__init__.py", Path(__file__).with_name("__init__.py"),
             Path(__file__).with_name("streaming.py")]
    files.extend(sorted((ROOT / "src/mapreduce").rglob("*.py")))
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for source in files:
            archive.write(source, source.relative_to(ROOT))


def docker_exec(service, *command, **kwargs):
    return subprocess.run([*COMPOSE, "exec", "-T", service, *command], cwd=ROOT,
                          check=True, **kwargs)


def hdfs(*args, **kwargs):
    return docker_exec("namenode", "hdfs", "dfs", *args, **kwargs)


def upload(source, target):
    with source.open("rb") as stream:
        hdfs("-put", "-", target, stdin=stream)


def run_streaming(frame, directory, hdfs_run, reducers):
    input_file = directory / "laps.jsonl"
    write_jsonl(input_file, frame.iter_rows(named=True))
    bundle = directory / "worker.zip"
    worker_archive(bundle)

    hdfs("-mkdir", "-p", hdfs_run)
    hdfs_input = f"{hdfs_run}/laps.jsonl"
    upload(input_file, hdfs_input)
    upload(bundle, f"{hdfs_run}/worker.zip")
    hdfs_output = f"{hdfs_run}/output"
    docker_exec(
        "resourcemanager", "mapred", "streaming",
        "-D", "mapreduce.job.name=F1 stint_stats",
        "-D", f"mapreduce.job.reduces={reducers}",
        "-files", f"{HDFS_URL}{hdfs_run}/worker.zip#worker.zip",
        "-input", hdfs_input, "-output", hdfs_output,
        "-cmdenv", "PYTHONPATH=worker.zip",
        "-mapper", "python3 -m src.hadoop.streaming map",
        "-reducer", "python3 -m src.hadoop.streaming reduce",
    )
    hdfs("-test", "-e", f"{hdfs_output}/_SUCCESS")
    result_file = directory / "streaming.tsv"
    with result_file.open("wb") as stream:
        hdfs("-cat", f"{hdfs_output}/part-*", stdout=stream)
    return result_file


def index_stints(rows):
    indexed = {}
    for key, value in rows:
        if key in indexed:
            raise ValueError(f"Duplicate stint: {key}")
        indexed[key] = value
    return indexed


def verify_stints(frame, actual, local_rows):
    reference = index_stints(
        ("|".join(str(row[name]) for name in stint_stats.KEYS), row)
        for row in polars_stints(frame).iter_rows(named=True)
    )
    local = index_stints(local_rows)
    if not reference:
        raise ValueError("No eligible stints in this sample")
    maximum_errors = {}
    for label, expected in [("polars", reference), ("local_mapreduce", local)]:
        if actual.keys() != expected.keys():
            missing = len(expected.keys() - actual.keys())
            extra = len(actual.keys() - expected.keys())
            raise ValueError(f"{label}: {missing} missing stints, {extra} extra stints")
        maximum = 0.
        for key, row in actual.items():
            baseline = expected[key]
            if any(row[field] != baseline[field] for field in ("n_laps", "compound")):
                raise ValueError(f"{label}: stint {key} has different lap counts or compound")
            error = abs(row["beta"] - baseline["beta"])
            if not math.isfinite(error) or error >= 1e-6:
                raise ValueError(f"{label}: stint {key} has beta error {error}")
            maximum = max(maximum, error)
        maximum_errors[label] = maximum
    return maximum_errors


def run(frame, output_dir, reducers):
    run_id = uuid4().hex
    hdfs_run = f"/f1-hadoop/{run_id}"
    print(f"HDFS run: {hdfs_run}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir) as workspace:
        work = Path(workspace)
        streaming_file = run_streaming(frame, work, hdfs_run, reducers)
        with streaming_file.open(encoding="utf-8") as lines:
            actual = index_stints(read_pairs(lines))
        parquet = work / "laps.parquet"
        frame.write_parquet(parquet, row_group_size=8192)
        local = work / "local"
        LocalJobRunner(workers=2).run(stint_stats.job(num_partitions=reducers), parquet, local)
        errors = verify_stints(frame, actual, read_output(local))

        result = work / "result"
        result.mkdir()
        write_jsonl(result / "stints.jsonl", sorted(actual.items()))
        report = dict(hdfs_output=f"{HDFS_URL}{hdfs_run}/output", reducers=reducers,
                      input_laps=frame.height, stints=len(actual), max_beta_abs_diff=errors)
        (result / "parity.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        result.rename(output_dir / run_id)
    print(json.dumps(report, indent=2))
    print(f"Results: {output_dir / run_id}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--demo", action="store_true", help="use a synthetic Grand Prix")
    source.add_argument("--input", type=Path, help=f"labeled Parquet file (default: {DEFAULT_INPUT})")
    parser.add_argument("--season", help="filter by season")
    parser.add_argument("--gp", help="filter by exact Grand Prix name")
    parser.add_argument("--reducers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/hadoop")
    args = parser.parse_args()
    if args.reducers < 1:
        parser.error("--reducers must be at least 1")

    if args.demo:
        input_frame = demo_frame().lazy()
    else:
        path = args.input or DEFAULT_INPUT
        if not path.is_file():
            parser.error(f"missing labeled Parquet: {path}; use --demo or supply --input")
        input_frame = pl.scan_parquet(path)
    if args.season:
        input_frame = input_frame.filter(pl.col("season").cast(pl.String) == args.season)
    if args.gp:
        input_frame = input_frame.filter(pl.col("gp") == args.gp)
    try:
        frame = input_frame.select(*stint_stats.KEYS, *stint_stats.LAP_COLUMNS).collect()
        if frame.is_empty():
            parser.error("no laps match the selected input and filters")
        run(frame, args.output_dir, args.reducers)
    except (OSError, ValueError, pl.exceptions.PolarsError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Hadoop run failed: {error}\n")


if __name__ == "__main__":
    main()
