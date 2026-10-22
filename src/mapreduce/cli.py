"""Run the three real lakehouse jobs with optional trace/fault demonstrations."""
import argparse
import json
from pathlib import Path

from src.ingest.config import ROOT
from .runner import LocalJobRunner
from .io import read_output
from .jobs import lap_aggregate, stint_stats, driver_profile
from .partitioner import HashPartitioner, IdentityPartitioner, RangePartitioner

JOBS = {"lap_aggregate": lap_aggregate, "stint_stats": stint_stats, "driver_profile": driver_profile}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    all_jobs = sub.add_parser("all", help="run cached real-data jobs and parity checks")
    all_jobs.add_argument("--force", action="store_true")
    run = sub.add_parser("run")
    run.add_argument("--job", choices=JOBS, required=True)
    run.add_argument("--input", type=Path)
    run.add_argument("--output", type=Path)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--partitions", type=int, default=4)
    run.add_argument("--partitioner", choices=["hash", "identity", "range"], default="hash")
    run.add_argument("--range-boundaries", nargs="*", default=[])
    run.add_argument("--trace", action="store_true")
    run.add_argument("--trace-sample", type=int)
    run.add_argument("--trace-dir", type=Path)
    run.add_argument("--no-combiner", action="store_true")
    run.add_argument("--inject-fault", nargs=3, metavar=("STAGE", "TASK", "ATTEMPTS"))
    run.add_argument("--max-attempts", type=int, default=3)
    run.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "all":
        import subprocess
        import sys
        for name in JOBS:
            cmd = [sys.executable, "-m", "src.mapreduce.cli", "run", "--job", name]
            if args.force:
                cmd.append("--force")
            if name == "stint_stats":
                cmd.append("--trace")
            subprocess.run(cmd, check=True, cwd=ROOT)
        subprocess.run([sys.executable, "-m", "src.mapreduce.evidence", "--silver"], check=True, cwd=ROOT)
        return
    input_path = args.input or (ROOT / "data/bronze/telemetry/season=2025" if args.job == "lap_aggregate"
                               else ROOT / "data/gold/labeled_frame.parquet")
    output = args.output or ROOT / "data/mapreduce/output" / args.job
    if output.exists() and not args.force:
        print(f"[skip] {output} exists; use --force to rebuild")
        return
    partitioner = {"hash": HashPartitioner(), "identity": IdentityPartitioner(),
                   "range": RangePartitioner(args.range_boundaries)}[args.partitioner]
    if args.partitioner == "range" and len(args.range_boundaries) != args.partitions - 1:
        parser.error("range partitioner needs --partitions minus one --range-boundaries")
    kwargs = dict(num_partitions=args.partitions, max_attempts=args.max_attempts, partitioner=partitioner)
    spec = JOBS[args.job].job(**kwargs, **({"combine": not args.no_combiner} if args.job == "lap_aggregate" else {}))
    fault = ((args.inject_fault[0], int(args.inject_fault[1]), int(args.inject_fault[2]))
             if args.inject_fault else None)
    if fault and fault[0] not in ("map", "reduce"):
        parser.error("fault stage must be map or reduce")
    result = LocalJobRunner(args.workers).run(spec, input_path, output, trace=args.trace,
        trace_sample=args.trace_sample, inject_fault=fault, overwrite=args.force)
    if args.trace_dir and args.trace:
        import shutil
        shutil.copytree(output / "trace", args.trace_dir, dirs_exist_ok=True)
    if args.job == "stint_stats":
        model = stint_stats.rollup(value for _, value in read_output(output))
        (output / "model.json").write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
