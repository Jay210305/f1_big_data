"""CLI: python -m src.nosql.cli --help."""
import argparse
import json
from pathlib import Path
from pymongo.errors import PyMongoError, BulkWriteError
from .connection import get_database
from .ingest import ROOT, ingest, quality_report
from .schemas import ensure_collections, SCHEMAS
from .indexes import create_indexes, explain_comparison


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", "--mongo-uri", dest="uri")
    parser.add_argument("--db", "--mongo-db", dest="db_name")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("health")
    load = sub.add_parser("ingest")
    load.add_argument("--scope", choices=["all", "race_sessions", "laps", "stints", "strategy_decisions"], default="all")
    load.add_argument("--season")
    load.add_argument("--gp")
    load.add_argument("--session")
    load.add_argument("--refresh-clusters", action="store_true",
                      help="invalidate the stored style cluster before ingesting")
    load.add_argument("--skip-if-unavailable", action="store_true")
    load.add_argument("--output", type=Path, default=ROOT/"data/mongo/ingest.json")
    for command in ("indexes", "index", "explain", "aggregate", "quality", "crud", "queries", "parity"):
        p = sub.add_parser(command)
        p.add_argument("--output", type=Path)
        if command == "index":
            p.add_argument("--explain", action="store_true")
    bench = sub.add_parser("benchmark")
    bench.add_argument("--n", type=int, default=100_000)
    bench.add_argument("--output", type=Path, default=ROOT/"data/mongo/benchmark.json")
    reset = sub.add_parser("reset")
    reset.add_argument("--force", action="store_true")
    delete = sub.add_parser("delete")
    delete.add_argument("--collection", choices=list(SCHEMAS), required=True)
    delete.add_argument("--season", required=True)
    delete.add_argument("--gp", required=True)
    delete.add_argument("--force", action="store_true")
    return parser


def _save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=str, allow_nan=False), encoding="utf-8")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "reset" and not args.force:
        print("Reset requires --force; no collections dropped.")
        return 2
    try:
        db = get_database(args.uri, args.db_name)
    except PyMongoError as exc:
        if getattr(args, "skip_if_unavailable", False):
            print(f"SKIP MongoDB unavailable: {exc}")
            return 0
        print(f"MongoDB unavailable: {exc}")
        return 1
    try:
        command = args.command
        if command == "health":
            print(f"OK MongoDB {db.client.server_info()['version']} database={db.name}")
            return 0
        if command == "ingest":
            result = ingest(db, args.scope, season=args.season, gp=args.gp, session=args.session,
                            refresh_clusters=args.refresh_clusters)
            _save(args.output, result)
            # Keep console short while preserving every per-batch timing in JSON.
            print(json.dumps({k: {field: value for field, value in v.items() if field != "batches"}
                              for k, v in result.items()}, indent=2))
            return 0
        if command in ("indexes", "index"):
            result = create_indexes(db)
            if getattr(args, "explain", False):
                result = explain_comparison(db)
        elif command == "explain":
            result = explain_comparison(db)
        elif command == "quality":
            result = quality_report(db)
        elif command == "benchmark":
            from .benchmark import benchmark
            result = benchmark(db, args.n)
        elif command == "aggregate":
            from .aggregation import stint_slopes, compound_pace, safety_car_share, race_control_counts, pace_facet
            result = {"stints": list(db.laps.aggregate(stint_slopes(), allowDiskUse=True)),
                      "compound_pace": list(db.laps.aggregate(compound_pace())),
                      "safety_car_share": list(db.laps.aggregate(safety_car_share())),
                      "race_control": list(db.race_sessions.aggregate(race_control_counts())),
                      "facet": list(db.laps.aggregate(pace_facet()))}
        elif command == "crud":
            from .queries import crud_demo
            result = crud_demo(db)
        elif command == "queries":
            from .queries import cookbook
            result = cookbook(db)
        elif command == "reset":
            for name in SCHEMAS:
                db.drop_collection(name)
            ensure_collections(db)
            result = create_indexes(db)
        elif command == "delete":
            from .queries import delete_matching
            result = delete_matching(db[args.collection], {"season": args.season, "gp": args.gp}, args.force)
        elif command == "parity":
            # Shared proof driver owns all three implementations and saved-model checks.
            from src.mapreduce.evidence import main as parity_main
            proof_args = ["--mongo", "--silver"]
            for flag, value in (("--mongo-uri", args.uri), ("--mongo-db", args.db_name),
                                ("--output", args.output)):
                if value is not None:
                    proof_args.extend([flag, str(value)])
            return parity_main(proof_args)
        default = ROOT/"data/mongo"/{"explain": "explain.json", "aggregate": "aggregations.json",
                                         "quality": "quality_report.json"}.get(command, command+".json")
        output = getattr(args, "output", None) or default
        _save(output, result)
        print(f"Saved {output}")
        return 0
    except BulkWriteError as exc:
        errors = exc.details.get("writeErrors", [])
        print(f"MongoDB {args.command} failed: {len(errors)} rejected documents; "
              f"first rejection: {errors[0].get('errInfo', errors[0].get('errmsg')) if errors else exc.details.get('writeConcernErrors')}")
        return 1
    except (PyMongoError, ValueError) as exc:
        print(f"MongoDB {args.command} failed: {exc}")
        return 1
    finally:
        db.client.close()


if __name__ == "__main__":
    raise SystemExit(main())
