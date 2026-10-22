"""Collect a read-only project snapshot; this is evidence, not a correctness verdict.

    python -m src.eval.audit_project --mongo

Parquet counts come from file footers; model metrics are identified as saved
artifacts, not newly trained estimates. Mongo access never changes collections.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

import polars as pl
import pyarrow.parquet as pq

from src.ingest.config import ROOT


def file_identity(path):
    return {"bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def parquet_inventory(path):
    files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    return {"files": len(files), "rows": sum(pq.read_metadata(f).num_rows for f in files),
            "bytes": sum(f.stat().st_size for f in files)}


def collect_snapshot(include_mongo=False):
    snapshot = {"collected_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(), "platform": platform.platform(),
                "packages": {}, "lakehouse": {}, "gold": {}, "saved_models": {},
                "source_sha256": {}, "saved_artifact_diagnostics": {}}
    for package in ("numpy", "polars", "pyarrow", "duckdb", "scikit-learn", "torch",
                    "xgboost", "lightgbm", "hdbscan", "streamlit", "pymongo"):
        try:
            snapshot["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            snapshot["packages"][package] = None
    for label, path in (("bronze_telemetry", ROOT / "data/bronze/telemetry"),
                         ("silver_laps", ROOT / "data/silver/stint_features"),
                         ("silver_events", ROOT / "data/silver/race_events")):
        if path.exists():
            snapshot["lakehouse"][label] = parquet_inventory(path)
    for path in sorted((ROOT / "data/bronze/telemetry").glob("season=*")):
        snapshot["lakehouse"][path.name] = parquet_inventory(path)
    grains = {"lap_features": ["season", "gp", "session", "driver", "lap"],
              "labeled_frame": ["season", "gp", "session", "driver", "lap"],
              "driver_style": ["season", "driver"],
              "stint_degradation": ["season", "gp", "session", "driver", "stint"],
              "environmental": ["season", "gp", "session"],
              "driver_embeddings": ["season", "driver"]}
    for path in sorted((ROOT / "data/gold").glob("*.parquet")):
        entry = {**parquet_inventory(path), "columns": len(pq.read_schema(path))}
        if path.stem in grains:
            keys = grains[path.stem]
            frame = pl.read_parquet(path, columns=keys)
            entry.update(grain=keys, unique_keys=frame.unique().height,
                         null_keys=frame.null_count().row(0, named=True))
        snapshot["gold"][path.stem] = entry
    for name in ("phase4_metrics.json", "phase5_metrics.json", "feature_meta.json",
                 "degradation_model.json", "phase7_backtest.json", "phase7_backtest_realized.json"):
        path = ROOT / "data/models" / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if name == "degradation_model.json":
            hazards = [(gp, lap, value) for gp, table in payload.get("sc_hazard", {}).get("lap_hazard", {}).items()
                       for lap, value in table.items()]
            invalid = [(gp, lap, value) for gp, lap, value in hazards if not 0 <= value <= 1]
            snapshot["saved_artifact_diagnostics"]["sc_hazards"] = {
                "cells": len(hazards), "out_of_probability_range": len(invalid),
                "maximum": max((value for _, _, value in hazards), default=None),
                "examples": invalid[:10]}
        if name == "phase7_backtest_realized.json":
            compounds = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}
            decisions = payload.get("decisions", [])
            snapshot["saved_artifact_diagnostics"]["realized_compound_parser"] = {
                "stored_decisions": len(decisions), "invalid_final_compounds": sum(
                    row.get("rec_final_comp") not in compounds for row in decisions),
                "examples": [row.get("rec_final_comp") for row in decisions[:5]]}
        if "decisions" in payload:
            payload = {**{k: v for k, v in payload.items() if k != "decisions"},
                       "stored_decisions": len(payload["decisions"])}
        snapshot["saved_models"][name] = {**file_identity(path), "saved_artifact": payload}
    for base in (ROOT / "src", ROOT / "tests"):
        for path in sorted(base.rglob("*.py")):
            snapshot["source_sha256"][path.relative_to(ROOT).as_posix()] = file_identity(path)["sha256"]
    snapshot["git"] = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                       "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
                       "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))}
    if include_mongo:
        from pymongo.errors import PyMongoError
        from src.nosql.connection import get_database
        db = None
        try:
            db = get_database()
            snapshot["mongo"] = {"available": True, "database": db.name,
                "server_version": db.client.server_info()["version"],
                "collections": {name: db[name].count_documents({}) for name in
                    ("race_sessions", "laps", "stints", "strategy_decisions", "bench_laps")}}
        except PyMongoError as exc:
            snapshot["mongo"] = {"available": False, "error_type": type(exc).__name__}
        finally:
            if db is not None:
                db.client.close()
    return snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mongo", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "data/models/project_audit_snapshot.json")
    args = parser.parse_args()
    snapshot = collect_snapshot(args.mongo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    print(f"Saved read-only audit snapshot: {args.output}")


if __name__ == "__main__":
    main()
