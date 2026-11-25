"""Replay real Bahrain 2025 laps and produce evidence without full retraining."""
import argparse
import json
from pathlib import Path

import polars as pl

from src.ingest.config import ROOT
from src.models import optimizer
from .replay import BAHRAIN_GP, bahrain_frame, reference_params
from .state_updater import new_state

DEMO_DB = "f1_tyre_demo"
DECISION_LAP = 12


def replay_decision(frame):
    state = new_state("2025", BAHRAIN_GP, "VER", 57, "SOFT")
    for row in frame.filter(pl.col("lap") <= DECISION_LAP).iter_rows(named=True):
        state.update_lap(row)
    current = state.as_optimizer_state()
    params = reference_params()
    scenarios = {"clean_air": optimizer.load_scenarios(params)["clean_air"]}
    best = optimizer.optimize(current, params, 150, scenarios)["clean_air"][0]
    observed = (frame.filter(pl.col("lap") >= DECISION_LAP)
                .group_by("stint", maintain_order=True)
                .agg(pl.col("compound").first(), pl.col("lap").max()))
    actual = [(row["compound"], row["lap"], index == 0)
              for index, row in enumerate(observed.iter_rows(named=True))]
    comparison = optimizer.compare_strategies(
        actual, best["_strat"], current, params, scenarios["clean_air"], 150)
    return {
        "_id": f"2025|{BAHRAIN_GP}|VER|demo_clean_air|{DECISION_LAP}",
        "season": "2025", "gp": BAHRAIN_GP, "driver": "VER", "lap": DECISION_LAP,
        "scenario": "demo_clean_air",
        "rec_strategy": best["strategy"],
        "actual_strategy": " | ".join(f"{compound}->{lap}" for compound, lap, _ in actual),
        "compounds": sorted({compound for compound, _, _ in best["_strat"]}),
        "expected_seconds": best["mean"], "p5": best["p5"], "p95": best["p95"],
        "delta": comparison["mean_delta"],
        "model": "dry_report_reference",
    }


def save_mongo(decision):
    from src.nosql.connection import get_database
    from src.nosql.indexes import create_indexes
    from src.nosql.ingest import upsert_documents
    from src.nosql.repository import Repository
    from src.nosql.schemas import ensure_collections

    db = get_database(db_name=DEMO_DB)
    try:
        ensure_collections(db)
        create_indexes(db)
        upsert_documents(db.strategy_decisions, [decision])
        stored = Repository(db).decisions(season=decision["season"], gp=decision["gp"],
                                         driver=decision["driver"])
        if not any(all(row.get(k) == v for k, v in decision.items() if k != "_id")
                   for row in stored):
            raise RuntimeError("Failed to read the demo decision back from MongoDB")
    finally:
        db.client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/demo")
    parser.add_argument("--mongo", action="store_true", help=f"verify history in {DEMO_DB}")
    args = parser.parse_args()

    frame = bahrain_frame()
    decision = replay_decision(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(args.output_dir / "bahrain_2025_ver.parquet")
    (args.output_dir / "recommendation.json").write_text(
        json.dumps(decision, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, allow_nan=False))
    if args.mongo:
        save_mongo(decision)
        print(f"MongoDB read-back verified in {DEMO_DB}")


if __name__ == "__main__":
    main()
