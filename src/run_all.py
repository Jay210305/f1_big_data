"""Phase 7 — End-to-end pipeline runner (raw JSON -> recommendation).

Chains all phases with skip-if-cached logic so it doubles as the runbook
executor. Each stage checks for its artifacts; use --force to rerun a stage.

    python -m src.run_all                  # run missing stages only
    python -m src.run_all --stage etl      # run a single stage
    python -m src.run_all --force          # rerun everything from scratch
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from src.ingest.config import ROOT

DATA = ROOT / "data"
BRONZE = DATA / "bronze" / "telemetry"
GOLD = DATA / "gold"
MODELS = DATA / "models"

STAGES = ["etl", "verify_schema", "gold", "rul", "style", "degradation",
          "optimizer"]


def _done(path: Path, min_size: int = 1000) -> bool:
    return path.exists() and (path.is_dir() and any(path.iterdir())
                              or path.is_file() and path.stat().st_size > min_size)


def _run(cmd: list[str]):
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(f"stage failed ({' '.join(cmd)}), rc={r.returncode}")
    print(f"<<< ok ({time.time()-t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=STAGES, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    stages = [args.stage] if args.stage else STAGES
    py = sys.executable
    t0 = time.time()

    plan = {
        "etl": (lambda: _done(BRONZE),
                [py, "-m", "src.ingest.etl", "--workers", "8"]),
        "verify_schema": (lambda: _done(ROOT / "data" / "bronze"
                                        / "phase2_schema_verify.json"),
                          [py, "-m", "src.ingest.verify_schema"]),
        "gold": (lambda: _done(GOLD / "lap_features.parquet"),
                 [py, "-m", "src.features.build_gold"]),
        "rul": (lambda: _done(MODELS / "phase4_metrics.json"),
                [py, "-m", "src.models.train_rul", "--epochs", "40"]),
        "style": (lambda: _done(MODELS / "phase5_metrics.json"),
                  [py, "-m", "src.models.train_style"]),
        "degradation": (lambda: _done(MODELS / "degradation_model.json"),
                        [py, "-m", "src.models.fit_degradation"]),
        "optimizer": (lambda: _done(MODELS / "phase6_demo.json"),
                      [py, "-m", "src.models.optimizer"]),
    }
    for s in stages:
        is_done, cmd = plan[s]
        if is_done() and not args.force:
            print(f"[skip] {s} (artifacts present)")
            continue
        _run(cmd)
    print(f"\nPipeline complete in {time.time()-t0:.0f}s. "
          f"Run the backtest: python -m src.models.backtest")


if __name__ == "__main__":
    main()
