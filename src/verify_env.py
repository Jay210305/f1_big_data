"""Phase 0 smoke test: verify the venv, dependencies, CUDA/GPU, and I/O.

Run (venv active, from repo root):
    python src/verify_env.py

It runs a series of independent checks and prints PASS / FAIL for each,
plus a final summary. No check aborts the script, so you can paste the whole
output back even if something fails.
"""
from __future__ import annotations

import importlib.metadata as im
import os
import sys

RESULTS: list[tuple[str, str, str]] = []  # (check, status, detail)


def record(check: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    RESULTS.append((check, status, detail))
    print(f"[{status}] {check}" + (f"  ->  {detail}" if detail else ""))


def attempt(check: str, fn) -> None:
    try:
        fn()
        record(check, True)
    except Exception as exc:  # noqa: BLE001
        record(check, False, f"{type(exc).__name__}: {exc}")


def main() -> None:
    print("=" * 72)
    print("Phase 0 — Environment & CUDA verification")
    print("=" * 72)

    # --- 1. Python / venv -------------------------------------------------
    print(f"\nPython:     {sys.version.split()[0]}  ({sys.executable})")
    in_venv = sys.prefix != sys.base_prefix
    record("Running inside a virtualenv (.venv active)", in_venv,
           f"prefix={sys.prefix}")

    # --- 2. Required packages -------------------------------------------
    required = [
        "numpy", "pandas", "polars", "pyarrow", "duckdb", "fastparquet",
        "zstandard", "scipy", "scikit-learn", "xgboost", "lightgbm",
        "hdbscan", "jupyter", "matplotlib", "seaborn",
    ]
    print(f"\nInstalled packages (spot-check):")
    for pkg in required:
        try:
            ver = im.version(pkg)
            record(f"import {pkg}", True, f"v{ver}")
        except Exception as exc:  # noqa: BLE001
            record(f"import {pkg}", False, str(exc))

    # --- 3. torch + CUDA (Blackwell sm_120) ------------------------------
    def check_torch() -> None:
        import torch
        record("import torch", True, f"v{torch.__version__}")
        record("torch sees a CUDA device", torch.cuda.is_available(),
               f"count={torch.cuda.device_count()}")
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            record("GPU name / capability", True, f"{name}  sm_{cap[0]}{cap[1]}")
            # tiny matmul to force a real CUDA kernel launch
            a = torch.randn(512, 512, device="cuda")
            b = torch.randn(512, 512, device="cuda")
            _ = a @ b
            torch.cuda.synchronize()
            record("CUDA kernel launch (matmul)", True)

    attempt("torch/CUDA", check_torch)

    # --- 4. XGBoost GPU hist ----------------------------------------------
    def check_xgboost() -> None:
        import numpy as np
        import xgboost as xgb
        X = np.random.RandomState(0).rand(200, 5)
        y = X[:, 0] * 2 + np.random.RandomState(1).rand(200) * 0.1
        for dev in ("cuda", "gpu"):
            try:
                m = xgb.XGBRegressor(n_estimators=5, max_depth=3,
                                     tree_method="hist", device=dev)
                m.fit(X, y)
                record(f"XGBoost {dev}", True, f"v{im.version('xgboost')}")
                return
            except Exception as exc:  # noqa: BLE001
                last = exc
        record("XGBoost GPU", False, str(last))

    attempt("xgboost GPU", check_xgboost)

    # --- 5. LightGBM GPU (required) ----------------------------------------
    def check_lightgbm() -> None:
        import numpy as np
        import lightgbm as lgb
        X = np.random.RandomState(0).rand(200, 5)
        y = X[:, 0] * 2
        m = lgb.LGBMRegressor(n_estimators=5, device="gpu")
        m.fit(X, y)
        record("LightGBM GPU", True, f"v{im.version('lightgbm')}")

    attempt("lightgbm", check_lightgbm)

    # --- 6. Parquet / DuckDB I/O round-trip --------------------------------
    def check_io() -> None:
        import duckdb
        import polars as pl

        data_dir = os.path.join(os.getcwd(), "data", "bronze")
        os.makedirs(data_dir, exist_ok=True)

        df = pl.DataFrame({
            "season": ["2025"] * 3,
            "lap": [1, 2, 3],
            "speed": [200.0, 205.0, 190.0],
        })
        path = os.path.join(data_dir, "_smoke.parquet")
        df.write_parquet(path, compression="zstd")
        back = pl.read_parquet(path)
        record("Polars parquet write/read (zstd)", back.shape == (3, 3),
               f"rows={back.shape[0]}")

        row = duckdb.sql(
            f"SELECT count(*) AS n FROM read_parquet('{path}')"
        ).fetchone()
        if row is None:
            raise RuntimeError("DuckDB returned no row for count query")
        n = row[0]
        record("DuckDB reads parquet", n == 3, f"count={n}")
        os.remove(path)

    attempt("parquet/DuckDB I/O", check_io)

    # --- 7. Directory tree --------------------------------------------------
    def check_tree() -> None:
        dirs = ["data/bronze", "data/silver", "data/gold",
                "notebooks", "annotations",
                "src/ingest", "src/features", "src/models", "src/eval"]
        for d in dirs:
            os.makedirs(os.path.join(os.getcwd(), d), exist_ok=True)
        record("Directory tree created", True, ", ".join(dirs))

    attempt("directory tree", check_tree)

    # --- Summary ------------------------------------------------------------
    passed = sum(1 for _, s, _ in RESULTS if s == "PASS")
    failed = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    print("\n" + "=" * 72)
    print(f"SUMMARY: {passed} passed, {failed} failed, {len(RESULTS)} total")
    if failed == 0:
        print("Phase 0 environment is READY  ->  GO to Phase 1")
    else:
        print("Fix the FAIL items above before continuing.")
    print("=" * 72)


if __name__ == "__main__":
    main()
