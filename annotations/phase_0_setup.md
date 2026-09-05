# Phase 0 — Environment & Repo Scaffolding (STOP)

- Date / author: 2026-08-27 / agent
- Status: [ ] Go  [ ] No-Go  [ ] Rework

## Files created

| File | Purpose |
|------|---------|
| `requirements.txt` | All deps (CPU data + ML), **except torch** (installed separately for CUDA) |
| `src/verify_env.py` | Smoke test that checks the whole environment |
| `src/__init__.py` | Package marker |
| `src/ingest/__init__.py` | Ingestion/ETL package skeleton |
| `src/features/__init__.py` | Feature-engineering package skeleton |
| `src/models/__init__.py` | Modeling package skeleton |
| `src/eval/__init__.py` | Evaluation package skeleton |

## Steps to run (in order)

Open PowerShell in the repo root, then:

```powershell
# 1. Create + activate the virtualenv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install torch with CUDA wheels (RTX 5070 Ti = Blackwell, needs cu128)
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128

# 3. Install everything else
python -m pip install -r requirements.txt

# 4. Run the smoke test
python src/verify_env.py
```

## What's going on

`verify_env.py` runs independent checks and prints `PASS`/`FAIL`/`SKIP` per check (it never aborts, so you get a full report):

1. **Venv check** — confirms you're inside `.venv`, not the global Python.
2. **Package imports** — prints the installed version of every required library.
3. **torch/CUDA** — verifies `torch.cuda.is_available()`, reads the GPU name + compute capability (should show `sm_120`), and launches a real CUDA matmul to prove the kernel path works.
4. **XGBoost GPU** — fits a tiny model with `tree_method="hist", device="cuda"` (falls back to `gpu`).
5. **LightGBM GPU** — tries `device="gpu"`, and if the wheel lacks GPU support it reports CPU as a graceful fallback.
6. **Parquet/DuckDB I/O** — writes a 3-row ZSTD Parquet to `data/bronze/`, reads it back with both Polars and DuckDB, then deletes it (confirms the lakehouse write/read path).
7. **Directory tree** — creates `data/{bronze,silver,gold}`, `notebooks/`, `annotations/`.

The final line is the Go/No-Go signal: if `0 failed`, we proceed to Phase 1.

## After you run it

- Paste the full output (or the `SUMMARY` line + any `FAIL` lines) into the **Results** section below.
- Tell me the result.
- If any `FAIL` appears — especially torch/CUDA or XGBoost GPU — paste that block and I'll fix the install steps.
- Open question for Phase 1: scope schema validation to **all sessions** or **Race only** first?

## Results (filled after running)

- Commands run (copy/paste):
- Results / output (from user):
- Metrics / numbers:
- Errors / blockers:
- Decisions made:
- Next-phase risks:
- Time spent / next step:
