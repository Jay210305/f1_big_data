# Plan: F1 Prescriptive Tyre & Stint Strategy System

**Methodology:** Stop-n-Go (checkpointed, reviewable phases).
**Purpose of this file:** Master roadmap. Tracked in git.
**Local annotations:** Each phase ends with a Markdown note in `annotations/` (gitignored, local-only).

## Division of Labor (workflow rule)

- **The agent writes the code** and, for each phase, delivers: (1) the scripts/files, (2) **step-by-step run instructions**, and (3) an **explanation of what the code does and why**.
- **The user runs the scripts/pipelines locally** and reports the results back.
- Results go into the phase's `annotations/phase_N_*.md` note, which the agent uses to make the Go / No-Go / Rework decision for the next phase.

> Rule: **Never start Phase N+1 until the Phase N annotation file is written (with the user's run results) and its "Go" decision is confirmed.** Each phase is a hard checkpoint — you stop, document, review, then decide Go / No-Go / Rework.

---

## Hardware & Constraints (binding for every phase)

- Storage: 513 GB NVMe SSD → keep intermediates lean, convert to Parquet (ZSTD) / DuckDB.
- RAM: 32 GB → streaming/chunked ingestion, **never `json.load()` a full season**.
- GPU: RTX 5070 Ti (CUDA) → PyTorch, XGBoost/LightGBM GPU hist, optional cuDF/RAPIDS.
- CPU: parallelized disk I/O + worker pools.
- Granularity: `season → grand_prix → session → driver → stint`.

## Stack (proposed)

- **Environment:** a project-local virtualenv (`.venv/`) — every script, pipeline, and dependency lives inside it; nothing installed globally.
- Ingestion/ETL: **Polars (lazy/streaming)** + **PyArrow** + **DuckDB**.
- Storage: Parquet (ZSTD) partitioned by `season/session`.
- Features: Polars + NumPy; optional cuDF for batch vector ops.
- Modeling: PyTorch (LSTM / Autoencoder), XGBoost/LightGBM (GPU), HDBSCAN, SciPy/Monte Carlo optimizer.
- Orchestration/notes: plain scripts + `annotations/*.md`.

**Venv commands (run once, from repo root):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # activate (PowerShell)
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
> `.venv/` is gitignored. All `python`/`pip` calls in later phases assume the venv is activated.

---

## Phase 0 — Environment & Repo Scaffolding

**Objective:** Reproducible local environment inside a venv, verified GPU/CUDA, and project skeleton.

Tasks:
- [ ] Create and activate `.venv` (see commands above).
- [ ] Pin dependencies in `requirements.txt`: polars, pyarrow, duckdb, numpy, pandas, scipy, scikit-learn, xgboost, lightgbm, torch (cu12x), hdbscan, fastparquet/zstd.
- [ ] Verify CUDA: `torch.cuda.is_available()`, `xgboost` with `device='cuda'`, `lightgbm` GPU.
- [ ] On Windows, build LightGBM from source with `USE_GPU=ON`; its Windows GPU backend uses OpenCL, not `device='cuda'`.
- [ ] Create directory tree: `src/{ingest,features,models,eval}`, `data/{bronze,silver,gold}`, `notebooks/`, `annotations/`.
- [ ] Add `.gitignore` rules for `.venv/`, annotations + data lakehouse (see `.gitignore`).

**Deliverable:** `annotations/phase_0_setup.md`

### LightGBM GPU on Windows

The PyPI LightGBM wheel is CPU-only. Install Visual Studio Build Tools with
the C++ workload, then install Boost binaries matching the MSVC toolset. CUDA
Toolkit provides the NVIDIA OpenCL headers and library. From PowerShell:

```powershell
git clone --recursive https://github.com/lightgbm-org/LightGBM .\LightGBM-gpu
cmake -B .\LightGBM-gpu\build -S .\LightGBM-gpu -A x64 `
	-DUSE_GPU=ON `
	-DBOOST_ROOT=C:/local/boost `
	-DBOOST_LIBRARYDIR=C:/local/boost/lib64-msvc-14.3 `
	-DOpenCL_INCLUDE_DIR="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/include" `
	-DOpenCL_LIBRARY="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/lib/x64/OpenCL.lib"
cmake --build .\LightGBM-gpu\build --target ALL_BUILD --config Release
```

Install the Python package from that source checkout into the project venv:

```powershell
& .\.venv\Scripts\python.exe -m pip uninstall -y lightgbm
Push-Location .\LightGBM-gpu\python-package
& ..\..\.venv\Scripts\python.exe -m pip install .
Pop-Location
```

The verification script deliberately fails until this build is installed.

**Definition of Done:** Imports run on CPU+GPU; a tiny smoke test ingests 1 lap JSON → Parquet.

**Stop-n-Go checkpoint:** Confirm CUDA + version pins before touching 57 GB.

---

## Phase 1 — Data Exploration & Schema Validation (bounded sample)

**Objective:** Lock down real schema across all 5 seasons (it drifts: `rel_distance`, `dataKey`, `ms1..ms3`, sprint sessions, pre-season). Use a bounded sample only.

Tasks:
- [ ] Walk `season/gp/session/driver` tree; enumerate sessions (Race, Qualifying, Practice 1-3, Sprint, Sprint Qualifying, Pre-Season Testing).
- [ ] Confirm JSON envelope: telemetry nested under `{"tel": {...}}`; laptimes/weather/corners are flat dicts-of-lists.
- [ ] Catalog field presence/missingness per year (especially `distance`, `DistanceToDriverAhead`, `DriverAhead`, `acc_*`).
- [ ] Confirm `laptimes.json` fields: `time, lap, compound, stint, s1,s2,s3, ms1..ms3, life, pos, status, pb`.
- [ ] Confirm `weather.json` channels: `wT, wAT, wH, wP, wR, wTT, wWD, wWS`.
- [ ] Define a canonical schema (union) and per-year normalizers.

**Deliverable:** `annotations/phase_1_exploration.md` (+ schema manifest `src/ingest/schema.py`)

**Definition of Done:** A written schema manifest that every later phase trusts; known edge cases enumerated (e.g., `"None"` strings in laptimes, missing telemetry channels).

**Stop-n-Go checkpoint:** Confirm scope decision (all sessions vs Race-only first pass).

---

## Phase 2 — Streaming ETL → Bronze/Silver Lakehouse

**Objective:** Convert ~57 GB / ~367k JSON into compact, partitioned Parquet + DuckDB with bounded memory.

Tasks:
- [ ] Memory-bounded iterator: walk files, parse one lap at a time (never load a season).
- [ ] Parallelize with a worker pool (CPU) + chunked writes; flush per `season/gp/session`.
- [ ] Bronze `telemetry_clean.parquet` — raw rows + partition keys (`season, gp, session, driver, lap`), minimal normalization (types, nulls).
- [ ] Silver `stint_features.parquet` — per-lap aggregates from telemetry + laptimes merge (compound, stint, life, sectors).
- [ ] Silver `race_events.parquet` — weather (track/ambient temp, rain, wind), rcm (flags/SC/VSC), corners.
- [ ] Compress with ZSTD; report on-disk footprint vs raw 57 GB.
- [ ] Write a DuckDB view/read layer for downstream queries.

**Deliverable:** `annotations/phase_2_lakehouse.md`

**Definition of Done:** Full 5-season ingestion completes within RAM budget; Parquet footprint confirmed (target ≤ ~10 GB); row counts match inventory (~367k laps).

**Stop-n-Go checkpoint:** Verify disk/RAM stayed within limits; sample-validate a few GPs.

---

## Phase 3 — Feature Engineering (Gold Feature Store)

**Objective:** Build the physical feature tables that feed all 3 models.

Tasks (feature groups):
- [ ] **Mechanical/Thermal stress:** G-force integrals (`|acc_x|, |acc_y|, acc_z`), wheelspin proxy (throttle vs speed gain), brake pressure energy, per-corner energy dissipation (join `corners.json`).
- [ ] **Dirty-air penalty:** from `DistanceToDriverAhead`/`DriverAhead` → time-in-wake, following-distance percentiles, DRS-train exposure.
- [ ] **Degradation kinetics:** per-stint lap-time delta, sector decay slope, tyre-life, cliff-onset flag.
- [ ] **Environmental:** track/ambient temp, temp slope, rain/wind from `race_events`.
- [ ] **Driving-style embeddings:** micro-throttle modulation, brake/turn-in signatures per driver.
- [ ] Write to `data/gold/*.parquet`; document feature dictionary.

**Deliverable:** `annotations/phase_3_features.md` (+ `src/features/*`)

**Definition of Done:** Deterministic feature tables; feature dictionary; no leakage (e.g., no future lap data in RUL labels).

**Stop-n-Go checkpoint:** Feature audit vs model requirements before modeling.

---

## Phase 4 — Model 1: RUL & Degradation Cliff Predictor

**Objective:** Predict remaining useful life of a tyre and cliff onset from per-lap features.

Tasks:
- [ ] Label engineering: RUL = laps until pit/compound change; cliff = degradation inflection.
- [ ] Train/val/test split by GP (temporal, avoid leakage).
- [ ] Baselines: XGBoost / LightGBM (GPU) on tabular features.
- [ ] Sequence model: PyTorch LSTM on per-lap sequences (GPU).
- [ ] Evaluate: MAE/RMSE on RUL, precision/recall on cliff; compare vs naïve (mean stint length).

**Deliverable:** `annotations/phase_4_rul.md`

**Definition of Done:** Trained model + metrics + saved artifacts (onnx/pickle/pt).

**Stop-n-Go checkpoint:** Decide whether LSTM justifies cost vs gradient boosting.

---

## Phase 5 — Model 2: Unsupervised Driving Style & Tyre Care Profiler

**Objective:** Cluster driving styles & tyre-care behaviour without labels.

Tasks:
- [ ] Build driver-level style features (brake/turn-in/throttle signatures).
- [ ] HDBSCAN clustering → style archetypes.
- [ ] TimeSeries Autoencoder (PyTorch) → latent embeddings + reconstruction-error anomaly flag.
- [ ] Interpret clusters (labels, tyre-life impact).

**Deliverable:** `annotations/phase_5_style.md`

**Definition of Done:** Stable clusters with interpretable profiles; embeddings exported to gold store.

**Stop-n-Go checkpoint:** Cluster quality review (silhouette/DBCV).

---

## Phase 6 — Model 3: Prescriptive Pit-Window & Stint Optimizer

**Objective:** Recommend compound + pit laps dynamically (clean air / DRS train / backmarker scenarios).

Tasks:
- [ ] Merge Model 1 (RUL) + Model 2 (style) + scenario context into a simulator.
- [ ] Monte Carlo simulation of race outcomes under candidate strategies.
- [ ] Constrained optimization (SciPy / custom DP) over pit windows & compounds.
- [ ] Output prescriptive recommendation + confidence band.

**Deliverable:** `annotations/phase_6_optimizer.md`

**Definition of Done:** End-to-end recommendation pipeline producing a ranked strategy per scenario.

**Stop-n-Go checkpoint:** Full-system demo on a sample GP.

---

## Phase 7 — Integration, Evaluation & Report

**Objective:** Wire everything, validate on held-out GPs, write final report.

Tasks:
- [ ] End-to-end run script: raw JSON → recommendation.
- [ ] Backtest against real 2021–2025 races (did prescribed strategy beat actual?).
- [ ] Final metrics, ablation, and report (LaTeX in `document/`).
- [ ] Cleanup intermediates; document runbook.

**Deliverable:** `annotations/phase_7_report.md`

**Definition of Done:** Reproducible runbook + final report + backtest results.

---

## Per-Phase Deliverable (what the agent hands you each time)

1. **Code** — scripts/modules under `src/` (plus any configs).
2. **Run instructions** — exact commands to execute, in order, with expected inputs/outputs.
3. **Explanation** — plain-language description of what the code does and why.

The user then runs it and pastes the results into the phase note below.

## Annotation Template (per phase)

Every `annotations/phase_N_*.md` contains (filled in after the user runs the code):

```markdown
# Phase N — <name> (STOP)

- Date / author:
- Status: [ ] Go  [ ] No-Go  [ ] Rework
- Commands run by user (copy/paste):
- Results / output (from user):
- Metrics / numbers:
- Errors / blockers:
- Decisions made:
- Next-phase risks:
- Time spent / next step:
```
