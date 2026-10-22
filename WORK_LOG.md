# Work Log — what was done, in order

A chronological record of the work carried out after the Phase 9 delivery, with the
numbers that were measured at each step. Every figure here was re-read from the
artifacts in `data/` or from a live run, not copied from a plan.

---

## Session 1 — The project as it stood

The project was reviewed against `PLAN.md`, `PLAN2.md`, `PLAN3.md` and the nine
sub-plans in `planning/`. The review produced a full map of the system and a
register of what was **not** correct.

**Verified as working end-to-end:** 585 sessions ingested (325,541,022 Bronze
telemetry rows, 16.69 GB), the three models, the optimizer, the MapReduce runtime,
the MongoDB layer, both LaTeX reports.

**25 defects** were tracked in the repository as strict `xfail` tests — reproduced
failures recorded as "the invariant we want". They were the honest backlog, and
they are listed in full in [`annotations/phase_9_5_defect_closure.md`](annotations/phase_9_5_defect_closure.md).

A full overview of the system was written to [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md).

---

## Session 2 — Closing the 25 defects

Every `xfail` marker was removed and every defect fixed, so the invariants became
ordinary assertions.

| Cluster | Defects | Outcome |
|---|---|---|
| Modelling | M01–M10 | Fixed in `optimizer.py`, `rul_data.py`, `style_data.py`, `fit_degradation.py`, `backtest_realized.py` |
| MapReduce | MR01–MR04 | Fixed in `mapreduce/verify.py`, `mapreduce/runner.py`, `mapreduce/evidence.py` |
| MongoDB / integration | I01, I02, M01m–M03m | Fixed in `nosql/*`, `run_all.py`, `export_labeled.py` |
| Documentation | report ↔ code drift | Every reported number reconciled with its artifact |

**Two further defects** surfaced while closing the others:

- A full Mongo ingest left stale documents behind when the source changed
  (200 `sc_hazard` decisions sitting next to 346 `clean_air` ones). A full ingest
  now **prunes** what the source no longer produces; a scoped ingest never deletes.
- `PermissionError` (WinError 5) in the MapReduce temp-tree cleanup masked the
  real job result. The cleanup now retries and then gives up quietly.

**Also delivered:** the four missing report figures (RUL predicted-vs-actual,
degradation curves, style cluster embeddings, backtest delta histogram) — nine
figures in total, all generated from real artifacts.

**Test suite:** 129 passed / 25 xfailed → **157 passed / 0 failed / 0 xfailed**.

---

## Session 3 — The calibration finding

### The lead

The realized-pace cross-check reported that the optimizer's compound choice was
biased towards the harder tyre (−8.1 s on the decisions where it diverged). That
looked like a compound-offset calibration error.

### Two corrections to my own diagnosis

**The metric was circular.** The realized delta is algebraically
`−(pace_offset[rec] − pace_offset[actual]) × post-pit laps`. It *restates the
model's own offsets* and can never falsify them. It was not an independent check.
It now says so in the code, the reports and the runbook, and reports the
informative statistic instead: the optimizer picks the **same compound in 47.1 %**
of the 327 decisions.

**The aggregate bias was tiny.** The first probe showed +179 s, which was a bug in
the probe itself (it compared against the labelled frame, which has fewer rows
than the Gold lap table). With the correct population the aggregate bias is
**+0.28 %** — small enough to hide a large structural error.

### The actual finding

The error only appears **per stint**:

| Per-stint residual (387 dry stints) | Before | After |
|---|---|---|
| mean | **+38.4 s** | **+0.6 s** |
| median | **+39.7 s** | **−1.7 s** |

Two structural errors in the simulator:

1. **The degradation was charged twice.** A stint inherited the previous stint's
   *terminal* pace, so that stint's degradation was billed once inside the
   inherited base and again by the degradation term. The error grew with the stop
   count and the stint length — and cancelled against error 2 in the aggregate.
2. **The pit-entry lap was the base pace.** The decision is normally taken on the
   lap the car enters the pits, 5–20 s slower than the flying pace. The backtest
   corrected this locally; **the optimizer CLI did not**.

The single root cause is the fixed **0.03 s/lap fuel correction**
(`fuel_corr = laptime + 0.03·lap`), which under-credits the pace the car
recovers as the fuel burns off. The coefficient that would absorb it was measured
(−0.04 s/lap ⇒ an effective burn-off of ≈0.07 s/lap) and deliberately **not
adopted**: it shifts every candidate equally, so it cancels in the ranking.

### Consequences

| Metric | Before | After |
|---|---|---|
| Backtest win rate | 77.3 % | **77.6 %** |
| Δ median | +13.8 s | **+18.0 s** (+31 %) |
| Δ mean | +1.1 s | **+7.6 s** |
| p10 / p90 | −72.6 / +37.0 s | **−63.7 / +44.8 s** |
| Stops recommended vs run | 1.51 vs 1.66 | **1.73 vs 1.66** |
| Matched / divergent strata | 80.3 % / 70.0 % | 77.3 % / **78.1 %** |

The systematic under-stopping disappeared, and the optimizer now wins slightly
*more* where it diverges from the field instead of less.

---

## Session 4 — Documentation

| Document | What it is |
|---|---|
| [`annotations/phase_9_5_defect_closure.md`](annotations/phase_9_5_defect_closure.md) | The 25 defects, their fixes and the measured before/after |
| [`annotations/phase_10_simulator_calibration.md`](annotations/phase_10_simulator_calibration.md) | The calibration work in full, including the two corrections to the diagnosis |
| [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) | The whole project, end to end, against the current artifacts |
| [`MAPREDUCE_Y_MONGODB.md`](MAPREDUCE_Y_MONGODB.md) | MapReduce y MongoDB en detalle (ES) |
| [`WORK_LOG.md`](WORK_LOG.md) | This file |

---

## Commits

| Commit | Scope |
|---|---|
| `2589372` | *(pre-existing)* Phase 8 feature corrections preserved before Phase 9 |
| `c351da6` | Phase 9 MapReduce + MongoDB, all 25 audited defects closed, 4 missing figures, every reported number reconciled — 83 files |
| `d6c6d53` | Simulator calibration against the realized lap times — 10 files |

---

## Final state

| Check | Result |
|---|---|
| Test suite | **157 passed, 0 failed, 0 xfailed** (12 consecutive clean runs) |
| MapReduce trace | 107,606 inputs → 106,735 pairs → 4,981 keys → 4 partitions, verified |
| MR ≡ Polars ≡ Mongo β | **3.55e-15 / 1.78e-15** over 4,628 stints |
| Silver reproduction | 69,459 rows, **0 differing** |
| Mongo | 576 / 44,298 / 107,606 / 313 / 100,000 docs, **0** schema violations |
| Mongo idempotency | 0 modified, 0 upserted, 0 pruned |
| Simulator calibration | per-stint residual **+0.6 s** mean, **−1.7 s** median |
| Backtest | **77.6 %** win rate, **+18.0 s** median paired gain |
| Reports | `main.pdf` 11 pp (ES), `strategy_engine_report.pdf` 9 pp (EN), both compile |

### Still open (honestly)

- The simulator has **no race-position term** — early stints are over-predicted,
  late ones under-predicted (no track temperature, no traffic).
- The **compound pace offset is not independently validated**: within-race
  evidence is too noisy (±2.9 s per stint pair) to resolve a 0.1–0.3 s effect,
  and the realized-pace metric is circular.
- The **wet compounds remain priors**; the backtest excludes the three
  rain-affected races for that reason.
- **CI has never been dispatched** to a hosted runner.
