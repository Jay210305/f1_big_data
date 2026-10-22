# Phase 9.4 — Reproducibility and defense (STOP)

- Date / author: 2026-10-21 / Codex
- Status: [x] Go  [ ] No-Go  [ ] Rework

## Files created / updated

`src/mapreduce/README.md`, `src/nosql/README.md`, `README.md`, `RUNBOOK.md`,
`F1_Data_Inventory.md`, `document/phase9_defense_notes.md`,
`document/phase9_traceability.md`, PLAN2/PLAN3 status and these checkpoints.

## Steps to run (in order)

Follow the runbook's Phase 9 section, then read the ten-slide defense outline
and the requirement → implementation → test/evidence → figure matrix.

## What's going on

The evidence separates measured single-node behavior from future deployment
claims. Full traces, independent references, genuine Mongo execution statistics,
input identity and deterministic retries support the laboratory requirements.

## After you run it

A fresh checkout needs the ignored raw dataset or the lakehouse artifacts
identified in the inventory, then the runbook reproduces evidence locally.
No raw telemetry or fabricated history is checked into Git.

## Results (filled after running)

- Full suite: 86 passed with real Mongo access. Pure tests and unavailable-server
  paths also run independently of the data lake.
- Both reports compile; all three figures were generated and visually inspected.
- Independent review findings fixed with regression tests: missing Silver columns,
  nonblocking saved-model mismatch, nullable OLS sample counts, CLI database/output
  forwarding and synthetic benchmark cache-count mismatch.
- Existing Phase 8 work committed separately before Phase 9. Phase 9 changes are
  left on the feature branch for review; no merge or remote publication performed.
- Limitations retained explicitly: historical decision truncation, different raw
  Gold/driver-style populations, missing cached wet model parameters, local-only
  deployment and client allocation memory measurements.
- CI service configuration is present; local test success does not assert that
  an unrun remote workflow passed.
