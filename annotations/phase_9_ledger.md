# PLAN3 implementation ledger

Plan: `PLAN3.md`; specifications: `planning/practice3_mapreduce.md`,
`planning/practice5_mongodb.md`, `planning/practice3_5_integration.md`.

- Baseline: 53 tests passed (`python -m pytest -q --basetemp=.tools/pytest-baseline`).
- Phase 8 corrections preserved in commit `2589372` before Phase 9 implementation.
- Work proceeds on `codex/phase9-mapreduce-mongodb` in the existing checkout: the
  untracked plans and ignored lakehouse/venv are needed by this implementation.
- Ruling: the supplied plan and subplans are the approved design; no additional
  design approval is needed for the user's explicit implementation request.
- Pre-flight: labeled export supplies both MR and Mongo; Mongo repository must
  retain full-session lap counts from unfiltered Gold, not filtered labeled laps.
- Ruling: keys and partitions are different quantities. Verify every key maps to
  exactly one reducer and every nonempty partition has a reduce task; do not assert
  distinct keys equals nonempty partitions (false when keys outnumber reducers).
- Ruling: mappers emit zero or many pairs; count input records separately from KV
  emissions. Combiner-aware verification compares map-to-combine reference results
  and the exact combine-to-shuffle multiset. Sampled traces cannot prove full coverage.
- Ruling: reproduce the actual estimator, including wet compounds, nearest-quantile
  IQR, sigma clipping to [0.02, 0.5], and sparse-compound fallback; the plan omits
  these details. Gold's raw per-stint slope has a different population/estimator.
- Backtest discrepancy: summary describes 313 decisions; writer truncates persisted
  history to `rows[:200]`. Existing history contains 200, not 313; no missing records
  will be fabricated. Fix future writes to retain the full history.

## Checkpoints

- Setup / export: complete; 107,606 × 85, MongoDB 7.0.45 / PyMongo 4.19.0.
- MapReduce runtime and verification gate: complete; map/reduce retry, failed
  overwrite preservation, full-trace conservation and independent reduction pass.
- Real MR jobs / parity: complete; 69,459 Silver laps, 4,628 eligible stints,
  70 profiles. Max Silver error 2.70e-13, MR/Polars error 3.55e-15.
- MongoDB / live evidence: complete; 576/44,298/107,606/200 real collections,
  100,000 benchmark docs; second/scoped upserts unchanged, zero schema violations,
  six indexed explain comparisons, bounded streaming allocation growth.
- Integration / reports / reproducibility: complete; five real states and
  recommendations match. Bahrain CLI ranked strategies and pit-monitor output
  are byte-identical between sources. Cached pipeline succeeds online/offline.
- Reports: native pdflatex succeeds (Spanish 9 pages, English 6), no unresolved
  references; all three generated figures visually inspected. App compiler timed out.
- Final independent review: actionable findings fixed and scoped re-review clear.
  Regression tests demonstrated missing Silver schema fields, saved-model drift,
  nullable covariance sample counts, CLI URI/db/output forwarding, synthetic cache
  size mismatch, and driver slopes mixing stint pace offsets before their fixes.
- Ruling: profile degradation is the mean of eligible per-stint slopes, avoiding
  cross-race pace offsets contaminating a pooled life/pace regression.
- Ruling: `tracemalloc` measures peak Python allocations, explicitly permitted by
  the Practice 5 subplan; no total-RSS assertion is claimed. Streaming uses 1.65 MB
  versus a list's 141.56 MB on the measured 100,000-document experiment.
- Ruling: Mongo orchestration always runs idempotent upserts (36 s cached quick
  path), instead of count-only caching that could hide changed values at identical
  row counts. Unavailable Mongo skips in 4 s. Backtest remains manual.
- Ruling: integration implementation proceeded alongside live Mongo experiments
  after the runtime gate; integration acceptance waited for real Mongo parity.
- Ruling: preserve native compile evidence when the app compiler times out; no
  additional compiler retries or inferred success are needed.
- Hosted CI has not been dispatched; local equivalent suite passes with live Mongo.

- Final verification: 86 tests passed in 12.54 s with live Mongo; compileall and diff whitespace checks pass.
