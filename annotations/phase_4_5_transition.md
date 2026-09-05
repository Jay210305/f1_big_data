# Phase 4 → 5 Transition — Structural Adjustments (ACTION ITEMS)

- Date / author: 2026-08-29 / agent
- Status: [x] Integrated into PLAN.md (Phase 5 & 6 tasks) — to be executed in Phase 5/6
- Source: "Phase 4 to 5 Transition: Annotations & Action Items for the Driving Style Profiler"

## Action item 1 — Strict in/out lap exclusion (Phase 5)

**Risk:** out-laps (tyre warming) / in-laps (coasting to pits) corrupt driving-style
profiles — a driver would falsely appear "gentle" or "erratic", poisoning HDBSCAN
clusters and autoencoder embeddings.

**Implementation plan (Phase 5 feature extraction):**
- Hard filter BEFORE any micro-telemetry or mechanical-stress aggregation:
  - drop laps with `laptime IS NULL` (~14%, Phase 3 finding)
  - drop out-laps: first lap of a stint (stint position 0 — pit exit)
  - drop in-laps: last valid lap of a stint (pit entry; heuristically = stint end)
- Session context restriction:
  - hot laps only: `laptime <= 1.07 × driver's session best` (107% rule)
  - exclude SC/VSC laps entirely via the rcm-derived lap set (as in Phase 4)

## Action item 2 — Dynamic track evolution proxy (Phase 5 & 6)

**Risk:** without a grip proxy, increasing pace / reduced degradation on Sunday
is misread as "gentle style" instead of a rubbered-in (grippier) track.

**SPEC UPDATE (rain wash-out, 2026-08-30):** the monotonic accumulation lacked a
physical reset — heavy rain washes deposited rubber off the line, returning the
track to a low-grip "green" state. Implemented formulation:

- Weekend fold (chronological, per session k in weekend order):
  `R_prior(S_k) = [R_prior(S_{k-1}) + w_{k-1} · V_{k-1} · (0.5)^W_{k-1}] · (0.3)^W_k`
  - γ_wash = 0.3 — a wet session washes away 70% of accumulated rubber
  - wet sessions lay rubber at 50% efficiency ((0.5)^W_{k-1})
- Lap level inside session k at lap t:
  `TEI(t) = [R_prior(S_k) + w_k · cum_session_laps(t)] · (0.6)^r_t`
  - r_t = 1 if any driver is on INTERMEDIATE/WET at lap number t (per-lap rain
    proxy from compound usage) → instantaneous grip × 0.6
- Wet-session flag W_s: wet-compound lap fraction > 0.02 OR any rain recorded
  in the session (rain_flag).
- Session weights unchanged: FP1 0.6 / FP2 0.8 / FP3 1.0 / SQ 1.1 /
  Quali 1.2 / Sprint 1.3 / Race 1.5.

**Implementation plan (original, superseded by the wash-out version above):**
- `cumulative_session_laps`: total valid laps by ALL drivers in the session up
  to the current lap index (not a simple lap-number heuristic).
- `track_evolution_index` in the Gold feature store as an Autoencoder covariate
  (disentangles track grip from style).

## Action item 3 — Cliff false-positive mitigation (Phase 6 optimizer)

**Context:** Phase 4 cliff ensemble (thr 0.353, recall-biased) ⇒ F1 0.747 ⇒
~34% of critical alerts are false positives.

**Implementation plan (Phase 6 strategy logic):**
- **Temporal confirmation (debouncing):** pit recommendation requires
  `cliff_probability > threshold` on ≥ 2 consecutive laps (hysteresis loop) —
  a single spike (traffic back-off, brief lock-up) never triggers a stop.
- **Joint probability rule:** if the cliff alert fires but the RUL estimate
  (LSTM in the critical zone) predicts > 15 laps remaining, classify as
  anomaly / temporary thermal degradation (e.g., dirty-air overheating), not
  physical wear — do not pit on it.

## Action item 4 — Computational strategy for micro-telemetry (Phase 5)

**Risk:** Autoencoder on 325M raw Bronze rows (10–20 Hz) on 32 GB RAM → OOM
or unusable epoch times.

**Implementation plan:**
- Materialize a per-lap representation in DuckDB/Polars (Silver-level view):
  - **Option B (chosen): micro-sector aggregation** — fixed-length vector per
    lap: telemetry aggregated over ~10 standardized distance bins (by
    `distance` channel) × {max lat G, mean throttle, brake energy, mean speed}.
    ~317k laps × ~40 values ≈ 13M floats — tiny, DataLoader-friendly.
  - Option A (fallback): decimate Bronze to 1–2 Hz via DuckDB before PyTorch.
- Applied AFTER the action-item-1 lap filter (in/out/SC laps excluded first).

## Integration status

| Item | PLAN.md | Phase 5/6 execution |
|------|---------|---------------------|
| 1. In/out lap exclusion | Phase 5 task added | pending Phase 5 |
| 2. Track evolution index | Phase 5 task added | pending Phase 5 |
| 3. Cliff debouncing + joint rule | Phase 6 tasks added | pending Phase 6 |
| 4. Micro-sector aggregation | Phase 5 task added | pending Phase 5 |

Referenced from `annotations/phase_4_rul.md` → Next-phase risks.
