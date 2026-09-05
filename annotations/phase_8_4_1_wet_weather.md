# Tier 4.1 — Wet-weather modeling

The degradation fit now includes `INTERMEDIATE` and `WET` stints. It derives two
bounded contextual features: `rain_intensity` (the measured `rain_flag`, with
zero as the historical fallback) and `track_drying_proxy` (the within-session
fraction of the track-evolution index). The model metadata records the
`WET_EFFICIENCY` and `INSTANT_WET` factors already used by the style pipeline.

The optimizer uses the same analytic Monte Carlo path for all five compounds,
with conservative defaults when an older degradation artifact lacks wet
parameters. Strategy enumeration and parsing accept wet compounds while
retaining the two-compound rule. The backtest no longer removes a whole race
because its wet fraction exceeds 2%; wet stints are scored when their data is
complete, and truncated/DNF stints retain the existing skip policy.

This is a first-order wet model, not a rainfall-fluid-dynamics model. OpenF1
or a future weather source can replace the binary rain fallback with measured
intensity without changing the optimizer contract.
