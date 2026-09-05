# Tier 3.1 — Pre-race planning

- Date: 2026-09-05
- Status: validated
- Seeding: longest Practice 2 stint per compound; `laptime` median is used because the current Gold Practice rows do not contain `fuel_corr`.
- Fallback: if the driver lacks a compound in FP2, use the longest FP2 stint for another driver at the same GP and expose `driver/field` in `pace_source`.

## Validation

Command:

```powershell
python -m src.models.optimizer --pre-race --season 2025 --gp "Bahrain Grand Prix" --driver VER --starting-compound MEDIUM --n-sims 20
```

Observed state: lap `0/57`, MEDIUM, 34.8 C, cluster 0. The optimizer produced ranked plans for clean air, DRS train, and backmarker scenarios. The clean-air best plan was `MEDIUM->18 | HARD->57` at 5789.2 s in this 20-trial smoke run.

The run used `Practice 2 longest stints (median laptime; driver, field, field)`: VER had no MEDIUM FP2 stint, so the documented field fallback supplied the missing compound priors.

For a like-for-like start on SOFT, the same command with
`--starting-compound SOFT` ranked `SOFT->15 | HARD->37 | HARD->57` first in
clean air at 5775.1 s in the 20-trial smoke run (the one-stop
`SOFT->18 | HARD->57` was 5775.2 s). The actual execution was
`SOFT->10 | HARD->26 | MEDIUM->57` (10, 16, and 31 laps respectively), so
the pre-race model selected a two-stop plan and did not reproduce the team's
compound sequence in this small validation run.

## Tier 2 branch point

Command:

```powershell
python -m src.models.optimizer --pre-race --sc-hazard --season 2025 --gp "Bahrain Grand Prix" --driver VER --starting-compound MEDIUM --n-sims 20
```

The branch output was:

```text
green flag : MEDIUM->18 | HARD->57 | 5789.2s (p5-p95 5723.5-5855.2)
SC/VSC     : MEDIUM->19 | MEDIUM->33 | HARD->57 | 5784.2s (p5-p95 5741.8-5845.8)
SC - green : -4.9s
```

This is a decision branch, not a claim that an SC will occur: the second
plan uses the empirical Tier 2 hazard Monte Carlo and discounted pit loss.

## Limitation

This is a practice pace prior, not a fuel-corrected race estimate. A future Gold-store change can add fuel correction for Practice and remove the fallback approximation.
