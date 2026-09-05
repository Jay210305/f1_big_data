# Tier 3.2 — What-if comparator

- Date: 2026-09-05
- Status: validated
- Monte Carlo mode: common random numbers, seed 42, clean-air scenario.

## Validation transcript

Command:

```powershell
python -m src.models.optimizer --season 2025 --gp "Bahrain Grand Prix" --driver VER --lap 12 --compare "HARD->18,MEDIUM->end" "HARD->23,SOFT->end" --n-sims 20
```

Output from the validation run:

```text
[compare] A vs B (positive delta means A is slower)
  A: HARD->18,MEDIUM->end
  B: HARD->23,SOFT->end
  delta A-B: +13.00s (90% interval -5.27..+34.88s)
  P(A faster): 15.0%
  profile A: HARD 6 laps (beta 0.033), MEDIUM 39 laps (beta 0.073)
  profile B: HARD 11 laps (beta 0.033), SOFT 34 laps (beta 0.106)
```

The parser resolves `end` to the race distance and rejects plans whose first compound differs from the current state or whose stints do not finish at the race distance.
