# Phase 8.1.1 — Safety Car & VSC Detection Coverage Audit (STOP)

- Date / author: 2026-09-05 / Antigravity
- Status: [x] Go  [ ] No-Go  [ ] Rework
- Deliverable for: Task 8.1.1 in `planning/tier_1_sc_vsc_correctness.md`

## Files created / modified

| File | Purpose |
|------|---------|
| `src/models/rul_data.py` | Extended `load_sc_laps()` query to include verified SC/VSC message patterns and category variants while guarding against steward penalty notices. |
| `src/ingest/schema.py` | Documented FastF1 categorical convention for `RCM_COLUMNS` (`cat='SafetyCar'` encapsulates both physical SC and VSC). |
| `tests/test_rul_data.py` | Added unit test `test_sc_vsc_detection_patterns()` verifying SQL logic filters real SC/VSC events and rejects steward infringement notices. |
| `planning/tier_1_sc_vsc_correctness.md` | Marked Task 8.1.1 subtasks 8.1.1.1 through 8.1.1.7 as complete. |
| `annotations/phase_8_1_1_sc_detection_audit.md` | Walkthrough, empirical audit findings, before/after counts, and execution log. |

## Steps to run (in order)

```powershell
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Run unit tests to verify detection logic and regression safety
python -m pytest tests/test_rul_data.py -v

# 3. Run full test suite across all modules
python -m pytest tests/ -q

# 4. Execute RUL data preparation pipeline to verify SC lap counts and stint filtering
python -m src.models.rul_data
```

## What's going on

Task 8.1.1 is the prerequisite foundational audit for Tier 1 (Safety Car / VSC Correctness). In the initial codebase, `load_sc_laps()` filtered Race Control Messages (RCM) using:
```sql
WHERE (cat = 'SafetyCar') OR (flag IN ('DOUBLE YELLOW', 'RED'))
```
This audit verified whether Virtual Safety Car (VSC) periods and physical Safety Car events were fully captured or whether data was silently contaminated.

### 1. Empirical RCM Token Audit
Using `src.ingest.duckdb_layer.open_lakehouse()`, we queried distinct categories, flags, and messages across all seasons (2021–2025):
- **Categories in raw data**: `CarEvent` (1,201), `Drs` (675), `Flag` (12,612), `Other` (10,671), `SafetyCar` (438).
- **Finding on VSC categorization**: FastF1 and FIA telemetry feed do **not** use `cat='VirtualSafetyCar'`. Instead, both physical Safety Car and Virtual Safety Car events are categorized under `cat='SafetyCar'`.
- **Exact message signatures under `cat='SafetyCar'`**:
  - `'VIRTUAL SAFETY CAR DEPLOYED'` (123 occurrences)
  - `'SAFETY CAR DEPLOYED'` (112 occurrences)
  - `'VIRTUAL SAFETY CAR ENDING'` (98 occurrences)
  - `'SAFETY CAR IN THIS LAP'` (94 occurrences)
  - `'SAFETY CAR THROUGH THE PIT LANE'` (11 occurrences)
- **Known VSC races verified**:
  - *2023 Austrian GP (Race)*: Lap 14 (`VIRTUAL SAFETY CAR DEPLOYED`), Lap 16 (`VIRTUAL SAFETY CAR ENDING`).
  - *2024 Australian GP (Race)*: Lap 17 (`VIRTUAL SAFETY CAR DEPLOYED`), Lap 18 (`VIRTUAL SAFETY CAR ENDING`), Lap 58 (`VIRTUAL SAFETY CAR DEPLOYED`).
  - *2024 Azerbaijan GP (Race)*: Lap 51 (`VIRTUAL SAFETY CAR DEPLOYED`).

### 2. Discrepancy Discovered in `cat='Other'`
While deployment/ending messages for VSC are under `cat='SafetyCar'`, several critical physical Safety Car restart and procedure messages were filed under `cat='Other'`, which caused the original filter to miss 46 non-null session-laps (and 13 distinct session-laps under Race/Sprint conditions):
- `'SAFETY CAR WILL ENTER PITS: ROLLING START PROCEDURE'` (e.g. 2022 Japan L5, 2023 Australia L58, 2023 Netherlands L66)
- `'SAFETY CAR WILL ENTER PITS: STANDING START'` (e.g. 2021 Britain L3, 2021 Saudi Arabia L14 & L16, 2022 Britain L2)
- `'SAFETY CAR WILL ENTER PITS: ROLLING START'` (e.g. 2021 Emilia Romagna L34, 2022 Monaco L2 & L32)
- `'ALL CARS TO FOLLOW THE SAFETY CAR THROUGH THE PIT LANE'` (e.g. 2025 Canada L67)
- `'LAPPED CARS ... TO OVERTAKE SAFETY CAR'` (e.g. 2021 Abu Dhabi L57)

### 3. Discrimination Against Steward Notices
Under `cat='Other'`, numerous messages mention `"SAFETY CAR"` or `"VIRTUAL SAFETY CAR"`, but represent steward incident logging or penalties issued laps after the fact (e.g. `"INCIDENT INVOLVING CAR 27 (HUL) NOTED - VIRTUAL SAFETY CAR INFRINGEMENT"`, `"FIA STEWARDS: 5 SECOND TIME PENALTY FOR CAR 11 (PER) - SAFETY CAR INFRINGEMENT"`).
The extended filter logic explicitly guards against these notices:
```sql
WHERE (
    cat IN ('SafetyCar', 'VirtualSafetyCar', 'VSC')
    OR (
        upper(msg) LIKE '%SAFETY CAR%'
        AND upper(msg) NOT LIKE '%INFRINGEMENT%'
        AND upper(msg) NOT LIKE '%PENALTY%'
        AND upper(msg) NOT LIKE '%UNDER INVESTIGATION%'
        AND upper(msg) NOT LIKE '%NOTED%'
        AND upper(msg) NOT LIKE '%REVIEWED%'
        AND upper(msg) NOT LIKE '%LIGHTS ON%'
    )
    OR (
        (upper(msg) LIKE '%VIRTUAL SAFETY CAR%' OR upper(msg) LIKE '%VSC%')
        AND upper(msg) NOT LIKE '%INFRINGEMENT%'
        AND upper(msg) NOT LIKE '%PENALTY%'
        AND upper(msg) NOT LIKE '%UNDER INVESTIGATION%'
        AND upper(msg) NOT LIKE '%NOTED%'
        AND upper(msg) NOT LIKE '%REVIEWED%'
        AND upper(msg) NOT LIKE '%TEST%'
    )
    OR (flag IN ('DOUBLE YELLOW', 'RED'))
)
```

## After you run it

Verify that `sc laps` detected in `rul_data.py` increases from 793 to 822 session-laps (+29 rows, covering the previously missed restart/pit procedure laps), and that stint filtering removes the contaminated stints without altering validation or test split integrity.

## Results (filled after running)

- **Commands run**:
  ```powershell
  python -m pytest tests/test_rul_data.py -v
  python -m pytest tests/ -q
  python -m src.models.rul_data
  ```

- **Results / output**:
  - `pytest tests/test_rul_data.py -v`: 7 passed in 0.28s (including new `test_sc_vsc_detection_patterns`).
  - `pytest tests/ -q`: 39 passed in 1.90s.
  - `python -m src.models.rul_data`:
    ```
    [prep] Race/Sprint laps: 138,449
    [prep] loading SC/VSC laps from rcm...
       sc laps: 822 session-laps affected
    [prep] loading lateral tyre-energy integral (cached Bronze scan)...
    [prep] stint filtering (SC/VSC ends, DNF-ish, short dry stints)...
       stint filtering: 138,449 -> 108,318 laps (21.8% removed)
    [prep] fuel correction + rolling features...
    [prep] performance-based labels (T_end, piecewise RUL, cliff)...
    train tabular: X=(60455, 80) target_rul mean=6.80 cliff+=0.424
          seqs: 2840 stints
    val   tabular: X=(25188, 80) target_rul mean=6.85 cliff+=0.412
          seqs: 1176 stints
    test  tabular: X=(22675, 80) target_rul mean=8.07 cliff+=0.383
          seqs: 1036 stints
    ```

- **Metrics / numbers**:
  | Metric | Before (Task 8.1.1) | After (Task 8.1.1) | Delta |
  |---|---|---|---|
  | `sc_laps.height` (total rows) | 793 | 822 | +29 |
  | `sc_laps` non-null valid laps | 477 | 506 | +29 |
  | Filtered Race/Sprint laps | 108,340 | 108,318 | -22 (more contaminated laps culled) |
  | Stints retained | 5,058 | 5,052 | -6 stints that ended under missed SC |
  | Train set rows | 60,477 | 60,455 | -22 |
  | Val set rows | 25,188 | 25,188 | 0 (stable) |
  | Test set rows | 22,675 | 22,675 | 0 (stable) |
  | Unit test suite | 38 passed | 39 passed | +1 regression test |

- **Errors / blockers**:
  - None. DuckDB and Polars executed reliably in `.venv`.

- **Decisions made**:
  - Confirmed FastF1's architecture places both physical SC and VSC under `cat='SafetyCar'`.
  - Added pattern matching for restart/procedure laps under `cat='Other'` (e.g. `SAFETY CAR WILL ENTER PITS`) to prevent off-pace restart laps from contaminating stint cliff calculations.
  - Excluded steward infringement/penalty notices (`INFRINGEMENT`, `PENALTY`, `UNDER INVESTIGATION`, `NOTED`) to prevent non-SC green-flag laps from being falsely tagged.

- **Next-phase risks**:
  - In Task 8.1.2, mid-stint SC/VSC laps (where cars run 20–40s off pace) must be excluded from rolling pace windows (`pace_roll_med_3`, etc.) and degradation slope fitting ($\beta$). The verified SC lap set from 8.1.1 will serve as the exact input mask for 8.1.2.

- **Time spent / next step**:
  - Time spent: ~25 minutes.
  - Next step: Task 8.1.2 in `planning/tier_1_sc_vsc_correctness.md` (Exclude mid-stint SC/VSC laps from rolling features and cliff detection).
