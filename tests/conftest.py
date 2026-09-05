"""Shared pytest fixtures — synthetic Polars frames (no 57 GB dataset dependency).

Makes the repo root importable and provides reusable small DataFrames so that
pure/leakage-critical functions are tested against controlled inputs.
"""
import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def laps_frame() -> pl.DataFrame:
    """A minimal Race/Sprint lap-level frame with the columns the RUL data
    helpers (`add_fuel_corr`, `filter_stints`, `compute_labels`) consume."""
    # one 8-lap Race stint + one 4-lap Sprint stint, two drivers
    rows = []
    for season, gp, session, driver, stint, laps, base in [
        ("2025", "Test GP", "Race", "VER", 1, 8, 100.0),
        ("2025", "Test GP", "Race", "HAM", 1, 4, 100.5),
        ("2025", "Test GP", "Sprint", "VER", 1, 4, 99.0),
    ]:
        for life in range(laps):
            rows.append({
                "season": season, "gp": gp, "session": session,
                "driver": driver, "stint": stint,
                "lap": life + 1, "life": life,
                "compound": "SOFT",
                "laptime": base + life * 0.1,
            })
    return pl.DataFrame(rows)


@pytest.fixture
def track_frame() -> pl.DataFrame:
    """A minimal lap_features-like frame for `_track_evolution` (TEI)."""
    rows = []
    for lap in range(1, 6):
        for driver in ("VER", "HAM"):
            rows.append({
                "season": "2025", "gp": "Test GP", "session": "Race",
                "lap": lap, "laptime": 95.0 + lap * 0.05,
                "compound": "SOFT", "rain_flag": 0,
            })
    return pl.DataFrame(rows)
