"""Unit tests for track-evolution-index (TEI) with rain wash-out + wet factors."""

import polars as pl

from src.models import style_data as SD


def test_track_evolution_monotonic_within_session():
    df = pl.DataFrame({
        "season": ["2025"] * 4, "gp": ["T"] * 4, "session": ["Race"] * 4,
        "lap": [1, 2, 3, 4], "laptime": [95.0] * 4,
        "compound": ["SOFT"] * 4, "rain_flag": [0] * 4,
    })
    tei = SD._track_evolution(df)
    vals = tei.sort("lap")["track_evolution_index"].to_list()
    # TEI accumulates with laps (rubber laid down -> non-decreasing)
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert vals[-1] > vals[0]


def test_track_evolution_wet_session_reset():
    """A wet session washes 70% of prior rubber before adding its own laps."""
    df = pl.DataFrame({
        "season": ["2025"] * 4, "gp": ["T"] * 4,
        "session": ["Practice 1", "Practice 1", "Practice 2", "Practice 2"],
        "lap": [1, 2, 1, 2], "laptime": [95.0] * 4,
        "compound": ["SOFT", "SOFT", "INTERMEDIATE", "INTERMEDIATE"],
        "rain_flag": [0, 0, 1, 1],
    })
    tei = SD._track_evolution(df)
    # second session (wet) is present and r_prior_rubber is washed
    p2 = tei.filter(pl.col("session") == "Practice 2")["r_prior_rubber"]
    p1 = tei.filter(pl.col("session") == "Practice 1")["track_evolution_index"].max()
    # wash: prior rubber multiplied by GAMMA_WASH, so P2 prior < P1 total
    assert p2[0] < p1


def test_instant_wet_halves_grip():
    """Wet tyres apply INSTANT_WET to the accumulated lap-level TEI base."""
    df = pl.DataFrame({
        "season": ["2025"] * 2, "gp": ["T"] * 2, "session": ["Race"] * 2,
        "lap": [1, 2], "laptime": [95.0, 95.0],
        "compound": ["SOFT", "INTERMEDIATE"], "rain_flag": [0, 0],
    })
    tei = SD._track_evolution(df)
    dry, wet = tei.sort("lap")["track_evolution_index"].to_list()
    expected_wet_base = 2 * SD.SESS_WEIGHT["Race"]
    assert wet == expected_wet_base * SD.INSTANT_WET
    assert wet < expected_wet_base
    assert SD.INSTANT_WET < 1.0
