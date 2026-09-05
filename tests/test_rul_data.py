"""Unit tests for RUL data-prep helpers (fuel correction, label edge cases).

No dataset dependency — pure functions exercised on synthetic stint frames.
"""
import polars as pl
import pytest

from src.models.rul_data import (RUL_MAX, add_fuel_corr, compute_labels,
                                 compute_rolling_features, feature_cols)


def _fuel_frame():
    return pl.DataFrame({
        "session_type": ["Race", "Race", "Sprint", "Sprint", "Practice"],
        "lap": [1, 10, 1, 10, 1],
        "laptime": [100.0, 100.0, 100.0, 100.0, 100.0],
    })


def test_add_fuel_corr_race_gets_alpha():
    df = add_fuel_corr(_fuel_frame())
    # Race and Sprint use the documented absolute lap correction.
    assert df["fuel_corr"][0] == 100.03
    assert df["fuel_corr"][1] == 100.30
    # Sprint also corrected
    assert df["fuel_corr"][2] == 100.03
    assert df["fuel_corr"][3] == 100.30


def test_add_fuel_corr_non_race_unchanged():
    df = add_fuel_corr(_fuel_frame())
    # Practice lap 1: uncorrected -> equals laptime
    assert df["fuel_corr"][4] == 100.0


def test_add_fuel_corr_alpha_sign_positive():
    df = add_fuel_corr(_fuel_frame())
    d = df["fuel_corr"][1] - df["fuel_corr"][0]
    assert d > 0 and abs(d - 9 * 0.03) < 1e-6


def _labeled_frame():
    """One stint whose fuel-corrected pace crosses the cliff on the last laps."""
    df = pl.DataFrame({
        "season": ["2025"] * 6, "gp": ["T"] * 6, "session": ["Race"] * 6,
        "driver": ["VER"] * 6, "stint": [1] * 6, "life": list(range(6)),
        "lap": [1, 2, 3, 4, 5, 6], "compound": ["SOFT"] * 6,
        "laptime": [100.1, 100.0, 100.1, 101.8, 102.5, 103.0],
        "s1": [25.0] * 6, "s2": [25.0] * 6, "s3": [25.0] * 6,
        "session_type": ["Race"] * 6,
    })
    df = add_fuel_corr(df)
    df = compute_rolling_features(df)
    return df


def test_compute_labels_cliff_detection():
    df = compute_labels(_labeled_frame())
    # cliffs are detected (pace_roll_med_3 crosses best + CLIFF_THR)
    assert "t_end_life" in df.columns
    assert df["target_rul"].min() >= 0
    # later laps have shorter RUL than earlier laps
    assert df["target_rul"][5] < df["target_rul"][0]


def test_compute_labels_rul_max_clipping():
    df = compute_labels(_labeled_frame())
    # SOFT cap is 15; no label may exceed RUL_MAX[compound]
    assert df["target_rul"].max() <= RUL_MAX["SOFT"]


def test_feature_cols_excludes_labels(monkeypatch=None):
    df = pl.DataFrame({
        "season": ["2025"], "gp": ["T"], "session": ["Race"],
        "lap": [1], "speed_mean": [300.0],
        "target_rul": [5.0], "pace_above_best": [0.0],
    })
    feats = feature_cols(df)
    assert "speed_mean" in feats
    assert "target_rul" not in feats
    assert "lap" not in feats


def test_sc_vsc_detection_patterns():
    """Verify DuckDB SQL pattern logic distinguishes real SC/VSC events from steward notices."""
    import duckdb
    con = duckdb.connect()
    raw = pl.DataFrame({
        "season": [2024] * 9,
        "gp": ["Test GP"] * 9,
        "session": ["Race"] * 9,
        "lap": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "cat": ["SafetyCar", "SafetyCar", "Other", "Flag", "Flag", "Other", "Other", "Flag", "Other"],
        "flag": ["None", "None", "None", "DOUBLE YELLOW", "RED", "None", "None", "CLEAR", "None"],
        "msg": [
            "VIRTUAL SAFETY CAR DEPLOYED",
            "SAFETY CAR IN THIS LAP",
            "SAFETY CAR WILL ENTER PITS: ROLLING START PROCEDURE",
            "DOUBLE YELLOW IN TRACK SECTOR 2",
            "RED FLAG",
            "INCIDENT INVOLVING CAR 27 (HUL) NOTED - VIRTUAL SAFETY CAR INFRINGEMENT",
            "VSC TEST CANCELLED",
            "CLEAR IN TRACK SECTOR 2",
            "FIA STEWARDS: 5 SECOND TIME PENALTY - SAFETY CAR INFRINGEMENT",
        ],
    })
    con.register("test_rcm", raw.to_pandas())
    res = con.execute("""
        SELECT lap FROM test_rcm
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
        ORDER BY lap
    """).df()
    matched_laps = list(res["lap"])
    # laps 1, 2, 3, 4, 5 should match; laps 6 (infringement), 7 (test), 8 (clear), 9 (penalty) must not
    assert matched_laps == [1, 2, 3, 4, 5]


def test_is_sc_lap_in_feature_cols():
    """Option B: is_sc_lap must be retained in X and not excluded."""
    from src.models.rul_data import CLIFF_DERIVED, EXCLUDED
    df = pl.DataFrame({
        "season": ["2025"], "gp": ["T"], "session": ["Race"],
        "lap": [1], "speed_mean": [300.0], "is_sc_lap": [0],
        "target_rul": [5.0], "pace_above_best": [0.0],
    })
    feats = feature_cols(df)
    assert "is_sc_lap" in feats
    assert "is_sc_lap" not in EXCLUDED
    assert "is_sc_lap" not in CLIFF_DERIVED


def test_mid_stint_sc_rolling_features_despiked():
    """Option B: Mid-stint SC laps (20-40s delta) must not spike rolling pace."""
    # Stint with normal pace ~100s, but laps 4-5 are under SC with 140s laptime
    df = pl.DataFrame({
        "season": ["2025"] * 7, "gp": ["T"] * 7, "session": ["Race"] * 7,
        "driver": ["VER"] * 7, "stint": [1] * 7, "life": list(range(1, 8)),
        "lap": [1, 2, 3, 4, 5, 6, 7], "compound": ["MEDIUM"] * 7,
        "laptime": [100.0, 100.2, 100.1, 140.0, 140.0, 100.3, 100.4],
        "s1": [25.0] * 7, "s2": [25.0] * 7, "s3": [25.0] * 7,
        "session_type": ["Race"] * 7,
        "is_sc_lap": [0, 0, 0, 1, 1, 0, 0],
    })
    df = add_fuel_corr(df)
    df = compute_rolling_features(df)
    # On green laps 6 and 7, pace_roll_med_3 must NOT be polluted by the 140s SC laps
    assert df.filter(pl.col("lap") == 6)["pace_roll_med_3"][0] < 105.0
    assert df.filter(pl.col("lap") == 7)["pace_roll_med_3"][0] < 105.0


def test_mid_stint_sc_does_not_trigger_false_cliff():
    """Option B: Slow SC laps must not trigger false cliff crossings."""
    # Healthy tyre where laps 4-6 are SC (140s), then resumes 100s green-flag racing
    df = pl.DataFrame({
        "season": ["2025"] * 8, "gp": ["T"] * 8, "session": ["Race"] * 8,
        "driver": ["VER"] * 8, "stint": [1] * 8, "life": list(range(1, 9)),
        "lap": [1, 2, 3, 4, 5, 6, 7, 8], "compound": ["MEDIUM"] * 8,
        "laptime": [100.0, 100.1, 100.2, 140.0, 140.0, 140.0, 100.3, 100.4],
        "s1": [25.0] * 8, "s2": [25.0] * 8, "s3": [25.0] * 8,
        "session_type": ["Race"] * 8,
        "is_sc_lap": [0, 0, 0, 1, 1, 1, 0, 0],
    })
    df = add_fuel_corr(df)
    df = compute_rolling_features(df)
    df = compute_labels(df)
    # No cliff should be detected on laps 4, 5, 6 (t_end_life is null)
    assert df["t_end_life"][0] is None
    assert not any(df["cliff_within_3"])

