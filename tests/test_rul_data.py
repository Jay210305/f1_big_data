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
