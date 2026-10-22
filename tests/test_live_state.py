from src.live.state_updater import new_state


def test_incremental_state_tracks_pace_temperature_and_slope():
    state = new_state("2025", "Test GP", "TST", 20, "MEDIUM")
    state.update_lap({"lap": 1, "laptime": 90.0, "compound": "MEDIUM",
                      "track_temp": 30.0})
    snapshot = state.update_lap({"lap": 2, "laptime": 90.2, "compound": "MEDIUM",
                                 "track_temp": 32.0})
    assert snapshot["fuel_corr"] == 90.26
    # track_temp_mean mirrors the batch optimizer: the CURRENT lap's track
    # temperature, not a session running average.
    assert snapshot["track_temp_mean"] == 32.0
    assert snapshot["track_temp_session_mean"] == 31.0
    assert snapshot["pace_above_best"] > 0
    assert snapshot["degradation_slope"] > 0
    assert snapshot["cur_lap"] == 2


def test_compound_change_resets_stint_aggregates_without_lakehouse_scan():
    state = new_state("2025", "Wet GP", "TST", 20, "INTERMEDIATE")
    state.update_lap({"lap": 5, "laptime": 100.0, "compound": "INTERMEDIATE"})
    snapshot = state.update_lap({"lap": 6, "laptime": 95.0, "compound": "WET"})
    assert snapshot["compound"] == "WET"
    assert snapshot["stint"] == 2
    assert snapshot["laps_into_stint"] == 1
    assert snapshot["compounds_used"] == ["INTERMEDIATE", "WET"]
