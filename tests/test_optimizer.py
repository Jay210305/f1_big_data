"""Unit tests for the optimizer's pure simulation helpers (no model artifacts)."""
import numpy as np
import polars as pl
import pytest

from src.models import optimizer as O


def test_stint_time_zero_laps():
    assert O.stint_time(90.0, 0.0, 0.1, 0, 10.0, 1.5, 0.15) == 0.0


def test_stint_time_basic_deg():
    # L=3, beta=0.1: deg = 0.1 * 3*4/2 = 0.6
    total = O.stint_time(90.0, 0.0, 0.1, 3, 10.0, 1.5, 0.15)
    assert abs(total - (3 * 90.0 + 0.6)) < 1e-9


def test_sample_beta_within_bounds():
    rng = np.random.default_rng(0)
    beta = O._sample_beta(rng, 0.1, 0.02, 1000)
    assert beta.shape == (1000,)
    assert beta.min() >= 0.005
    assert beta.max() <= min(0.1 * 4.0 + 0.15, 1.0) + 1e-9


def test_sample_cliff_shape_and_censor():
    rng = np.random.default_rng(0)
    cfg = {"censor_prob": 0.3, "quantiles": [5.0, 10.0, 15.0]}
    out = O._sample_cliff(rng, cfg, 1000)
    assert out.shape == (1000,)
    # censored samples are inf, non-censored are drawn from quantiles
    assert (out[~np.isinf(out)].min()) >= 0.0


def test_enumerate_strategies_respects_two_compound_rule():
    """A dry race must be run on at least two compounds (F1 rule).

    The zero-stop candidate is therefore only legal when the compounds used
    so far already satisfy the rule; otherwise the candidate set contains a
    two-compound plan only.
    """
    state = {"cur_lap": 10, "n_laps": 57, "compound": "MEDIUM",
             "compounds_used": {"MEDIUM"}}
    strats = O.enumerate_strategies(state)
    assert not any(len(s) == 1 and s[0][2] is True for s in strats)
    for s in strats:
        # every candidate ends up on at least two compounds in total
        assert len(state["compounds_used"] | {c for c, _, _ in s}) >= 2
        # every non-current segment ends before or at race end
        for comp, end_lap, cur in s:
            assert 0 < end_lap <= state["n_laps"]

    # with two compounds already used, staying on the current one is legal
    state2 = dict(state, compounds_used={"MEDIUM", "SOFT"})
    strats2 = O.enumerate_strategies(state2)
    assert any(len(s) == 1 and s[0][2] is True for s in strats2)


def test_enumerate_strategies_excludes_wet_tyres_in_dry_context():
    state = {"cur_lap": 10, "n_laps": 57, "compound": "MEDIUM",
             "compounds_used": {"MEDIUM"}}
    assert all(c in O.DRY for s in O.enumerate_strategies(state)
               for c, _, _ in s)
    wet_state = dict(state, wet=True)
    assert any(c in ("INTERMEDIATE", "WET") for s in O.enumerate_strategies(wet_state)
               for c, _, _ in s)


def test_sample_sc_windows_uses_conditional_hazard_and_seed():
    cfg = {"lap_hazard": {"3": 1.0}, "duration_laps": [2]}
    starts, durations = O._sample_sc_windows(
        np.random.default_rng(4), cfg, 8, 10)
    assert np.all(starts == 3)
    assert np.all(durations == 2)


def test_load_scenarios_selects_circuit_hazard():
    params = {"sc_hazard": {
        "lap_hazard": {"Test GP": {"3": 1.0}},
        "duration_laps": [2],
    }}
    scenario = O.load_scenarios(params, include_sc=True,
                                circuit="Test GP")["sc_hazard"]
    assert scenario["sc_hazard"]["lap_hazard"] == {"3": 1.0}


def test_branch_point_summary_selects_green_and_sc_plans():
    results = {
        "clean_air": [{"strategy": "MEDIUM->20 | HARD->50", "mean": 100.0,
                       "p5": 95.0, "p95": 105.0}],
        "sc_hazard": [{"strategy": "MEDIUM->25 | HARD->50", "mean": 94.0,
                       "p5": 90.0, "p95": 99.0}],
    }
    branch = O.branch_point_summary(results)
    assert branch["green_flag"]["strategy"] == "MEDIUM->20 | HARD->50"
    assert branch["sc_window"]["strategy"] == "MEDIUM->25 | HARD->50"
    assert branch["sc_minus_green_mean"] == -6.0


def test_simulate_discounts_pit_loss_inside_sc_window():
    """The discounted pit loss applies to a stop that falls INSIDE the SC window.

    The pit happens at the END of the stint that precedes it, so for the
    strategy [(MEDIUM, 5, True), (SOFT, 10, False)] the stop is at lap 5 —
    an SC at lap 10 happens after the pit and must NOT discount it.
    """
    state = {"cur_lap": 5, "n_laps": 15, "compound": "MEDIUM",
             "cluster": -1, "temp": 30.0, "laps_into_stint": 1,
             "base_pace": 90.0, "in_cliff_now": False}
    params = {
        "beta": {"SOFT": 0.1, "MEDIUM": 0.1, "HARD": 0.1},
        "sigma": {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0},
        "cluster_offset": {}, "temp_coef": 0.0, "temp_mean": 30.0,
        "pace_offset": {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0},
        "cliff": {c: {"censor_prob": 1.0, "quantiles": [10.0]}
                  for c in O.DRY},
    }
    strategy = [("MEDIUM", 5, True), ("SOFT", 10, False)]
    scen = {"beta_mult": 1.0, "pace_pen": 0.0, "pit_loss_sc": 7.0}
    green = O.simulate(strategy, state, params, scen, 4,
                       np.random.default_rng(1))
    # SC covering the pit lap (5) -> discounted pit loss
    on_pit = O.simulate(strategy, state, params, scen, 4,
                        np.random.default_rng(1),
                        sc_windows=(np.full(4, 5.0), np.full(4, 1.0)))
    assert np.allclose(green - on_pit, O.PIT_LOSS - 7.0)
    # SC after the pit (lap 10) -> the stop already happened, no discount
    after_pit = O.simulate(strategy, state, params, scen, 4,
                           np.random.default_rng(1),
                           sc_windows=(np.full(4, 10.0), np.full(4, 1.0)))
    assert np.allclose(green, after_pit)


def test_subsequent_stint_does_not_inherit_the_previous_tyres_degradation():
    """A new stint must start from the car's FRESH pace.

    The previous tyre's degradation must not be charged twice: once inside
    the inherited base pace and again by the degradation term. Calibrating
    the simulator against the realized lap times of the executed strategies
    showed this error as a median per-stint over-prediction of ~40 s.
    """
    params = {
        "beta": {"SOFT": 0.10, "MEDIUM": 0.10, "HARD": 0.10},
        "sigma": {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0},
        "cluster_offset": {}, "temp_coef": 0.0, "temp_mean": 30.0,
        "pace_offset": {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.5},
        # censor_prob = 1.0 -> the cliff never fires, so the simulated time
        # is deterministic
        "cliff": {c: {"censor_prob": 1.0, "quantiles": [20.0]} for c in O.COMPOUNDS},
    }
    state = {"cur_lap": 5, "n_laps": 40, "compound": "MEDIUM",
             "compounds_used": ["MEDIUM"], "stint": 1, "laps_into_stint": 11,
             "cluster": -1, "temp": 30.0, "base_pace": 92.0, "in_cliff_now": False}
    strategy = [("MEDIUM", 5, True), ("HARD", 40, False)]
    total = float(O.simulate(strategy, state, params,
                             O.SCENARIOS["clean_air"], 1,
                             np.random.default_rng(0))[0])

    k0 = state["laps_into_stint"] - 1          # laps already run on the old tyre
    fresh = state["base_pace"] - params["beta"]["MEDIUM"] * k0
    length = 40 - 5
    expected = (length * (fresh + 0.5 - 0.0)
                + params["beta"]["HARD"] * length * (length + 1) / 2.0
                + O.PIT_LOSS + O.NEW_STINT_WARMUP)   # one stop
    assert total == pytest.approx(expected, abs=1e-6)
    # the naive (buggy) reference charges the old tyre's degradation again
    naive = (length * (state["base_pace"] + 0.5)
             + params["beta"]["HARD"] * length * (length + 1) / 2.0
             + O.PIT_LOSS + O.NEW_STINT_WARMUP)
    assert naive - total == pytest.approx(length * params["beta"]["MEDIUM"] * k0, abs=1e-6)


def test_get_state_uses_a_flying_lap_not_the_pit_entry_lap():
    """The decision lap is often the pit-ENTRY lap, which runs seconds slow.

    Using it as the base pace inflates the whole evaluation by the in-lap
    penalty times the number of remaining laps.
    """
    rows = []
    for stint, laps in ((1, range(1, 11)), (2, range(11, 21))):
        for lap in laps:
            rows.append(dict(season="2025", gp="G", session="Race", driver="D",
                              stint=stint, life=lap - 1, lap=lap,
                              compound="SOFT", fuel_corr=90.0 if lap < 10 else 110.0,
                              pace_above_best=0.0 if lap < 10 else 20.0,
                              track_temp_mean=30.0, cluster=-1, rain_flag=0))
    frame = pl.DataFrame(rows)
    state = O.get_state(frame, "2025", "G", "Race", "D", 10)
    # lap 10 is the pit-entry lap: its pace (110 s) and pace_above_best (20 s)
    # must not be used as the base
    assert state["base_pace"] == pytest.approx(90.0)
    assert state["in_cliff_now"] is False


def test_parse_strategy_resolves_end_and_marks_current():
    state = {"cur_lap": 0, "n_laps": 20, "compound": "SOFT"}
    strategy = O.parse_strategy("soft->8,medium->end", state)
    assert strategy == [("SOFT", 8, True), ("MEDIUM", 20, False)]


def test_parse_strategy_rejects_wrong_current_compound():
    state = {"cur_lap": 0, "n_laps": 20, "compound": "SOFT"}
    try:
        O.parse_strategy("MEDIUM->8,HARD->end", state)
    except ValueError as exc:
        assert "current compound" in str(exc)
    else:
        raise AssertionError("expected current-compound validation")


def test_prerace_state_uses_longest_fp2_stints(monkeypatch, tmp_path):
    practice = pl.DataFrame({
        "season": ["2025"] * 7, "gp": ["Test GP"] * 7,
        "session": ["Practice 2"] * 7, "driver": ["TST"] * 7,
        "stint": [1, 1, 1, 2, 2, 2, 2], "compound": ["SOFT"] * 3 + ["MEDIUM"] * 4,
        "laptime": [91.0, 90.0, 92.0, 92.0, 93.0, 91.0, 94.0],
        "track_temp_mean": [30.0] * 7, "lap": list(range(1, 8)),
    })
    race = practice.select(["season", "gp", "lap"]).with_columns(
        pl.lit("Race").alias("session"), pl.lit("TST").alias("driver"))
    monkeypatch.setattr("src.models.rul_data.load_lap_features",
                        lambda: pl.concat([practice, race], how="diagonal_relaxed"))
    state = O.get_prerace_state("2025", "Test GP", "TST", "MEDIUM")
    assert state["cur_lap"] == 0
    assert state["n_laps"] == 7
    assert state["base_pace"] == 92.5
    assert state["pace_source"].startswith("Practice 2")
