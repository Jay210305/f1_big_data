"""Seeded model/simulation audit, with strict xfails for reproduced defects.

Passing samples provide bounded empirical evidence, not universal proofs.
The xfails state desired invariants and must fail until the underlying defect
is fixed; they deliberately do not bless today's counterexamples as behavior.
"""
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.live.state_updater import new_state
from src.models import fit_degradation as FD
from src.models import optimizer as O
from src.models import rul_data as RD
from src.models.style_data import _track_evolution
from src.models.train_rul import _asym_grad_hess, nasa_phm_score, reg_metrics


def state():
    return dict(season="2025", gp="Audit GP", session="Race", driver="TST",
                cur_lap=5, n_laps=20, compound="MEDIUM", compounds_used=["MEDIUM"],
                stint=1, laps_into_stint=4, cluster=-1, temp=30.0,
                base_pace=90.0, in_cliff_now=False)


def params():
    return dict(beta={c: 0.1 for c in O.COMPOUNDS},
                sigma={c: 0.0 for c in O.COMPOUNDS}, cluster_offset={},
                temp_coef=0.0, temp_mean=30.0, pace_offset={},
                cliff={c: dict(censor_prob=1.0, quantiles=[20.0])
                       for c in O.COMPOUNDS}, cliff_jump=1.5, cliff_slope=0.15)


def lap_frame(sc=(0, 0, 0, 0, 0, 0)):
    return pl.DataFrame([
        dict(season="2025", gp="Audit GP", session="Race", driver="TST",
             stint=1, life=i, lap=i + 1, compound="SOFT", rain_flag=0,
             fuel_corr=90.0 + i * 0.2, s1=30.0 + i * 0.1,
             s2=30.0 + i * 0.1, s3=30.0 + i * 0.1, is_sc_lap=flag)
        for i, flag in enumerate(sc)
    ])


def test_seeded_stint_closed_form_matches_integer_lap_sum():
    rng = np.random.default_rng(20261021)
    for _ in range(200):
        length = int(rng.integers(0, 81))
        onset = int(rng.integers(-2, 85))
        base, pen, beta, jump, slope = rng.uniform(0, 2, 5)
        expected = sum(base + pen + beta * k
                       + (jump + slope * (k - max(onset, 0))
                          if k > max(onset, 0) else 0)
                       for k in range(1, length + 1))
        actual = O.stint_time(base, pen, beta, length, onset, jump, slope)
        assert actual == pytest.approx(expected)


def test_seeded_asymmetric_objective_is_derivative_of_weighted_square():
    rng = np.random.default_rng(17)
    truth = rng.normal(size=200)
    prediction = truth + rng.choice([-1, 1], 200) * rng.uniform(0.1, 4, 200)
    residual = prediction - truth
    weight = np.where(residual > 0, 2.5, 1.0)
    grad, hess = _asym_grad_hess(truth, prediction)
    eps = 1e-5
    loss = lambda p: weight * (p - truth) ** 2
    np.testing.assert_allclose(grad, (loss(prediction + eps) - loss(prediction - eps)) / (2 * eps))
    np.testing.assert_allclose(hess, 2 * weight)
    assert nasa_phm_score([0.0], [2.0]) > nasa_phm_score([2.0], [0.0])


def test_regression_metrics_match_independent_nonconstant_reference():
    truth = np.random.default_rng(5).uniform(0, 30, 200)
    prediction = truth + np.random.default_rng(6).normal(size=200)
    result = reg_metrics(truth, prediction)
    assert result["MAE"] == pytest.approx(np.abs(truth - prediction).mean())
    assert result["R2"] == pytest.approx(1 - np.sum((truth - prediction) ** 2)
                                        / np.sum((truth - truth.mean()) ** 2))
    assert result["n_crit"] + result["n_noncrit"] == result["n"] == len(truth)


def test_green_rolling_features_are_prefix_invariant():
    full = lap_frame()
    expected = RD.compute_rolling_features(full)
    for end in range(1, full.height + 1):
        prefix = RD.compute_rolling_features(full.head(end))
        for col in set(expected.columns) - set(full.columns):
            np.testing.assert_allclose(prefix[col].to_numpy(), expected[col].head(end).to_numpy(),
                                       equal_nan=True)


def test_seeded_labels_are_bounded_and_monotone_within_stint():
    rng = np.random.default_rng(902)
    for compound in [*O.COMPOUNDS, "UNKNOWN"]:
        for _ in range(10):
            frame = lap_frame((0,) * 40).with_columns(
                pl.lit(compound).alias("compound"),
                pl.Series("fuel_corr", 90 + np.cumsum(rng.normal(0.05, 0.4, 40))))
            labeled = RD.compute_labels(RD.compute_rolling_features(frame))
            rul = labeled["target_rul"].to_numpy()
            assert np.all((rul >= 0) & (rul <= RD.RUL_MAX.get(compound, RD.DEFAULT_RUL_MAX)))
            assert np.all(np.isin(np.diff(rul), [-1, 0]))
            assert np.all(np.diff(labeled["cliff_within_3"].cast(pl.Int32).to_numpy()) >= 0)


def test_train_only_sequence_scaling_flattening_and_inference_encoding_agree():
    frame = pl.DataFrame([
        dict(season=year, gp="Audit GP", session="Race", session_type="Race",
             driver="TST", compound="SOFT", stint=1, life=life, lap=life + 1,
             signal=value, target_rul=float(3 - life), cliff_within_3=life >= 1)
        for year, values in [("2021", [10.0, 20.0, 30.0]),
                             ("2024", [1e8, 2e8, 3e8]), ("2025", [4e8, 5e8, 6e8])]
        for life, value in enumerate(values)
    ])
    tab, feats, codes = RD.make_tabular(frame)
    sequence, _, (means, stds) = RD.make_sequences(frame, feats, codes)
    assert means["signal"] == 20.0
    assert stds["signal"] == 10.0
    np.testing.assert_array_equal(np.concatenate(sequence["test"]["y_rul"]), tab["test"][1])
    meta = dict(features=feats, codes=codes, norm_means=means, norm_stds=stds)
    inferred = O.encode_sequence(frame.filter(pl.col("season") == "2025"), meta)
    np.testing.assert_array_equal(inferred[0], sequence["test"]["X"][0])


def test_identical_strategy_paired_comparison_is_exactly_zero():
    cfg = params()
    cfg["sigma"] = {c: 0.05 for c in O.COMPOUNDS}
    strategy = [("MEDIUM", 10, True), ("HARD", 20, False)]
    result = O.compare_strategies(strategy, strategy, state(), cfg,
                                  O.SCENARIOS["clean_air"], 128)
    np.testing.assert_array_equal(result["delta"], np.zeros(128))


def test_seeded_candidate_structure_and_analytic_simulation_finiteness():
    rng = np.random.default_rng(10)
    for _ in range(20):
        current = int(rng.integers(0, 60))
        sample = {**state(), "cur_lap": current, "n_laps": current + int(rng.integers(1, 50))}
        strategies = O.enumerate_strategies(sample)
        assert 1 <= len(strategies) <= 400
        for strategy in strategies:
            ends = [sample["cur_lap"]] + [segment[1] for segment in strategy]
            assert all(b > a for a, b in zip(ends, ends[1:]))
            assert ends[-1] == sample["n_laps"]
            totals = O.simulate(strategy, sample, params(), O.SCENARIOS["clean_air"],
                                2, rng)
            assert np.isfinite(totals).all()


def test_initial_sc_features_do_not_depend_on_future_green_pace():
    frame = lap_frame((1, 1, 0, 0, 0, 0))
    changed = frame.with_columns(pl.when(pl.col("life") >= 2)
                                  .then(pl.col("fuel_corr") + 50)
                                  .otherwise(pl.col("fuel_corr")).alias("fuel_corr"))
    before = RD.compute_rolling_features(frame)["pace_roll_mean_5"][0]
    after = RD.compute_rolling_features(changed)["pace_roll_mean_5"][0]
    assert before == after


def test_sc_discount_applies_at_actual_pit_boundary():
    strategy = [("MEDIUM", 10, True), ("HARD", 20, False)]
    scen = {**O.SCENARIOS["clean_air"], "pit_loss_sc": 7.0}
    green = O.simulate(strategy, state(), params(), scen, 4, np.random.default_rng(1))
    sc = O.simulate(strategy, state(), params(), scen, 4, np.random.default_rng(1),
                    sc_windows=(np.full(4, 10.0), np.ones(4)))
    np.testing.assert_allclose(green - sc, np.full(4, O.PIT_LOSS - 7.0))


def test_sc_only_at_race_finish_does_not_discount_earlier_stop():
    strategy = [("MEDIUM", 10, True), ("HARD", 20, False)]
    scen = {**O.SCENARIOS["clean_air"], "pit_loss_sc": 7.0}
    green = O.simulate(strategy, state(), params(), scen, 4, np.random.default_rng(1))
    sc = O.simulate(strategy, state(), params(), scen, 4, np.random.default_rng(1),
                    sc_windows=(np.full(4, 20.0), np.ones(4)))
    np.testing.assert_allclose(green, sc)


def test_dry_candidates_use_two_compounds_when_only_one_used():
    for strategy in O.enumerate_strategies(state()):
        assert len(set(state()["compounds_used"]) | {c for c, _, _ in strategy}) >= 2


def test_dry_context_candidates_do_not_include_wet_tyres():
    assert all(c in O.DRY for strategy in O.enumerate_strategies(state())
               for c, _, _ in strategy)


def test_hazard_includes_event_race_in_event_lap_exposure(monkeypatch):
    sc = pl.DataFrame(dict(season=["2024"], gp=["Audit GP"], session=["Race"], lap=[3]))
    laps = pl.DataFrame(dict(season=["2024"], gp=["Audit GP"], session=["Race"], lap=[10]))
    monkeypatch.setattr(RD, "load_sc_laps", lambda: sc)
    monkeypatch.setattr(FD, "load_lap_features", lambda: laps)
    result = FD._fit_sc_hazard()
    assert result["lap_hazard"]["Audit GP"].get("3", 0) == 1.0


def test_fitted_hazard_is_bounded_probability(monkeypatch):
    sc = pl.DataFrame(dict(season=["2022", "2023"], gp=["Audit GP"] * 2,
                           session=["Race"] * 2, lap=[3, 3]))
    laps = pl.DataFrame(dict(season=["2022", "2023", "2024"], gp=["Audit GP"] * 3,
                             session=["Race"] * 3, lap=[10] * 3))
    monkeypatch.setattr(RD, "load_sc_laps", lambda: sc)
    monkeypatch.setattr(FD, "load_lap_features", lambda: laps)
    result = FD._fit_sc_hazard()
    assert all(0 <= value <= 1 for value in result["lap_hazard"]["Audit GP"].values())


def test_early_track_evolution_does_not_change_when_future_rain_changes():
    previous = dict(season="2025", gp="Audit GP", session="Practice 1", lap=1,
                    laptime=90.0, compound="SOFT", rain_flag=0)
    race1 = {**previous, "session": "Race", "lap": 1}
    race2 = {**previous, "session": "Race", "lap": 2}
    frame = pl.DataFrame([previous, race1, race2])
    wet_future = pl.DataFrame([previous, race1, {**race2, "rain_flag": 1}])
    pick = lambda f: _track_evolution(f).filter((pl.col("session") == "Race")
                                               & (pl.col("lap") == 1))["track_evolution_index"][0]
    assert pick(frame) == pick(wet_future)


def test_same_compound_new_stint_resets_live_state():
    live = new_state("2025", "Audit GP", "TST", 20)
    live.update_lap(dict(lap=5, laptime=93.0, compound="MEDIUM", stint=1))
    result = live.update_lap(dict(lap=6, laptime=90.0, compound="MEDIUM", stint=2))
    assert result["stint"] == 2
    assert result["laps_into_stint"] == 1


def test_live_batch_current_stint_age_parity():
    frame = lap_frame().with_columns(pl.col("fuel_corr").alias("laptime"),
                                    pl.lit(0.0).alias("pace_above_best"),
                                    pl.lit(30.0).alias("track_temp_mean"),
                                    pl.lit(-1).alias("cluster"))
    live = new_state("2025", "Audit GP", "TST", 6, "SOFT")
    for row in frame.iter_rows(named=True):
        live.update_lap(row)
    batch = O.get_state(frame, "2025", "Audit GP", "Race", "TST", 6)
    assert live.as_optimizer_state()["laps_into_stint"] == batch["laps_into_stint"]


def test_live_temperature_matches_batch_current_lap():
    live = new_state("2025", "Audit GP", "TST", 20)
    live.update_lap(dict(lap=1, laptime=90.0, compound="MEDIUM", track_temp=30.0))
    result = live.update_lap(dict(lap=2, laptime=90.0, compound="MEDIUM", track_temp=40.0))
    assert result["temp"] == 40.0


def test_saved_degradation_model_covers_all_runtime_compounds():
    model_path = Path(__file__).resolve().parents[1] / "data/models/degradation_model.json"
    if not model_path.exists():
        pytest.skip("saved model not present")
    saved = json.loads(model_path.read_text())
    for name in ("beta", "sigma", "cliff", "pace_offset"):
        assert set(O.COMPOUNDS) <= set(saved[name])


def test_saved_realized_report_records_real_final_compounds():
    report_path = Path(__file__).resolve().parents[1] / "data/models/phase7_backtest_realized.json"
    if not report_path.exists():
        pytest.skip("saved report not present")
    report = json.loads(report_path.read_text())
    assert all(row["rec_final_comp"] in O.COMPOUNDS for row in report["decisions"])


def test_beta_sampler_retains_physical_lower_bound_for_negative_adjusted_mean():
    beta = O._sample_beta(np.random.default_rng(3), -0.1, 0.02, 128)
    assert np.all(beta >= 0.005)
