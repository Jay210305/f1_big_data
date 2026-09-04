"""Unit tests for the optimizer's pure simulation helpers (no model artifacts)."""
import numpy as np

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


def test_enumerate_strategies_includes_zero_stop():
    state = {"cur_lap": 10, "n_laps": 57, "compound": "MEDIUM",
             "compounds_used": {"MEDIUM"}}
    strats = O.enumerate_strategies(state)
    assert any(len(s) == 1 and s[0][2] is True for s in strats)
    # every non-current segment ends before or at race end
    for s in strats:
        for comp, end_lap, cur in s:
            assert 0 < end_lap <= state["n_laps"]
