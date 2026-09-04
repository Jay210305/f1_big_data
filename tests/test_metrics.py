"""Unit tests for RUL/Cliff metrics and the asymmetric objective."""
import numpy as np

from src.models import train_rul as T


def test_nasa_phm_perfect_is_zero():
    y = np.array([5.0, 10.0, 3.0])
    assert T.nasa_phm_score(y, y) == 0.0


def test_nasa_phm_penalizes_late_more():
    y = np.array([5.0])
    early = T.nasa_phm_score(y, np.array([4.0]))   # underestimates RUL
    late = T.nasa_phm_score(y, np.array([6.0]))    # overestimates RUL
    assert late > early > 0


def test_reg_metrics_perfect():
    y = np.array([1.0, 2.0, 3.0])
    m = T.reg_metrics(y, y)
    assert m["MAE"] == 0.0
    assert m["RMSE"] == 0.0
    assert abs(m["R2"] - 1.0) < 1e-9
    assert m["PHM"] == 0.0


def test_reg_metrics_critical_zone():
    y = np.array([1.0, 2.0, 20.0])
    p = np.array([1.0, 3.0, 20.0])
    m = T.reg_metrics(y, p)
    assert m["n_crit"] == 2
    assert abs(m["MAE_crit"] - 0.5) < 1e-9


def test_clf_metrics_perfect():
    y = np.array([1, 1, 0, 0])
    p = np.array([1, 1, 0, 0])
    prob = np.array([0.9, 0.8, 0.2, 0.1])
    m = T.clf_metrics(y, p, prob)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["auc"] == 1.0


def test_best_f1_threshold_in_range():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.1, 0.4, 0.6, 0.9])
    thr, f1, points = T._best_f1_threshold(y, prob)
    assert 0.0 <= thr <= 1.0
    assert 0.0 <= f1 <= 1.0
    assert len(points) > 0


def test_asym_grad_hess_late_heavier():
    y = np.array([0.0, 0.0])
    grad, hess = T._asym_grad_hess(y, np.array([-1.0, 1.0]))
    # residual <= 0 -> C1; residual > 0 -> C2 (C2 > C1)
    assert hess[0] < hess[1]
