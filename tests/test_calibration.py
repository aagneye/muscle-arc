import numpy as np

from muscle_arc.geometry.calibration import apply_calibration, fit_linear_calibration, loo_gain


def test_fit_recovers_known_bias_with_shrink() -> None:
    gt = np.linspace(10, 30, 35)
    pred = gt + 2.0
    gain, offset = fit_linear_calibration(pred, gt, shrink=0.5)
    assert abs(gain - 1.0) < 1e-6
    assert abs(offset - (-1.0)) < 1e-6  # half of the -2 correction
    assert np.allclose(apply_calibration(pred, gain, offset), gt + 1.0)


def test_loo_helps_on_systematic_bias() -> None:
    rng = np.random.default_rng(0)
    gt = rng.uniform(10, 30, 35)
    pred = gt + 2.0 + rng.normal(0, 0.3, 35)
    assert loo_gain(pred, gt, shrink=0.0) < -1.0  # MAE drops by more than 1 unit


def test_loo_no_gain_when_already_calibrated() -> None:
    rng = np.random.default_rng(1)
    gt = rng.uniform(10, 30, 35)
    pred = gt + rng.normal(0, 0.1, 35)
    assert loo_gain(pred, gt, shrink=0.5) > -0.01


def test_pooled_fit_recovers_group_bias_and_pools_small_groups() -> None:
    from muscle_arc.geometry.calibration import GLOBAL, apply_pooled, fit_pooled_calibration

    rng = np.random.default_rng(2)
    gt = rng.uniform(10, 30, 60)
    groups = np.array(["A"] * 40 + ["B"] * 18 + ["C"] * 2)
    pred = gt + np.where(groups == "A", 3.0, -3.0) + rng.normal(0, 0.1, 60)
    params = fit_pooled_calibration(pred, gt, groups, shrink=0.0, pool=5.0)
    # A (n=40): 40/45 of its own -3 correction plus 5/45 of the global one
    assert params["A"][1] < -2.0 and params["B"][1] > 1.5
    assert params["C"] == params[GLOBAL]  # too small to fit: global map
    cal = apply_pooled(pred, groups, params)
    assert np.mean(np.abs(cal - gt)[groups == "A"]) < np.mean(np.abs(pred - gt)[groups == "A"]) / 3
    assert np.allclose(apply_pooled(np.array([20.0]), np.array(["unknown"]), params),
                       params[GLOBAL][0] * 20.0 + params[GLOBAL][1])


def test_loo_pooled_mae_improves_on_group_bias() -> None:
    from muscle_arc.geometry.calibration import loo_pooled_mae

    rng = np.random.default_rng(3)
    gt = rng.uniform(40, 90, 50)
    groups = np.array(["A"] * 25 + ["B"] * 25)
    pred = gt + np.where(groups == "A", 6.0, -6.0) + rng.normal(0, 1.0, 50)
    before, after = loo_pooled_mae(pred, gt, groups, shrink=0.0, pool=5.0)
    assert before > 5.0 and after < 2.0
