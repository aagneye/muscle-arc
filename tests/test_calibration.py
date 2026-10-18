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
