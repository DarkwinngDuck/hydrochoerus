import math

import numpy as np
import pytest

from hydrochoerus.core.metrics import metrics, r_factor, r_squared, ss_res, ss_tot

experimental = np.array([1.0, 2.0, 3.0])
fitted = np.array([1.0, 2.0, 4.0])


def test_should_sum_square_differences():
    assert ss_res(experimental, fitted) == pytest.approx(1.0)


def test_should_return_same_results():
    assert ss_res(fitted, experimental) == ss_res(experimental, fitted)


def test_should_return_spread_around_mean():
    assert ss_tot(experimental) == pytest.approx(2.0)


def test_should_return_zero_for_constant():
    assert ss_tot(np.array([2.0, 2.0, 2.0])) == 0.0


def test_should_return_r_squared():
    assert r_squared(1.0, 2.0) == pytest.approx(0.5)


def test_should_return_nan_if_total_negative():
    assert math.isnan(r_squared(1.0, -2.0))


def test_should_return_negative_r_squared():
    assert r_squared(4.0, 2.0) == pytest.approx(-1.0)


def test_should_return_expected_r_factor():
    expected = 1 / 6
    assert r_factor(experimental, fitted) == expected


def test_should_return_nan_if_experimental_sum_is_zero():
    assert math.isnan(r_factor([0, 0, 0], fitted))


def test_should_return_correct_metrics_for_identical_arrays():
    result = metrics(experimental, experimental)
    assert result.residual == 0.0
    assert result.r_squared == 1.0
    assert result.r_factor == 0.0


def test_should_return_correct_metrics():
    result = metrics(experimental, fitted)
    assert result.residual == pytest.approx(1.0)
    assert result.r_squared == pytest.approx(0.5)
    assert result.r_factor == pytest.approx(1 / 6)


def test_should_raise_error_if_empty_arrays():
    with pytest.raises(ValueError, match="empty"):
        metrics(np.array([]), np.array([]))


def test_should_raise_error_if_shape_is_different():
    with pytest.raises(ValueError, match="shapes differ"):
        metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
