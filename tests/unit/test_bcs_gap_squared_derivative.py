import pytest

from hydrochoerus.core.bcs_gap import (
    bcs_reduced_gap,
    bcs_reduced_gap_squared_derivative,
)


@pytest.mark.parametrize("T_reduced", [1.0, 1.2])
def test_should_be_exactly_zero_at_and_above_tc(T_reduced):
    expected = 0.0
    assert bcs_reduced_gap_squared_derivative(T_reduced) == expected


@pytest.mark.parametrize("T_reduced", [0.2, 0.5, 0.8, 0.95])
def test_should_agree_with_finite_difference_of_gap_squared(T_reduced):
    step = 1e-4
    above = bcs_reduced_gap(T_reduced + step) ** 2
    below = bcs_reduced_gap(T_reduced - step) ** 2
    expected = (above - below) / (2 * step)
    assert bcs_reduced_gap_squared_derivative(T_reduced) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("T_reduced", [0.3, 0.6, 0.9, 0.99])
def test_should_be_negative_below_tc(T_reduced):
    assert bcs_reduced_gap_squared_derivative(T_reduced) < 0
