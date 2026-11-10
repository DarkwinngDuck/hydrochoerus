import math

import pytest

from hydrochoerus.core.specific_heat import ces_integrand


def test_should_return_correct_value():
    eps_reduced, gap_reduced, T_reduced, alpha = 3.0, 4.0, 5.0, 1.0
    gap_squared_derivative = -2.0
    f = 1.0 / (math.e + 1.0)
    expected = f * (1.0 - f) * 6.0
    assert ces_integrand(
        eps_reduced, gap_reduced, gap_squared_derivative, T_reduced, alpha
    ) == pytest.approx(expected, rel=1e-15)


def test_should_reduce_to_energy_term_without_slope():
    eps_reduced, gap_reduced, T_reduced, alpha = 3.0, 4.0, 5.0, 1.0
    gap_squared_derivative = 0.0
    f = 1.0 / (math.e + 1.0)
    expected = f * (1.0 - f) * 5.0
    assert ces_integrand(
        eps_reduced, gap_reduced, gap_squared_derivative, T_reduced, alpha
    ) == pytest.approx(expected, rel=1e-15)
