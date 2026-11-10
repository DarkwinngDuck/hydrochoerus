import math

import pytest

from hydrochoerus.consts import ALPHA_BCS
from hydrochoerus.core.bcs_gap import (
    REDUCED_ENERGY_CUTOFF,
    gap_squared_derivative_numerator_integrand,
    integrate_over_energy,
)


def test_should_integrate_constant_over_whole_cutoff_range():
    gap_reduced = 0.5
    T_reduced = 0.5
    expected = REDUCED_ENERGY_CUTOFF

    def mock_integrand(eps_reduced, gap_reduced, T_reduced):
        return 1.0

    assert integrate_over_energy(mock_integrand, gap_reduced, T_reduced) == pytest.approx(expected)


@pytest.mark.parametrize("T_reduced", [0.5, 0.05, 0.005])
def test_should_find_narrow_peak_near_zero_energy(T_reduced):
    gap_reduced = 0.5
    expected = 1.0

    def mock_integrand(eps_reduced, gap_reduced, T_reduced):
        return math.exp(-eps_reduced / T_reduced) / T_reduced

    assert integrate_over_energy(mock_integrand, gap_reduced, T_reduced) == pytest.approx(
        expected, rel=1e-8
    )


@pytest.mark.parametrize("T_reduced", [0.1, 0.5, 0.9])
def test_should_give_closed_form_for_sech_squared_without_gap(T_reduced):
    gap_reduced = 1e-12
    expected = 2.0 * T_reduced / ALPHA_BCS
    result = integrate_over_energy(
        gap_squared_derivative_numerator_integrand, gap_reduced, T_reduced
    )
    assert result == pytest.approx(expected, rel=1e-6)
