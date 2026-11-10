import math

import pytest

from hydrochoerus.consts import ALPHA_BCS
from hydrochoerus.core.bcs_gap import (
    gap_squared_derivative_numerator_integrand as integrand,
)


def test_should_be_one_at_zero_energy_without_gap():
    eps_reduced = 0.0
    gap_reduced = 0.0
    T_reduced = 0.5
    expected = 1.0
    assert integrand(eps_reduced, gap_reduced, T_reduced) == pytest.approx(expected)


def test_should_match_one_over_cosh_squared():
    eps_reduced = 1.0
    gap_reduced = 0.0
    T_reduced = 0.5
    expected = 1.0 / math.cosh(ALPHA_BCS) ** 2
    assert integrand(eps_reduced, gap_reduced, T_reduced) == pytest.approx(expected)


@pytest.mark.parametrize(("eps_reduced", "gap_reduced"), [(3.0, 4.0), (5.0, 0.0), (0.0, 5.0)])
def test_should_depend_only_on_quasiparticle_energy(eps_reduced, gap_reduced):
    T_reduced = 2.0
    expected = integrand(5.0, 0.0, T_reduced)
    assert integrand(eps_reduced, gap_reduced, T_reduced) == pytest.approx(expected)


def test_should_be_zero_far_from_fermi_level():
    eps_reduced = 500.0
    gap_reduced = 1.0
    T_reduced = 0.1
    expected = 0.0
    assert integrand(eps_reduced, gap_reduced, T_reduced) == expected


def test_should_fall_as_energy_rises():
    gap_reduced = 0.5
    T_reduced = 0.5
    eps_reduced_low = 0.2
    eps_reduced_high = 0.8
    low_energy = integrand(eps_reduced_low, gap_reduced, T_reduced)
    high_energy = integrand(eps_reduced_high, gap_reduced, T_reduced)
    assert high_energy < low_energy


def test_should_rise_as_temperature_rises():
    eps_reduced = 0.5
    gap_reduced = 0.5
    T_reduced_low = 0.3
    T_reduced_high = 0.9
    low_t_integrand = integrand(eps_reduced, gap_reduced, T_reduced_low)
    high_t_integrand = integrand(eps_reduced, gap_reduced, T_reduced_high)
    assert low_t_integrand < high_t_integrand
