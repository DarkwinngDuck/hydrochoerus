import pytest

from hydrochoerus.consts import ALPHA_BCS
from hydrochoerus.core.bcs_gap import (
    gap_squared_derivative_denominator_integrand as integrand,
)


def test_should_be_negative():
    eps_reduced = 0.3
    gap_reduced = 0.9
    T_reduced = 0.5
    assert integrand(eps_reduced, gap_reduced, T_reduced) < 0


@pytest.mark.parametrize(("eps_reduced", "gap_reduced"), [(3.0, 4.0), (0.0, 5.0)])
def test_should_depend_only_on_quasiparticle_energy(eps_reduced, gap_reduced):
    T_reduced = 2.0
    expected = integrand(5.0, 0.0, T_reduced)
    assert integrand(eps_reduced, gap_reduced, T_reduced) == pytest.approx(expected)


def test_should_stay_finite_near_zero_energy():
    eps_reduced = 0.0
    gap_reduced = 1e-3
    T_reduced = 0.5
    expected = -(ALPHA_BCS**2) / (12 * T_reduced)  # Taylor series limit
    assert integrand(eps_reduced, gap_reduced, T_reduced) == pytest.approx(expected, rel=1e-4)


def test_should_reduce_to_tanh_term_far_from_fermi_level():
    eps_reduced = 500.0
    gap_reduced = 0.0
    T_reduced = 0.1
    E_reduced = 500.0
    expected = -(T_reduced**2) / (ALPHA_BCS * E_reduced**3)
    assert integrand(eps_reduced, gap_reduced, T_reduced) == pytest.approx(expected)
