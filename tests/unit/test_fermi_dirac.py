import math

import pytest

from hydrochoerus.consts import ALPHA_BCS, FERMI_ENERGY
from hydrochoerus.core.fermi_dirac import fermi_dirac


@pytest.mark.parametrize("T_reduced", [0.1, 0.5, 0.99])
def test_half_filled_at_fermi_level(T_reduced):
    assert fermi_dirac(FERMI_ENERGY, T_reduced, ALPHA_BCS) == 0.5


def test_overflow_x():
    reduced_energy = 100.0
    T_reduced = 0.1
    alpha = 1.0
    expected_x = alpha * reduced_energy / T_reduced
    expected = math.exp(-expected_x)
    assert fermi_dirac(reduced_energy, T_reduced, alpha) == expected


def test_symmetry():
    reduced_energy = 0.3
    T_reduced = 0.2
    above_fermi = fermi_dirac(reduced_energy, T_reduced, ALPHA_BCS)
    below_fermi = fermi_dirac(-reduced_energy, T_reduced, ALPHA_BCS)
    assert above_fermi + below_fermi == pytest.approx(1.0)
