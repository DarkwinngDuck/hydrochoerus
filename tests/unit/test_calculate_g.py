import pytest

from hydrochoerus.consts import ALPHA_BCS
from hydrochoerus.core.bcs_gap import calculate_g


def test_should_match_hand_computed_value():
    E_reduced = 2.0
    T_reduced = 0.5
    g_expected = 2.0 * ALPHA_BCS
    assert calculate_g(E_reduced, T_reduced) == pytest.approx(g_expected)


def test_should_be_one_when_energy_is_2kt():
    T_reduced = 0.4
    E_reduced = 2.0 * T_reduced / ALPHA_BCS
    g_expected = 1.0
    assert calculate_g(E_reduced, T_reduced) == pytest.approx(g_expected)


def test_should_be_zero_at_zero_energy():
    E_reduced = 0.0
    T_reduced = 0.5
    g_expected = 0.0
    assert calculate_g(E_reduced, T_reduced) == g_expected


def test_should_double_when_energy_doubles():
    E_reduced = 1.6
    T_reduced = 0.3
    g_expected = 2.0 * calculate_g(E_reduced / 2, T_reduced)
    assert calculate_g(E_reduced, T_reduced) == g_expected


def test_should_be_twice_lower_when_temperature_doubles():
    E_reduced = 1.6
    T_reduced = 0.6
    g_expected = 0.5 * calculate_g(E_reduced, T_reduced / 2)
    assert calculate_g(E_reduced, T_reduced) == g_expected
