import pytest

from hydrochoerus.consts import ALPHA_BCS
from hydrochoerus.core.specific_heat import ces


@pytest.mark.parametrize("T_reduced", [0.3, 0.6, 0.9, 0.99])
def test_should_reduce_to_normal_state_for_tiny_alpha(T_reduced):
    expected = T_reduced
    assert ces(T_reduced, alpha=1e-3) == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize("T_reduced", [1.0, 1.5, 3.0])
def test_should_be_exactly_normal_state_above_tc(T_reduced):
    expected = T_reduced
    assert ces(T_reduced, ALPHA_BCS) == expected


def test_should_jump_above_normal_state_just_below_tc():
    T_reduced = 0.999
    assert ces(T_reduced, ALPHA_BCS) > T_reduced


@pytest.mark.parametrize("T_reduced", [0.05, 0.1, 0.5, 0.99])
def test_should_be_positive(T_reduced):
    assert ces(T_reduced, ALPHA_BCS) > 0.0


def test_should_vanish_at_low_temperature():
    T_reduced = 0.1
    assert ces(T_reduced, ALPHA_BCS) < 1e-4
