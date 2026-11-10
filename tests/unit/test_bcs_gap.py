import pytest

from hydrochoerus.core.bcs_gap import bcs_reduced_gap


def test_should_be_zero_at_tc():
    T_reduced = 1.0
    expected_gap = 0.0
    assert bcs_reduced_gap(T_reduced) == expected_gap


def test_should_be_zero_above_tc():
    T_reduced = 1.2
    expected_gap = 0.0
    assert bcs_reduced_gap(T_reduced) == expected_gap


def test_should_be_one_low_temperature():
    T_reduced = 0.1
    expected = 1.0
    assert bcs_reduced_gap(T_reduced) == pytest.approx(expected, abs=1e-4)
