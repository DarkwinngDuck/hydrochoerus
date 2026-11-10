import pytest

from hydrochoerus.core.bcs_gap import bcs_reduced_gap

# TODO: reference paper
BCS_TABLE = [
    (0.25, 0.9992),
    (0.5, 0.9569),
    (0.8, 0.7111),
    (0.9, 0.5265),
    (0.999, 0.0568),
]


@pytest.mark.parametrize(("T_reduced", "expected_gap"), BCS_TABLE)
def test_should_match_bcs_table(T_reduced, expected_gap):
    assert bcs_reduced_gap(T_reduced) == pytest.approx(expected_gap, abs=1e-4)


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
