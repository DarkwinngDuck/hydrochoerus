import math

import pytest

from hydrochoerus.core.bcs_gap import sech_squared


def test_should_be_one_at_zero():
    g = 0.0
    expected = 1.0
    assert sech_squared(g) == expected


def test_should_match_exact_value_at_ln_two():
    g = math.log(2.0)
    expected = 0.64
    assert sech_squared(g) == pytest.approx(expected)


@pytest.mark.parametrize("g", [0.1, 1.3, 5.0])
def test_should_match_one_over_cosh_squared(g):
    expected = 1.0 / math.cosh(g) ** 2
    assert sech_squared(g) == pytest.approx(expected)


def test_should_be_zero_for_huge_g():
    g = 5000.0
    expected = 0.0
    assert sech_squared(g) == expected
