from hydrochoerus.consts import ALPHA_BCS
from hydrochoerus.core.gap_ratio import (
    alpha_from_gap_ratio,
    gap_ratio_from_alpha,
)


def test_should_return_correct_gap_ratio():
    expected = 2 * ALPHA_BCS
    assert gap_ratio_from_alpha(ALPHA_BCS) == expected


def test_should_return_correct_alpha():
    expected = ALPHA_BCS
    assert alpha_from_gap_ratio(gap_ratio_from_alpha(ALPHA_BCS)) == expected
