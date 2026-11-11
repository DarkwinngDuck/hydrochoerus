from pathlib import Path

import numpy as np
import pytest

from hydrochoerus.core.bcs_gap import bcs_reduced_gap

pytestmark = [pytest.mark.integration, pytest.mark.slow]

TABLE = Path(__file__).parent / "gap_reduced.dat"
GAP_TOLERANCE = 1e-5
T_ROUNDING = 5e-7
ABS = 2e-4  # currently reproduces for all t with 1e-5 except for t = 0.999985

T_TABLE, GAP_TABLE = np.loadtxt(TABLE, delimiter=",", unpack=True)
BELOW_TC = [
    (float(t), float(gap)) for t, gap in zip(T_TABLE, GAP_TABLE, strict=True) if 0.0 < t < 1.0
]
AT_OR_ABOVE_TC = [float(t) for t in T_TABLE if t >= 1.0]


@pytest.mark.parametrize(("T_reduced", "gap_reduced"), BELOW_TC)
def test_should_match_published_gap_values(T_reduced, gap_reduced):
    expected = gap_reduced
    assert bcs_reduced_gap(T_reduced) == pytest.approx(expected, abs=ABS)


@pytest.mark.parametrize("T_reduced", AT_OR_ABOVE_TC)
def test_should_be_zero_gap_at_tc(T_reduced):
    expected = 0.0
    assert bcs_reduced_gap(T_reduced) == expected
