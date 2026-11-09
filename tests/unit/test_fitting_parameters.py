import pytest

from hydrochoerus.consts import ALPHA_BCS
from hydrochoerus.model.fitting_parameter import FittingParameter

alpha_bcs = FittingParameter(
    "alpha_bcs",
    "2 delta /k_B T_c",
    default_guess=ALPHA_BCS,
    bounds=(0.01, 5.0),
    scale_ratio=2,
)
two_gap_weight_ratio = FittingParameter(
    "weight ration two gap model", "x", default_guess=0.5, bounds=(0.0, 1.0)
)
anisotropy = FittingParameter("anisotropy", "epsilon", default_guess=0.5, bounds=(-10.0, 0.99))


def test_gap_ratio_scaled_correctly():
    assert alpha_bcs.scale(ALPHA_BCS) == pytest.approx(ALPHA_BCS * 2)


def test_gap_ratio_converts_correctly():
    assert alpha_bcs.from_scaled(alpha_bcs.scale(ALPHA_BCS)) == pytest.approx(ALPHA_BCS)


@pytest.mark.parametrize("parameter", [two_gap_weight_ratio, anisotropy])
def test_shape_parameters_are_not_scalled(parameter):
    test_value = 0.2
    assert parameter.scale(test_value) == test_value


@pytest.mark.parametrize("parameter", [two_gap_weight_ratio, anisotropy])
def test_every_gap_ratio_is_scaled_correctly(parameter):
    assert parameter.scale_ratio == 1.0


def test_scaled_bounds_are_correct():
    assert alpha_bcs.scale_bounds() == pytest.approx((0.02, 10.0))
    assert two_gap_weight_ratio.scale_bounds() == (0.0, 1.0)


def test_default_guess_outside_bounds_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        FittingParameter("test-parameter-name", "test-parameter-symbol", 5.0, bounds=(0.0, 1.0))


def test_inverted_bounds_are_rejected():
    with pytest.raises(ValueError, match="bound"):
        FittingParameter("test-parameter-name", "test-parameter-symbol", 1.0, bounds=(1.0, 0.0))


def test_scale_ratio_is_frozen():
    with pytest.raises(AttributeError):
        alpha_bcs.scale_ratio = 1.0
