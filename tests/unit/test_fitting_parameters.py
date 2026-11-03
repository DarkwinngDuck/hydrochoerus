import pytest

from hydrochoerus.model import FittingParameter

alpha_bcs_const = 1.764
alpha_bcs = FittingParameter(
    "alpha_bcs", "2 delta /k_B T_c", default_guess=1.764, bounds=(0.01, 5.0), scale=2
)
two_gap_weight_ratio = FittingParameter(
    "weight ration two gap model", "x", default_guess=0.5, bounds=(0.0, 1.0)
)
anisotropy = FittingParameter("anisotropy", "epsilon", default_guess=0.5, bounds=(-10.0, 0.99))


def test_shown_gap_ration_is_doubled():
    assert alpha_bcs.shown(alpha_bcs_const) == pytest.approx(alpha_bcs_const * 2)


def test_gap_ratio_converts_correctly():
    assert alpha_bcs.from_shown(alpha_bcs.shown(alpha_bcs_const)) == pytest.approx(alpha_bcs_const)


@pytest.mark.parametrize("parameter", [two_gap_weight_ratio, anisotropy])
def test_shape_parameters_are_not_scalled(parameter):
    test_value = 0.2
    assert parameter.shown(test_value) == test_value


@pytest.mark.parametrize("parameter", [two_gap_weight_ratio, anisotropy])
def test_every_gap_ratio_is_scaled_correctly(parameter):
    assert parameter.scale == 1.0


def test_shown_bounds_are_scaled():
    assert alpha_bcs.shown_bounds() == pytest.approx((0.02, 10.0))
    assert two_gap_weight_ratio.shown_bounds() == (0.0, 1.0)


def test_default_guess_outside_bounds_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        FittingParameter("test-parameter-name", "test-parameter-symbol", 5.0, bounds=(0.0, 1.0))


def test_inverted_bounds_are_rejected():
    with pytest.raises(ValueError, match="low < high"):
        FittingParameter("test-parameter-name", "test-parameter-symbol", 1.0, bounds=(1.0, 0.0))


def test_scale_parameter_is_frozen():
    with pytest.raises(AttributeError):
        alpha_bcs.scale = 1.0
