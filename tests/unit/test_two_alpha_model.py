from hydrochoerus.consts import ALPHA_BCS
from hydrochoerus.model.models import ModelType, ParameterType
from hydrochoerus.model.two_alpha_model import TwoAlpha


def test_should_have_two_alpha_type():
    expected = ModelType.TWO_ALPHA
    assert TwoAlpha().type is expected


def test_should_list_parameters_in_order():
    expected = ("alpha1", "alpha2", "x")
    assert TwoAlpha.parameter_names() == expected


def test_should_hold_defaults():
    expected = TwoAlpha(alpha1=ALPHA_BCS, alpha2=ALPHA_BCS, x=1.0)
    assert TwoAlpha() == expected


def test_should_bound_gaps():
    expected = ([0.01, 0.01], [20.0, 20.0])
    assert TwoAlpha.bounds(["alpha1", "alpha2"]) == expected


def test_should_bound_weight_between_0_and_1():
    expected = ([0.0], [1.0])
    assert TwoAlpha.bounds(["x"]) == expected


def test_should_mark_gaps_as_gap():
    expected = (ParameterType.GAP, ParameterType.GAP)
    assert (TwoAlpha.ptype("alpha1"), TwoAlpha.ptype("alpha2")) == expected


def test_should_mark_x_as_weight():
    expected = ParameterType.WEIGHT
    assert TwoAlpha.ptype("x") is expected


def test_should_swap_gaps_and_weight_when_alpha1_is_smaller():
    expected = TwoAlpha(alpha1=2.2, alpha2=0.6, x=0.75)
    assert TwoAlpha(alpha1=0.6, alpha2=2.2, x=0.25).normalised() == expected


def test_should_keep_itself_when_alpha1_is_larger():
    model = TwoAlpha(alpha1=2.2, alpha2=0.6, x=0.25)
    assert model.normalised() is model


def test_should_keep_itself_when_gaps_are_equal():
    model = TwoAlpha(alpha1=1.0, alpha2=1.0, x=0.25)
    assert model.normalised() is model
